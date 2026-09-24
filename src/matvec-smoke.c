#include <errno.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_tile.h"
#include "safetensors.h"

static int load_qwen_q_proj_row(const char *path, int8_t *quantized,
                                float *original, float *scale)
{
    float max_abs = 0;
    int i;

    if (safetensors_read_bf16(path,
            "model.layers.0.self_attn.q_proj.weight", 0,
            QWEN3_HIDDEN_SIZE, original))
        return -1;
    for (i = 0; i < QWEN3_HIDDEN_SIZE; i++) {
        float magnitude;
        magnitude = original[i] < 0 ? -original[i] : original[i];
        if (magnitude > max_abs)
            max_abs = magnitude;
    }
    if (max_abs == 0 || max_abs > 1000)
        return -1;
    *scale = max_abs / 127.0f;
    for (i = 0; i < QWEN3_HIDDEN_SIZE; i++) {
        float value = original[i] / *scale;
        int rounded = (int)(value + (value >= 0 ? 0.5f : -0.5f));
        if (rounded > 127)
            rounded = 127;
        if (rounded < -127)
            rounded = -127;
        quantized[i] = (int8_t)rounded;
    }
    return 0;
}

int main(int argc, char **argv)
{
    struct bpf_object *obj = NULL;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_tile work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    int8_t model_row[QWEN3_HIDDEN_SIZE];
    float model_original[QWEN3_HIDDEN_SIZE], original_reference = 0;
    float model_scale = 0;
    int64_t expected = 0;
    const uint32_t key = 0;
    int prog_fd, map_fd, tile_index, i, mode, rc = 1;

    if ((argc != 2 && argc != 3 && argc != 4) ||
        (argc == 4 && strcmp(argv[3], "--q24") != 0)) {
        fprintf(stderr, "usage: %s build/qwen3_matvec.bpf.o [model.safetensors [--q24]]\n", argv[0]);
        return 2;
    }
    mode = argc == 4 ? 2 : (argc == 3 ? 1 : 0);
    if (mode && load_qwen_q_proj_row(argv[2], model_row,
                                        model_original, &model_scale)) {
        fprintf(stderr, "could not load Qwen3 layer-0 Q projection row from %s\n", argv[2]);
        return 1;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF object failed: %s\n", strerror(errno));
        goto done;
    }
    prog = bpf_object__find_program_by_name(obj, "qwen3_matvec_tile");
    map = bpf_object__find_map_by_name(obj, "tile");
    if (!prog || !map) {
        fprintf(stderr, "BPF program or map missing\n");
        goto done;
    }
    prog_fd = bpf_program__fd(prog);
    map_fd = bpf_map__fd(map);
    if (mode) {
        work.fixed_point_mode = mode;
        work.weight_scale_q24 = (int32_t)(model_scale * (1 << 24) + 0.5f);
    }
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int8_t activation = (int8_t)((tile_index * 17 + i) % 21 - 10);
            work.activation[i] = activation;
            work.activation_q16[i] = (int32_t)activation * (1 << 16);
            float raw_weight = mode
                ? model_original[tile_index * QWEN3_TILE_WIDTH + i] : 0;
            work.weight[i] = mode
                ? model_row[tile_index * QWEN3_TILE_WIDTH + i]
                : (int8_t)((tile_index * 7 + i * 3) % 19 - 9);
            work.weight_q24[i] = (int32_t)(raw_weight * (1 << 24) +
                (raw_weight >= 0 ? 0.5f : -0.5f));
            expected += (int64_t)(mode ? work.activation_q16[i] : activation)
                        * (mode == 2 ? work.weight_q24[i] : work.weight[i]);
            if (mode)
                original_reference += activation * raw_weight;
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY)) {
            perror("bpf_map_update_elem");
            goto done;
        }
        if (bpf_prog_test_run_opts(prog_fd, &opts)) {
            perror("bpf_prog_test_run_opts");
            goto done;
        }
        if (bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("bpf_map_lookup_elem");
            goto done;
        }
    }
    if (work.accumulator != expected ||
        work.completed_tiles != QWEN3_HIDDEN_TILES) {
        fprintf(stderr, "kernel mismatch: got %lld/%u, expected %lld/%u\n",
                (long long)work.accumulator, work.completed_tiles,
                (long long)expected, QWEN3_HIDDEN_TILES);
        goto done;
    }
    if (mode && work.output_q16 !=
        (mode == 2 ? expected >> 24
                   : (expected * work.weight_scale_q24) >> 24)) {
        fprintf(stderr, "kernel fixed-point scale mismatch\n");
        goto done;
    }
    printf("kernel matvec mode=%d: %u MACs, sum=%lld, tiles=%u (matches C reference)%s\n",
           mode, QWEN3_HIDDEN_SIZE, (long long)work.accumulator,
           work.completed_tiles, mode ? "; actual Qwen3 layer-0 Q-projection row" : "");
    if (mode)
        printf("output Q16: %lld (%.9g); original BF16 reference: %.9g; Q8 weight scale: %.9g\n",
               (long long)work.output_q16, work.output_q16 / 65536.0,
               original_reference, model_scale);
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
