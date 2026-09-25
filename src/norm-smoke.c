#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_norm.h"
#include "safetensors.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj = NULL;
    struct bpf_program *full_prog;
    struct bpf_program *heads_prog;
    struct bpf_map *map;
    struct bpf_map *heads_map;
    struct qwen3_norm_state work = {0};
    struct qwen3_qk_norm_state *heads = NULL;
    int32_t *expected = NULL;
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    float weights[QWEN3_HIDDEN_SIZE];
    double sum_sq = 0, max_error = 0;
    const uint32_t key = 0;
    int full_fd, map_fd, i, count, head, set, rc = 1;

    if ((argc != 3 && argc != 4) ||
        (argc == 4 && strcmp(argv[3], "--qk") != 0)) {
        fprintf(stderr, "usage: %s build/qwen3_norm.bpf.o model.safetensors [--qk]\n", argv[0]);
        return 2;
    }
    count = argc == 4 ? QWEN3_TILE_WIDTH : QWEN3_HIDDEN_SIZE;
    if (safetensors_read_bf16(argv[2],
            argc == 4 ? "model.layers.0.self_attn.q_norm.weight"
                      : "model.layers.0.input_layernorm.weight",
            0, (size_t)count, weights)) {
        fprintf(stderr, "could not load Qwen3 layer-0 RMSNorm weights\n");
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
    full_prog = bpf_object__find_program_by_name(obj, "qwen3_rms_full");
    heads_prog = bpf_object__find_program_by_name(obj, "qwen3_qk_norm_heads");
    map = bpf_object__find_map_by_name(obj, "norm");
    heads_map = bpf_object__find_map_by_name(obj, "qk_norm");
    if (!full_prog || !heads_prog || !map || !heads_map) {
        fprintf(stderr, "BPF RMSNorm program or map missing\n");
        goto done;
    }
    full_fd = bpf_program__fd(full_prog);
    map_fd = bpf_map__fd(map);
    work.total_tiles = count / QWEN3_TILE_WIDTH;
    for (i = 0; i < count; i++) {
        int value = ((i / QWEN3_TILE_WIDTH) * 17 +
                     i % QWEN3_TILE_WIDTH) % 21 - 10;
        float w = weights[i];
        work.activation_q16[i] = value * (1 << 16);
        work.weight_q20[i] = (int32_t)(w * (1 << 20) +
            (w >= 0 ? 0.5f : -0.5f));
        sum_sq += (double)value * value;
    }
    if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
        bpf_prog_test_run_opts(full_fd, &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work) ||
        work.completed_tiles != 2 * work.total_tiles ||
        !work.inv_rms_q16) {
        fprintf(stderr, "kernel did not complete RMSNorm\n");
        goto done;
    }
    for (i = 0; i < count; i++) {
        int value = ((i / QWEN3_TILE_WIDTH) * 17 +
                     i % QWEN3_TILE_WIDTH) % 21 - 10;
        double reference = value * weights[i] /
            sqrt(sum_sq / count + 1e-6);
        double observed = work.output_q16[i] / 65536.0;
        double error = fabs(observed - reference);
        if (error > max_error)
            max_error = error;
    }
    printf("kernel RMSNorm: %d elements, inv_rms_q16=%llu, max_abs_error=%.9g\n",
           count, (unsigned long long)work.inv_rms_q16, max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel RMSNorm exceeds one-vector error budget\n");
        goto done;
    }
    heads = calloc(1, sizeof(*heads));
    expected = calloc(QWEN3_QK_HEADS * QWEN3_TILE_WIDTH, sizeof(*expected));
    if (!heads || !expected)
        goto done;
    for (set = 0; set < 2; set++) {
        if (safetensors_read_bf16(argv[2],
                set ? "model.layers.0.self_attn.k_norm.weight"
                    : "model.layers.0.self_attn.q_norm.weight",
                0, QWEN3_TILE_WIDTH, weights))
            goto done;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            heads->weight_q20[set][i] = (int32_t)(
                weights[i] * (1 << 20) +
                (weights[i] >= 0 ? 0.5f : -0.5f));
    }
    for (head = 0; head < QWEN3_QK_HEADS; head++) {
        memset(&work, 0, sizeof(work));
        work.total_tiles = 1;
        set = head >= QWEN3_Q_HEADS;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            work.activation_q16[i] =
                (((head * 17 + i) % 21) - 10) * (1 << 16);
            heads->activation_q16[head * QWEN3_TILE_WIDTH + i] =
                work.activation_q16[i];
            work.weight_q20[i] = heads->weight_q20[set][i];
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
            bpf_prog_test_run_opts(full_fd, &opts) ||
            bpf_map_lookup_elem(map_fd, &key, &work) ||
            work.completed_tiles != 2)
            goto done;
        memcpy(expected + head * QWEN3_TILE_WIDTH, work.output_q16,
               QWEN3_TILE_WIDTH * sizeof(*expected));
    }
    if (bpf_map_update_elem(bpf_map__fd(heads_map), &key, heads, BPF_ANY) ||
        bpf_prog_test_run_opts(bpf_program__fd(heads_prog), &opts) ||
        bpf_map_lookup_elem(bpf_map__fd(heads_map), &key, heads) ||
        heads->completed_heads != QWEN3_QK_HEADS ||
        memcmp(heads->output_q16, expected,
               QWEN3_QK_HEADS * QWEN3_TILE_WIDTH * sizeof(*expected))) {
        fprintf(stderr, "batched Q/K RMSNorm differs from per-head BPF\n");
        goto done;
    }
    puts("batched Q/K RMSNorm: 24 heads match per-head BPF exactly");
    memset(&work, 0, sizeof(work));
    work.total_tiles = QWEN3_HIDDEN_TILES + 1;
    if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
        bpf_prog_test_run_opts(full_fd, &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work) ||
        work.completed_tiles || work.inv_rms_q16) {
        fprintf(stderr, "kernel accepted oversized RMSNorm input\n");
        goto done;
    }
    rc = 0;
done:
    free(heads);
    free(expected);
    bpf_object__close(obj);
    return rc;
}
