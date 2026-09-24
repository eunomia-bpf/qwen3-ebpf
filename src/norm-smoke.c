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
    struct bpf_program *accum_prog, *finalize_prog, *apply_prog;
    struct bpf_map *map;
    struct qwen3_norm_state work = {0};
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
    int accum_fd, finalize_fd, apply_fd, map_fd, tile_index, i, rc = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: %s build/qwen3_norm.bpf.o model.safetensors\n", argv[0]);
        return 2;
    }
    if (safetensors_read_bf16(argv[2],
            "model.layers.0.input_layernorm.weight", 0,
            QWEN3_HIDDEN_SIZE, weights)) {
        fprintf(stderr, "could not load Qwen3 layer-0 input RMSNorm weights\n");
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
    accum_prog = bpf_object__find_program_by_name(obj, "qwen3_rms_accumulate");
    finalize_prog = bpf_object__find_program_by_name(obj, "qwen3_rms_finalize");
    apply_prog = bpf_object__find_program_by_name(obj, "qwen3_rms_apply");
    map = bpf_object__find_map_by_name(obj, "norm");
    if (!accum_prog || !finalize_prog || !apply_prog || !map) {
        fprintf(stderr, "BPF RMSNorm programs or map missing\n");
        goto done;
    }
    accum_fd = bpf_program__fd(accum_prog);
    finalize_fd = bpf_program__fd(finalize_prog);
    apply_fd = bpf_program__fd(apply_prog);
    map_fd = bpf_map__fd(map);
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int value = (tile_index * 17 + i) % 21 - 10;
            work.activation_q16[i] = value * (1 << 16);
            sum_sq += (double)value * value;
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
            bpf_prog_test_run_opts(accum_fd, &opts) ||
            bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("BPF RMSNorm accumulation");
            goto done;
        }
    }
    if (work.completed_tiles != QWEN3_HIDDEN_TILES ||
        bpf_prog_test_run_opts(finalize_fd, &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work) || !work.inv_rms_q16) {
        fprintf(stderr, "kernel did not finalize RMSNorm\n");
        goto done;
    }
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int index = tile_index * QWEN3_TILE_WIDTH + i;
            int value = (tile_index * 17 + i) % 21 - 10;
            float w = weights[index];
            work.activation_q16[i] = value * (1 << 16);
            work.weight_q20[i] = (int32_t)(w * (1 << 20) +
                (w >= 0 ? 0.5f : -0.5f));
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
            bpf_prog_test_run_opts(apply_fd, &opts) ||
            bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("BPF RMSNorm apply");
            goto done;
        }
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int index = tile_index * QWEN3_TILE_WIDTH + i;
            int value = (tile_index * 17 + i) % 21 - 10;
            double reference = value * weights[index] /
                sqrt(sum_sq / QWEN3_HIDDEN_SIZE + 1e-6);
            double observed = work.output_q16[i] / 65536.0;
            double error = fabs(observed - reference);
            if (error > max_error)
                max_error = error;
        }
    }
    printf("kernel RMSNorm: 1024 elements, inv_rms_q16=%llu, max_abs_error=%.9g\n",
           (unsigned long long)work.inv_rms_q16, max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel RMSNorm exceeds one-vector error budget\n");
        goto done;
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
