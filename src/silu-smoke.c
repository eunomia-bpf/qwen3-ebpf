#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_silu.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_silu_state work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    const uint32_t key = 0;
    double max_error = 0;
    int map_fd, prog_fd, tile_index, i, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_silu.bpf.o\n", argv[0]);
        return 2;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF object failed\n");
        goto done;
    }
    prog = bpf_object__find_program_by_name(obj, "qwen3_silu_apply");
    map = bpf_object__find_map_by_name(obj, "silu");
    if (!prog || !map) {
        fprintf(stderr, "BPF SiLU program or map missing\n");
        goto done;
    }
    prog_fd = bpf_program__fd(prog);
    map_fd = bpf_map__fd(map);
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int index = tile_index * QWEN3_TILE_WIDTH + i;
            work.input_q16[i] = ((index % 257) - 128) * 4096;
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
            bpf_prog_test_run_opts(prog_fd, &opts) ||
            bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("BPF SiLU");
            goto done;
        }
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            double x = work.input_q16[i] / 65536.0;
            double reference = x / (1.0 + exp(-x));
            double error = fabs(work.output_q16[i] / 65536.0 - reference);
            if (error > max_error)
                max_error = error;
        }
    }
    printf("kernel SiLU: 1024 inputs in [-8,8], max_abs_error=%.9g\n", max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel SiLU exceeds one-vector error budget\n");
        goto done;
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
