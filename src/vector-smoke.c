#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_vector.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *add, *multiply, *argmax;
    struct bpf_map *map;
    struct qwen3_vector_state work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    const uint32_t key = 0;
    int map_fd, i, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_vector.bpf.o\n", argv[0]);
        return 2;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF vector programs failed\n");
        goto done;
    }
    add = bpf_object__find_program_by_name(obj, "qwen3_vector_add");
    multiply = bpf_object__find_program_by_name(obj, "qwen3_vector_multiply");
    argmax = bpf_object__find_program_by_name(obj, "qwen3_vector_argmax");
    map = bpf_object__find_map_by_name(obj, "vector");
    if (!add || !multiply || !argmax || !map) {
        fprintf(stderr, "BPF vector program or map missing\n");
        goto done;
    }
    map_fd = bpf_map__fd(map);
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        work.left_q16[i] = (i - 64) * 65536;
        work.right_q16[i] = (i % 7 - 3) * 32768;
    }
    if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
        bpf_prog_test_run_opts(bpf_program__fd(add), &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work)) {
        perror("BPF vector add");
        goto done;
    }
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        if (work.output_q16[i] != work.left_q16[i] + work.right_q16[i]) {
            fprintf(stderr, "BPF vector add mismatch at %d\n", i);
            goto done;
        }
    }
    if (bpf_prog_test_run_opts(bpf_program__fd(multiply), &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work)) {
        perror("BPF vector multiply");
        goto done;
    }
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        int32_t expected = (int32_t)(((int64_t)work.left_q16[i] *
                                      work.right_q16[i]) >> 16);
        if (work.output_q16[i] != expected) {
            fprintf(stderr, "BPF vector multiply mismatch at %d\n", i);
            goto done;
        }
    }
    work.best_q16 = INT_MIN;
    work.base_index = 256;
    if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
        bpf_prog_test_run_opts(bpf_program__fd(argmax), &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work)) {
        perror("BPF vector argmax");
        goto done;
    }
    if (work.best_index != 383 || work.best_q16 != work.left_q16[127]) {
        fprintf(stderr, "BPF vector argmax mismatch\n");
        goto done;
    }
    printf("kernel vector add/multiply/argmax: 128 elements (matches C assertions)\n");
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
