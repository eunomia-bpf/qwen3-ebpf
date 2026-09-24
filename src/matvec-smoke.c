#include <errno.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_tile.h"

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
    int64_t expected = 0;
    const uint32_t key = 0;
    int prog_fd, map_fd, tile_index, i, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_matvec.bpf.o\n", argv[0]);
        return 2;
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
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            work.activation[i] = (int8_t)((tile_index * 17 + i) % 21 - 10);
            work.weight[i] = (int8_t)((tile_index * 7 + i * 3) % 19 - 9);
            expected += (int64_t)work.activation[i] * work.weight[i];
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
    printf("kernel Q8 matvec: %u MACs, sum=%lld, tiles=%u (matches C reference)\n",
           QWEN3_HIDDEN_SIZE, (long long)work.accumulator, work.completed_tiles);
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
