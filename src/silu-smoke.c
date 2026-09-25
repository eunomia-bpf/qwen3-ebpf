#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_silu.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_silu_state *work;
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    double max_error = 0;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(*work) + page - 1) / page) * page;
    const int counts[] = {QWEN3_TILE_WIDTH, QWEN3_SILU_MAX};
    int map_fd, prog_fd, count, case_index, i, rc = 1;

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
    work = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, map_fd, 0);
    if (work == MAP_FAILED) {
        perror("BPF SiLU mmap");
        goto done;
    }
    for (case_index = 0; case_index < 2; case_index++) {
        count = counts[case_index];
        for (i = 0; i < count; i++)
            work->input_q16[i] = ((i % 257) - 128) * 4096;
        work->count = (uint32_t)count;
        work->completed_tiles = 0;
        if (bpf_prog_test_run_opts(prog_fd, &opts) ||
            work->completed_tiles != (uint32_t)(count / QWEN3_TILE_WIDTH)) {
            perror("BPF SiLU");
            goto unmap;
        }
        for (i = 0; i < count; i++) {
            double x = work->input_q16[i] / 65536.0;
            double reference = x / (1.0 + exp(-x));
            double error = fabs(work->output_q16[i] / 65536.0 - reference);
            if (error > max_error)
                max_error = error;
        }
    }
    printf("kernel SiLU: 128 and 3072 inputs in [-8,8], max_abs_error=%.9g\n", max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel SiLU exceeds one-vector error budget\n");
        goto unmap;
    }
    rc = 0;
unmap:
    munmap(work, length);
done:
    bpf_object__close(obj);
    return rc;
}
