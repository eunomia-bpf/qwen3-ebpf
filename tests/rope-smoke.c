#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_rope.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_rope_state work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    const uint32_t key = 0;
    const int positions[] = {0, 1, 7, 63};
    const int head_counts[] = {1, QWEN3_ROPE_HEADS};
    double max_error = 0;
    int prog_fd, map_fd, p, h, head, i, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_rope.bpf.o\n", argv[0]);
        return 2;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF RoPE program failed\n");
        goto done;
    }
    prog = bpf_object__find_program_by_name(obj, "qwen3_rope_apply");
    map = bpf_object__find_map_by_name(obj, "rope");
    if (!prog || !map) {
        fprintf(stderr, "BPF RoPE program or map missing\n");
        goto done;
    }
    prog_fd = bpf_program__fd(prog);
    map_fd = bpf_map__fd(map);
    for (h = 0; h < (int)(sizeof(head_counts) / sizeof(head_counts[0])); h++) {
        for (p = 0; p < (int)(sizeof(positions) / sizeof(positions[0])); p++) {
            work.heads = (uint32_t)head_counts[h];
            work.completed = 0;
            for (head = 0; head < head_counts[h]; head++)
                for (i = 0; i < QWEN3_TILE_WIDTH; i++)
                    work.input_q16[head * QWEN3_TILE_WIDTH + i] =
                        (i - 64 + head) * 2048;
            for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
                double angle = positions[p] * pow(1000000.0, -(double)i / 64.0);
                work.cosine_q20[i] = (int32_t)round(cos(angle) * (1 << 20));
                work.sine_q20[i] = (int32_t)round(sin(angle) * (1 << 20));
            }
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                bpf_prog_test_run_opts(prog_fd, &opts) ||
                bpf_map_lookup_elem(map_fd, &key, &work)) {
                perror("BPF RoPE");
                goto done;
            }
            if (work.completed != work.heads) {
                fprintf(stderr, "BPF RoPE completed %u of %u heads\n",
                        work.completed, work.heads);
                goto done;
            }
            for (head = 0; head < head_counts[h]; head++) {
                int base = head * QWEN3_TILE_WIDTH;
                for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
                    double angle = positions[p] * pow(1000000.0, -(double)i / 64.0);
                    double c = cos(angle), s = sin(angle);
                    double x = work.input_q16[base + i] / 65536.0;
                    double y = work.input_q16[base + i + QWEN3_TILE_WIDTH / 2] / 65536.0;
                    double first = x * c - y * s;
                    double second = y * c + x * s;
                    double error_first = fabs(work.output_q16[base + i] / 65536.0 - first);
                    double error_second = fabs(work.output_q16[base + i + QWEN3_TILE_WIDTH / 2] /
                                               65536.0 - second);
                    if (error_first > max_error) max_error = error_first;
                    if (error_second > max_error) max_error = error_second;
                }
            }
        }
    }
    printf("kernel RoPE: 1 and 24 heads at positions 0,1,7,63; max_abs_error=%.9g\n", max_error);
    if (max_error > 0.0001) {
        fprintf(stderr, "kernel RoPE exceeds one-vector error budget\n");
        goto done;
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
