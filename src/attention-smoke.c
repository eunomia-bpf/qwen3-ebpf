#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_attention.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *step;
    struct bpf_map *map;
    struct qwen3_attention_state work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    const uint32_t key = 0;
    double scores[16] = {0}, values[16][QWEN3_TILE_WIDTH] = {{0}};
    double max_error = 0;
    int step_fd, map_fd, token, i, length, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_attention.bpf.o\n", argv[0]);
        return 2;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF attention program failed\n");
        goto done;
    }
    step = bpf_object__find_program_by_name(obj, "qwen3_attention_step");
    map = bpf_object__find_map_by_name(obj, "attention");
    if (!step || !map) {
        fprintf(stderr, "BPF attention program or map missing\n");
        goto done;
    }
    step_fd = bpf_program__fd(step);
    map_fd = bpf_map__fd(map);
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work.query_q16[i] = ((i * 7) % 31 - 15) * 32768;
    for (length = 1; length <= 16; length *= 2) {
        work.seen = 0;
        work.mass_q16 = 0;
        max_error = 0;
        for (token = 0; token < length; token++) {
            scores[token] = 0;
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                work.key_q16[i] = ((i * (token + 3)) % 29 - 14) * 16384;
                work.value_q16[i] = ((i * (token + 5)) % 23 - 11) * 8192;
                values[token][i] = work.value_q16[i] / 65536.0;
                scores[token] +=
                    (work.query_q16[i] / 65536.0) *
                    (work.key_q16[i] / 65536.0) / sqrt(128.0);
            }
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                bpf_prog_test_run_opts(step_fd, &opts) ||
                bpf_map_lookup_elem(map_fd, &key, &work)) {
                perror("BPF attention step");
                goto done;
            }
        }
        if (work.seen != (uint32_t)length) {
            fprintf(stderr, "kernel attention processed wrong number of keys\n");
            goto done;
        }
        {
            double maximum = scores[0], denominator = 0;
            for (token = 1; token < length; token++)
                if (scores[token] > maximum)
                    maximum = scores[token];
            for (token = 0; token < length; token++)
                denominator += exp(scores[token] - maximum);
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                double expected = 0;
                for (token = 0; token < length; token++)
                    expected += exp(scores[token] - maximum) * values[token][i];
                expected /= denominator;
                double error = fabs(work.output_q16[i] / 65536.0 - expected);
                if (error > max_error) max_error = error;
            }
        }
        printf("kernel %d-token attention: max_abs_error=%.9g\n",
               length, max_error);
        if (max_error > 0.005) {
            fprintf(stderr, "kernel attention exceeds one-head error budget\n");
            goto done;
        }
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
