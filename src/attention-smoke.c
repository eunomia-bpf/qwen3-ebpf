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
    struct bpf_program *cached;
    struct bpf_map *map, *kv_map;
    struct qwen3_attention_state work = {0};
    struct qwen3_kv_pair pair = {0};
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
    int step_fd, cached_fd, map_fd, kv_fd, token, i, length, rc = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s build/qwen3_attention.bpf.o\n", argv[0]);
        return 2;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    kv_map = bpf_object__find_map_by_name(obj, "kv");
    if (!kv_map || bpf_map__set_max_entries(kv_map,
            16 * QWEN3_ATTENTION_LAYERS * QWEN3_ATTENTION_KV_HEADS))
        goto done;
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF attention program failed\n");
        goto done;
    }
    step = bpf_object__find_program_by_name(obj, "qwen3_attention_step");
    cached = bpf_object__find_program_by_name(obj, "qwen3_attention_cached");
    map = bpf_object__find_map_by_name(obj, "attention");
    if (!step || !cached || !map) {
        fprintf(stderr, "BPF attention program or map missing\n");
        goto done;
    }
    step_fd = bpf_program__fd(step);
    cached_fd = bpf_program__fd(cached);
    map_fd = bpf_map__fd(map);
    kv_fd = bpf_map__fd(kv_map);
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work.query_q16[i] = ((i * 7) % 31 - 15) * 32768;
    for (length = 1; length <= 16; length *= 2) {
        work.seen = 0;
        work.mass_q16 = 0;
        max_error = 0;
        for (token = 0; token < length; token++) {
            uint32_t slot = (uint32_t)token * QWEN3_ATTENTION_LAYERS *
                            QWEN3_ATTENTION_KV_HEADS;
            scores[token] = 0;
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                work.key_q16[i] = ((i * (token + 3)) % 29 - 14) * 16384;
                work.value_q16[i] = ((i * (token + 5)) % 23 - 11) * 8192;
                values[token][i] = work.value_q16[i] / 65536.0;
                scores[token] +=
                    (work.query_q16[i] / 65536.0) *
                    (work.key_q16[i] / 65536.0) / sqrt(128.0);
            }
            memcpy(pair.key_q16, work.key_q16, sizeof(pair.key_q16));
            memcpy(pair.value_q16, work.value_q16, sizeof(pair.value_q16));
            if (bpf_map_update_elem(kv_fd, &slot, &pair, BPF_ANY))
                goto done;
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
        {
            int32_t old_output[QWEN3_TILE_WIDTH];
            memcpy(old_output, work.output_q16, sizeof(old_output));
            work.seen = 0;
            work.mass_q16 = 0;
            work.layer = 0;
            work.kv_head = 0;
            work.base_position = 0;
            work.step_count = (uint32_t)length;
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                bpf_prog_test_run_opts(cached_fd, &opts) ||
                bpf_map_lookup_elem(map_fd, &key, &work) ||
                work.seen != (uint32_t)length ||
                memcmp(old_output, work.output_q16, sizeof(old_output))) {
                fprintf(stderr, "cached attention differs at length %d\n", length);
                goto done;
            }
        }
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
