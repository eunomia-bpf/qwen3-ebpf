#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_attention.h"

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *step;
    struct bpf_program *cached;
    struct bpf_program *all_heads;
    struct bpf_map *map, *kv_map, *all_heads_map;
    struct qwen3_attention_state work = {0};
    struct qwen3_attention_heads_state *heads = NULL;
    int32_t *expected_heads = NULL;
    struct qwen3_kv_pair pair = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    const uint32_t key = 0;
    const int lengths[] = {1, 2, 4, 8, 16, QWEN3_ATTENTION_CHUNK + 1};
    double scores[QWEN3_ATTENTION_CHUNK + 1] = {0};
    double values[QWEN3_ATTENTION_CHUNK + 1][QWEN3_TILE_WIDTH] = {{0}};
    double max_error = 0;
    int step_fd, cached_fd, map_fd, kv_fd, token, i, length, case_index;
    int head, kv_head, all_heads_fd, all_heads_map_fd, start, rc = 1;

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
            (QWEN3_ATTENTION_CHUNK + 1) * QWEN3_ATTENTION_LAYERS *
            QWEN3_ATTENTION_KV_HEADS))
        goto done;
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF attention program failed\n");
        goto done;
    }
    step = bpf_object__find_program_by_name(obj, "qwen3_attention_step");
    cached = bpf_object__find_program_by_name(obj, "qwen3_attention_cached");
    all_heads = bpf_object__find_program_by_name(obj, "qwen3_attention_all_heads");
    map = bpf_object__find_map_by_name(obj, "attention");
    all_heads_map = bpf_object__find_map_by_name(obj, "attention_heads");
    if (!step || !cached || !all_heads || !map || !all_heads_map) {
        fprintf(stderr, "BPF attention program or map missing\n");
        goto done;
    }
    step_fd = bpf_program__fd(step);
    cached_fd = bpf_program__fd(cached);
    all_heads_fd = bpf_program__fd(all_heads);
    map_fd = bpf_map__fd(map);
    all_heads_map_fd = bpf_map__fd(all_heads_map);
    kv_fd = bpf_map__fd(kv_map);
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work.query_q16[i] = ((i * 7) % 31 - 15) * 32768;
    for (case_index = 0; case_index < (int)(sizeof(lengths) / sizeof(lengths[0]));
         case_index++) {
        length = lengths[case_index];
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
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY))
                goto done;
            for (token = 0; token < length; token += QWEN3_ATTENTION_CHUNK) {
                work.base_position = (uint32_t)token;
                work.step_count = (uint32_t)(length - token < QWEN3_ATTENTION_CHUNK
                    ? length - token : QWEN3_ATTENTION_CHUNK);
                if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                    bpf_prog_test_run_opts(cached_fd, &opts) ||
                    bpf_map_lookup_elem(map_fd, &key, &work) ||
                    work.seen != (uint32_t)(token + work.step_count)) {
                    fprintf(stderr, "cached attention stopped at position %d\n", token);
                    goto done;
                }
            }
            if (work.seen != (uint32_t)length ||
                memcmp(old_output, work.output_q16, sizeof(old_output))) {
                fprintf(stderr, "cached attention differs at length %d\n", length);
                goto done;
            }
        }
    }
    heads = calloc(1, sizeof(*heads));
    expected_heads = calloc(QWEN3_Q_HEADS * QWEN3_TILE_WIDTH,
                            sizeof(*expected_heads));
    if (!heads || !expected_heads)
        goto done;
    length = QWEN3_ATTENTION_CHUNK + 1;
    for (token = 0; token < length; token++) {
        for (kv_head = 0; kv_head < QWEN3_KV_HEADS; kv_head++) {
            uint32_t slot = (uint32_t)token * QWEN3_ATTENTION_LAYERS *
                            QWEN3_ATTENTION_KV_HEADS + kv_head;
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                pair.key_q16[i] =
                    ((i * (token + 3) + kv_head * 7) % 29 - 14) * 16384;
                pair.value_q16[i] =
                    ((i * (token + 5) + kv_head * 11) % 23 - 11) * 8192;
            }
            if (bpf_map_update_elem(kv_fd, &slot, &pair, BPF_ANY))
                goto done;
        }
    }
    for (head = 0; head < QWEN3_Q_HEADS; head++) {
        memset(&work, 0, sizeof(work));
        work.kv_head = (uint32_t)(head / 2);
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            work.query_q16[i] = ((i * 7 + head * 3) % 31 - 15) * 32768;
            heads->heads[head].query_q16[i] = work.query_q16[i];
        }
        for (start = 0; start < length; start += QWEN3_ATTENTION_CHUNK) {
            work.base_position = (uint32_t)start;
            work.step_count = (uint32_t)(length - start < QWEN3_ATTENTION_CHUNK
                ? length - start : QWEN3_ATTENTION_CHUNK);
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                bpf_prog_test_run_opts(cached_fd, &opts) ||
                bpf_map_lookup_elem(map_fd, &key, &work) ||
                work.seen != (uint32_t)(start + work.step_count))
                goto done;
        }
        memcpy(expected_heads + head * QWEN3_TILE_WIDTH,
               work.output_q16, sizeof(work.output_q16));
    }
    for (start = 0; start < length; start += QWEN3_ATTENTION_CHUNK) {
        heads->base_position = (uint32_t)start;
        heads->step_count = (uint32_t)(length - start < QWEN3_ATTENTION_CHUNK
            ? length - start : QWEN3_ATTENTION_CHUNK);
        if (bpf_map_update_elem(all_heads_map_fd, &key, heads, BPF_ANY) ||
            bpf_prog_test_run_opts(all_heads_fd, &opts) ||
            bpf_map_lookup_elem(all_heads_map_fd, &key, heads) ||
            heads->completed_positions != (uint32_t)(start + heads->step_count))
            goto done;
    }
    for (head = 0; head < QWEN3_Q_HEADS; head++) {
        if (heads->heads[head].seen != (uint32_t)length ||
            memcmp(heads->heads[head].output_q16,
                   expected_heads + head * QWEN3_TILE_WIDTH,
                   sizeof(heads->heads[head].output_q16))) {
            fprintf(stderr, "batched attention differs at head %d\n", head);
            goto done;
        }
    }
    puts("batched attention: 16 heads and 257 positions match per-head BPF");
    memset(heads, 0, sizeof(*heads));
    heads->store_kv = 1;
    heads->current_position = QWEN3_ATTENTION_CHUNK;
    heads->base_position = QWEN3_ATTENTION_CHUNK;
    heads->step_count = 1;
    for (kv_head = 0; kv_head < QWEN3_KV_HEADS; kv_head++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            heads->current_kv[kv_head].key_q16[i] =
                ((i + kv_head) % 17 - 8) * 8192;
            heads->current_kv[kv_head].value_q16[i] =
                (kv_head * 13 + i % 7) * 4096;
        }
    }
    if (bpf_map_update_elem(all_heads_map_fd, &key, heads, BPF_ANY) ||
        bpf_prog_test_run_opts(all_heads_fd, &opts) ||
        bpf_map_lookup_elem(all_heads_map_fd, &key, heads) ||
        heads->stored_kv != QWEN3_KV_HEADS ||
        heads->completed_positions != 1)
        goto done;
    for (kv_head = 0; kv_head < QWEN3_KV_HEADS; kv_head++) {
        uint32_t slot = (uint32_t)QWEN3_ATTENTION_CHUNK *
                        QWEN3_ATTENTION_LAYERS * QWEN3_ATTENTION_KV_HEADS +
                        (uint32_t)kv_head;
        if (bpf_map_lookup_elem(kv_fd, &slot, &pair) ||
            memcmp(&pair, &heads->current_kv[kv_head], sizeof(pair))) {
            fprintf(stderr, "kernel KV write differs at head %d\n", kv_head);
            goto done;
        }
        for (head = kv_head * 2; head < kv_head * 2 + 2; head++) {
            if (heads->heads[head].seen != 1 ||
                memcmp(heads->heads[head].output_q16, pair.value_q16,
                       sizeof(pair.value_q16)))
                goto done;
        }
    }
    puts("kernel KV write: eight heads stored and consumed exactly");
    rc = 0;
done:
    free(heads);
    free(expected_heads);
    bpf_object__close(obj);
    return rc;
}
