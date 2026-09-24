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
    struct bpf_program *score, *apply;
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
    double reference_score[2] = {0}, max_error = 0;
    int score_fd, apply_fd, map_fd, token, i, rc = 1;

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
        fprintf(stderr, "loading BPF attention programs failed\n");
        goto done;
    }
    score = bpf_object__find_program_by_name(obj, "qwen3_attention_score");
    apply = bpf_object__find_program_by_name(obj, "qwen3_attention_apply");
    map = bpf_object__find_map_by_name(obj, "attention");
    if (!score || !apply || !map) {
        fprintf(stderr, "BPF attention program or map missing\n");
        goto done;
    }
    score_fd = bpf_program__fd(score);
    apply_fd = bpf_program__fd(apply);
    map_fd = bpf_map__fd(map);
    work.context_length = 2;
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work.query_q16[i] = ((i * 7) % 31 - 15) * 32768;
    for (token = 0; token < 2; token++) {
        work.score_index = token;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            work.key_q16[i] = ((i * (token + 3)) % 29 - 14) * 16384;
            work.value_q16[token][i] =
                ((i * (token + 5)) % 23 - 11) * 8192;
            reference_score[token] +=
                (work.query_q16[i] / 65536.0) *
                (work.key_q16[i] / 65536.0) / sqrt(128.0);
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
            bpf_prog_test_run_opts(score_fd, &opts) ||
            bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("BPF attention score");
            goto done;
        }
    }
    if (bpf_prog_test_run_opts(apply_fd, &opts) ||
        bpf_map_lookup_elem(map_fd, &key, &work)) {
        perror("BPF attention apply");
        goto done;
    }
    {
        double maximum = reference_score[0] > reference_score[1]
            ? reference_score[0] : reference_score[1];
        double e0 = exp(reference_score[0] - maximum);
        double e1 = exp(reference_score[1] - maximum);
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            double expected =
                (e0 * work.value_q16[0][i] + e1 * work.value_q16[1][i]) /
                (e0 + e1) / 65536.0;
            double error = fabs(work.output_q16[i] / 65536.0 - expected);
            if (error > max_error) max_error = error;
        }
    }
    printf("kernel 2-token attention: scores %.6g/%.6g, max_abs_error=%.9g\n",
           work.score_q16[0] / 65536.0, work.score_q16[1] / 65536.0,
           max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel attention exceeds one-head error budget\n");
        goto done;
    }
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
