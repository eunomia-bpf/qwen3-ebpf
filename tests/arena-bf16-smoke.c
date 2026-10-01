#include <errno.h>
#include <limits.h>
#include <math.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_arena_bf16.h"
#include "qwen3_attention.h"
#include "qwen3_arena_bf16.skel.h"
#include "safetensors.h"

static const uint32_t rope_positions[4] = {0, 1, 128, 40959};

static int run_kernel(struct qwen3_arena_bf16_bpf *skel,
                      struct qwen3_arena_bf16_work *work, int rows)
{
    const __u8 packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };

    work->rows = (__u32)rows;
    work->completed = 0;
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_bf16_rows), &opts)) {
        perror("arena BF16 test-run");
        return -1;
    }
    return work->completed == (__u32)rows ? 0 : -1;
}

static void populate_lut(struct qwen3_arena_bf16_bpf *skel)
{
    uint32_t bits;

    for (bits = 0; bits <= UINT16_MAX; bits++) {
        uint32_t float_bits = bits << 16;
        float value;
        double scaled, scaled16, scaled20;

        memcpy(&value, &float_bits, sizeof(value));
        scaled = (double)value * 16777216.0;
        skel->arena->q24_by_bf16[bits] =
            !isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN
            ? 0 : (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
        scaled16 = (double)value * 65536.0;
        skel->arena->q16_by_bf16[bits] =
            !isfinite(scaled16) || scaled16 > INT32_MAX || scaled16 < INT32_MIN
            ? 0 : (int32_t)(scaled16 + (scaled16 >= 0 ? 0.5 : -0.5));
        scaled20 = (double)value * 1048576.0;
        skel->arena->q20_by_bf16[bits] =
            !isfinite(scaled20) || scaled20 > INT32_MAX || scaled20 < INT32_MIN
            ? 0 : (int32_t)(scaled20 + (scaled20 >= 0 ? 0.5 : -0.5));
    }
}

static uint64_t reference_isqrt64(uint64_t value)
{
    uint64_t root = 1ULL << 20;

    for (int i = 0; i < 24; i++) {
        root = (root + value / root) >> 1;
        root |= 1;
    }
    return root;
}

static uint64_t reference_exp_negative_q32(uint32_t delta_q16)
{
    uint64_t result;

    if (!delta_q16)
        return 1ULL << 32;
    if (delta_q16 >= (16U << 16))
        return 0;
    result = (1ULL << 32) - ((uint64_t)delta_q16 << 6);
    for (int i = 0; i < 10; i++)
        result = (result * result) >> 32;
    return result;
}

static int run_synthetic(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work)
{
    static const __u16 samples[] = {0x3d00, 0xbd00, 0x3c00, 0};
    int row, col;

    memset(work, 0, sizeof(*work));
    work->cols = QWEN3_ARENA_BF16_COLS;
    for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++)
        for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++)
            skel->arena->weight_bf16[row][col] =
                samples[(row + col) % (sizeof(samples) / sizeof(samples[0]))];
    for (int cols = QWEN3_ARENA_BF16_COLS;
         cols >= QWEN3_ARENA_BF16_COLS - 1; cols--) {
        work->cols = cols;
        if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
            return -1;
        for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
            __s64 expected = 0;
            for (col = 0; col < cols; col++) {
                __u16 bits = skel->arena->weight_bf16[row][col];
                expected += (__s64)work->input_q16[col] *
                            skel->arena->q24_by_bf16[bits];
            }
            if (work->output_q16[row] != expected >> 24) {
                fprintf(stderr, "arena BF16 row %d mismatch at %d columns\n",
                        row, cols);
                return -1;
            }
        }
    }
    return 0;
}

static int run_resident_synthetic(struct qwen3_arena_bf16_bpf *skel,
                                   struct qwen3_arena_bf16_work *work)
{
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    static const __u16 samples[] = {0x3d00, 0xbd00, 0x3c00, 0};
    int row, col;

    memset(work, 0, sizeof(*work));
    work->model_elements = 256;
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_allocate_model),
            &opts) || opts.retval || !skel->bss->model_bf16) {
        fprintf(stderr, "arena BF16 resident allocation failed\n");
        return -1;
    }
    work->cols = 128;
    work->resident_weights = 1;
    for (col = 0; col < 128; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < 2; row++)
        for (col = 0; col < 128; col++)
            skel->bss->model_bf16[row * 128 + col] =
                samples[(row + col) % 4];
    if (run_kernel(skel, work, 2))
        return -1;
    for (row = 0; row < 2; row++) {
        __s64 expected = 0;
        for (col = 0; col < 128; col++) {
            __u16 bits = skel->bss->model_bf16[row * 128 + col];
            expected += (__s64)work->input_q16[col] *
                        skel->arena->q24_by_bf16[bits];
        }
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "arena BF16 resident row %d mismatch\n", row);
            return -1;
        }
    }
    return 0;
}

static int run_xdp_event(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work,
                         const int32_t *expected, uint32_t total,
                         const int32_t *expected_inputs,
                         const int32_t *expected_embeddings,
                         const uint32_t *token_ids,
                         const int32_t *expected_cosine,
                         const int32_t *expected_sine,
                         const int32_t *expected_attention,
                         const int32_t *expected_projected,
                         const int32_t *expected_hidden,
                         const int32_t *expected_post_norm,
                         const int32_t *expected_gate,
                         const int32_t *expected_up,
                         const int32_t *expected_product,
                         const int32_t *expected_down)
{
    const __u32 key = 0;
    unsigned char payload[12] = {'Q', '3', 'B', 'P'};
    const char ignored[] = "Q3XX";
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    struct qwen3_event_state job = {0};
    struct bpf_link *link = NULL;
    struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port = htons(49002),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    struct timespec begin, now;
    __s64 elapsed_ns;
    unsigned int ifindex = if_nametoindex("lo");
    int fd = -1, rc = -1;
    uint32_t i, attempt;

    if (!ifindex || !total || total > QWEN3_ARENA_EVENT_OUTPUTS ||
        (work->event_use_token && (!expected_inputs ||
                                   !expected_embeddings || !token_ids))) {
        fprintf(stderr, "invalid loopback interface or matrix row count\n");
        return -1;
    }
    work->matrix_total_rows = total;
    if (bpf_prog_test_run_opts(bpf_program__fd(skel->progs.qwen3_event_init),
                               &opts) || opts.retval) {
        fprintf(stderr, "BPF workqueue initialization failed\n");
        return -1;
    }
    link = bpf_program__attach_xdp(skel->progs.qwen3_event_xdp, ifindex);
    if (!link || libbpf_get_error(link)) {
        fprintf(stderr, "XDP attach to container loopback failed: %ld\n",
                link ? libbpf_get_error(link) : -1L);
        link = NULL;
        goto done;
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("XDP event socket");
        goto done;
    }
    if (sendto(fd, ignored, sizeof(ignored) - 1, 0,
               (struct sockaddr *)&dst, sizeof(dst)) != sizeof(ignored) - 1 ||
        bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
        job.status != QWEN3_EVENT_READY || job.requests != 0) {
        fprintf(stderr, "unrecognized XDP packet started a request\n");
        goto done;
    }
    if (work->event_use_token) {
        uint32_t invalid_token = work->embedding_vocab;

        payload[4] = (unsigned char)(invalid_token >> 24);
        payload[5] = (unsigned char)(invalid_token >> 16);
        payload[6] = (unsigned char)(invalid_token >> 8);
        payload[7] = (unsigned char)invalid_token;
        if (sendto(fd, payload, 8, 0, (struct sockaddr *)&dst,
                   sizeof(dst)) != 8 ||
            bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
            job.status != QWEN3_EVENT_READY || job.requests != 0) {
            fprintf(stderr, "out-of-range token started a request\n");
            goto done;
        }
    }
    if (work->event_rope) {
        uint32_t invalid_position = 40960;

        payload[4] = payload[5] = payload[6] = payload[7] = 0;
        payload[8] = (unsigned char)(invalid_position >> 24);
        payload[9] = (unsigned char)(invalid_position >> 16);
        payload[10] = (unsigned char)(invalid_position >> 8);
        payload[11] = (unsigned char)invalid_position;
        if (sendto(fd, payload, sizeof(payload), 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != sizeof(payload) ||
            bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
            job.status != QWEN3_EVENT_READY || job.requests != 0) {
            fprintf(stderr, "out-of-range position started a request\n");
            goto done;
        }
    }
    if (work->event_attention) {
        payload[4] = payload[5] = payload[6] = payload[7] = 0;
        payload[8] = payload[9] = payload[10] = 0;
        payload[11] = 2;
        if (sendto(fd, payload, sizeof(payload), 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != sizeof(payload) ||
            bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
            job.status != QWEN3_EVENT_READY || job.requests != 0) {
            fprintf(stderr, "non-contiguous position started attention\n");
            goto done;
        }
        payload[10] = 1;
        payload[11] = 0;
        if (sendto(fd, payload, sizeof(payload), 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != sizeof(payload) ||
            bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
            job.status != QWEN3_EVENT_READY || job.requests != 0) {
            fprintf(stderr, "out-of-cache position started attention\n");
            goto done;
        }
    }
    for (attempt = 0; attempt < 10; attempt++) {
        uint32_t scenario = attempt % (work->event_attention ? 2u : 4u);
        uint32_t expected_stride = work->event_qkv ? 4096 : total;
        const int32_t *request_expected = expected +
            (work->event_rope ? scenario * expected_stride :
             work->event_use_token ? (attempt % 2) * expected_stride : 0);
        uint32_t token_id = work->event_use_token ? token_ids[attempt % 2] : 0;
        size_t payload_size = work->event_rope ? 12 :
                              work->event_use_token ? 8 : 4;
        uint32_t position = rope_positions[scenario];

        payload[4] = (unsigned char)(token_id >> 24);
        payload[5] = (unsigned char)(token_id >> 16);
        payload[6] = (unsigned char)(token_id >> 8);
        payload[7] = (unsigned char)token_id;
        payload[8] = (unsigned char)(position >> 24);
        payload[9] = (unsigned char)(position >> 16);
        payload[10] = (unsigned char)(position >> 8);
        payload[11] = (unsigned char)position;
        if (work->event_qkv) {
            for (i = 0; i < 2048; i++)
                work->query_q16[i] = INT32_MIN;
            for (i = 0; i < QWEN3_ARENA_TOKEN_WIDTH; i++) {
                work->key_q16[i] = INT32_MIN;
                work->value_q16[i] = INT32_MIN;
            }
        } else {
            for (i = 0; i < total; i++)
                work->matrix_output_q16[i] = INT32_MIN;
        }
        work->completed = 0;
        if (clock_gettime(CLOCK_MONOTONIC, &begin))
            goto done;
        if (sendto(fd, payload, payload_size, 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != (ssize_t)payload_size) {
            perror("XDP event sendto");
            goto done;
        }
        do {
            if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job)) {
                perror("XDP event lookup");
                goto done;
            }
            if (job.status == QWEN3_EVENT_ERROR) {
                fprintf(stderr, "XDP event callback/start failed\n");
                goto done;
            }
            if (job.status == QWEN3_EVENT_DONE &&
                job.completed_requests == attempt + 1)
                break;
            usleep(50);
            if (clock_gettime(CLOCK_MONOTONIC, &now))
                goto done;
            elapsed_ns = (__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
                         now.tv_nsec - begin.tv_nsec;
        } while (elapsed_ns < 30000000000LL);
        if (job.status != QWEN3_EVENT_DONE || job.requests != attempt + 1 ||
            job.completed_requests != attempt + 1 ||
            job.finish_ns < job.start_ns ||
            (work->event_qkv && work->event_stage !=
             (work->event_attention ? QWEN3_EVENT_STAGE_DOWN :
                                      QWEN3_EVENT_STAGE_V)) ||
            (work->event_qkv && work->completed_qk_heads != 24) ||
            (work->event_rope &&
             (work->completed_rope_heads != 24 ||
              work->event_position != position)) ||
            (work->event_attention &&
             (work->completed_attention_heads != 16 ||
              work->attention_past_cursor != position + 1 ||
              work->event_next_position != position + 1)) ||
            work->base_index != (work->event_qkv ?
                QWEN3_ARENA_TOKEN_WIDTH : total) ||
            work->completed != ((total - 1) % QWEN3_ARENA_BF16_ROWS + 1)) {
            fprintf(stderr, "XDP event state mismatch: status=%llu requests=%u completions=%u base=%u completed=%u\n",
                    (unsigned long long)job.status, job.requests,
                    job.completed_requests, work->base_index, work->completed);
            goto done;
        }
        if (work->event_rope) {
            for (i = 0; i < 64; i++) {
                if (work->rope_cosine_q20[i] !=
                        expected_cosine[scenario * 64 + i] ||
                    work->rope_sine_q20[i] !=
                        expected_sine[scenario * 64 + i]) {
                    fprintf(stderr, "XDP RoPE coefficient %u mismatch at position %u: cos=%d/%d sin=%d/%d\n",
                            i, position, work->rope_cosine_q20[i],
                            expected_cosine[scenario * 64 + i],
                            work->rope_sine_q20[i],
                            expected_sine[scenario * 64 + i]);
                    goto done;
                }
            }
        }
        for (i = 0; i < expected_stride; i++) {
            int32_t observed = work->event_qkv ?
                (i < 2048 ? work->query_q16[i] :
                 i < 3072 ? work->key_q16[i - 2048] :
                            work->value_q16[i - 3072]) :
                work->matrix_output_q16[i];

            if (observed != request_expected[i]) {
                fprintf(stderr, "XDP event row %u mismatch: BPF=%d C=%d\n",
                        i, observed, request_expected[i]);
                goto done;
            }
        }
        if (work->event_attention) {
            if (!position) {
                for (uint32_t head = 0; head < 16; head++) {
                    int64_t dot = 0;

                    for (i = 0; i < 128; i++)
                        dot += (int64_t)request_expected[head * 128 + i] *
                            request_expected[2048 + (head / 2) * 128 + i];
                    int64_t score = ((dot >> 16) * 5793) >> 16;
                    if (work->attention_max_score_q16[head] != score) {
                        fprintf(stderr, "XDP attention position 0 head %u score mismatch: BPF=%d C=%lld\n",
                                head, work->attention_max_score_q16[head],
                                (long long)score);
                        goto done;
                    }
                }
            }
            for (uint32_t kv_head = 0; kv_head < 8; kv_head++) {
                uint32_t slot = position * 8 + kv_head;
                struct qwen3_kv_pair pair;

                if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event_kv),
                                        &slot, &pair))
                    goto done;
                for (i = 0; i < 128; i++) {
                    if (pair.key_q16[i] !=
                            request_expected[2048 + kv_head * 128 + i] ||
                        pair.value_q16[i] !=
                            request_expected[3072 + kv_head * 128 + i]) {
                        fprintf(stderr, "XDP cached KV mismatch at position %u head %u col %u\n",
                                position, kv_head, i);
                        goto done;
                    }
                }
            }
            for (i = 0; i < 2048; i++) {
                int32_t observed = work->attention_output_q16[i];
                int32_t wanted = expected_attention[scenario * 2048 + i];

                if (observed != wanted) {
                    fprintf(stderr, "XDP attention position %u row %u mismatch: BPF=%d C=%d\n",
                            position, i, observed, wanted);
                    goto done;
                }
            }
            for (i = 0; i < QWEN3_ARENA_TOKEN_WIDTH; i++) {
                if (work->projected_q16[i] !=
                        expected_projected[scenario * QWEN3_ARENA_TOKEN_WIDTH + i] ||
                    work->down_q16[i] !=
                        expected_down[scenario * QWEN3_ARENA_TOKEN_WIDTH + i] ||
                    work->post_norm_q16[i] !=
                        expected_post_norm[scenario * QWEN3_ARENA_TOKEN_WIDTH + i] ||
                    work->hidden_q16[i] !=
                        expected_hidden[scenario * QWEN3_ARENA_TOKEN_WIDTH + i]) {
                    fprintf(stderr, "XDP attention residual position %u row %u mismatch\n",
                            position, i);
                    goto done;
                }
            }
            for (i = 0; i < QWEN3_ARENA_EVENT_OUTPUTS; i++) {
                uint32_t offset = scenario * QWEN3_ARENA_EVENT_OUTPUTS + i;

                if (work->gate_q16[i] != expected_gate[offset] ||
                    work->up_q16[i] != expected_up[offset] ||
                    work->product_q16[i] != expected_product[offset] ||
                    work->input_q16[i] != expected_product[offset]) {
                    fprintf(stderr, "XDP MLP position %u row %u mismatch: gate=%d/%d up=%d/%d product=%d/%d\n",
                            position, i, work->gate_q16[i], expected_gate[offset],
                            work->up_q16[i], expected_up[offset],
                            work->product_q16[i], expected_product[offset]);
                    goto done;
                }
            }
            for (i = 0; i < 16; i++)
                if (work->attention_seen[i] != position + 1) {
                    fprintf(stderr, "XDP attention head %u saw %u positions\n",
                            i, work->attention_seen[i]);
                    goto done;
                }
        }
        if (work->event_use_token) {
            for (i = 0; i < QWEN3_ARENA_TOKEN_WIDTH; i++) {
                int32_t wanted_input = work->event_attention ?
                    expected_product[scenario * QWEN3_ARENA_EVENT_OUTPUTS + i] :
                    expected_inputs[(attempt % 2) * QWEN3_ARENA_TOKEN_WIDTH + i];

                if (work->input_q16[i] != wanted_input) {
                    fprintf(stderr, "XDP normalized input mismatch at %u\n", i);
                    goto done;
                }
                if (work->embedding_q16[i] !=
                    expected_embeddings[(attempt % 2) * QWEN3_ARENA_TOKEN_WIDTH + i]) {
                    fprintf(stderr, "XDP embedding mismatch at %u\n", i);
                    goto done;
                }
            }
        }
        if (clock_gettime(CLOCK_MONOTONIC, &now))
            goto done;
        elapsed_ns = (__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
                     now.tv_nsec - begin.tv_nsec;
        printf("XDP request %u -> %u resident projection rows: host %.3f ms, kernel %.3f ms\n",
               attempt + 1,
               expected_stride + (work->event_attention ? 8192u : 0u),
               (double)elapsed_ns / 1000000.0,
               (double)(job.finish_ns - job.start_ns) / 1000000.0);
        if (work->event_attention && attempt == 0) {
            payload[8] = payload[9] = payload[10] = 0;
            payload[11] = 2;
            if (sendto(fd, payload, 12, 0, (struct sockaddr *)&dst,
                       sizeof(dst)) != 12 ||
                bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
                job.status != QWEN3_EVENT_DONE || job.requests != 1 ||
                job.completed_requests != 1) {
                fprintf(stderr, "skipped attention position started work\n");
                goto done;
            }
        }
    }
    rc = 0;
done:
    if (fd >= 0)
        close(fd);
    bpf_link__destroy(link);
    return rc;
}

static int reference_two_token_attention(const int32_t qkv[4][4096],
                                         int32_t output[2][2048])
{
    double max_error = 0;

    for (int head = 0; head < 16; head++) {
        int qbase = head * 128;
        int kbase = 2048 + (head / 2) * 128;
        int vbase = 3072 + (head / 2) * 128;
        int64_t dot[2] = {0}, score[2], delta;
        uint64_t mass, weight, alpha_q24;
        double probability;

        for (int i = 0; i < 128; i++) {
            int64_t query = qkv[1][qbase + i];

            output[0][qbase + i] = qkv[0][vbase + i];
            dot[0] += query * qkv[0][kbase + i];
            dot[1] += query * qkv[1][kbase + i];
        }
        for (int past = 0; past < 2; past++) {
            score[past] = ((dot[past] >> 16) * 5793) >> 16;
            if (score[past] > INT32_MAX || score[past] < INT32_MIN)
                return -1;
        }
        delta = score[1] - score[0];
        if (delta > 0) {
            mass = reference_exp_negative_q32((uint32_t)delta) >> 16;
            weight = 1ULL << 16;
        } else {
            mass = 1ULL << 16;
            weight = reference_exp_negative_q32((uint32_t)-delta) >> 16;
        }
        alpha_q24 = (weight << 24) / (mass + weight);
        probability = 1.0 / (1.0 + exp((double)(dot[0] - dot[1]) /
                            (65536.0 * 65536.0 * sqrt(128.0))));
        for (int i = 0; i < 128; i++) {
            int64_t first = qkv[0][vbase + i];
            int64_t second = qkv[1][vbase + i];
            double exact = first + probability * (second - first);
            double error;

            output[1][qbase + i] = (int32_t)(first +
                (((second - first) * (int64_t)alpha_q24) >> 24));
            error = fabs((output[1][qbase + i] - exact) / 65536.0);
            if (error > max_error)
                max_error = error;
        }
    }
    printf("first-layer two-token attention max_abs_error=%.9g\n", max_error);
    return max_error <= 0.005 ? 0 : -1;
}

static int32_t reference_silu_q16(int32_t x)
{
    uint64_t magnitude = x < 0 ? -(int64_t)x : x;
    uint64_t exponential, denominator, sigmoid;

    if (!x)
        return 0;
    if (magnitude >= (16ULL << 16))
        return x > 0 ? x : 0;
    exponential = (1ULL << 32) - (magnitude << 6);
    for (int i = 0; i < 10; i++)
        exponential = (exponential * exponential) >> 32;
    denominator = (1ULL << 32) + exponential;
    sigmoid = x >= 0 ? (1ULL << 48) / denominator :
              (exponential << 16) / denominator;
    return (int32_t)(((int64_t)x * (int64_t)sigmoid) >> 16);
}

static int reference_one_token_layer(struct safetensors_file *file,
                                  const uint64_t projection_byte[3],
                                  const int32_t norm_q20[1024],
                                  const int32_t head_norm_q20[2][128],
                                  uint64_t o_byte,
                                  const int32_t post_norm_q20[1024],
                                  const uint64_t mlp_byte[3],
                                  const int32_t first_hidden[1024],
                                  int32_t second_hidden[1024],
                                  int32_t second_key[1024],
                                  int32_t second_value[1024])
{
    const int rows[3] = {2048, 1024, 1024};
    const int offsets[3] = {0, 2048, 3072};
    int32_t input[1024], qkv[4096], attention[2048], intermediate[3072];
    int32_t projected[2][3072], q24[3072];
    uint64_t sum_sq = 0, inv_rms;

    for (int i = 0; i < 1024; i++)
        sum_sq += (uint64_t)((int64_t)first_hidden[i] * first_hidden[i]);
    inv_rms = (1ULL << 32) /
        reference_isqrt64(sum_sq / QWEN3_ARENA_TOKEN_WIDTH + 4295);
    for (int i = 0; i < 1024; i++) {
        int64_t scaled = ((int64_t)first_hidden[i] * (int64_t)inv_rms) >> 16;

        input[i] = (int32_t)((scaled * norm_q20[i]) >> 20);
    }
    for (int projection = 0; projection < 3; projection++)
        for (int row = 0; row < rows[projection]; row++) {
            int64_t sum = 0;

            if (safetensors_read_bf16_q24_at(file, projection_byte[projection],
                    (uint64_t)row * 1024, 1024, q24)) {
                fprintf(stderr, "second-layer projection read failed: %d/%d\n",
                        projection, row);
                return -1;
            }
            for (int col = 0; col < 1024; col++)
                sum += (int64_t)input[col] * q24[col];
            sum >>= 24;
            if (sum > INT32_MAX || sum < INT32_MIN) {
                fprintf(stderr, "second-layer projection overflow: %d/%d\n",
                        projection, row);
                return -1;
            }
            qkv[offsets[projection] + row] = (int32_t)sum;
        }
    for (int head = 0; head < 24; head++) {
        int set = head >= 16;
        int base = set ? 2048 + (head - 16) * 128 : head * 128;

        sum_sq = 0;
        for (int col = 0; col < 128; col++)
            sum_sq += (uint64_t)((int64_t)qkv[base + col] * qkv[base + col]);
        inv_rms = (1ULL << 32) / reference_isqrt64(sum_sq / 128 + 4295);
        for (int col = 0; col < 128; col++) {
            int64_t scaled = ((int64_t)qkv[base + col] *
                              (int64_t)inv_rms) >> 16;

            qkv[base + col] =
                (int32_t)((scaled * head_norm_q20[set][col]) >> 20);
        }
    }
    /* At position zero RoPE is the identity and causal attention has one KV. */
    memcpy(second_key, qkv + 2048, 1024 * sizeof(*second_key));
    memcpy(second_value, qkv + 3072, 1024 * sizeof(*second_value));
    for (int head = 0; head < 16; head++)
        for (int col = 0; col < 128; col++)
            attention[head * 128 + col] =
                second_value[(head / 2) * 128 + col];
    for (int row = 0; row < 1024; row++) {
        int64_t sum = 0;

        if (safetensors_read_bf16_q24_at(file, o_byte,
                (uint64_t)row * 2048, 2048, q24)) {
            fprintf(stderr, "second-layer output read failed: %d\n", row);
            return -1;
        }
        for (int col = 0; col < 2048; col++)
            sum += (int64_t)attention[col] * q24[col];
        sum >>= 24;
        sum += first_hidden[row];
        if (sum > INT32_MAX || sum < INT32_MIN) {
            fprintf(stderr, "second-layer output overflow: %d\n", row);
            return -1;
        }
        second_hidden[row] = (int32_t)sum;
    }
    sum_sq = 0;
    for (int i = 0; i < 1024; i++)
        sum_sq += (uint64_t)((int64_t)second_hidden[i] * second_hidden[i]);
    inv_rms = (1ULL << 32) /
        reference_isqrt64(sum_sq / QWEN3_ARENA_TOKEN_WIDTH + 4295);
    for (int i = 0; i < 1024; i++) {
        int64_t scaled = ((int64_t)second_hidden[i] *
                          (int64_t)inv_rms) >> 16;

        input[i] = (int32_t)((scaled * post_norm_q20[i]) >> 20);
    }
    for (int set = 0; set < 2; set++)
        for (int row = 0; row < 3072; row++) {
            int64_t sum = 0;

            if (safetensors_read_bf16_q24_at(file, mlp_byte[set],
                    (uint64_t)row * 1024, 1024, q24)) {
                fprintf(stderr, "second-layer MLP read failed: %d/%d\n",
                        set, row);
                return -1;
            }
            for (int col = 0; col < 1024; col++)
                sum += (int64_t)input[col] * q24[col];
            sum >>= 24;
            if (sum > INT32_MAX || sum < INT32_MIN) {
                fprintf(stderr, "second-layer MLP overflow: %d/%d\n",
                        set, row);
                return -1;
            }
            projected[set][row] = (int32_t)sum;
        }
    for (int i = 0; i < 3072; i++)
        intermediate[i] = (int32_t)(((int64_t)reference_silu_q16(
            projected[0][i]) * projected[1][i]) >> 16);
    for (int row = 0; row < 1024; row++) {
        int64_t sum = 0;

        if (safetensors_read_bf16_q24_at(file, mlp_byte[2],
                (uint64_t)row * 3072, 3072, q24)) {
            fprintf(stderr, "second-layer down read failed: %d\n", row);
            return -1;
        }
        for (int col = 0; col < 3072; col++)
            sum += (int64_t)intermediate[col] * q24[col];
        sum = (sum >> 24) + second_hidden[row];
        if (sum > INT32_MAX || sum < INT32_MIN) {
            fprintf(stderr, "second-layer down overflow: %d\n", row);
            return -1;
        }
        second_hidden[row] = (int32_t)sum;
    }
    return 0;
}

static int run_xdp_layer_transition(struct qwen3_arena_bf16_bpf *skel,
                                    struct qwen3_arena_bf16_work *work,
                                    uint32_t token_id,
                                    const struct qwen3_arena_layer_weights *layers,
                                    uint32_t layer_count,
                                    uint32_t expected_requests,
                                    const int32_t *first_qkv,
                                    const int32_t *last_hidden,
                                    const int32_t *last_key,
                                    const int32_t *last_value,
                                    const int32_t *final_norm,
                                    uint32_t expected_token,
                                    int64_t expected_logit)
{
    struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port = htons(49002),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    struct qwen3_event_state job = {0};
    struct bpf_link *link = NULL;
    unsigned char payload[12] = {'Q', '3', 'B', 'P',
        (unsigned char)(token_id >> 24), (unsigned char)(token_id >> 16),
        (unsigned char)(token_id >> 8), (unsigned char)token_id};
    struct timespec begin, now;
    unsigned int ifindex = if_nametoindex("lo");
    uint32_t key, differing = 0;
    int64_t reply_elapsed_ns = 0;
    int fd = -1, poll_fd = -1, rc = -1;

    if (!ifindex || !layers || layer_count < 2 ||
        layer_count > QWEN3_EVENT_MAX_LAYERS || !first_qkv ||
        !last_hidden || !last_key || !last_value)
        return -1;
    for (key = 0; key < layer_count; key++)
        if (bpf_map_update_elem(bpf_map__fd(skel->maps.event_layer_weights),
                                &key, &layers[key], BPF_ANY))
            return -1;
    work->event_layer_count = layer_count;
    work->event_final_logits = final_norm != NULL;
    link = bpf_program__attach_xdp(skel->progs.qwen3_event_xdp, ifindex);
    if (!link || libbpf_get_error(link)) {
        link = NULL;
        goto done;
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0 || clock_gettime(CLOCK_MONOTONIC, &begin) ||
        sendto(fd, payload, sizeof(payload), 0,
               (struct sockaddr *)&dst, sizeof(dst)) != sizeof(payload))
        goto done;
    do {
        key = 0;
        if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job) ||
            job.status == QWEN3_EVENT_ERROR)
            goto done;
        if (job.status == QWEN3_EVENT_DONE &&
            job.completed_requests == expected_requests)
            break;
        usleep(100);
        if (clock_gettime(CLOCK_MONOTONIC, &now))
            goto done;
    } while ((__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
             now.tv_nsec - begin.tv_nsec < 30000000000LL);
    if (job.status != QWEN3_EVENT_DONE ||
        job.requests != expected_requests ||
        job.completed_requests != expected_requests ||
        work->event_layer != layer_count - 1 ||
        work->event_stage != (final_norm ? QWEN3_EVENT_STAGE_LOGITS :
                             QWEN3_EVENT_STAGE_DOWN) ||
        work->event_next_position != 1)
        goto done;
    if (final_norm) {
        if (job.result_token_id != expected_token ||
            job.result_logit_q16 != expected_logit ||
            work->best_index != expected_token ||
            work->best_q16 != expected_logit)
            goto done;
        for (uint32_t col = 0; col < QWEN3_ARENA_TOKEN_WIDTH; col++)
            if (work->input_q16[col] != final_norm[col])
                goto done;
        {
            unsigned char poll[20] = {'Q', '3', 'B', 'R'};
            unsigned char response[20];
            struct timeval timeout = {.tv_sec = 2};
            struct sockaddr_in sender;
            socklen_t sender_len = sizeof(sender);
            uint32_t received_token, received_request;
            uint64_t received_logit = 0;
            ssize_t received;

            poll[4] = expected_requests >> 24;
            poll[5] = expected_requests >> 16;
            poll[6] = expected_requests >> 8;
            poll[7] = expected_requests;
            poll_fd = socket(AF_INET, SOCK_DGRAM, 0);
            if (poll_fd < 0 ||
                setsockopt(poll_fd, SOL_SOCKET, SO_RCVTIMEO,
                           &timeout, sizeof(timeout)) ||
                sendto(poll_fd, poll, sizeof(poll), 0,
                       (struct sockaddr *)&dst, sizeof(dst)) != sizeof(poll))
                goto done;
            received = recvfrom(poll_fd, response, sizeof(response), 0,
                                (struct sockaddr *)&sender, &sender_len);
            if (received != sizeof(response) ||
                sender.sin_port != htons(49002) ||
                memcmp(response, "Q3BA", 4)) {
                fprintf(stderr, "XDP result packet missing: recv=%zd errno=%d\n",
                        received, errno);
                goto done;
            }
            received_token = ((uint32_t)response[4] << 24) |
                             ((uint32_t)response[5] << 16) |
                             ((uint32_t)response[6] << 8) | response[7];
            for (int i = 0; i < 8; i++)
                received_logit = (received_logit << 8) | response[8 + i];
            received_request = ((uint32_t)response[16] << 24) |
                               ((uint32_t)response[17] << 16) |
                               ((uint32_t)response[18] << 8) | response[19];
            if (received_token != expected_token ||
                (int64_t)received_logit != expected_logit ||
                received_request != expected_requests)
                goto done;
            if (clock_gettime(CLOCK_MONOTONIC, &now))
                goto done;
            reply_elapsed_ns = (int64_t)(now.tv_sec - begin.tv_sec) *
                               1000000000 + now.tv_nsec - begin.tv_nsec;
        }
    }
    for (uint32_t head = 0; head < QWEN3_ATTENTION_KV_HEADS; head++) {
        struct qwen3_kv_pair pair[2];

        for (uint32_t layer = 0; layer < 2; layer++) {
            key = (layer ? layer_count - 1 : 0) *
                  QWEN3_EVENT_KV_LIMIT * QWEN3_ATTENTION_KV_HEADS + head;
            if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event_kv),
                                    &key, &pair[layer]))
                goto done;
        }
        for (uint32_t col = 0; col < 128; col++) {
            uint32_t index = head * 128 + col;

            if (pair[0].key_q16[col] != first_qkv[2048 + index] ||
                pair[0].value_q16[col] != first_qkv[3072 + index] ||
                pair[1].key_q16[col] != last_key[index] ||
                pair[1].value_q16[col] != last_value[index] ||
                pair[1].key_q16[col] != work->key_q16[index] ||
                pair[1].value_q16[col] != work->value_q16[index]) {
                fprintf(stderr, "XDP layer KV mismatch head=%u col=%u first K=%d/%d V=%d/%d last K=%d/%d V=%d/%d\n",
                        head, col, pair[0].key_q16[col],
                        first_qkv[2048 + index], pair[0].value_q16[col],
                        first_qkv[3072 + index], pair[1].key_q16[col],
                        last_key[index], pair[1].value_q16[col],
                        last_value[index]);
                goto done;
            }
            differing |= pair[0].key_q16[col] != pair[1].key_q16[col] ||
                         pair[0].value_q16[col] != pair[1].value_q16[col];
        }
    }
    if (!differing)
        goto done;
    for (uint32_t col = 0; col < QWEN3_ARENA_TOKEN_WIDTH; col++)
        if (work->hidden_q16[col] != last_hidden[col]) {
            fprintf(stderr, "XDP layer hidden mismatch col=%u BPF=%d C=%d\n",
                    col, work->hidden_q16[col], last_hidden[col]);
            goto done;
        }
    if (final_norm)
        printf("XDP %u-layer vocabulary argmax and UDP reply passed: token=%u logit=%lld host=%.3f ms layers=%.3f ms vocab=%.3f ms cpu=%u\n",
               layer_count, job.result_token_id,
               (long long)job.result_logit_q16,
               (double)reply_elapsed_ns / 1000000.0,
               (double)(job.layers_finish_ns - job.start_ns) / 1000000.0,
               (double)(job.finish_ns - job.layers_finish_ns) / 1000000.0,
               job.work_cpu);
    else
        printf("XDP %u-layer arithmetic and per-layer KV isolation passed\n",
               layer_count);
    rc = 0;
done:
    if (rc)
        fprintf(stderr, "XDP %u-layer transition failed: status=%llu requests=%u completions=%u layer=%u stage=%u base=%u next=%u\n",
                layer_count, (unsigned long long)job.status, job.requests,
                job.completed_requests, work->event_layer,
                work->event_stage, work->base_index,
                work->event_next_position);
    if (fd >= 0)
        close(fd);
    if (poll_fd >= 0)
        close(poll_fd);
    bpf_link__destroy(link);
    work->event_layer_count = 0;
    work->event_final_logits = 0;
    return rc;
}

static int run_xdp_prefill(struct qwen3_arena_bf16_bpf *skel,
                           struct qwen3_arena_bf16_work *work)
{
    struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port = htons(49002),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    struct timeval timeout = {.tv_sec = 2};
    struct qwen3_event_state job;
    struct bpf_link *link = NULL;
    struct timespec begin, now;
    unsigned int ifindex = if_nametoindex("lo");
    uint32_t key = 0, expected;
    int fd = -1, poll_fd = -1, rc = -1;

    if (!ifindex || bpf_map_lookup_elem(bpf_map__fd(skel->maps.event),
                                        &key, &job))
        return -1;
    expected = job.completed_requests;
    work->event_layer_count = QWEN3_EVENT_MAX_LAYERS;
    work->event_final_logits = 1;
    link = bpf_program__attach_xdp(skel->progs.qwen3_event_xdp, ifindex);
    if (!link || libbpf_get_error(link)) {
        link = NULL;
        goto done;
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    poll_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0 || poll_fd < 0 ||
        setsockopt(poll_fd, SOL_SOCKET, SO_RCVTIMEO,
                   &timeout, sizeof(timeout)))
        goto done;
    for (uint32_t pos = 0; pos < 2; pos++) {
        unsigned char request[12] = {'Q', '3', 'B', pos ? 'P' : 'F'};
        unsigned char poll[20] = {'Q', '3', 'B', 'R'};
        unsigned char reply[20];
        uint32_t count = ++expected;
        ssize_t received;

        request[7] = pos;
        request[11] = pos;
        poll[4] = count >> 24;
        poll[5] = count >> 16;
        poll[6] = count >> 8;
        poll[7] = count;
        if (clock_gettime(CLOCK_MONOTONIC, &begin) ||
            sendto(fd, request, sizeof(request), 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != sizeof(request))
            goto done;
        do {
            if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event),
                                    &key, &job) ||
                job.status == QWEN3_EVENT_ERROR)
                goto done;
            if (job.status == QWEN3_EVENT_DONE &&
                job.completed_requests == count)
                break;
            usleep(100);
            if (clock_gettime(CLOCK_MONOTONIC, &now))
                goto done;
        } while ((__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
                 now.tv_nsec - begin.tv_nsec < 30000000000LL);
        if (job.status != QWEN3_EVENT_DONE ||
            job.completed_requests != count ||
            work->event_next_position != pos + 1 ||
            (pos == 0 && (work->event_stage != QWEN3_EVENT_STAGE_DOWN ||
                          job.result_token_id != 0 ||
                          job.result_logit_q16 != 0)) ||
            (pos == 1 && (job.result_token_id != 220 ||
                          job.result_logit_q16 != 746817)))
            goto done;
        if (sendto(poll_fd, poll, sizeof(poll), 0,
                   (struct sockaddr *)&dst, sizeof(dst)) != sizeof(poll))
            goto done;
        received = recvfrom(poll_fd, reply, sizeof(reply), 0, NULL, NULL);
        if (received != sizeof(reply) ||
            memcmp(reply, pos ? "Q3BA" : "Q3BK", 4) ||
            reply[16] != poll[4] || reply[17] != poll[5] ||
            reply[18] != poll[6] || reply[19] != poll[7])
            goto done;
        if (pos == 0)
            for (int i = 4; i < 16; i++)
                if (reply[i])
                    goto done;
    }
    puts("XDP prefill acknowledgement and final token passed");
    rc = 0;
done:
    if (rc)
        fprintf(stderr, "XDP prefill failed: status=%llu completions=%u next=%u\n",
                (unsigned long long)job.status, job.completed_requests,
                work->event_next_position);
    if (fd >= 0)
        close(fd);
    if (poll_fd >= 0)
        close(poll_fd);
    bpf_link__destroy(link);
    work->event_layer_count = 0;
    work->event_final_logits = 0;
    return rc;
}

static int find_matrix_offset(struct safetensors_file *file, const char *name,
                              uint64_t elements, uint64_t *byte)
{
    uint64_t found;

    return safetensors_find_bf16(file, name, byte, &found) ||
           found != elements ? -1 : 0;
}

static int find_norm_q20(struct safetensors_file *file, const char *name,
                         uint32_t count, uint64_t *byte, int32_t *out)
{
    float values[QWEN3_ARENA_TOKEN_WIDTH];

    if (count > QWEN3_ARENA_TOKEN_WIDTH ||
        find_matrix_offset(file, name, count, byte) ||
        safetensors_read_bf16_at(file, *byte, 0, count, values))
        return -1;
    for (uint32_t i = 0; i < count; i++) {
        double scaled = (double)values[i] * 1048576.0;

        if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
            return -1;
        out[i] = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
    }
    return 0;
}

static int find_layer_weights(struct safetensors_file *file, int layer,
                              struct qwen3_arena_layer_weights *weights,
                              uint64_t projection_byte[3],
                              int32_t norm_q20[1024],
                              int32_t head_norm_q20[2][128],
                              uint64_t *o_byte,
                              int32_t post_norm_q20[1024],
                              uint64_t mlp_byte[3])
{
    char name[96];
    uint64_t byte;
    const char *projection[3] = {"q_proj", "k_proj", "v_proj"};
    const char *mlp[3] = {"gate_proj", "up_proj", "down_proj"};

    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer);
    if (find_norm_q20(file, name, 1024, &byte, norm_q20))
        return -1;
    weights->input_norm_first_bf16 = byte / 2;
    snprintf(name, sizeof(name),
             "model.layers.%d.post_attention_layernorm.weight", layer);
    if (find_norm_q20(file, name, 1024, &byte, post_norm_q20))
        return -1;
    weights->post_norm_first_bf16 = byte / 2;
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer);
    if (find_norm_q20(file, name, 128, &byte, head_norm_q20[0]))
        return -1;
    weights->q_norm_first_bf16 = byte / 2;
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer);
    if (find_norm_q20(file, name, 128, &byte, head_norm_q20[1]))
        return -1;
    weights->k_norm_first_bf16 = byte / 2;
    for (int i = 0; i < 3; i++) {
        uint64_t elements = (uint64_t)(i ? 1024 : 2048) * 1024;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.%s.weight",
                 layer, projection[i]);
        if (find_matrix_offset(file, name, elements, &projection_byte[i]))
            return -1;
    }
    weights->q_first_bf16 = projection_byte[0] / 2;
    weights->k_first_bf16 = projection_byte[1] / 2;
    weights->v_first_bf16 = projection_byte[2] / 2;
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer);
    if (find_matrix_offset(file, name, 1024ULL * 2048, o_byte))
        return -1;
    weights->o_first_bf16 = *o_byte / 2;
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "model.layers.%d.mlp.%s.weight",
                 layer, mlp[i]);
        if (find_matrix_offset(file, name, 3072ULL * 1024, &mlp_byte[i]))
            return -1;
    }
    weights->gate_first_bf16 = mlp_byte[0] / 2;
    weights->up_first_bf16 = mlp_byte[1] / 2;
    weights->down_first_bf16 = mlp_byte[2] / 2;
    return 0;
}

static int reference_final_logits(struct safetensors_file *file,
                                  const int32_t hidden[1024],
                                  uint64_t embedding_byte,
                                  uint64_t *norm_byte,
                                  int32_t normalized[1024],
                                  uint32_t *token_id,
                                  int64_t *best_logit)
{
    int32_t norm_q20[1024], q24[1024];
    uint64_t sum_sq = 0, inv_rms;

    if (find_norm_q20(file, "model.norm.weight", 1024,
                      norm_byte, norm_q20))
        return -1;
    for (int i = 0; i < 1024; i++)
        sum_sq += (uint64_t)((int64_t)hidden[i] * hidden[i]);
    inv_rms = (1ULL << 32) / reference_isqrt64(sum_sq / 1024 + 4295);
    for (int i = 0; i < 1024; i++) {
        int64_t scaled = ((int64_t)hidden[i] * (int64_t)inv_rms) >> 16;

        normalized[i] = (int32_t)((scaled * norm_q20[i]) >> 20);
    }
    *token_id = 0;
    *best_logit = INT32_MIN;
    for (uint32_t row = 0; row < QWEN3_EVENT_VOCAB; row++) {
        int64_t sum = 0;

        if (safetensors_read_bf16_q24_at(file, embedding_byte,
                (uint64_t)row * 1024, 1024, q24))
            return -1;
        for (int col = 0; col < 1024; col++)
            sum += (int64_t)normalized[col] * q24[col];
        sum >>= 24;
        if (sum > *best_logit) {
            *best_logit = sum;
            *token_id = row;
        }
    }
    return 0;
}

static int run_xdp_model(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work,
                         const char *path, int with_attention)
{
    struct safetensors_file file;
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    struct qwen3_arena_layer_weights layers[QWEN3_EVENT_MAX_LAYERS] = {0};
    int32_t (*layer_hidden)[QWEN3_ARENA_TOKEN_WIDTH] = NULL;
    int32_t (*layer_key)[QWEN3_ARENA_TOKEN_WIDTH] = NULL;
    int32_t (*layer_value)[QWEN3_ARENA_TOKEN_WIDTH] = NULL;
    uint64_t projection_byte[3], projection_elements[3], projection_first[3];
    uint64_t embedding_byte, embedding_elements, cursor;
    uint64_t norm_byte, norm_elements, norm_first;
    uint64_t o_byte, o_elements;
    uint64_t post_norm_byte, post_norm_elements;
    uint64_t mlp_byte[3], mlp_elements[3], mlp_first[3];
    uint64_t head_norm_byte[2], head_norm_first[2], head_norm_elements[2];
    int32_t q24[1024], hidden[2][1024], normalized[2][1024];
    int32_t norm_q20[1024], head_norm_q20[2][128];
    int32_t expected[2][4096], norm_expected[2][4096];
    int32_t event_expected[4][4096], attention_expected[2][2048];
    int32_t projected_expected[2][1024], residual_expected[2][1024];
    int32_t post_norm_expected[2][1024], post_norm_q20[1024];
    int32_t mlp_expected[2][2][QWEN3_ARENA_EVENT_OUTPUTS];
    int32_t product_expected[2][QWEN3_ARENA_EVENT_OUTPUTS];
    int32_t down_expected[2][QWEN3_ARENA_TOKEN_WIDTH];
    int32_t final_hidden_expected[2][QWEN3_ARENA_TOKEN_WIDTH];
    int32_t running_hidden[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t next_hidden[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t next_key[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t next_value[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t final_norm_expected[QWEN3_ARENA_TOKEN_WIDTH];
    uint64_t final_norm_byte = 0;
    uint32_t final_token_expected = 0;
    int64_t final_logit_expected = 0;
    int32_t second_norm_q20[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t second_post_norm_q20[QWEN3_ARENA_TOKEN_WIDTH];
    int32_t second_head_norm_q20[2][128];
    uint64_t second_projection_byte[3], second_mlp_byte[3], second_o_byte;
    int32_t o_q24[QWEN3_ARENA_EVENT_OUTPUTS];
    int32_t rope_cosine[4][64], rope_sine[4][64];
    int32_t direct_outputs[4096];
    float embedding[1024], norm_weights[1024], post_norm_weights[1024];
    float head_norm_weights[2][128];
    const uint8_t *embedding_raw, *norm_raw;
    const uint32_t token_ids[2] = {0, 151935};
    const char *projection_names[3] = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    };
    const char *head_norm_names[2] = {
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    };
    const char *mlp_names[3] = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    };
    const int projection_rows[3] = {2048, 1024, 1024};
    const int output_offsets[3] = {0, 2048, 3072};
    int row, col, token, projection, rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (safetensors_find_bf16(&file, "model.embed_tokens.weight",
            &embedding_byte, &embedding_elements) ||
        embedding_elements != 151936ULL * QWEN3_ARENA_TOKEN_WIDTH ||
        safetensors_find_bf16(&file, "model.layers.0.input_layernorm.weight",
            &norm_byte, &norm_elements) ||
        norm_elements != QWEN3_ARENA_TOKEN_WIDTH ||
        safetensors_read_bf16_at(&file, norm_byte, 0,
                                 QWEN3_ARENA_TOKEN_WIDTH, norm_weights))
        goto done;
    for (projection = 0; projection < 3; projection++)
        if (safetensors_find_bf16(&file, projection_names[projection],
                &projection_byte[projection],
                &projection_elements[projection]) ||
            projection_elements[projection] !=
                (uint64_t)projection_rows[projection] * QWEN3_ARENA_TOKEN_WIDTH)
            goto done;
    for (int set = 0; set < 2; set++)
        if (safetensors_find_bf16(&file, head_norm_names[set],
                &head_norm_byte[set], &head_norm_elements[set]) ||
            head_norm_elements[set] != 128 ||
            safetensors_read_bf16_at(&file, head_norm_byte[set], 0, 128,
                                     head_norm_weights[set]))
            goto done;
    if (with_attention &&
        (safetensors_find_bf16(&file,
            "model.layers.0.self_attn.o_proj.weight", &o_byte,
            &o_elements) || o_elements != 1024ULL * 2048 ||
         safetensors_find_bf16(&file,
            "model.layers.0.post_attention_layernorm.weight",
            &post_norm_byte, &post_norm_elements) ||
         post_norm_elements != QWEN3_ARENA_TOKEN_WIDTH ||
         safetensors_read_bf16_at(&file, post_norm_byte, 0,
            QWEN3_ARENA_TOKEN_WIDTH, post_norm_weights)))
        goto done;
    if (with_attention)
        for (int set = 0; set < 3; set++)
            if (safetensors_find_bf16(&file, mlp_names[set],
                    &mlp_byte[set], &mlp_elements[set]) ||
                mlp_elements[set] !=
                    (uint64_t)QWEN3_ARENA_EVENT_OUTPUTS * QWEN3_ARENA_TOKEN_WIDTH)
                goto done;
    embedding_raw = file.mapping + 8 + file.header_length + embedding_byte;
    norm_raw = file.mapping + 8 + file.header_length + norm_byte;
    memset(work, 0, sizeof(*work));
    if (with_attention) {
        uint64_t data_bytes = file.mapping_size - 8 - file.header_length;

        if ((data_bytes & 1) ||
            data_bytes / 2 > QWEN3_ARENA_MODEL_MAX_BF16)
            goto done;
        work->model_elements = data_bytes / 2;
    } else {
        work->model_elements = embedding_elements + norm_elements;
        for (projection = 0; projection < 3; projection++)
            work->model_elements += projection_elements[projection];
        work->model_elements += head_norm_elements[0] + head_norm_elements[1];
    }
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_allocate_model),
            &opts) || opts.retval || !skel->bss->model_bf16) {
        fprintf(stderr, "XDP model arena allocation failed\n");
        goto done;
    }
    if (with_attention) {
        memcpy(skel->bss->model_bf16,
               file.mapping + 8 + file.header_length,
               work->model_elements * 2);
        work->embedding_first_bf16 = embedding_byte / 2;
        for (projection = 0; projection < 3; projection++)
            projection_first[projection] = projection_byte[projection] / 2;
        norm_first = norm_byte / 2;
        for (int set = 0; set < 2; set++)
            head_norm_first[set] = head_norm_byte[set] / 2;
        work->o_first_bf16 = o_byte / 2;
        work->post_norm_first_bf16 = post_norm_byte / 2;
        for (int set = 0; set < 3; set++)
            mlp_first[set] = mlp_byte[set] / 2;
        work->gate_first_bf16 = mlp_first[0];
        work->up_first_bf16 = mlp_first[1];
        work->down_first_bf16 = mlp_first[2];
    } else {
        memcpy(skel->bss->model_bf16, embedding_raw, embedding_elements * 2);
        cursor = embedding_elements;
        for (projection = 0; projection < 3; projection++) {
            const uint8_t *raw = file.mapping + 8 + file.header_length +
                                 projection_byte[projection];

            projection_first[projection] = cursor;
            memcpy(skel->bss->model_bf16 + cursor, raw,
                   projection_elements[projection] * 2);
            cursor += projection_elements[projection];
        }
        norm_first = cursor;
        memcpy(skel->bss->model_bf16 + cursor, norm_raw, norm_elements * 2);
        cursor += norm_elements;
        for (int set = 0; set < 2; set++) {
            const uint8_t *raw = file.mapping + 8 + file.header_length +
                                 head_norm_byte[set];

            head_norm_first[set] = cursor;
            memcpy(skel->bss->model_bf16 + cursor, raw,
                   head_norm_elements[set] * 2);
            cursor += head_norm_elements[set];
        }
    }
    work->cols = 1024;
    work->rows = QWEN3_ARENA_BF16_ROWS;
    work->resident_weights = 1;
    work->weight_first_bf16 = projection_first[0];
    work->q_first_bf16 = projection_first[0];
    work->k_first_bf16 = projection_first[1];
    work->v_first_bf16 = projection_first[2];
    work->norm_first_bf16 = norm_first;
    work->q_norm_first_bf16 = head_norm_first[0];
    work->k_norm_first_bf16 = head_norm_first[1];
    work->embedding_vocab = 151936;
    work->event_use_token = 1;
    work->event_qkv = 1;
    work->event_rope = 1;
    work->event_attention = with_attention;
    if (with_attention) {
        layers[0].input_norm_first_bf16 = work->norm_first_bf16;
        layers[0].q_first_bf16 = work->q_first_bf16;
        layers[0].k_first_bf16 = work->k_first_bf16;
        layers[0].v_first_bf16 = work->v_first_bf16;
        layers[0].o_first_bf16 = work->o_first_bf16;
        layers[0].post_norm_first_bf16 = work->post_norm_first_bf16;
        layers[0].gate_first_bf16 = work->gate_first_bf16;
        layers[0].up_first_bf16 = work->up_first_bf16;
        layers[0].down_first_bf16 = work->down_first_bf16;
        layers[0].q_norm_first_bf16 = work->q_norm_first_bf16;
        layers[0].k_norm_first_bf16 = work->k_norm_first_bf16;
    }
    for (col = 0; col < 1024; col++) {
        double scaled = (double)norm_weights[col] * 1048576.0;

        if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
            goto done;
        norm_q20[col] = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
        if (with_attention) {
            scaled = (double)post_norm_weights[col] * 1048576.0;
            if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
                goto done;
            post_norm_q20[col] =
                (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
        }
    }
    for (int set = 0; set < 2; set++)
        for (col = 0; col < 128; col++) {
            double scaled = (double)head_norm_weights[set][col] * 1048576.0;

            if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
                goto done;
            head_norm_q20[set][col] =
                (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
        }
    for (token = 0; token < 2; token++) {
        uint64_t sum_sq_q32 = 0, inv_rms_q16;

        if (safetensors_read_bf16_at(&file, embedding_byte,
                (uint64_t)token_ids[token] * QWEN3_ARENA_TOKEN_WIDTH,
                QWEN3_ARENA_TOKEN_WIDTH, embedding))
            goto done;
        for (col = 0; col < 1024; col++) {
            double scaled = (double)embedding[col] * 65536.0;

            if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
                goto done;
            hidden[token][col] =
                (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
            sum_sq_q32 += (uint64_t)((int64_t)hidden[token][col] *
                                     hidden[token][col]);
        }
        inv_rms_q16 = (1ULL << 32) /
            reference_isqrt64(sum_sq_q32 / QWEN3_ARENA_TOKEN_WIDTH + 4295);
        for (col = 0; col < 1024; col++) {
            int64_t scaled = ((int64_t)hidden[token][col] *
                              (int64_t)inv_rms_q16) >> 16;
            normalized[token][col] =
                (int32_t)((scaled * norm_q20[col]) >> 20);
        }
        for (projection = 0; projection < 3; projection++) {
            for (row = 0; row < projection_rows[projection]; row++) {
                __s64 sum = 0;

                if (safetensors_read_bf16_q24_at(&file,
                        projection_byte[projection],
                        (uint64_t)row * QWEN3_ARENA_TOKEN_WIDTH,
                        QWEN3_ARENA_TOKEN_WIDTH, q24))
                    goto done;
                for (col = 0; col < 1024; col++)
                    sum += (__s64)normalized[token][col] * q24[col];
                sum >>= 24;
                if (sum > INT32_MAX || sum < INT32_MIN)
                    goto done;
                expected[token][output_offsets[projection] + row] =
                    (int32_t)sum;
            }
        }
        memcpy(norm_expected[token], expected[token], sizeof(expected[token]));
        for (int head = 0; head < 24; head++) {
            int set = head >= 16;
            int base = set ? 2048 + (head - 16) * 128 : head * 128;
            uint64_t sum_sq_q32 = 0, inv_rms_q16;

            for (int i = 0; i < 128; i++) {
                int64_t x = expected[token][base + i];
                sum_sq_q32 += (uint64_t)(x * x);
            }
            inv_rms_q16 = (1ULL << 32) /
                reference_isqrt64(sum_sq_q32 / 128 + 4295);
            for (int i = 0; i < 128; i++) {
                int64_t scaled = ((int64_t)expected[token][base + i] *
                                  (int64_t)inv_rms_q16) >> 16;
                norm_expected[token][base + i] =
                    (int32_t)((scaled * head_norm_q20[set][i]) >> 20);
            }
        }
    }
    for (int scenario = 0; scenario < 4; scenario++) {
        int selected_token = scenario % 2;
        int position = rope_positions[scenario];

        memcpy(event_expected[scenario], norm_expected[selected_token],
               sizeof(norm_expected[selected_token]));
        for (int i = 0; i < 64; i++) {
            double angle = position * pow(1000000.0, -(double)i / 64.0);

            rope_cosine[scenario][i] = (int32_t)round(cos(angle) * (1 << 20));
            rope_sine[scenario][i] = (int32_t)round(sin(angle) * (1 << 20));
        }
        for (int head = 0; head < 24; head++) {
            int base = head < 16 ? head * 128 : 2048 + (head - 16) * 128;

            for (int i = 0; i < 64; i++) {
                int64_t first = event_expected[scenario][base + i];
                int64_t second = event_expected[scenario][base + i + 64];
                int64_t cosine = rope_cosine[scenario][i];
                int64_t sine = rope_sine[scenario][i];

                event_expected[scenario][base + i] =
                    (int32_t)((first * cosine - second * sine) >> 20);
                event_expected[scenario][base + i + 64] =
                    (int32_t)((second * cosine + first * sine) >> 20);
            }
        }
    }
    if (with_attention &&
        reference_two_token_attention(event_expected, attention_expected))
        goto done;
    if (with_attention)
        for (token = 0; token < 2; token++) {
            uint64_t sum_sq_q32 = 0, inv_rms_q16;

            for (row = 0; row < 1024; row++) {
                int64_t sum = 0, residual;

                if (safetensors_read_bf16_q24_at(&file, o_byte,
                        (uint64_t)row * 2048, 2048, o_q24))
                    goto done;
                for (col = 0; col < 2048; col++)
                    sum += (int64_t)attention_expected[token][col] * o_q24[col];
                sum >>= 24;
                residual = (int64_t)hidden[token][row] + sum;
                if (sum > INT32_MAX || sum < INT32_MIN ||
                    residual > INT32_MAX || residual < INT32_MIN)
                    goto done;
                projected_expected[token][row] = (int32_t)sum;
                residual_expected[token][row] = (int32_t)residual;
                sum_sq_q32 += (uint64_t)(residual * residual);
            }
            inv_rms_q16 = (1ULL << 32) /
                reference_isqrt64(sum_sq_q32 / QWEN3_ARENA_TOKEN_WIDTH + 4295);
            for (col = 0; col < 1024; col++) {
                int64_t scaled = ((int64_t)residual_expected[token][col] *
                                  (int64_t)inv_rms_q16) >> 16;

                post_norm_expected[token][col] =
                    (int32_t)((scaled * post_norm_q20[col]) >> 20);
            }
            for (int set = 0; set < 2; set++)
                for (row = 0; row < QWEN3_ARENA_EVENT_OUTPUTS; row++) {
                    int64_t sum = 0;

                    if (safetensors_read_bf16_q24_at(&file, mlp_byte[set],
                            (uint64_t)row * QWEN3_ARENA_TOKEN_WIDTH,
                            QWEN3_ARENA_TOKEN_WIDTH, q24))
                        goto done;
                    for (col = 0; col < QWEN3_ARENA_TOKEN_WIDTH; col++)
                        sum += (int64_t)post_norm_expected[token][col] *
                               q24[col];
                    sum >>= 24;
                    if (sum > INT32_MAX || sum < INT32_MIN)
                        goto done;
                    mlp_expected[set][token][row] = (int32_t)sum;
                }
            for (row = 0; row < QWEN3_ARENA_EVENT_OUTPUTS; row++) {
                int32_t activated =
                    reference_silu_q16(mlp_expected[0][token][row]);

                product_expected[token][row] =
                    (int32_t)(((int64_t)activated *
                        mlp_expected[1][token][row]) >> 16);
            }
            for (row = 0; row < QWEN3_ARENA_TOKEN_WIDTH; row++) {
                int64_t sum = 0, residual;

                if (safetensors_read_bf16_q24_at(&file, mlp_byte[2],
                        (uint64_t)row * QWEN3_ARENA_EVENT_OUTPUTS,
                        QWEN3_ARENA_EVENT_OUTPUTS, o_q24))
                    goto done;
                for (col = 0; col < QWEN3_ARENA_EVENT_OUTPUTS; col++)
                    sum += (int64_t)product_expected[token][col] * o_q24[col];
                sum >>= 24;
                residual = (int64_t)residual_expected[token][row] + sum;
                if (sum > INT32_MAX || sum < INT32_MIN ||
                    residual > INT32_MAX || residual < INT32_MIN)
                    goto done;
                down_expected[token][row] = (int32_t)sum;
                final_hidden_expected[token][row] = (int32_t)residual;
            }
        }
    if (with_attention) {
        layer_hidden = calloc(QWEN3_EVENT_MAX_LAYERS, sizeof(*layer_hidden));
        layer_key = calloc(QWEN3_EVENT_MAX_LAYERS, sizeof(*layer_key));
        layer_value = calloc(QWEN3_EVENT_MAX_LAYERS, sizeof(*layer_value));
        if (!layer_hidden || !layer_key || !layer_value)
            goto done;
        memcpy(layer_hidden[0], final_hidden_expected[0],
               sizeof(layer_hidden[0]));
        memcpy(layer_key[0], event_expected[0] + 2048,
               sizeof(layer_key[0]));
        memcpy(layer_value[0], event_expected[0] + 3072,
               sizeof(layer_value[0]));
        memcpy(running_hidden, final_hidden_expected[0],
               sizeof(running_hidden));
        for (int layer = 1; layer < QWEN3_EVENT_MAX_LAYERS; layer++) {
            if (find_layer_weights(&file, layer, &layers[layer],
                    second_projection_byte, second_norm_q20,
                    second_head_norm_q20, &second_o_byte,
                    second_post_norm_q20, second_mlp_byte) ||
                reference_one_token_layer(&file, second_projection_byte,
                    second_norm_q20, second_head_norm_q20, second_o_byte,
                    second_post_norm_q20, second_mlp_byte,
                    running_hidden, next_hidden, next_key, next_value)) {
                fprintf(stderr, "layer %d reference failed\n", layer);
                goto done;
            }
            memcpy(running_hidden, next_hidden, sizeof(running_hidden));
            memcpy(layer_hidden[layer], next_hidden, sizeof(layer_hidden[layer]));
            memcpy(layer_key[layer], next_key, sizeof(layer_key[layer]));
            memcpy(layer_value[layer], next_value,
                   sizeof(layer_value[layer]));
        }
        if (reference_final_logits(&file,
                layer_hidden[QWEN3_EVENT_MAX_LAYERS - 1],
                embedding_byte, &final_norm_byte, final_norm_expected,
                &final_token_expected, &final_logit_expected))
            goto done;
        work->final_norm_first_bf16 = final_norm_byte / 2;
    }
    for (int attempt = 0; attempt < 10; attempt++) {
        struct timespec begin, end;
        __s64 elapsed_ns;
        token = attempt % 2;

        memcpy(work->input_q16, normalized[token], sizeof(normalized[token]));
        if (clock_gettime(CLOCK_MONOTONIC, &begin))
            goto done;
        for (projection = 0; projection < 3; projection++) {
            work->weight_first_bf16 = projection_first[projection];
            for (row = 0; row < projection_rows[projection];
                 row += QWEN3_ARENA_BF16_ROWS) {
                work->base_index = row;
                if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
                    goto done;
                for (int i = 0; i < QWEN3_ARENA_BF16_ROWS; i++) {
                    if (work->output_q16[i] > INT32_MAX ||
                        work->output_q16[i] < INT32_MIN)
                        goto done;
                    direct_outputs[output_offsets[projection] + row + i] =
                        (int32_t)work->output_q16[i];
                }
            }
        }
        if (clock_gettime(CLOCK_MONOTONIC, &end))
            goto done;
        for (row = 0; row < 4096; row++)
            if (direct_outputs[row] != expected[token][row]) {
                fprintf(stderr, "direct model row %d mismatch\n", row);
                goto done;
            }
        elapsed_ns = (__s64)(end.tv_sec - begin.tv_sec) * 1000000000 +
                     end.tv_nsec - begin.tv_nsec;
        printf("BPF_PROG_RUN request %d -> 4096 QKV rows: %.3f ms\n",
               attempt + 1, (double)elapsed_ns / 1000000.0);
    }
    rc = run_xdp_event(skel, work, &event_expected[0][0], 2048,
                       &normalized[0][0], &hidden[0][0], token_ids,
                       &rope_cosine[0][0], &rope_sine[0][0],
                       with_attention ? &attention_expected[0][0] : NULL,
                       with_attention ? &projected_expected[0][0] : NULL,
                       with_attention ? &final_hidden_expected[0][0] : NULL,
                       with_attention ? &post_norm_expected[0][0] : NULL,
                       with_attention ? &mlp_expected[0][0][0] : NULL,
                       with_attention ? &mlp_expected[1][0][0] : NULL,
                       with_attention ? &product_expected[0][0] : NULL,
                       with_attention ? &down_expected[0][0] : NULL);
    if (!rc && with_attention)
        for (uint32_t count = 2; count <= QWEN3_EVENT_MAX_LAYERS; count++) {
            rc = run_xdp_layer_transition(skel, work, token_ids[0],
                                          layers, count, count + 9,
                                          event_expected[0],
                                          layer_hidden[count - 1],
                                          layer_key[count - 1],
                                          layer_value[count - 1],
                                          NULL, 0, 0);
            if (rc)
                break;
        }
    if (!rc && with_attention)
        for (uint32_t repeat = 0; repeat < 3; repeat++) {
            rc = run_xdp_layer_transition(skel, work, token_ids[0],
                                          layers, QWEN3_EVENT_MAX_LAYERS,
                                          38 + repeat, event_expected[0],
                                          layer_hidden[27], layer_key[27],
                                          layer_value[27],
                                          final_norm_expected,
                                          final_token_expected,
                                          final_logit_expected);
            if (rc)
                break;
        }
    if (!rc && with_attention)
        rc = run_xdp_prefill(skel, work);
done:
    free(layer_hidden);
    free(layer_key);
    free(layer_value);
    safetensors_close(&file);
    return rc;
}

static int run_model_rows(struct qwen3_arena_bf16_bpf *skel,
                          struct qwen3_arena_bf16_work *work,
                          const char *path)
{
    struct safetensors_file file;
    uint64_t first_byte, elements;
    int32_t q24[1024];
    const uint8_t *raw;
    __s64 expected[QWEN3_ARENA_BF16_ROWS] = {0};
    int row, col, rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (safetensors_find_bf16(&file,
            "model.layers.0.self_attn.q_proj.weight", &first_byte,
            &elements) || elements != 2048ULL * 1024)
        goto done;
    raw = file.mapping + 8 + file.header_length + first_byte;
    memset(work, 0, sizeof(*work));
    work->cols = 1024;
    for (col = 0; col < 1024; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        if (safetensors_read_bf16_q24_at(&file, first_byte,
                (uint64_t)row * 1024, 1024, q24))
            goto done;
        for (col = 0; col < 1024; col++) {
            __u16 bits = (__u16)raw[2 * (row * 1024 + col)] |
                         ((__u16)raw[2 * (row * 1024 + col) + 1] << 8);
            skel->arena->weight_bf16[row][col] = bits;
            if (skel->arena->q24_by_bf16[bits] != q24[col]) {
                fprintf(stderr, "arena BF16 lookup differs at row %d col %d\n",
                        row, col);
                goto done;
            }
            expected[row] += (__s64)work->input_q16[col] * q24[col];
        }
    }
    if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
        goto done;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        if (work->output_q16[row] != expected[row] >> 24) {
            fprintf(stderr, "arena BF16 real row %d differs from Q24 input\n",
                    row);
            goto done;
        }
    }
    printf("arena BF16 model rows: %d exact Q24 outputs; first Q16=%lld\n",
           QWEN3_ARENA_BF16_ROWS, (long long)work->output_q16[0]);
    rc = 0;
done:
    safetensors_close(&file);
    return rc;
}

int main(int argc, char **argv)
{
    struct qwen3_arena_bf16_bpf *skel;
    struct qwen3_arena_bf16_work *work;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(*work) + page - 1) / page) * page;
    int rc = 1;
    int32_t expected[2];

    if (argc > 3 || (argc == 3 && strcmp(argv[1], "--xdp-model") &&
                      strcmp(argv[1], "--xdp-attention")))
        return 2;
    skel = qwen3_arena_bf16_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "arena BF16 BPF object load failed\n");
        return 1;
    }
    work = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED,
                bpf_map__fd(skel->maps.work), 0);
    if (work == MAP_FAILED) {
        perror("arena BF16 work mmap");
        goto done;
    }
    populate_lut(skel);
    if (argc == 3) {
        int with_attention = !strcmp(argv[1], "--xdp-attention");

        if (!run_xdp_model(skel, work, argv[2], with_attention)) {
            puts(with_attention ? "arena BF16 XDP attention smoke passed" :
                                  "arena BF16 XDP model smoke passed");
            rc = 0;
        }
        goto unmap;
    }
    if (!run_synthetic(skel, work) &&
        !run_resident_synthetic(skel, work)) {
        expected[0] = (int32_t)work->output_q16[0];
        expected[1] = (int32_t)work->output_q16[1];
        if (argc == 1 ||
            (strcmp(argv[1], "--xdp-loopback") == 0 ?
             !run_xdp_event(skel, work, expected, 2, NULL, NULL, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                            NULL, NULL) :
             !run_model_rows(skel, work, argv[1]))) {
            puts("arena BF16 smoke passed");
            rc = 0;
        }
    }
unmap:
    munmap(work, length);
done:
    qwen3_arena_bf16_bpf__destroy(skel);
    return rc;
}
