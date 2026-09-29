#include <limits.h>
#include <math.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_arena_bf16.h"
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
    if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
        return -1;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        __s64 expected = 0;
        for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++) {
            __u16 bits = skel->arena->weight_bf16[row][col];
            expected += (__s64)work->input_q16[col] *
                        skel->arena->q24_by_bf16[bits];
        }
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "arena BF16 row %d mismatch\n", row);
            return -1;
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
                         const int32_t *expected_sine)
{
    const __u32 key = 0;
    unsigned char payload[12] = {'Q', '3', 'B', 'P'};
    const char ignored[] = "Q3XX";
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    struct qwen3_event_state job;
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
    for (attempt = 0; attempt < 10; attempt++) {
        uint32_t expected_stride = work->event_qkv ? 4096 : total;
        const int32_t *request_expected = expected +
            (work->event_rope ? (attempt % 4) * expected_stride :
             work->event_use_token ? (attempt % 2) * expected_stride : 0);
        uint32_t token_id = work->event_use_token ? token_ids[attempt % 2] : 0;
        size_t payload_size = work->event_rope ? 12 :
                              work->event_use_token ? 8 : 4;
        uint32_t position = rope_positions[attempt % 4];

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
            (work->event_qkv && work->event_stage != QWEN3_EVENT_STAGE_V) ||
            (work->event_qkv && work->completed_qk_heads != 24) ||
            (work->event_rope &&
             (work->completed_rope_heads != 24 ||
              work->event_position != position)) ||
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
                        expected_cosine[(attempt % 4) * 64 + i] ||
                    work->rope_sine_q20[i] !=
                        expected_sine[(attempt % 4) * 64 + i]) {
                    fprintf(stderr, "XDP RoPE coefficient %u mismatch at position %u: cos=%d/%d sin=%d/%d\n",
                            i, position, work->rope_cosine_q20[i],
                            expected_cosine[(attempt % 4) * 64 + i],
                            work->rope_sine_q20[i],
                            expected_sine[(attempt % 4) * 64 + i]);
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
        if (work->event_use_token) {
            for (i = 0; i < QWEN3_ARENA_TOKEN_WIDTH; i++) {
                if (work->input_q16[i] !=
                    expected_inputs[(attempt % 2) * QWEN3_ARENA_TOKEN_WIDTH + i]) {
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
               attempt + 1, expected_stride, (double)elapsed_ns / 1000000.0,
               (double)(job.finish_ns - job.start_ns) / 1000000.0);
    }
    rc = 0;
done:
    if (fd >= 0)
        close(fd);
    bpf_link__destroy(link);
    return rc;
}

static int run_xdp_model(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work,
                         const char *path)
{
    struct safetensors_file file;
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    uint64_t projection_byte[3], projection_elements[3], projection_first[3];
    uint64_t embedding_byte, embedding_elements, cursor;
    uint64_t norm_byte, norm_elements, norm_first;
    uint64_t head_norm_byte[2], head_norm_first[2], head_norm_elements[2];
    int32_t q24[1024], hidden[2][1024], normalized[2][1024];
    int32_t norm_q20[1024], head_norm_q20[2][128];
    int32_t expected[2][4096], norm_expected[2][4096];
    int32_t event_expected[4][4096];
    int32_t rope_cosine[4][64], rope_sine[4][64];
    int32_t direct_outputs[4096];
    float embedding[1024], norm_weights[1024], head_norm_weights[2][128];
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
    embedding_raw = file.mapping + 8 + file.header_length + embedding_byte;
    norm_raw = file.mapping + 8 + file.header_length + norm_byte;
    memset(work, 0, sizeof(*work));
    work->model_elements = embedding_elements + norm_elements;
    for (projection = 0; projection < 3; projection++)
        work->model_elements += projection_elements[projection];
    work->model_elements += head_norm_elements[0] + head_norm_elements[1];
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_allocate_model),
            &opts) || opts.retval || !skel->bss->model_bf16) {
        fprintf(stderr, "XDP model arena allocation failed\n");
        goto done;
    }
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
    for (col = 0; col < 1024; col++) {
        double scaled = (double)norm_weights[col] * 1048576.0;

        if (!isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN)
            goto done;
        norm_q20[col] = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
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
                       &rope_cosine[0][0], &rope_sine[0][0]);
done:
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

    if (argc > 3 || (argc == 3 && strcmp(argv[1], "--xdp-model")))
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
        if (!run_xdp_model(skel, work, argv[2])) {
            puts("arena BF16 XDP model smoke passed");
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
