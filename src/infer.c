#include <limits.h>
#include <float.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_norm.h"
#include "qwen3_rope.h"
#include "qwen3_attention.h"
#include "qwen3_silu.h"
#include "qwen3_tile.h"
#include "qwen3_batch.h"
#include "qwen3_vector.h"
#include "qwen3_tokenizer.h"
#include "safetensors.h"

#define QWEN3_LAYERS 28
#define QWEN3_INTERMEDIATE 3072
#define QWEN3_ATTENTION_WIDTH 2048
#define QWEN3_VOCAB 151936
#define QWEN3_EOS_TOKEN_ID 151645
#define QWEN3_CONTEXT_LIMIT 40960

static int trace_enabled;

struct kernel_operator {
    struct bpf_object *object;
    int map_fd;
    int program_fd[3];
};

struct qwen3_engine {
    struct safetensors_file model;
    struct kernel_operator batch;
    void *batch_mapping;
    size_t batch_mapping_len;
    struct qwen3_batch_work *batch_work;
    void *norm_mapping;
    size_t norm_mapping_len;
    struct qwen3_norm_state *norm_work;
    void *qk_norm_mapping;
    size_t qk_norm_mapping_len;
    struct qwen3_qk_norm_state *qk_norm_work;
    int qk_norm_program_fd;
    void *silu_mapping;
    size_t silu_mapping_len;
    struct qwen3_silu_state *silu_work;
    void *vector_mapping;
    size_t vector_mapping_len;
    struct qwen3_vector_state *vector_work;
    void *rope_mapping;
    size_t rope_mapping_len;
    struct qwen3_rope_state *rope_work;
    void *attention_mapping;
    size_t attention_mapping_len;
    struct qwen3_attention_state *attention_work;
    void *kv_mapping;
    size_t kv_mapping_len;
    struct qwen3_kv_pair *kv_pairs;
    struct kernel_operator norm;
    struct kernel_operator silu;
    struct kernel_operator vector;
    struct kernel_operator rope;
    struct kernel_operator attention;
};

static size_t cache_offset(int position, int layer, int head)
{
    return ((size_t)position * QWEN3_LAYERS + layer) * QWEN3_KV_HEADS + head;
}

static int call_kernel(int fd)
{
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    return bpf_prog_test_run_opts(fd, &opts);
}

static int open_operator(struct kernel_operator *op, const char *path,
                         const char *map_name, const char *const *programs,
                         int count, uint32_t kv_slots)
{
    struct bpf_map *map;
    struct bpf_map *kv_map;
    int i;

    op->object = bpf_object__open_file(path, NULL);
    if (!op->object || libbpf_get_error(op->object)) {
        op->object = NULL;
        return -1;
    }
    if (kv_slots) {
        kv_map = bpf_object__find_map_by_name(op->object, "kv");
        if (!kv_map || bpf_map__set_max_entries(kv_map, kv_slots))
            return -1;
    }
    if (bpf_object__load(op->object))
        return -1;
    map = bpf_object__find_map_by_name(op->object, map_name);
    if (!map)
        return -1;
    op->map_fd = bpf_map__fd(map);
    for (i = 0; i < count; i++) {
        struct bpf_program *program =
            bpf_object__find_program_by_name(op->object, programs[i]);
        if (!program)
            return -1;
        op->program_fd[i] = bpf_program__fd(program);
    }
    return 0;
}

static void *map_shared(int fd, size_t bytes, size_t *length)
{
    long page = sysconf(_SC_PAGESIZE);
    void *mapping;

    if (page <= 0 || bytes > SIZE_MAX - (size_t)page)
        return NULL;
    *length = ((bytes + (size_t)page - 1) / (size_t)page) * (size_t)page;
    mapping = mmap(NULL, *length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return mapping == MAP_FAILED ? NULL : mapping;
}

static void close_engine(struct qwen3_engine *engine)
{
    if (engine->batch_mapping)
        munmap(engine->batch_mapping, engine->batch_mapping_len);
    if (engine->norm_mapping)
        munmap(engine->norm_mapping, engine->norm_mapping_len);
    if (engine->qk_norm_mapping)
        munmap(engine->qk_norm_mapping, engine->qk_norm_mapping_len);
    if (engine->silu_mapping)
        munmap(engine->silu_mapping, engine->silu_mapping_len);
    if (engine->vector_mapping)
        munmap(engine->vector_mapping, engine->vector_mapping_len);
    if (engine->rope_mapping)
        munmap(engine->rope_mapping, engine->rope_mapping_len);
    if (engine->attention_mapping)
        munmap(engine->attention_mapping, engine->attention_mapping_len);
    if (engine->kv_mapping)
        munmap(engine->kv_mapping, engine->kv_mapping_len);
    bpf_object__close(engine->batch.object);
    bpf_object__close(engine->norm.object);
    bpf_object__close(engine->silu.object);
    bpf_object__close(engine->vector.object);
    bpf_object__close(engine->rope.object);
    bpf_object__close(engine->attention.object);
    safetensors_close(&engine->model);
}

static int open_engine(struct qwen3_engine *engine, const char *model_path,
                       int total_positions)
{
    static const char *batch_programs[] = {"qwen3_batch_rows"};
    static const char *norm_programs[] = {"qwen3_rms_full"};
    static const char *silu_programs[] = {"qwen3_silu_apply"};
    static const char *vector_programs[] = {
        "qwen3_vector_add", "qwen3_vector_multiply"
    };
    static const char *rope_programs[] = {"qwen3_rope_apply"};
    static const char *attention_programs[] = {
        "qwen3_attention_step", "qwen3_attention_cached"
    };
    struct bpf_map *kv_map;
    struct bpf_map *qk_norm_map;
    struct bpf_program *qk_norm_program;
    uint32_t kv_slots = (uint32_t)total_positions * QWEN3_LAYERS * QWEN3_KV_HEADS;

    if (safetensors_open(&engine->model, model_path) ||
        open_operator(&engine->batch, "build/qwen3_batch.bpf.o",
                      "batch", batch_programs, 1, 0) ||
        open_operator(&engine->norm, "build/qwen3_norm.bpf.o",
                      "norm", norm_programs, 1, 0) ||
        open_operator(&engine->silu, "build/qwen3_silu.bpf.o",
                      "silu", silu_programs, 1, 0) ||
        open_operator(&engine->vector, "build/qwen3_vector.bpf.o",
                      "vector", vector_programs, 2, 0) ||
        open_operator(&engine->rope, "build/qwen3_rope.bpf.o",
                      "rope", rope_programs, 1, 0) ||
        open_operator(&engine->attention, "build/qwen3_attention.bpf.o",
                      "attention", attention_programs, 2, kv_slots))
        return -1;
    kv_map = bpf_object__find_map_by_name(engine->attention.object, "kv");
    qk_norm_map = bpf_object__find_map_by_name(engine->norm.object, "qk_norm");
    qk_norm_program = bpf_object__find_program_by_name(
        engine->norm.object, "qwen3_qk_norm_heads");
    if (!kv_map || !qk_norm_map || !qk_norm_program)
        return -1;
    engine->qk_norm_program_fd = bpf_program__fd(qk_norm_program);
    engine->batch_mapping = map_shared(engine->batch.map_fd,
        sizeof(struct qwen3_batch_work), &engine->batch_mapping_len);
    engine->norm_mapping = map_shared(engine->norm.map_fd,
        sizeof(struct qwen3_norm_state), &engine->norm_mapping_len);
    engine->qk_norm_mapping = map_shared(bpf_map__fd(qk_norm_map),
        sizeof(struct qwen3_qk_norm_state), &engine->qk_norm_mapping_len);
    engine->silu_mapping = map_shared(engine->silu.map_fd,
        sizeof(struct qwen3_silu_state), &engine->silu_mapping_len);
    engine->vector_mapping = map_shared(engine->vector.map_fd,
        sizeof(struct qwen3_vector_state), &engine->vector_mapping_len);
    engine->rope_mapping = map_shared(engine->rope.map_fd,
        sizeof(struct qwen3_rope_state), &engine->rope_mapping_len);
    engine->attention_mapping = map_shared(engine->attention.map_fd,
        sizeof(struct qwen3_attention_state), &engine->attention_mapping_len);
    engine->kv_mapping = map_shared(bpf_map__fd(kv_map),
        (size_t)kv_slots * sizeof(struct qwen3_kv_pair),
        &engine->kv_mapping_len);
    if (!engine->batch_mapping || !engine->norm_mapping ||
        !engine->qk_norm_mapping ||
        !engine->silu_mapping ||
        !engine->vector_mapping || !engine->rope_mapping ||
        !engine->attention_mapping ||
        !engine->kv_mapping)
        return -1;
    engine->batch_work = engine->batch_mapping;
    engine->norm_work = engine->norm_mapping;
    engine->qk_norm_work = engine->qk_norm_mapping;
    engine->silu_work = engine->silu_mapping;
    engine->vector_work = engine->vector_mapping;
    engine->rope_work = engine->rope_mapping;
    engine->attention_work = engine->attention_mapping;
    engine->kv_pairs = engine->kv_mapping;
    return 0;
}

static int convert_q16(float value, int32_t *out)
{
    double scaled = (double)value * 65536.0;
    if (scaled > INT32_MAX || scaled < INT32_MIN)
        return -1;
    *out = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
    return 0;
}

static int convert_q20(float value, int32_t *out)
{
    double scaled = (double)value * 1048576.0;
    if (scaled > INT32_MAX || scaled < INT32_MIN)
        return -1;
    *out = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
    return 0;
}

static int load_embedding(struct qwen3_engine *engine, uint32_t token_id,
                          int32_t *hidden)
{
    uint64_t first_byte, elements;
    float row[QWEN3_HIDDEN_SIZE];
    int i;

    if (token_id >= QWEN3_VOCAB ||
        safetensors_find_bf16(&engine->model, "model.embed_tokens.weight",
                              &first_byte, &elements) ||
        elements != (uint64_t)QWEN3_VOCAB * QWEN3_HIDDEN_SIZE ||
        safetensors_read_bf16_at(&engine->model, first_byte,
                                 (uint64_t)token_id * QWEN3_HIDDEN_SIZE,
                                 QWEN3_HIDDEN_SIZE, row))
        return -1;
    for (i = 0; i < QWEN3_HIDDEN_SIZE; i++)
        if (convert_q16(row[i], &hidden[i]))
            return -1;
    return 0;
}

static int kernel_norm(struct qwen3_engine *engine, const char *name,
                       const int32_t *input, int count, int32_t *output)
{
    struct qwen3_norm_state *work = engine->norm_work;
    float weights[QWEN3_HIDDEN_SIZE];
    int i;

    if (count <= 0 || count > QWEN3_HIDDEN_SIZE ||
        count % QWEN3_TILE_WIDTH)
        return -1;
    memset(work, 0, sizeof(*work));
    work->total_tiles = count / QWEN3_TILE_WIDTH;

    {
        uint64_t first_byte, elements;
        if (safetensors_find_bf16(&engine->model, name,
                                  &first_byte, &elements) ||
            elements != (uint64_t)count ||
            safetensors_read_bf16_at(&engine->model, first_byte, 0,
                                     (size_t)count, weights))
            return -1;
    }
    if (trace_enabled) {
        float low = FLT_MAX, high = -FLT_MAX;
        for (i = 0; i < count; i++) {
            if (weights[i] < low) low = weights[i];
            if (weights[i] > high) high = weights[i];
        }
        fprintf(stderr, "norm %s weight range: %.6g .. %.6g\n", name, low, high);
    }
    for (i = 0; i < count; i++) {
        work->activation_q16[i] = input[i];
        if (convert_q20(weights[i], &work->weight_q20[i]))
            return -1;
    }
    if (call_kernel(engine->norm.program_fd[0]) ||
        work->completed_tiles != 2 * work->total_tiles ||
        !work->inv_rms_q16)
        return -1;
    if (trace_enabled)
        fprintf(stderr, "norm %s sum_sq_q32=%llu inv_rms_q16=%llu\n",
                name, (unsigned long long)work->sum_sq_q32,
                (unsigned long long)work->inv_rms_q16);
    memcpy(output, work->output_q16, (size_t)count * sizeof(*output));
    return 0;
}

static int kernel_qk_norm(struct qwen3_engine *engine, int layer,
                          int32_t *query, int32_t *key_vectors)
{
    struct qwen3_qk_norm_state *work = engine->qk_norm_work;
    float weights[QWEN3_TILE_WIDTH];
    char name[128];
    uint64_t first_byte, elements;
    int set, i;

    memcpy(work->activation_q16, query, sizeof(*query) * QWEN3_ATTENTION_WIDTH);
    memcpy(work->activation_q16 + QWEN3_ATTENTION_WIDTH, key_vectors,
           sizeof(*key_vectors) * QWEN3_HIDDEN_SIZE);
    for (set = 0; set < 2; set++) {
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.%s_norm.weight",
                 layer, set ? "k" : "q");
        if (safetensors_find_bf16(&engine->model, name,
                                  &first_byte, &elements) ||
            elements != QWEN3_TILE_WIDTH ||
            safetensors_read_bf16_at(&engine->model, first_byte, 0,
                                     QWEN3_TILE_WIDTH, weights))
            return -1;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            if (convert_q20(weights[i], &work->weight_q20[set][i]))
                return -1;
    }
    work->completed_heads = 0;
    if (call_kernel(engine->qk_norm_program_fd) ||
        work->completed_heads != QWEN3_QK_HEADS)
        return -1;
    memcpy(query, work->output_q16,
           sizeof(*query) * QWEN3_ATTENTION_WIDTH);
    memcpy(key_vectors, work->output_q16 + QWEN3_ATTENTION_WIDTH,
           sizeof(*key_vectors) * QWEN3_HIDDEN_SIZE);
    return 0;
}

static int kernel_rope(struct qwen3_engine *engine, int32_t *query,
                       int32_t *key_vectors, int position)
{
    struct qwen3_rope_state *work = engine->rope_work;
    int i;

    memcpy(work->input_q16, query,
           QWEN3_ATTENTION_WIDTH * sizeof(*query));
    memcpy(work->input_q16 + QWEN3_ATTENTION_WIDTH, key_vectors,
           QWEN3_HIDDEN_SIZE * sizeof(*key_vectors));
    work->heads = QWEN3_ROPE_HEADS;
    work->completed = 0;
    for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
        double angle = position * pow(1000000.0, -(double)i / 64.0);
        work->cosine_q20[i] = (int32_t)round(cos(angle) * (1 << 20));
        work->sine_q20[i] = (int32_t)round(sin(angle) * (1 << 20));
    }
    if (call_kernel(engine->rope.program_fd[0]) ||
        work->completed != QWEN3_ROPE_HEADS)
        return -1;
    memcpy(query, work->output_q16,
           QWEN3_ATTENTION_WIDTH * sizeof(*query));
    memcpy(key_vectors, work->output_q16 + QWEN3_ATTENTION_WIDTH,
           QWEN3_HIDDEN_SIZE * sizeof(*key_vectors));
    return 0;
}

static int kernel_attention_head(struct qwen3_engine *engine,
                                 int layer, int position, int head,
                                 const int32_t *query, int32_t *output)
{
    struct qwen3_attention_state *work = engine->attention_work;
    int start;

    memset(work, 0, sizeof(*work));
    memcpy(work->query_q16, query, sizeof(work->query_q16));
    work->layer = (uint32_t)layer;
    work->kv_head = (uint32_t)(head / 2);
    for (start = 0; start <= position; start += QWEN3_ATTENTION_CHUNK) {
        work->base_position = (uint32_t)start;
        work->step_count = (uint32_t)(position + 1 - start < QWEN3_ATTENTION_CHUNK
            ? position + 1 - start : QWEN3_ATTENTION_CHUNK);
        if (call_kernel(engine->attention.program_fd[1]) ||
            work->seen != (uint32_t)(start + work->step_count))
            return -1;
    }
    memcpy(output, work->output_q16, sizeof(work->output_q16));
    return 0;
}

static int kernel_matrix(struct qwen3_engine *engine, const char *name,
                         int rows, int cols, const int32_t *input,
                         int repeat_v, int32_t *output)
{
    uint64_t first_byte, elements;
    struct qwen3_batch_work *work = engine->batch_work;
    int row, batch_row, i, rc = -1;

    if (cols % QWEN3_TILE_WIDTH || rows <= 0 || cols <= 0 ||
        cols > QWEN3_BATCH_COLS ||
        safetensors_find_bf16(&engine->model, name,
                              &first_byte, &elements) ||
        elements != (uint64_t)rows * cols) {
        fprintf(stderr, "matrix metadata mismatch for %s\n", name);
        return -1;
    }
    work->cols = (uint32_t)cols;
    for (i = 0; i < cols; i++)
        work->input_q16[i] = input[repeat_v
            ? (i / QWEN3_TILE_WIDTH / 2) * QWEN3_TILE_WIDTH +
              i % QWEN3_TILE_WIDTH
            : i];
    for (row = 0; row < rows; row += QWEN3_BATCH_ROWS) {
        work->rows = (uint32_t)(rows - row < QWEN3_BATCH_ROWS
            ? rows - row : QWEN3_BATCH_ROWS);
        work->base_index = (uint32_t)row;
        work->completed = 0;
        for (batch_row = 0; batch_row < (int)work->rows; batch_row++) {
            if (safetensors_read_bf16_q24_at(&engine->model, first_byte,
                    (uint64_t)(row + batch_row) * cols,
                    (size_t)cols, work->weight_q24[batch_row])) {
                fprintf(stderr, "matrix Q24 read failed for %s row %d\n",
                        name, row + batch_row);
                goto done;
            }
        }
        if (call_kernel(engine->batch.program_fd[0])) {
            perror("BPF matrix batch");
            fprintf(stderr, "matrix %s row %d\n", name, row);
            goto done;
        }
        if (work->completed != work->rows)
            goto done;
        for (batch_row = 0; batch_row < (int)work->rows; batch_row++) {
            if (work->output_q16[batch_row] > INT32_MAX ||
                work->output_q16[batch_row] < INT32_MIN)
                goto done;
            output[row + batch_row] = (int32_t)work->output_q16[batch_row];
        }
    }
    rc = 0;
done:
    return rc;
}

static int kernel_silu(struct qwen3_engine *engine, const int32_t *input,
                       int count, int32_t *output)
{
    struct qwen3_silu_state *work = engine->silu_work;

    if (count <= 0 || count > QWEN3_SILU_MAX ||
        count % QWEN3_TILE_WIDTH)
        return -1;
    memcpy(work->input_q16, input, (size_t)count * sizeof(*input));
    work->count = (uint32_t)count;
    work->completed_tiles = 0;
    if (call_kernel(engine->silu.program_fd[0]) ||
        work->completed_tiles != (uint32_t)(count / QWEN3_TILE_WIDTH))
        return -1;
    memcpy(output, work->output_q16, (size_t)count * sizeof(*output));
    return 0;
}

static int kernel_vector(struct qwen3_engine *engine, int multiply,
                         const int32_t *left, const int32_t *right,
                         int count, int32_t *output)
{
    struct qwen3_vector_state *work = engine->vector_work;

    if (count <= 0 || count > QWEN3_VECTOR_MAX ||
        count % QWEN3_TILE_WIDTH)
        return -1;
    memcpy(work->left_q16, left, (size_t)count * sizeof(*left));
    memcpy(work->right_q16, right, (size_t)count * sizeof(*right));
    work->count = (uint32_t)count;
    work->completed_tiles = 0;
    if (call_kernel(engine->vector.program_fd[multiply ? 1 : 0]) ||
        work->completed_tiles != (uint32_t)(count / QWEN3_TILE_WIDTH))
        return -1;
    memcpy(output, work->output_q16, (size_t)count * sizeof(*output));
    return 0;
}

static double elapsed(const struct timespec *start)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec - start->tv_sec +
           (now.tv_nsec - start->tv_nsec) / 1e9;
}

static void diagnostic_range(const char *name, const int32_t *values, int count)
{
    int32_t low = INT32_MAX, high = INT32_MIN;
    int i;
    for (i = 0; i < count; i++) {
        if (values[i] < low)
            low = values[i];
        if (values[i] > high)
            high = values[i];
    }
    fprintf(stderr, "%s Q16 range: %.6g .. %.6g\n",
            name, low / 65536.0, high / 65536.0);
}

int main(int argc, char **argv)
{
    struct qwen3_engine engine = {0};
    struct qwen3_tokenizer *tokenizer = NULL;
    int32_t hidden[QWEN3_HIDDEN_SIZE], normed[QWEN3_HIDDEN_SIZE];
    int32_t query[QWEN3_ATTENTION_WIDTH], key_vectors[QWEN3_HIDDEN_SIZE];
    int32_t v[QWEN3_HIDDEN_SIZE], attended[QWEN3_ATTENTION_WIDTH];
    int32_t projected[QWEN3_HIDDEN_SIZE];
    int32_t gate[QWEN3_INTERMEDIATE], up[QWEN3_INTERMEDIATE];
    int32_t activated[QWEN3_INTERMEDIATE], product[QWEN3_INTERMEDIATE];
    int32_t down[QWEN3_HIDDEN_SIZE], *logits = NULL;
    char name[128];
    struct timespec start;
    uint32_t *token_ids = NULL, next_id = 0;
    unsigned long parsed_token;
    char *endptr;
    const char *dump_path = NULL;
    FILE *report = stdout;
    int32_t best_logit = 0;
    int layer, position, head, token_count = 0, generate_count = 1;
    int total_positions, emitted_count = 0, last_processed = -1;
    int arg, rc = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s model.safetensors token_id... [--generate count] [--dump-logits output.i32]\n"
                        "   or: %s model.safetensors --tokenizer tokenizer.json --prompt text [--generate count] [--dump-logits output.i32]\n",
                argv[0], argv[0]);
        return 2;
    }
    if (argc >= 6 && strcmp(argv[2], "--tokenizer") == 0 &&
        strcmp(argv[4], "--prompt") == 0) {
        size_t count;
        tokenizer = qwen3_tokenizer_open(argv[3]);
        if (!tokenizer ||
            qwen3_tokenizer_encode(tokenizer, argv[5], &token_ids, &count) ||
            !count || count > QWEN3_CONTEXT_LIMIT) {
            fprintf(stderr, "could not encode prompt within model context\n");
            goto done;
        }
        token_count = (int)count;
        report = stderr;
        arg = 6;
    } else {
        token_ids = calloc((size_t)argc - 2, sizeof(*token_ids));
        if (!token_ids)
            return 1;
        arg = 2;
        while (arg < argc && strncmp(argv[arg], "--", 2) != 0 &&
               token_count < QWEN3_CONTEXT_LIMIT) {
            errno = 0;
            parsed_token = strtoul(argv[arg], &endptr, 10);
            if (errno || endptr == argv[arg] || *endptr ||
                parsed_token >= QWEN3_VOCAB) {
                fprintf(stderr, "input token ID must be below %d\n", QWEN3_VOCAB);
                rc = 2;
                goto done;
            }
            token_ids[token_count++] = (uint32_t)parsed_token;
            arg++;
        }
    }
    if (arg < argc && strcmp(argv[arg], "--generate") == 0 && arg + 1 < argc) {
        errno = 0;
        parsed_token = strtoul(argv[arg + 1], &endptr, 10);
        if (errno || endptr == argv[arg + 1] || *endptr ||
            parsed_token < 1 || parsed_token > QWEN3_CONTEXT_LIMIT) {
            fprintf(stderr, "generation count must be between 1 and %d\n",
                    QWEN3_CONTEXT_LIMIT);
            rc = 2;
            goto done;
        }
        generate_count = (int)parsed_token;
        arg += 2;
    }
    if (arg < argc && strcmp(argv[arg], "--dump-logits") == 0 &&
        arg + 1 < argc) {
        dump_path = argv[arg + 1];
        arg += 2;
    }
    if (!token_count || arg != argc ||
        generate_count > QWEN3_CONTEXT_LIMIT - token_count + 1) {
        fprintf(stderr, "expected token IDs and options within %d positions\n",
                QWEN3_CONTEXT_LIMIT);
        rc = 2;
        goto done;
    }
    total_positions = token_count + generate_count - 1;
    {
        uint32_t *resized = realloc(token_ids,
                                     ((size_t)total_positions + 1) * sizeof(*token_ids));
        if (!resized)
            goto done;
        token_ids = resized;
    }
    trace_enabled = getenv("QWEN3_TRACE") != NULL;
    if (open_engine(&engine, argv[1], total_positions)) {
        fprintf(stderr, "could not load model or BPF operators\n");
        goto done;
    }
    logits = malloc((size_t)QWEN3_VOCAB * sizeof(*logits));
    if (!logits) {
        fprintf(stderr, "could not allocate inference state\n");
        goto done;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (position = 0; position < total_positions; position++) {
        if (load_embedding(&engine, token_ids[position], hidden)) {
            fprintf(stderr, "could not load embedding for position %d\n", position);
            goto done;
        }
        if (trace_enabled)
            diagnostic_range("embedding", hidden, QWEN3_HIDDEN_SIZE);
        for (layer = 0; layer < QWEN3_LAYERS; layer++) {
        snprintf(name, sizeof(name),
                 "model.layers.%d.input_layernorm.weight", layer);
        if (kernel_norm(&engine, name, hidden, QWEN3_HIDDEN_SIZE, normed))
            goto layer_fail;
        if (trace_enabled && layer == 0)
            diagnostic_range("input norm", normed, QWEN3_HIDDEN_SIZE);
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.q_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_ATTENTION_WIDTH,
                          QWEN3_HIDDEN_SIZE, normed, 0, query))
            goto layer_fail;
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.k_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_HIDDEN_SIZE,
                          QWEN3_HIDDEN_SIZE, normed, 0, key_vectors))
            goto layer_fail;
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.v_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_HIDDEN_SIZE,
                          QWEN3_HIDDEN_SIZE, normed, 0, v))
            goto layer_fail;
        if (trace_enabled && layer == 0)
            diagnostic_range("V projection", v, QWEN3_HIDDEN_SIZE);
        if (kernel_qk_norm(&engine, layer, query, key_vectors))
            goto layer_fail;
        if (kernel_rope(&engine, query, key_vectors, position))
            goto layer_fail;
        for (head = 0; head < QWEN3_KV_HEADS; head++) {
            struct qwen3_kv_pair *pair = engine.kv_pairs +
                cache_offset(position, layer, head);
            memcpy(pair->key_q16, key_vectors + head * QWEN3_TILE_WIDTH,
                   sizeof(pair->key_q16));
            memcpy(pair->value_q16, v + head * QWEN3_TILE_WIDTH,
                   sizeof(pair->value_q16));
        }
        for (head = 0; head < QWEN3_Q_HEADS; head++) {
            if (kernel_attention_head(&engine, layer, position, head,
                    query + head * QWEN3_TILE_WIDTH,
                    attended + head * QWEN3_TILE_WIDTH))
                goto layer_fail;
        }
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.o_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_HIDDEN_SIZE,
                          QWEN3_ATTENTION_WIDTH, attended, 0, projected) ||
            kernel_vector(&engine, 0, hidden, projected,
                          QWEN3_HIDDEN_SIZE, hidden))
            goto layer_fail;
        if (trace_enabled && layer == 0)
            diagnostic_range("attention residual", hidden, QWEN3_HIDDEN_SIZE);
        snprintf(name, sizeof(name),
                 "model.layers.%d.post_attention_layernorm.weight", layer);
        if (kernel_norm(&engine, name, hidden, QWEN3_HIDDEN_SIZE, normed))
            goto layer_fail;
        if (trace_enabled && layer == 0)
            diagnostic_range("post-attention norm", normed, QWEN3_HIDDEN_SIZE);
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.gate_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_INTERMEDIATE,
                          QWEN3_HIDDEN_SIZE, normed, 0, gate))
            goto layer_fail;
        if (trace_enabled && layer == 0)
            diagnostic_range("gate projection", gate, QWEN3_INTERMEDIATE);
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.up_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_INTERMEDIATE,
                          QWEN3_HIDDEN_SIZE, normed, 0, up) ||
            kernel_silu(&engine, gate, QWEN3_INTERMEDIATE, activated) ||
            kernel_vector(&engine, 1, activated, up,
                          QWEN3_INTERMEDIATE, product))
            goto layer_fail;
        if (trace_enabled && layer == 0) {
            diagnostic_range("up projection", up, QWEN3_INTERMEDIATE);
            diagnostic_range("SiLU gate", activated, QWEN3_INTERMEDIATE);
            diagnostic_range("MLP product", product, QWEN3_INTERMEDIATE);
        }
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.down_proj.weight", layer);
        if (kernel_matrix(&engine, name, QWEN3_HIDDEN_SIZE,
                          QWEN3_INTERMEDIATE, product, 0, down) ||
            kernel_vector(&engine, 0, hidden, down,
                          QWEN3_HIDDEN_SIZE, hidden))
            goto layer_fail;
        fprintf(report, "position %d/%d layer %d/%d complete (%.3f s)\n",
               position + 1, total_positions, layer + 1, QWEN3_LAYERS,
               elapsed(&start));
        fflush(report);
        }
        if (position < token_count - 1)
            continue;
        engine.batch_work->track_argmax = 1;
        engine.batch_work->best_q16 = INT32_MIN;
        engine.batch_work->best_index = 0;
        if (kernel_norm(&engine, "model.norm.weight", hidden,
                    QWEN3_HIDDEN_SIZE, normed) ||
            kernel_matrix(&engine, "model.embed_tokens.weight",
                      QWEN3_VOCAB, QWEN3_HIDDEN_SIZE,
                      normed, 0, logits)) {
            fprintf(stderr, "final norm, vocabulary projection or argmax failed\n");
            goto done;
        }
        next_id = engine.batch_work->best_index;
        best_logit = (int32_t)engine.batch_work->best_q16;
        engine.batch_work->track_argmax = 0;
        if (dump_path) {
        FILE *dump = fopen(dump_path, "wb");
        size_t written;
        int close_rc;
        if (!dump) {
            fprintf(stderr, "could not write logit dump\n");
            goto done;
        }
        written = fwrite(logits, sizeof(*logits), QWEN3_VOCAB, dump);
        close_rc = fclose(dump);
        if (written != QWEN3_VOCAB || close_rc) {
            fprintf(stderr, "could not write logit dump\n");
            goto done;
        }
        }
        emitted_count++;
        last_processed = position;
        fprintf(report, "generated_token=%d token_id=%u logit_q16=%d\n",
               emitted_count, next_id, best_logit);
        fflush(report);
        if (tokenizer && next_id != QWEN3_EOS_TOKEN_ID) {
            unsigned char *piece = NULL;
            size_t length;
            if (qwen3_tokenizer_decode(tokenizer, next_id, &piece, &length)) {
                fprintf(stderr, "could not decode generated token %u\n", next_id);
                goto done;
            }
            if (fwrite(piece, 1, length, stdout) != length) {
                free(piece);
                fprintf(stderr, "could not write decoded token\n");
                goto done;
            }
            fflush(stdout);
            free(piece);
        }
        if (next_id == QWEN3_EOS_TOKEN_ID)
            break;
        if (position + 1 < total_positions)
            token_ids[position + 1] = next_id;
    }
    if (tokenizer)
        fputc('\n', stdout);
    fprintf(report, "prompt_tokens=%d generated_tokens=%d last_context_length=%d last_input_token_id=%u next_token_id=%u best_logit_q16=%d elapsed=%.3f s\n",
           token_count, emitted_count, last_processed + 1,
           token_ids[last_processed], next_id, best_logit, elapsed(&start));
    rc = 0;
    goto done;
layer_fail:
    fprintf(stderr, "kernel forward failed at position %d layer %d near %s\n",
            position, layer, name);
done:
    free(token_ids);
    free(logits);
    qwen3_tokenizer_close(tokenizer);
    close_engine(&engine);
    return rc;
}
