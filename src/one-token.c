#include <limits.h>
#include <float.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_norm.h"
#include "qwen3_rope.h"
#include "qwen3_attention.h"
#include "qwen3_silu.h"
#include "qwen3_tile.h"
#include "qwen3_vector.h"
#include "safetensors.h"

#define QWEN3_LAYERS 28
#define QWEN3_INTERMEDIATE 3072
#define QWEN3_ATTENTION_WIDTH 2048
#define QWEN3_VOCAB 151936
#define QWEN3_Q_HEADS 16
#define QWEN3_KV_HEADS 8
#define QWEN3_CONTEXT_LIMIT 2

static int trace_enabled;

struct kernel_operator {
    struct bpf_object *object;
    int map_fd;
    int program_fd[3];
};

struct qwen3_engine {
    struct safetensors_file model;
    struct kernel_operator matrix;
    struct kernel_operator norm;
    struct kernel_operator silu;
    struct kernel_operator vector;
    struct kernel_operator rope;
    struct kernel_operator attention;
};

struct qwen3_cache {
    int32_t key[QWEN3_CONTEXT_LIMIT][QWEN3_LAYERS][QWEN3_KV_HEADS]
               [QWEN3_TILE_WIDTH];
    int32_t value[QWEN3_CONTEXT_LIMIT][QWEN3_LAYERS][QWEN3_KV_HEADS]
                 [QWEN3_TILE_WIDTH];
};

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
                         int count)
{
    struct bpf_map *map;
    int i;

    op->object = bpf_object__open_file(path, NULL);
    if (!op->object || libbpf_get_error(op->object)) {
        op->object = NULL;
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

static void close_engine(struct qwen3_engine *engine)
{
    bpf_object__close(engine->matrix.object);
    bpf_object__close(engine->norm.object);
    bpf_object__close(engine->silu.object);
    bpf_object__close(engine->vector.object);
    bpf_object__close(engine->rope.object);
    bpf_object__close(engine->attention.object);
    safetensors_close(&engine->model);
}

static int open_engine(struct qwen3_engine *engine, const char *model_path)
{
    static const char *matrix_programs[] = {"qwen3_matvec_tile"};
    static const char *norm_programs[] = {
        "qwen3_rms_accumulate", "qwen3_rms_finalize", "qwen3_rms_apply"
    };
    static const char *silu_programs[] = {"qwen3_silu_apply"};
    static const char *vector_programs[] = {
        "qwen3_vector_add", "qwen3_vector_multiply", "qwen3_vector_argmax"
    };
    static const char *rope_programs[] = {"qwen3_rope_apply"};
    static const char *attention_programs[] = {
        "qwen3_attention_score", "qwen3_attention_apply"
    };

    if (safetensors_open(&engine->model, model_path) ||
        open_operator(&engine->matrix, "build/qwen3_matvec.bpf.o",
                      "tile", matrix_programs, 1) ||
        open_operator(&engine->norm, "build/qwen3_norm.bpf.o",
                      "norm", norm_programs, 3) ||
        open_operator(&engine->silu, "build/qwen3_silu.bpf.o",
                      "silu", silu_programs, 1) ||
        open_operator(&engine->vector, "build/qwen3_vector.bpf.o",
                      "vector", vector_programs, 3) ||
        open_operator(&engine->rope, "build/qwen3_rope.bpf.o",
                      "rope", rope_programs, 1) ||
        open_operator(&engine->attention, "build/qwen3_attention.bpf.o",
                      "attention", attention_programs, 2))
        return -1;
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

static int convert_q24(float value, int32_t *out)
{
    double scaled = (double)value * 16777216.0;
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
    struct qwen3_norm_state work = {0};
    float weights[QWEN3_HIDDEN_SIZE];
    const uint32_t key = 0;
    int tile, i;

    if (count <= 0 || count > QWEN3_HIDDEN_SIZE ||
        count % QWEN3_TILE_WIDTH)
        return -1;
    work.total_tiles = count / QWEN3_TILE_WIDTH;

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
    for (tile = 0; tile < (int)work.total_tiles; tile++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work.activation_q16[i] = input[tile * QWEN3_TILE_WIDTH + i];
        if (bpf_map_update_elem(engine->norm.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->norm.program_fd[0]) ||
            bpf_map_lookup_elem(engine->norm.map_fd, &key, &work))
            return -1;
    }
    if (work.completed_tiles != work.total_tiles ||
        call_kernel(engine->norm.program_fd[1]) ||
        bpf_map_lookup_elem(engine->norm.map_fd, &key, &work) ||
        !work.inv_rms_q16)
        return -1;
    if (trace_enabled)
        fprintf(stderr, "norm %s sum_sq_q32=%llu inv_rms_q16=%llu\n",
                name, (unsigned long long)work.sum_sq_q32,
                (unsigned long long)work.inv_rms_q16);
    for (tile = 0; tile < (int)work.total_tiles; tile++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int index = tile * QWEN3_TILE_WIDTH + i;
            work.activation_q16[i] = input[index];
            if (convert_q20(weights[index], &work.weight_q20[i]))
                return -1;
        }
        if (bpf_map_update_elem(engine->norm.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->norm.program_fd[2]) ||
            bpf_map_lookup_elem(engine->norm.map_fd, &key, &work))
            return -1;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            output[tile * QWEN3_TILE_WIDTH + i] = work.output_q16[i];
    }
    return 0;
}

static int kernel_rope(struct qwen3_engine *engine, const int32_t *input,
                       int position, int32_t *output)
{
    struct qwen3_rope_state work = {0};
    const uint32_t key = 0;
    int i;

    memcpy(work.input_q16, input, sizeof(work.input_q16));
    for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
        double angle = position * pow(1000000.0, -(double)i / 64.0);
        work.cosine_q20[i] = (int32_t)round(cos(angle) * (1 << 20));
        work.sine_q20[i] = (int32_t)round(sin(angle) * (1 << 20));
    }
    if (bpf_map_update_elem(engine->rope.map_fd, &key, &work, BPF_ANY) ||
        call_kernel(engine->rope.program_fd[0]) ||
        bpf_map_lookup_elem(engine->rope.map_fd, &key, &work))
        return -1;
    memcpy(output, work.output_q16, sizeof(work.output_q16));
    return 0;
}

static int kernel_attention_head(struct qwen3_engine *engine,
                                 const struct qwen3_cache *cache,
                                 int layer, int position, int head,
                                 const int32_t *query, int32_t *output)
{
    struct qwen3_attention_state work = {0};
    const uint32_t key = 0;
    int past, kv_head = head / 2;

    work.context_length = position + 1;
    memcpy(work.query_q16, query, sizeof(work.query_q16));
    for (past = 0; past <= position; past++) {
        work.score_index = past;
        memcpy(work.key_q16, cache->key[past][layer][kv_head],
               sizeof(work.key_q16));
        memcpy(work.value_q16[past], cache->value[past][layer][kv_head],
               sizeof(work.value_q16[past]));
        if (bpf_map_update_elem(engine->attention.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->attention.program_fd[0]) ||
            bpf_map_lookup_elem(engine->attention.map_fd, &key, &work))
            return -1;
    }
    if (call_kernel(engine->attention.program_fd[1]) ||
        bpf_map_lookup_elem(engine->attention.map_fd, &key, &work))
        return -1;
    memcpy(output, work.output_q16, sizeof(work.output_q16));
    return 0;
}

static int kernel_matrix(struct qwen3_engine *engine, const char *name,
                         int rows, int cols, const int32_t *input,
                         int repeat_v, int32_t *output)
{
    uint64_t first_byte, elements;
    const uint32_t key = 0;
    float *weights;
    int row, tile, i, rc = -1;

    if (cols % QWEN3_TILE_WIDTH || rows <= 0 || cols <= 0 ||
        safetensors_find_bf16(&engine->model, name,
                              &first_byte, &elements) ||
        elements != (uint64_t)rows * cols) {
        fprintf(stderr, "matrix metadata mismatch for %s\n", name);
        return -1;
    }
    weights = malloc((size_t)cols * sizeof(*weights));
    if (!weights)
        return -1;
    for (row = 0; row < rows; row++) {
        struct qwen3_tile work = {0};
        if (safetensors_read_bf16_at(&engine->model, first_byte,
                (uint64_t)row * cols, (size_t)cols, weights)) {
            fprintf(stderr, "matrix read failed for %s row %d\n", name, row);
            goto done;
        }
        work.fixed_point_mode = 2;
        work.total_tiles = cols / QWEN3_TILE_WIDTH;
        for (tile = 0; tile < (int)work.total_tiles; tile++) {
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                int index = tile * QWEN3_TILE_WIDTH + i;
                int input_index = repeat_v
                    ? (index / QWEN3_TILE_WIDTH / 2) * QWEN3_TILE_WIDTH +
                      index % QWEN3_TILE_WIDTH
                    : index;
                work.activation_q16[i] = input[input_index];
                if (convert_q24(weights[index], &work.weight_q24[i])) {
                    fprintf(stderr, "matrix Q24 conversion failed for %s row %d col %d value %.9g\n",
                            name, row, index, weights[index]);
                    goto done;
                }
            }
            if (bpf_map_update_elem(engine->matrix.map_fd, &key, &work, BPF_ANY) ||
                call_kernel(engine->matrix.program_fd[0]) ||
                bpf_map_lookup_elem(engine->matrix.map_fd, &key, &work)) {
                perror("BPF matrix tile");
                fprintf(stderr, "matrix %s row %d tile %d\n", name, row, tile);
                goto done;
            }
        }
        if (work.completed_tiles != work.total_tiles ||
            work.output_q16 > INT32_MAX || work.output_q16 < INT32_MIN) {
            fprintf(stderr, "matrix %s row %d output out of Q16 range: %lld\n",
                    name, row, (long long)work.output_q16);
            goto done;
        }
        output[row] = (int32_t)work.output_q16;
    }
    rc = 0;
done:
    free(weights);
    return rc;
}

static int kernel_silu(struct qwen3_engine *engine, const int32_t *input,
                       int count, int32_t *output)
{
    struct qwen3_silu_state work = {0};
    const uint32_t key = 0;
    int tile, i;

    if (count % QWEN3_TILE_WIDTH)
        return -1;
    for (tile = 0; tile < count / QWEN3_TILE_WIDTH; tile++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work.input_q16[i] = input[tile * QWEN3_TILE_WIDTH + i];
        if (bpf_map_update_elem(engine->silu.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->silu.program_fd[0]) ||
            bpf_map_lookup_elem(engine->silu.map_fd, &key, &work))
            return -1;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            output[tile * QWEN3_TILE_WIDTH + i] = work.output_q16[i];
    }
    return 0;
}

static int kernel_vector(struct qwen3_engine *engine, int multiply,
                         const int32_t *left, const int32_t *right,
                         int count, int32_t *output)
{
    struct qwen3_vector_state work = {0};
    const uint32_t key = 0;
    int tile, i;

    if (count % QWEN3_TILE_WIDTH)
        return -1;
    for (tile = 0; tile < count / QWEN3_TILE_WIDTH; tile++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            int index = tile * QWEN3_TILE_WIDTH + i;
            work.left_q16[i] = left[index];
            work.right_q16[i] = right[index];
        }
        if (bpf_map_update_elem(engine->vector.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->vector.program_fd[multiply ? 1 : 0]) ||
            bpf_map_lookup_elem(engine->vector.map_fd, &key, &work))
            return -1;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            output[tile * QWEN3_TILE_WIDTH + i] = work.output_q16[i];
    }
    return 0;
}

static int kernel_argmax(struct qwen3_engine *engine, const int32_t *logits,
                         uint32_t *best_id, int32_t *best_logit)
{
    struct qwen3_vector_state work = {.best_q16 = INT32_MIN};
    const uint32_t key = 0;
    int tile, i;

    for (tile = 0; tile < QWEN3_VOCAB / QWEN3_TILE_WIDTH; tile++) {
        work.base_index = tile * QWEN3_TILE_WIDTH;
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work.left_q16[i] = logits[work.base_index + i];
        if (bpf_map_update_elem(engine->vector.map_fd, &key, &work, BPF_ANY) ||
            call_kernel(engine->vector.program_fd[2]) ||
            bpf_map_lookup_elem(engine->vector.map_fd, &key, &work))
            return -1;
    }
    *best_id = work.best_index;
    *best_logit = work.best_q16;
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
    struct qwen3_cache *cache = NULL;
    int32_t hidden[QWEN3_HIDDEN_SIZE], normed[QWEN3_HIDDEN_SIZE];
    int32_t query[QWEN3_ATTENTION_WIDTH], key_vectors[QWEN3_HIDDEN_SIZE];
    int32_t v[QWEN3_HIDDEN_SIZE], attended[QWEN3_ATTENTION_WIDTH];
    int32_t projected[QWEN3_HIDDEN_SIZE];
    int32_t gate[QWEN3_INTERMEDIATE], up[QWEN3_INTERMEDIATE];
    int32_t activated[QWEN3_INTERMEDIATE], product[QWEN3_INTERMEDIATE];
    int32_t down[QWEN3_HIDDEN_SIZE], *logits = NULL;
    char name[128];
    struct timespec start;
    uint32_t token_ids[QWEN3_CONTEXT_LIMIT], next_id;
    unsigned long parsed_token;
    char *endptr;
    const char *dump_path = NULL;
    int32_t best_logit;
    int layer, position, head, token_count = 0, arg, rc = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s model.safetensors token_id [token_id] [--dump-logits output.i32]\n", argv[0]);
        return 2;
    }
    arg = 2;
    while (arg < argc && strcmp(argv[arg], "--dump-logits") != 0 &&
           token_count < QWEN3_CONTEXT_LIMIT) {
        errno = 0;
        parsed_token = strtoul(argv[arg], &endptr, 10);
        if (errno || endptr == argv[arg] || *endptr ||
            parsed_token >= QWEN3_VOCAB) {
            fprintf(stderr, "input token ID must be below %d\n", QWEN3_VOCAB);
            return 2;
        }
        token_ids[token_count++] = (uint32_t)parsed_token;
        arg++;
    }
    if (!token_count || (arg < argc &&
        (strcmp(argv[arg], "--dump-logits") != 0 || arg + 2 != argc))) {
        fprintf(stderr, "expected one or two token IDs and optional --dump-logits path\n");
        return 2;
    }
    if (arg < argc)
        dump_path = argv[arg + 1];
    trace_enabled = getenv("QWEN3_TRACE") != NULL;
    if (open_engine(&engine, argv[1])) {
        fprintf(stderr, "could not load model or BPF operators\n");
        goto done;
    }
    logits = malloc((size_t)QWEN3_VOCAB * sizeof(*logits));
    cache = calloc(1, sizeof(*cache));
    if (!logits || !cache) {
        fprintf(stderr, "could not allocate inference state\n");
        goto done;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (position = 0; position < token_count; position++) {
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
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.q_norm.weight", layer);
        for (head = 0; head < QWEN3_Q_HEADS; head++) {
            if (kernel_norm(&engine, name,
                    query + head * QWEN3_TILE_WIDTH,
                    QWEN3_TILE_WIDTH,
                    query + head * QWEN3_TILE_WIDTH))
                goto layer_fail;
        }
        snprintf(name, sizeof(name),
                 "model.layers.%d.self_attn.k_norm.weight", layer);
        for (head = 0; head < QWEN3_KV_HEADS; head++) {
            if (kernel_norm(&engine, name,
                    key_vectors + head * QWEN3_TILE_WIDTH,
                    QWEN3_TILE_WIDTH,
                    key_vectors + head * QWEN3_TILE_WIDTH))
                goto layer_fail;
        }
        for (head = 0; head < QWEN3_Q_HEADS; head++) {
            if (kernel_rope(&engine, query + head * QWEN3_TILE_WIDTH,
                            position, query + head * QWEN3_TILE_WIDTH))
                goto layer_fail;
        }
        for (head = 0; head < QWEN3_KV_HEADS; head++) {
            if (kernel_rope(&engine, key_vectors + head * QWEN3_TILE_WIDTH,
                            position,
                            key_vectors + head * QWEN3_TILE_WIDTH))
                goto layer_fail;
        }
        memcpy(cache->key[position][layer], key_vectors,
               sizeof(key_vectors));
        memcpy(cache->value[position][layer], v, sizeof(v));
        for (head = 0; head < QWEN3_Q_HEADS; head++) {
            if (kernel_attention_head(&engine, cache, layer, position, head,
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
        printf("position %d/%d layer %d/%d complete (%.3f s)\n",
               position + 1, token_count, layer + 1, QWEN3_LAYERS,
               elapsed(&start));
        fflush(stdout);
        }
    }
    if (kernel_norm(&engine, "model.norm.weight", hidden,
                    QWEN3_HIDDEN_SIZE, normed) ||
        kernel_matrix(&engine, "model.embed_tokens.weight",
                      QWEN3_VOCAB, QWEN3_HIDDEN_SIZE,
                      normed, 0, logits) ||
        kernel_argmax(&engine, logits, &next_id, &best_logit)) {
        fprintf(stderr, "final norm, vocabulary projection or argmax failed\n");
        goto done;
    }
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
    printf("context_length=%d last_input_token_id=%u next_token_id=%u best_logit_q16=%d elapsed=%.3f s\n",
           token_count, token_ids[token_count - 1], next_id,
           best_logit, elapsed(&start));
    rc = 0;
    goto done;
layer_fail:
    fprintf(stderr, "kernel forward failed at position %d layer %d near %s\n",
            position, layer, name);
done:
    free(cache);
    free(logits);
    close_engine(&engine);
    return rc;
}
