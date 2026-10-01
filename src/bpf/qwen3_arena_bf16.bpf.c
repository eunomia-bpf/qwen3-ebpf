#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_arena_bf16.h"
#include "qwen3_attention.h"

#define __arena __attribute__((address_space(1)))

struct {
    __uint(type, BPF_MAP_TYPE_ARENA);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, QWEN3_ARENA_MODEL_MAX_BF16 / 2048u + 512u);
#ifdef __TARGET_ARCH_arm64
    __ulong(map_extra, 0x1ull << 32);
#else
    __ulong(map_extra, 0x1ull << 44);
#endif
} arena SEC(".maps");

__u16 __arena weight_bf16[QWEN3_ARENA_BF16_ROWS][QWEN3_ARENA_BF16_COLS];
__s32 __arena q24_by_bf16[1 << 16];
__s32 __arena q16_by_bf16[1 << 16];
__s32 __arena q20_by_bf16[1 << 16];
__u16 __arena *model_bf16;

/* round(2^48 * 1000000^(-i/64)), the model's RoPE inverse frequencies. */
static const __u64 rope_freq_q48[64] = {
    281474976710656ULL, 226824411032627ULL, 182784679624242ULL, 147295606117681ULL,
    118697013481525ULL, 95651061024706ULL, 77079660277856ULL, 62114042070220ULL,
    50054115552574ULL, 40335717983353ULL, 32504223224635ULL, 26193274354827ULL,
    21107645510731ULL, 17009431236861ULL, 13706917280491ULL, 11045612208776ULL,
    8901020307485ULL, 7172817677893ULL, 5780159089968ULL, 4657896046669ULL,
    3753529140613ULL, 3024752134498ULL, 2437472877500ULL, 1964218476214ULL,
    1582850114114ULL, 1275527398856ULL, 1027873789644ULL, 828304063389ULL,
    667482358573ULL, 537885444125ULL, 433450783059ULL, 349292927307ULL,
    281474976711ULL, 226824411033ULL, 182784679624ULL, 147295606118ULL,
    118697013482ULL, 95651061025ULL, 77079660278ULL, 62114042070ULL,
    50054115553ULL, 40335717983ULL, 32504223225ULL, 26193274355ULL,
    21107645511ULL, 17009431237ULL, 13706917280ULL, 11045612209ULL,
    8901020307ULL, 7172817678ULL, 5780159090ULL, 4657896047ULL,
    3753529141ULL, 3024752134ULL, 2437472878ULL, 1964218476ULL,
    1582850114ULL, 1275527399ULL, 1027873790ULL, 828304063ULL,
    667482359ULL, 537885444ULL, 433450783ULL, 349292927ULL,
};

extern void *bpf_arena_alloc_pages(void *map, void *addr, __u32 page_cnt,
                                   int node_id, __u64 flags) __ksym;
extern int bpf_wq_init(struct bpf_wq *wq, void *map, unsigned int flags) __ksym;
extern int bpf_wq_start(struct bpf_wq *wq, unsigned int flags) __ksym;
extern int bpf_wq_set_callback_impl(struct bpf_wq *wq,
    int (*callback)(void *map, int *key, void *value),
    unsigned int flags__k, void *aux__ign) __ksym;

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_arena_bf16_work);
} work SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_event_state);
} event SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, QWEN3_EVENT_MAX_LAYERS * QWEN3_EVENT_KV_LIMIT * QWEN3_ATTENTION_KV_HEADS);
    __type(key, __u32);
    __type(value, struct qwen3_kv_pair);
} event_kv SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, QWEN3_EVENT_MAX_LAYERS);
    __type(key, __u32);
    __type(value, struct qwen3_arena_layer_weights);
} event_layer_weights SEC(".maps");

static __always_inline int event_begin_layer(struct qwen3_arena_bf16_work *state,
                                              __u32 layer)
{
    struct qwen3_arena_layer_weights *weights;

    if (state->event_layer_count) {
        if (layer >= state->event_layer_count ||
            state->event_layer_count > QWEN3_EVENT_MAX_LAYERS)
            return -1;
        weights = bpf_map_lookup_elem(&event_layer_weights, &layer);
        if (!weights)
            return -1;
        state->norm_first_bf16 = weights->input_norm_first_bf16;
        state->q_first_bf16 = weights->q_first_bf16;
        state->k_first_bf16 = weights->k_first_bf16;
        state->v_first_bf16 = weights->v_first_bf16;
        state->o_first_bf16 = weights->o_first_bf16;
        state->post_norm_first_bf16 = weights->post_norm_first_bf16;
        state->gate_first_bf16 = weights->gate_first_bf16;
        state->up_first_bf16 = weights->up_first_bf16;
        state->down_first_bf16 = weights->down_first_bf16;
        state->q_norm_first_bf16 = weights->q_norm_first_bf16;
        state->k_norm_first_bf16 = weights->k_norm_first_bf16;
    }
    state->event_layer = layer;
    state->event_stage = QWEN3_EVENT_STAGE_Q;
    state->weight_first_bf16 = state->q_first_bf16;
    state->matrix_total_rows = 2048;
    state->base_index = 0;
    state->rows = QWEN3_ARENA_BF16_ROWS;
    state->cols = QWEN3_ARENA_TOKEN_WIDTH;
    return 0;
}

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_row = row & (QWEN3_ARENA_BF16_ROWS - 1);
    __u16 __arena *weights;
    __s64 sum = 0;
    int i;

    (void)ctx;
    if (!state || row >= state->rows)
        return 1;
    if (state->resident_weights) {
        __u64 first = state->weight_first_bf16 +
                      (__u64)(state->base_index + row) * state->cols;
        if (!model_bf16 || first > state->model_elements ||
            state->cols > state->model_elements - first)
            return 1;
        weights = model_bf16 + first;
    } else {
        weights = weight_bf16[bounded_row];
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_ARENA_BF16_COLS; i++) {
        __u16 bits;

        if (i >= state->cols)
            break;
        bits = weights[i];
        sum += (__s64)state->input_q16[i] * q24_by_bf16[bits];
    }
    state->output_q16[bounded_row] = sum >> 24;
    if (state->track_argmax &&
        state->output_q16[bounded_row] > state->best_q16) {
        state->best_q16 = state->output_q16[bounded_row];
        state->best_index = state->base_index + row;
    }
    state->completed++;
    return 0;
}

SEC("syscall")
int qwen3_arena_allocate_model(void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 pages;

    (void)ctx;
    if (!state || model_bf16 || !state->model_elements ||
        state->model_elements > QWEN3_ARENA_MODEL_MAX_BF16)
        return -1;
    pages = (state->model_elements + 2047u) / 2048u;
    model_bf16 = (__u16 __arena *)bpf_arena_alloc_pages(&arena, 0,
                                                        pages, -1, 0);
    return model_bf16 ? 0 : -1;
}

SEC("socket")
int qwen3_arena_bf16_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS)
        return 0;
    bpf_loop(state->rows, compute_row, &callback_ctx, 0);
    return 0;
}

static long store_output_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 index;
    __s64 output;

    if (!state || row >= state->rows) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    /* The mask makes the verifier retain a bounded 64-bit map offset. */
    index = (state->base_index + row) & 4095u;
    if (index >= QWEN3_ARENA_EVENT_OUTPUTS) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    output = state->output_q16[row & (QWEN3_ARENA_BF16_ROWS - 1)];
    if (output > 2147483647LL || output < -2147483648LL) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    if (!state->event_qkv)
        state->matrix_output_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_Q && index < 2048)
        state->query_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_K &&
             index < QWEN3_ARENA_TOKEN_WIDTH)
        state->key_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_V &&
             index < QWEN3_ARENA_TOKEN_WIDTH)
        state->value_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_O &&
             index < QWEN3_ARENA_TOKEN_WIDTH) {
        __s64 residual = (__s64)state->hidden_q16[index] + output;

        if (residual > 2147483647LL || residual < -2147483648LL) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        state->projected_q16[index] = (__s32)output;
        state->hidden_q16[index] = (__s32)residual;
    }
    else if (state->event_stage == QWEN3_EVENT_STAGE_GATE)
        state->gate_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_UP)
        state->up_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_DOWN &&
             index < QWEN3_ARENA_TOKEN_WIDTH) {
        __s64 residual = (__s64)state->hidden_q16[index] + output;

        if (residual > 2147483647LL || residual < -2147483648LL) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        state->down_q16[index] = (__s32)output;
        state->hidden_q16[index] = (__s32)residual;
    }
    else {
        *(__u32 *)ctx = 1;
        return 1;
    }
    return 0;
}

static long load_token_embedding(__u32 col, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u64 first;
    __u32 bounded_col = col & (QWEN3_ARENA_TOKEN_WIDTH - 1);
    __u16 bits;

    if (!state || !model_bf16 ||
        state->event_token_id >= state->embedding_vocab) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    first = state->embedding_first_bf16 +
            (__u64)state->event_token_id * QWEN3_ARENA_TOKEN_WIDTH + bounded_col;
    if (first >= state->model_elements) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    bits = model_bf16[first];
    state->embedding_q16[bounded_col] =
        q16_by_bf16[bits];
    state->hidden_q16[bounded_col] = state->embedding_q16[bounded_col];
    return 0;
}

static __always_inline __u64 event_isqrt64(__u64 value)
{
    __u64 root = 1ULL << 20;
    int i;

#pragma clang loop unroll(disable)
    for (i = 0; i < 24; i++) {
        root = (root + value / root) >> 1;
        root |= 1;
    }
    return root;
}

static long event_norm_sum(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 7;
    __u64 sum = 0;
    int i;

    if (!state) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __s64 x = (state->event_stage == QWEN3_EVENT_STAGE_O ||
                    state->event_stage == QWEN3_EVENT_STAGE_FINAL_NORM ||
                    state->event_layer) ?
            state->hidden_q16[bounded_tile * 128 + i] :
            state->embedding_q16[bounded_tile * 128 + i];
        sum += (__u64)(x * x);
    }
    state->norm_sum_sq_q32 += sum;
    return 0;
}

static long event_norm_apply(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 7;
    int i;

    if (!state || !model_bf16) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u32 index = bounded_tile * 128 + i;
        __u64 first = (state->event_stage == QWEN3_EVENT_STAGE_FINAL_NORM ?
                       state->final_norm_first_bf16 :
                       state->event_stage == QWEN3_EVENT_STAGE_O ?
                       state->post_norm_first_bf16 : state->norm_first_bf16) + index;
        __s64 normalized;
        __u16 bits;

        if (first >= state->model_elements) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        bits = model_bf16[first];
        normalized = ((__s64)((state->event_stage == QWEN3_EVENT_STAGE_O ||
                      state->event_stage == QWEN3_EVENT_STAGE_FINAL_NORM ||
                      state->event_layer) ?
                      state->hidden_q16[index] :
                      state->embedding_q16[index]) *
                      (__s64)state->norm_inv_rms_q16) >> 16;
        state->input_q16[index] =
            (__s32)((normalized * q20_by_bf16[bits]) >> 20);
        if (state->event_stage == QWEN3_EVENT_STAGE_O)
            state->post_norm_q16[index] = state->input_q16[index];
    }
    return 0;
}

static long event_qk_norm_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_head = head & 31;
    __u32 is_key, base;
    __u64 sum = 0, inv_rms;
    int i;

    if (!state || !model_bf16 || bounded_head >= 24) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    is_key = bounded_head >= 16;
    base = (bounded_head & 15) * 128;
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __s64 x = is_key ? state->key_q16[base + i] :
                             state->query_q16[base + i];
        sum += (__u64)(x * x);
    }
    inv_rms = (1ULL << 32) / event_isqrt64(sum / 128 + 4295);
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u64 first = (is_key ? state->k_norm_first_bf16 :
                               state->q_norm_first_bf16) + i;
        __u16 bits;
        __s64 x, normalized;

        if (first >= state->model_elements) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        bits = model_bf16[first];
        x = is_key ? state->key_q16[base + i] :
                     state->query_q16[base + i];
        normalized = (x * (__s64)inv_rms) >> 16;
        if (is_key)
            state->key_q16[base + i] =
                (__s32)((normalized * q20_by_bf16[bits]) >> 20);
        else
            state->query_q16[base + i] =
                (__s32)((normalized * q20_by_bf16[bits]) >> 20);
    }
    state->completed_qk_heads++;
    return 0;
}

static __always_inline __s64 event_div_signed(__s64 value, __u32 divisor)
{
    return value < 0 ? -(__s64)((__u64)(-value) / divisor) :
                        (__s64)((__u64)value / divisor);
}

static long event_rope_angle(__u32 index, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    const __s64 two_pi_q48 = 1768559438007110LL;
    const __s64 pi_q48 = 884279719003555LL;
    const __s64 half_pi_q48 = 442139859501778LL;
    __u32 bounded_index = index & 63;
    __s64 angle, x, x2, sin_term, cos_term, sin_sum, cos_sum;
    __u32 negate_cos = 0;
    int n;

    if (!state || index >= 64 || state->event_position >= 40960) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    angle = ((__u64)state->event_position * rope_freq_q48[bounded_index]) %
            two_pi_q48;
    if (angle > pi_q48)
        angle -= two_pi_q48;
    if (angle > half_pi_q48) {
        angle = pi_q48 - angle;
        negate_cos = 1;
    } else if (angle < -half_pi_q48) {
        angle = -pi_q48 - angle;
        negate_cos = 1;
    }
    x = event_div_signed(angle, 1u << 18);
    x2 = (x * x) >> 30;
    sin_term = x;
    cos_term = 1LL << 30;
    sin_sum = sin_term;
    cos_sum = cos_term;
#pragma clang loop unroll(disable)
    for (n = 1; n <= 6; n++) {
        sin_term = -event_div_signed((sin_term * x2) >> 30,
                                     (2 * n) * (2 * n + 1));
        cos_term = -event_div_signed((cos_term * x2) >> 30,
                                     (2 * n - 1) * (2 * n));
        sin_sum += sin_term;
        cos_sum += cos_term;
    }
    if (negate_cos)
        cos_sum = -cos_sum;
    state->rope_sine_q20[bounded_index] =
        sin_sum >= 0 ? (__s32)((sin_sum + 512) >> 10) :
                       -(__s32)((-sin_sum + 512) >> 10);
    state->rope_cosine_q20[bounded_index] =
        cos_sum >= 0 ? (__s32)((cos_sum + 512) >> 10) :
                       -(__s32)((-cos_sum + 512) >> 10);
    return 0;
}

static long event_rope_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_head = head & 31;
    __u32 is_key, base;
    int i;

    if (!state || bounded_head >= 24) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    is_key = bounded_head >= 16;
    base = (bounded_head & 15) * 128;
#pragma clang loop unroll(disable)
    for (i = 0; i < 64; i++) {
        __s64 first = is_key ? state->key_q16[base + i] :
                               state->query_q16[base + i];
        __s64 second = is_key ? state->key_q16[base + i + 64] :
                                state->query_q16[base + i + 64];
        __s64 cosine = state->rope_cosine_q20[i];
        __s64 sine = state->rope_sine_q20[i];
        __s32 rotated_first = (__s32)((first * cosine - second * sine) >> 20);
        __s32 rotated_second = (__s32)((second * cosine + first * sine) >> 20);

        if (is_key) {
            state->key_q16[base + i] = rotated_first;
            state->key_q16[base + i + 64] = rotated_second;
        } else {
            state->query_q16[base + i] = rotated_first;
            state->query_q16[base + i + 64] = rotated_second;
        }
    }
    state->completed_rope_heads++;
    return 0;
}

static __always_inline __u64 event_exp_negative_q32(__u32 delta_q16)
{
    __u64 result;
    int i;

    if (!delta_q16)
        return 1ULL << 32;
    if (delta_q16 >= (16U << 16))
        return 0;
    result = (1ULL << 32) - ((__u64)delta_q16 << 6);
#pragma clang loop unroll(disable)
    for (i = 0; i < 10; i++)
        result = (result * result) >> 32;
    return result;
}

static __always_inline __u32 event_q_slot(__u32 base, __u32 offset)
{
    __u32 index = base + offset;

    asm volatile ("" : "+r"(index));
    return index & 2047;
}

static long event_store_kv_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_head = head & 7;
    __u32 slot;
    struct qwen3_kv_pair *pair;
    int i;

    if (!state || head >= QWEN3_ATTENTION_KV_HEADS ||
        state->event_position >= QWEN3_EVENT_KV_LIMIT) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    slot = (state->event_layer * QWEN3_EVENT_KV_LIMIT +
            state->event_position) * QWEN3_ATTENTION_KV_HEADS + bounded_head;
    pair = bpf_map_lookup_elem(&event_kv, &slot);
    if (!pair) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u32 index = bounded_head * 128 + i;

        asm volatile ("" : "+r"(index));
        index &= QWEN3_ARENA_TOKEN_WIDTH - 1;
        pair->key_q16[i] = state->key_q16[index];
        pair->value_q16[i] = state->value_q16[index];
    }
    return 0;
}

static long event_attention_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_head = head;
    __u32 base, kv_head, slot, past;
    struct qwen3_kv_pair *pair;
    __s64 dot = 0, score, delta;
    __u64 mass, weight, new_mass, alpha_q24;
    int i;

    asm volatile ("" : "+r"(bounded_head));
    bounded_head &= 15;
    if (!state || head >= 16 ||
        state->event_position >= QWEN3_EVENT_KV_LIMIT ||
        state->attention_past_cursor > state->event_position) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    base = bounded_head * 128;
    kv_head = bounded_head >> 1;
    past = state->attention_past_cursor;
    slot = (state->event_layer * QWEN3_EVENT_KV_LIMIT + past) *
           QWEN3_ATTENTION_KV_HEADS + kv_head;
    pair = bpf_map_lookup_elem(&event_kv, &slot);
    if (!pair || state->attention_seen[bounded_head] != past) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++)
        dot += (__s64)state->query_q16[event_q_slot(base, i)] *
               pair->key_q16[i];
    score = ((dot >> 16) * 5793) >> 16;
    if (score > 2147483647LL || score < -2147483648LL) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    if (!past) {
#pragma clang loop unroll(disable)
        for (i = 0; i < 128; i++)
            state->attention_output_q16[event_q_slot(base, i)] =
                pair->value_q16[i];
        state->attention_max_score_q16[bounded_head] = (__s32)score;
        state->attention_mass_q16[bounded_head] = 1ULL << 16;
    } else {
        delta = score - state->attention_max_score_q16[bounded_head];
        mass = state->attention_mass_q16[bounded_head];
        if (delta > 0) {
            __u64 factor = event_exp_negative_q32((__u32)delta);
            __u64 old_mass = mass;

            mass = ((old_mass >> 16) * factor) >> 16;
            mass += ((old_mass & 65535) * factor) >> 32;
            weight = 1ULL << 16;
            state->attention_max_score_q16[bounded_head] = (__s32)score;
        } else {
            weight = event_exp_negative_q32((__u32)-delta) >> 16;
        }
        new_mass = mass + weight;
        if (!new_mass) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        alpha_q24 = (weight << 24) / new_mass;
#pragma clang loop unroll(disable)
        for (i = 0; i < 128; i++) {
            __u32 index = event_q_slot(base, i);
            __s64 difference = (__s64)pair->value_q16[i] -
                               state->attention_output_q16[index];
            state->attention_output_q16[index] +=
                (__s32)((difference * (__s64)alpha_q24) >> 24);
        }
        state->attention_mass_q16[bounded_head] = new_mass;
    }
    state->attention_seen[bounded_head]++;
    state->completed_attention_heads++;
    return 0;
}

static long event_attention_to_input(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 15;
    int i;

    if (!state || tile >= 16) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u32 index = event_q_slot(bounded_tile * 128, i);

        state->input_q16[index] = state->attention_output_q16[index];
    }
    return 0;
}

static __always_inline __s32 event_silu_q16(__s32 x)
{
    __u64 magnitude = x < 0 ? -(__s64)x : x;
    __u64 exponential, denominator, sigmoid;
    int i;

    if (!x)
        return 0;
    if (magnitude >= (16ULL << 16))
        return x > 0 ? x : 0;
    exponential = (1ULL << 32) - (magnitude << 6);
#pragma clang loop unroll(disable)
    for (i = 0; i < 10; i++)
        exponential = (exponential * exponential) >> 32;
    denominator = (1ULL << 32) + exponential;
    sigmoid = x >= 0 ? (1ULL << 48) / denominator :
             (exponential << 16) / denominator;
    return (__s32)(((__s64)x * (__s64)sigmoid) >> 16);
}

static long event_mlp_product(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 31;
    int i;

    if (!state || tile >= 24) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u32 index = bounded_tile * 128 + i;
        __s32 activated;

        asm volatile ("" : "+r"(index));
        index &= 4095u;
        if (index >= QWEN3_ARENA_EVENT_OUTPUTS) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        activated = event_silu_q16(state->gate_q16[index]);

        state->product_q16[index] = (__s32)(((__s64)activated *
                                             state->up_q16[index]) >> 16);
        state->input_q16[index] = state->product_q16[index];
    }
    return 0;
}

static int event_callback(void *map, int *key, void *value)
{
    const __u32 work_key = 0;
    struct qwen3_event_state *job = value;
    struct qwen3_arena_bf16_work *state =
        bpf_map_lookup_elem(&work, &work_key);
    __u32 callback_ctx = 0;
    __u32 total, base, rows;

    (void)map;
    (void)key;
    job->work_cpu = bpf_get_smp_processor_id();
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    if (state->event_attention &&
        state->event_stage == QWEN3_EVENT_STAGE_ATTENTION) {
        if (state->event_position >= QWEN3_EVENT_KV_LIMIT ||
            state->attention_past_cursor > state->event_position) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->completed_attention_heads = 0;
        bpf_loop(16, event_attention_head, &callback_ctx, 0);
        if (callback_ctx || state->completed_attention_heads != 16) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->attention_past_cursor++;
        if (state->attention_past_cursor <= state->event_position) {
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->o_first_bf16 > state->model_elements ||
            2048ULL * QWEN3_ARENA_TOKEN_WIDTH >
                state->model_elements - state->o_first_bf16) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        callback_ctx = 0;
        bpf_loop(16, event_attention_to_input, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->event_stage = QWEN3_EVENT_STAGE_O;
        state->weight_first_bf16 = state->o_first_bf16;
        state->matrix_total_rows = QWEN3_ARENA_TOKEN_WIDTH;
        state->base_index = 0;
        state->rows = QWEN3_ARENA_BF16_ROWS;
        state->cols = 2048;
        if (bpf_wq_start(&job->work, 0))
            job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    total = state->matrix_total_rows ? state->matrix_total_rows : state->rows;
    base = state->base_index;
    rows = state->rows;
    if (total > (state->event_stage == QWEN3_EVENT_STAGE_LOGITS ?
                 QWEN3_EVENT_VOCAB : QWEN3_ARENA_EVENT_OUTPUTS) ||
        base >= total ||
        rows > total - base ||
        (state->event_qkv &&
         (state->event_stage < QWEN3_EVENT_STAGE_Q ||
          state->event_stage > QWEN3_EVENT_STAGE_LOGITS))) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    if (state->event_use_token && base == 0 &&
        (!state->event_qkv || state->event_stage == QWEN3_EVENT_STAGE_Q)) {
        if (!state->event_layer) {
            if (state->cols != QWEN3_ARENA_TOKEN_WIDTH ||
                state->event_token_id >= state->embedding_vocab ||
                state->embedding_first_bf16 > state->model_elements ||
                (__u64)state->embedding_vocab * QWEN3_ARENA_TOKEN_WIDTH >
                    state->model_elements - state->embedding_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            bpf_loop(QWEN3_ARENA_TOKEN_WIDTH, load_token_embedding,
                     &callback_ctx, 0);
            if (callback_ctx) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
        }
        if (state->norm_first_bf16 > state->model_elements ||
            QWEN3_ARENA_TOKEN_WIDTH >
                state->model_elements - state->norm_first_bf16) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->norm_sum_sq_q32 = 0;
        callback_ctx = 0;
        bpf_loop(8, event_norm_sum, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->norm_inv_rms_q16 =
            (1ULL << 32) / event_isqrt64(state->norm_sum_sq_q32 /
                                          QWEN3_ARENA_TOKEN_WIDTH + 4295);
        callback_ctx = 0;
        bpf_loop(8, event_norm_apply, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
    }
    state->completed = 0;
    bpf_loop(rows, compute_row, &callback_ctx, 0);
    if (state->completed != rows) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    if (state->event_stage != QWEN3_EVENT_STAGE_LOGITS) {
        callback_ctx = 0;
        bpf_loop(rows, store_output_row, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
    }
    base += rows;
    if (base == total) {
        if (state->event_qkv && state->event_stage < QWEN3_EVENT_STAGE_V) {
            if (state->event_stage == QWEN3_EVENT_STAGE_Q) {
                state->event_stage = QWEN3_EVENT_STAGE_K;
                state->weight_first_bf16 = state->k_first_bf16;
            } else if (state->event_stage == QWEN3_EVENT_STAGE_K) {
                state->event_stage = QWEN3_EVENT_STAGE_V;
                state->weight_first_bf16 = state->v_first_bf16;
            } else {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->matrix_total_rows = QWEN3_ARENA_TOKEN_WIDTH;
            state->base_index = 0;
            state->rows = QWEN3_ARENA_BF16_ROWS;
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->event_qkv && state->event_stage == QWEN3_EVENT_STAGE_V) {
            if (state->q_norm_first_bf16 > state->model_elements ||
                128 > state->model_elements - state->q_norm_first_bf16 ||
                state->k_norm_first_bf16 > state->model_elements ||
                128 > state->model_elements - state->k_norm_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->completed_qk_heads = 0;
            callback_ctx = 0;
            bpf_loop(24, event_qk_norm_head, &callback_ctx, 0);
            if (callback_ctx || state->completed_qk_heads != 24) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            if (state->event_rope) {
                callback_ctx = 0;
                bpf_loop(64, event_rope_angle, &callback_ctx, 0);
                if (callback_ctx) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
                state->completed_rope_heads = 0;
                bpf_loop(24, event_rope_head, &callback_ctx, 0);
                if (callback_ctx || state->completed_rope_heads != 24) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
            }
            if (state->event_attention) {
                callback_ctx = 0;
                bpf_loop(8, event_store_kv_head, &callback_ctx, 0);
                if (callback_ctx) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
                __builtin_memset(state->attention_seen, 0,
                                 sizeof(state->attention_seen));
                state->attention_past_cursor = 0;
                state->event_stage = QWEN3_EVENT_STAGE_ATTENTION;
                state->base_index = base;
                if (bpf_wq_start(&job->work, 0))
                    job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
        }
        state->base_index = base;
        if (state->event_attention &&
            state->event_stage == QWEN3_EVENT_STAGE_O) {
            if (state->post_norm_first_bf16 > state->model_elements ||
                QWEN3_ARENA_TOKEN_WIDTH >
                    state->model_elements - state->post_norm_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->norm_sum_sq_q32 = 0;
            callback_ctx = 0;
            bpf_loop(8, event_norm_sum, &callback_ctx, 0);
            if (callback_ctx) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->norm_inv_rms_q16 =
                (1ULL << 32) / event_isqrt64(state->norm_sum_sq_q32 /
                                              QWEN3_ARENA_TOKEN_WIDTH + 4295);
            callback_ctx = 0;
            bpf_loop(8, event_norm_apply, &callback_ctx, 0);
            if (callback_ctx) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            if (state->gate_first_bf16 > state->model_elements ||
                (__u64)QWEN3_ARENA_EVENT_OUTPUTS * QWEN3_ARENA_TOKEN_WIDTH >
                    state->model_elements - state->gate_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->event_stage = QWEN3_EVENT_STAGE_GATE;
            state->weight_first_bf16 = state->gate_first_bf16;
            state->matrix_total_rows = QWEN3_ARENA_EVENT_OUTPUTS;
            state->base_index = 0;
            state->rows = QWEN3_ARENA_BF16_ROWS;
            state->cols = QWEN3_ARENA_TOKEN_WIDTH;
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->event_attention &&
            state->event_stage == QWEN3_EVENT_STAGE_GATE) {
            if (state->up_first_bf16 > state->model_elements ||
                (__u64)QWEN3_ARENA_EVENT_OUTPUTS * QWEN3_ARENA_TOKEN_WIDTH >
                    state->model_elements - state->up_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->event_stage = QWEN3_EVENT_STAGE_UP;
            state->weight_first_bf16 = state->up_first_bf16;
            state->base_index = 0;
            state->rows = QWEN3_ARENA_BF16_ROWS;
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->event_attention &&
            state->event_stage == QWEN3_EVENT_STAGE_UP) {
            if (state->down_first_bf16 > state->model_elements ||
                (__u64)QWEN3_ARENA_TOKEN_WIDTH * QWEN3_ARENA_EVENT_OUTPUTS >
                    state->model_elements - state->down_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            callback_ctx = 0;
            bpf_loop(24, event_mlp_product, &callback_ctx, 0);
            if (callback_ctx) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->event_stage = QWEN3_EVENT_STAGE_DOWN;
            state->weight_first_bf16 = state->down_first_bf16;
            state->matrix_total_rows = QWEN3_ARENA_TOKEN_WIDTH;
            state->base_index = 0;
            state->rows = QWEN3_ARENA_BF16_ROWS;
            state->cols = QWEN3_ARENA_EVENT_OUTPUTS;
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->event_attention &&
            state->event_stage == QWEN3_EVENT_STAGE_DOWN) {
            if (state->event_layer_count &&
                state->event_layer + 1 < state->event_layer_count) {
                if (event_begin_layer(state, state->event_layer + 1) ||
                    bpf_wq_start(&job->work, 0))
                    job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            if (state->event_final_logits && state->event_emit_logits) {
                job->layers_finish_ns = bpf_ktime_get_ns();
                if (state->final_norm_first_bf16 > state->model_elements ||
                    QWEN3_ARENA_TOKEN_WIDTH >
                        state->model_elements - state->final_norm_first_bf16 ||
                    state->embedding_first_bf16 > state->model_elements ||
                    (__u64)QWEN3_EVENT_VOCAB * QWEN3_ARENA_TOKEN_WIDTH >
                        state->model_elements - state->embedding_first_bf16) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
                state->event_stage = QWEN3_EVENT_STAGE_FINAL_NORM;
                state->norm_sum_sq_q32 = 0;
                callback_ctx = 0;
                bpf_loop(8, event_norm_sum, &callback_ctx, 0);
                if (callback_ctx) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
                state->norm_inv_rms_q16 =
                    (1ULL << 32) / event_isqrt64(state->norm_sum_sq_q32 /
                                                  QWEN3_ARENA_TOKEN_WIDTH + 4295);
                callback_ctx = 0;
                bpf_loop(8, event_norm_apply, &callback_ctx, 0);
                if (callback_ctx) {
                    job->status = QWEN3_EVENT_ERROR;
                    return 0;
                }
                state->event_stage = QWEN3_EVENT_STAGE_LOGITS;
                state->weight_first_bf16 = state->embedding_first_bf16;
                state->matrix_total_rows = QWEN3_EVENT_VOCAB;
                state->base_index = 0;
                state->rows = QWEN3_ARENA_BF16_ROWS;
                state->cols = QWEN3_ARENA_TOKEN_WIDTH;
                state->track_argmax = 1;
                state->best_q16 = -2147483648LL;
                state->best_index = 0;
                if (bpf_wq_start(&job->work, 0))
                    job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->event_next_position = state->event_position + 1;
        }
        if (state->event_stage == QWEN3_EVENT_STAGE_LOGITS) {
            job->result_token_id = state->best_index;
            job->result_logit_q16 = state->best_q16;
            state->event_next_position = state->event_position + 1;
            state->track_argmax = 0;
        }
        job->finish_ns = bpf_ktime_get_ns();
        job->completed_requests++;
        job->status = QWEN3_EVENT_DONE;
        return 0;
    }
    state->base_index = base;
    state->rows = total - base < QWEN3_ARENA_BF16_ROWS ?
        total - base : QWEN3_ARENA_BF16_ROWS;
    if (bpf_wq_start(&job->work, 0))
        job->status = QWEN3_EVENT_ERROR;
    return 0;
}

SEC("syscall")
int qwen3_event_init(void *ctx)
{
    const __u32 key = 0;
    struct qwen3_event_state *job = bpf_map_lookup_elem(&event, &key);

    (void)ctx;
    if (!job || bpf_wq_init(&job->work, &event, 0) ||
        bpf_wq_set_callback_impl(&job->work, event_callback, 0, 0))
        return -1;
    job->status = QWEN3_EVENT_READY;
    return 0;
}

SEC("xdp")
int qwen3_event_xdp(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *end = (void *)(long)ctx->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    struct udphdr *udp;
    unsigned char *payload;
    const __u32 key = 0;
    struct qwen3_event_state *job;
    struct qwen3_arena_bf16_work *state;
    __u64 old_status;
    __u32 token_id = 0;
    __u32 position = 0;

    if ((void *)(eth + 1) > end || eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;
    ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > end || ip->ihl != 5 || ip->protocol != IPPROTO_UDP)
        return XDP_PASS;
    udp = (void *)(ip + 1);
    if ((void *)(udp + 1) > end || udp->dest != bpf_htons(49002))
        return XDP_PASS;
    payload = (void *)(udp + 1);
    if (payload + 4 > (unsigned char *)end ||
        payload[0] != 'Q' || payload[1] != '3' ||
        payload[2] != 'B')
        return XDP_PASS;
    state = bpf_map_lookup_elem(&work, &key);
    if (!state)
        return XDP_PASS;
    if (payload[3] == 'R') {
        __u32 request_number;
        __u32 token;
        __u64 logit;
        __be32 address;
        __be16 port;
        __u8 mac[ETH_ALEN];

        if (!state->event_final_logits ||
            payload + 20 > (unsigned char *)end ||
            bpf_ntohs(udp->len) < sizeof(*udp) + 20 ||
            bpf_ntohs(ip->tot_len) < sizeof(*ip) + sizeof(*udp) + 20)
            return XDP_PASS;
        job = bpf_map_lookup_elem(&event, &key);
        if (!job || job->status != QWEN3_EVENT_DONE)
            return XDP_DROP;
        request_number = ((__u32)payload[4] << 24) |
                         ((__u32)payload[5] << 16) |
                         ((__u32)payload[6] << 8) | payload[7];
        if (request_number != job->completed_requests)
            return XDP_DROP;
        token = job->result_token_id;
        logit = (__u64)job->result_logit_q16;
        payload[3] = state->event_emit_logits ? 'A' : 'K';
        payload[4] = token >> 24;
        payload[5] = token >> 16;
        payload[6] = token >> 8;
        payload[7] = token;
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; i++)
            payload[8 + i] = logit >> (56 - 8 * i);
        payload[16] = request_number >> 24;
        payload[17] = request_number >> 16;
        payload[18] = request_number >> 8;
        payload[19] = request_number;
        __builtin_memcpy(mac, eth->h_source, ETH_ALEN);
        __builtin_memcpy(eth->h_source, eth->h_dest, ETH_ALEN);
        __builtin_memcpy(eth->h_dest, mac, ETH_ALEN);
        address = ip->saddr;
        ip->saddr = ip->daddr;
        ip->daddr = address;
        port = udp->source;
        udp->source = udp->dest;
        udp->dest = port;
        udp->check = 0;
        return XDP_TX;
    }
    if (payload[3] != 'P' &&
        (payload[3] != 'F' || !state->event_final_logits))
        return XDP_PASS;
    if (state->event_use_token) {
        if (payload + 8 > (unsigned char *)end)
            return XDP_PASS;
        token_id = ((__u32)payload[4] << 24) |
                   ((__u32)payload[5] << 16) |
                   ((__u32)payload[6] << 8) | payload[7];
        if (token_id >= state->embedding_vocab)
            return XDP_PASS;
    }
    if (state->event_rope) {
        if (!state->event_use_token || payload + 12 > (unsigned char *)end)
            return XDP_PASS;
        position = ((__u32)payload[8] << 24) |
                   ((__u32)payload[9] << 16) |
                   ((__u32)payload[10] << 8) | payload[11];
        if (position >= 40960)
            return XDP_PASS;
    }
    if (state->event_attention &&
        (!state->event_qkv || !state->event_rope ||
         position >= QWEN3_EVENT_KV_LIMIT ||
         (position && position != state->event_next_position)))
        return XDP_PASS;
    job = bpf_map_lookup_elem(&event, &key);
    if (!job)
        return XDP_PASS;
    old_status = __sync_val_compare_and_swap(&job->status,
                                              QWEN3_EVENT_READY,
                                              QWEN3_EVENT_RUNNING);
    if (old_status != QWEN3_EVENT_READY &&
        (old_status != QWEN3_EVENT_DONE ||
         __sync_val_compare_and_swap(&job->status,
                                      QWEN3_EVENT_DONE,
                                      QWEN3_EVENT_RUNNING) !=
                                      QWEN3_EVENT_DONE))
        return XDP_PASS;
    state->base_index = 0;
    state->event_token_id = token_id;
    state->event_position = position;
    state->event_emit_logits = payload[3] == 'P';
    job->result_token_id = 0;
    job->result_logit_q16 = 0;
    if (state->event_qkv) {
        if (event_begin_layer(state, 0)) {
            job->status = QWEN3_EVENT_ERROR;
            return XDP_PASS;
        }
    }
    if (state->matrix_total_rows)
        state->rows = state->matrix_total_rows < QWEN3_ARENA_BF16_ROWS ?
            state->matrix_total_rows : QWEN3_ARENA_BF16_ROWS;
    job->requests++;
    job->start_ns = bpf_ktime_get_ns();
    job->finish_ns = 0;
    job->layers_finish_ns = 0;
    if (bpf_wq_start(&job->work, 0))
        job->status = QWEN3_EVENT_ERROR;
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
