#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_attention.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_attention_state);
} attention SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_attention_heads_state);
} attention_heads SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_kv_pair);
} kv SEC(".maps");

static __always_inline __u64 exp_negative_q32(__u32 delta_q16)
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

static __always_inline void attention_accumulate(struct qwen3_attention_state *work,
                                                  const __s32 *key_q16,
                                                  const __s32 *value_q16)
{
    __s64 dot = 0;
    __s64 score, delta;
    __u64 mass, weight, new_mass, alpha_q24;
    int i;

    if (work->seen >= QWEN3_ATTENTION_CONTEXT_LIMIT)
        return;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        dot += (__s64)work->query_q16[i] * key_q16[i];
    /* Q16 dot scaled by 1/sqrt(128), also Q16 (5793/65536). */
    score = ((dot >> 16) * 5793) >> 16;
    if (score > 2147483647LL || score < -2147483648LL)
        return;
    work->score_q16 = (__s32)score;
    if (!work->seen) {
#pragma clang loop unroll(disable)
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work->output_q16[i] = value_q16[i];
        work->max_score_q16 = work->score_q16;
        work->mass_q16 = 1ULL << 16;
        work->seen = 1;
        return;
    }
    delta = score - work->max_score_q16;
    if (delta > 0) {
        __u64 factor = exp_negative_q32((__u32)delta);
        mass = ((work->mass_q16 >> 16) * factor) >> 16;
        mass += ((work->mass_q16 & 65535) * factor) >> 32;
        weight = 1ULL << 16;
        work->max_score_q16 = work->score_q16;
    } else {
        mass = work->mass_q16;
        weight = exp_negative_q32((__u32)-delta) >> 16;
    }
    new_mass = mass + weight;
    if (!new_mass)
        return;
    alpha_q24 = (weight << 24) / new_mass;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 difference = (__s64)value_q16[i] - work->output_q16[i];
        __s64 next = (__s64)work->output_q16[i] +
                     ((difference * (__s64)alpha_q24) >> 24);
        work->output_q16[i] = (__s32)next;
    }
    work->mass_q16 = new_mass;
    work->seen++;
}

SEC("socket")
int qwen3_attention_step(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);

    (void)skb;
    if (work)
        attention_accumulate(work, work->key_q16, work->value_q16);
    return 0;
}

static long cached_step(__u32 index, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);
    struct qwen3_kv_pair *pair;
    __u32 past, slot;

    (void)ctx;
    if (!work || index >= QWEN3_ATTENTION_CHUNK)
        return 1;
    past = work->base_position + index;
    if (past >= QWEN3_ATTENTION_CONTEXT_LIMIT ||
        work->layer >= QWEN3_ATTENTION_LAYERS ||
        work->kv_head >= QWEN3_ATTENTION_KV_HEADS)
        return 1;
    slot = (past * QWEN3_ATTENTION_LAYERS + work->layer) *
           QWEN3_ATTENTION_KV_HEADS + work->kv_head;
    pair = bpf_map_lookup_elem(&kv, &slot);
    if (!pair)
        return 1;
    attention_accumulate(work, pair->key_q16, pair->value_q16);
    return 0;
}

SEC("socket")
int qwen3_attention_cached(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->step_count ||
        work->step_count > QWEN3_ATTENTION_CHUNK ||
        work->base_position > QWEN3_ATTENTION_CONTEXT_LIMIT - work->step_count)
        return 0;
    bpf_loop(work->step_count, cached_step, &callback_ctx, 0);
    return 0;
}

static long cached_all_heads_step(__u32 index, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_attention_heads_state *work =
        bpf_map_lookup_elem(&attention_heads, &key);
    struct qwen3_kv_pair *pair;
    __u32 past, slot;
    int kv_head;

    (void)ctx;
    if (!work || index >= QWEN3_ATTENTION_CHUNK ||
        index >= work->step_count)
        return 1;
    past = work->base_position + index;
    if (past >= QWEN3_ATTENTION_CONTEXT_LIMIT ||
        work->layer >= QWEN3_ATTENTION_LAYERS)
        return 1;
#pragma clang loop unroll(disable)
    for (kv_head = 0; kv_head < QWEN3_ATTENTION_KV_HEADS; kv_head++) {
        __u32 bounded_kv_head = (__u32)kv_head &
                               (QWEN3_ATTENTION_KV_HEADS - 1);
        slot = (past * QWEN3_ATTENTION_LAYERS + work->layer) *
               QWEN3_ATTENTION_KV_HEADS + bounded_kv_head;
        pair = bpf_map_lookup_elem(&kv, &slot);
        if (!pair)
            return 1;
        attention_accumulate(&work->heads[bounded_kv_head * 2],
                             pair->key_q16, pair->value_q16);
        attention_accumulate(&work->heads[bounded_kv_head * 2 + 1],
                             pair->key_q16, pair->value_q16);
    }
    work->completed_positions++;
    return 0;
}

SEC("socket")
int qwen3_attention_all_heads(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_heads_state *work =
        bpf_map_lookup_elem(&attention_heads, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->step_count ||
        work->step_count > QWEN3_ATTENTION_CHUNK ||
        work->base_position > QWEN3_ATTENTION_CONTEXT_LIMIT - work->step_count)
        return 0;
    bpf_loop(work->step_count, cached_all_heads_step, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
