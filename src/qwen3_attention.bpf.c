#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_attention.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_attention_state);
} attention SEC(".maps");

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

SEC("socket")
int qwen3_attention_step(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);
    __s64 dot = 0;
    __s64 score, delta;
    __u64 mass, weight, new_mass, alpha_q24;
    int i;

    (void)skb;
    if (!work || work->seen >= 65536)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        dot += (__s64)work->query_q16[i] * work->key_q16[i];
    /* Q16 dot scaled by 1/sqrt(128), also Q16 (5793/65536). */
    score = ((dot >> 16) * 5793) >> 16;
    if (score > 2147483647LL || score < -2147483648LL)
        return 0;
    work->score_q16 = (__s32)score;
    if (!work->seen) {
#pragma clang loop unroll(disable)
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work->output_q16[i] = work->value_q16[i];
        work->max_score_q16 = work->score_q16;
        work->mass_q16 = 1ULL << 16;
        work->seen = 1;
        return 0;
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
        return 0;
    alpha_q24 = (weight << 24) / new_mass;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 difference = (__s64)work->value_q16[i] - work->output_q16[i];
        __s64 next = (__s64)work->output_q16[i] +
                     ((difference * (__s64)alpha_q24) >> 24);
        work->output_q16[i] = (__s32)next;
    }
    work->mass_q16 = new_mass;
    work->seen++;
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
