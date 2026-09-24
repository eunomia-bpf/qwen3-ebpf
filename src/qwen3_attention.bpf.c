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
int qwen3_attention_score(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);
    __s64 dot = 0;
    int i;

    (void)skb;
    if (!work || work->score_index >= 2)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        dot += (__s64)work->query_q16[i] * work->key_q16[i];
    /* Q16 dot scaled by 1/sqrt(128), also Q16 (5793/65536). */
    work->score_q16[work->score_index] = (__s32)((dot * 5793) >> 32);
    return 0;
}

SEC("socket")
int qwen3_attention_apply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_attention_state *work = bpf_map_lookup_elem(&attention, &key);
    __s32 maximum, delta0, delta1;
    __u64 weight0, weight1, denominator;
    int i;

    (void)skb;
    if (!work || work->context_length < 1 || work->context_length > 2)
        return 0;
    if (work->context_length == 1) {
#pragma clang loop unroll(disable)
        for (i = 0; i < QWEN3_TILE_WIDTH; i++)
            work->output_q16[i] = work->value_q16[0][i];
        return 0;
    }
    maximum = work->score_q16[0] > work->score_q16[1]
        ? work->score_q16[0] : work->score_q16[1];
    delta0 = maximum - work->score_q16[0];
    delta1 = maximum - work->score_q16[1];
    weight0 = exp_negative_q32((__u32)delta0);
    weight1 = exp_negative_q32((__u32)delta1);
    denominator = weight0 + weight1;
    if (!denominator)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 numerator = (__s64)work->value_q16[0][i] * (__s64)weight0 +
                          (__s64)work->value_q16[1][i] * (__s64)weight1;
        __u64 magnitude = numerator < 0 ? (__u64)(-numerator) : (__u64)numerator;
        __s64 result = (__s64)(magnitude / denominator);
        work->output_q16[i] = (__s32)(numerator < 0 ? -result : result);
    }
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
