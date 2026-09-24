#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_norm.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_norm_state);
} norm SEC(".maps");

static __always_inline __u64 isqrt64(__u64 value)
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

SEC("socket")
int qwen3_rms_accumulate(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    __u64 sum = 0;
    int i;

    (void)skb;
    if (!work)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 x = work->activation_q16[i];
        sum += (__u64)(x * x);
    }
    work->sum_sq_q32 += sum;
    work->completed_tiles++;
    return 0;
}

SEC("socket")
int qwen3_rms_finalize(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    __u64 root;

    (void)skb;
    if (!work || work->completed_tiles != QWEN3_HIDDEN_TILES)
        return 0;
    /* eps=1e-6, expressed in Q32 as 4295. */
    root = isqrt64(work->sum_sq_q32 / QWEN3_HIDDEN_SIZE + 4295);
    work->inv_rms_q16 = (1ULL << 32) / root;
    return 0;
}

SEC("socket")
int qwen3_rms_apply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    int i;

    (void)skb;
    if (!work || work->completed_tiles != QWEN3_HIDDEN_TILES ||
        !work->inv_rms_q16)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 normalized =
            ((__s64)work->activation_q16[i] *
             (__s64)work->inv_rms_q16) >> 16;
        work->output_q16[i] =
            (__s32)((normalized * work->weight_q20[i]) >> 20);
    }
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
