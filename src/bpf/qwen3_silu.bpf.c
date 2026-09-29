#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_silu.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_silu_state);
} silu SEC(".maps");

static __always_inline __s32 silu_q16(__s32 x)
{
    __u64 magnitude = x < 0 ? -(__s64)x : x;
    __u64 exponential, denominator, sigmoid;
    int i;

    if (!x)
        return 0;
    if (magnitude >= (16ULL << 16))
        return x > 0 ? x : 0;
    /* exp(-|x|) = lim_n (1 - |x|/n)^n, n=1024, in Q32. */
    exponential = (1ULL << 32) - (magnitude << 6);
#pragma clang loop unroll(disable)
    for (i = 0; i < 10; i++)
        exponential = (exponential * exponential) >> 32;
    denominator = (1ULL << 32) + exponential;
    sigmoid = x >= 0
        ? (1ULL << 48) / denominator
        : (exponential << 16) / denominator;
    return (__s32)(((__s64)x * (__s64)sigmoid) >> 16);
}

static long apply_tile(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_silu_state *work = bpf_map_lookup_elem(&silu, &key);
    __u32 bounded_tile = tile & 31;
    __u32 base;
    int i;

    (void)ctx;
    if (!work || bounded_tile >= QWEN3_SILU_MAX / QWEN3_TILE_WIDTH)
        return 1;
    base = bounded_tile * QWEN3_TILE_WIDTH;
    if (base >= work->count)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __u32 index = base + i;
        if (index >= QWEN3_SILU_MAX)
            break;
        work->output_q16[index] = silu_q16(work->input_q16[index]);
    }
    work->completed_tiles++;
    return 0;
}

SEC("socket")
int qwen3_silu_apply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_silu_state *work = bpf_map_lookup_elem(&silu, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->count || work->count > QWEN3_SILU_MAX ||
        work->count % QWEN3_TILE_WIDTH)
        return 0;
    bpf_loop(work->count / QWEN3_TILE_WIDTH, apply_tile, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
