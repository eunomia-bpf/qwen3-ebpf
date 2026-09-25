#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_norm.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_norm_state);
} norm SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_qk_norm_state);
} qk_norm SEC(".maps");

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

static long accumulate_tile(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    __u64 sum = 0;
    __u32 bounded_tile = tile & (QWEN3_HIDDEN_TILES - 1);
    int i;

    (void)ctx;
    if (!work || tile >= work->total_tiles)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 x = work->activation_q16[bounded_tile * QWEN3_TILE_WIDTH + i];
        sum += (__u64)(x * x);
    }
    work->sum_sq_q32 += sum;
    work->completed_tiles++;
    return 0;
}

static long apply_tile(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    __u32 bounded_tile = tile & (QWEN3_HIDDEN_TILES - 1);
    int i;

    (void)ctx;
    if (!work || tile >= work->total_tiles)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __u32 index = bounded_tile * QWEN3_TILE_WIDTH + i;
        __s64 normalized =
            ((__s64)work->activation_q16[index] *
             (__s64)work->inv_rms_q16) >> 16;
        work->output_q16[index] =
            (__s32)((normalized * work->weight_q20[index]) >> 20);
    }
    work->completed_tiles++;
    return 0;
}

SEC("socket")
int qwen3_rms_full(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_norm_state *work = bpf_map_lookup_elem(&norm, &key);
    __u32 callback_ctx = 0;
    __u64 root;

    (void)skb;
    if (!work || !work->total_tiles || work->total_tiles > QWEN3_HIDDEN_TILES)
        return 0;
    bpf_loop(work->total_tiles, accumulate_tile, &callback_ctx, 0);
    if (work->completed_tiles != work->total_tiles)
        return 0;
    /* eps=1e-6, expressed in Q32 as 4295. */
    root = isqrt64(work->sum_sq_q32 /
                   (work->total_tiles * QWEN3_TILE_WIDTH) + 4295);
    work->inv_rms_q16 = (1ULL << 32) / root;
    if (!work->inv_rms_q16)
        return 0;
    bpf_loop(work->total_tiles, apply_tile, &callback_ctx, 0);
    return 0;
}

static long normalize_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_qk_norm_state *work = bpf_map_lookup_elem(&qk_norm, &key);
    __u32 bounded_head = head & 31;
    __u32 base, weight_set;
    __u64 sum = 0, root, inv_rms;
    int i;

    (void)ctx;
    if (!work)
        return 1;
    /* bpf_loop supplies 0..23; keep the map offset visibly bounded. */
    if (bounded_head >= QWEN3_QK_HEADS)
        bounded_head = 0;
    base = bounded_head * QWEN3_TILE_WIDTH;
    weight_set = bounded_head >= QWEN3_Q_HEADS;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __s64 x = work->activation_q16[base + i];
        sum += (__u64)(x * x);
    }
    root = isqrt64(sum / QWEN3_TILE_WIDTH + 4295);
    inv_rms = (1ULL << 32) / root;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __u32 index = base + i;
        __s64 normalized =
            ((__s64)work->activation_q16[index] * (__s64)inv_rms) >> 16;
        work->output_q16[index] =
            (__s32)((normalized * work->weight_q20[weight_set][i]) >> 20);
    }
    work->completed_heads++;
    return 0;
}

SEC("socket")
int qwen3_qk_norm_heads(struct __sk_buff *skb)
{
    __u32 callback_ctx = 0;
    const __u32 key = 0;
    struct qwen3_qk_norm_state *work = bpf_map_lookup_elem(&qk_norm, &key);

    (void)skb;
    if (!work)
        return 0;
    bpf_loop(QWEN3_QK_HEADS, normalize_head, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
