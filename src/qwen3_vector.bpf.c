#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_vector.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_vector_state);
} vector SEC(".maps");

static long add_tile(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    __u32 base;
    int i;

    (void)ctx;
    if (!work || tile >= QWEN3_VECTOR_MAX / QWEN3_TILE_WIDTH)
        return 1;
    base = tile * QWEN3_TILE_WIDTH;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __u32 index = (base + i) & (QWEN3_VECTOR_CAPACITY - 1);
        if (base + i >= work->count)
            return 1;
        work->output_q16[index] = work->left_q16[index] +
                                  work->right_q16[index];
    }
    work->completed_tiles++;
    return 0;
}

static long multiply_tile(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    __u32 base;
    int i;

    (void)ctx;
    if (!work || tile >= QWEN3_VECTOR_MAX / QWEN3_TILE_WIDTH)
        return 1;
    base = tile * QWEN3_TILE_WIDTH;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        __u32 index = (base + i) & (QWEN3_VECTOR_CAPACITY - 1);
        if (base + i >= work->count)
            return 1;
        work->output_q16[index] = (__s32)(((__s64)work->left_q16[index] *
                                            work->right_q16[index]) >> 16);
    }
    work->completed_tiles++;
    return 0;
}

SEC("socket")
int qwen3_vector_add(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    __u32 ctx = 0;

    (void)skb;
    if (!work || !work->count || work->count > QWEN3_VECTOR_MAX ||
        work->count % QWEN3_TILE_WIDTH)
        return 0;
    bpf_loop(work->count / QWEN3_TILE_WIDTH, add_tile, &ctx, 0);
    return 0;
}

SEC("socket")
int qwen3_vector_multiply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    __u32 ctx = 0;

    (void)skb;
    if (!work || !work->count || work->count > QWEN3_VECTOR_MAX ||
        work->count % QWEN3_TILE_WIDTH)
        return 0;
    bpf_loop(work->count / QWEN3_TILE_WIDTH, multiply_tile, &ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
