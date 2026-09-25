#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_rope.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_rope_state);
} rope SEC(".maps");

static long rotate_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_rope_state *work = bpf_map_lookup_elem(&rope, &key);
    __u32 base;
    int i;

    (void)ctx;
    if (!work || head >= work->heads || head >= QWEN3_ROPE_HEADS)
        return 1;
    base = (head & (QWEN3_ROPE_CAPACITY - 1)) * QWEN3_TILE_WIDTH;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
        __u32 first_index = (base + i) &
                            (QWEN3_ROPE_CAPACITY * QWEN3_TILE_WIDTH - 1);
        __u32 second_index = (base + i + QWEN3_TILE_WIDTH / 2) &
                             (QWEN3_ROPE_CAPACITY * QWEN3_TILE_WIDTH - 1);
        __s64 first = work->input_q16[first_index];
        __s64 second = work->input_q16[second_index];
        __s64 cosine = work->cosine_q20[i];
        __s64 sine = work->sine_q20[i];
        work->output_q16[first_index] =
            (__s32)((first * cosine - second * sine) >> 20);
        work->output_q16[second_index] =
            (__s32)((second * cosine + first * sine) >> 20);
    }
    work->completed++;
    return 0;
}

SEC("socket")
int qwen3_rope_apply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_rope_state *work = bpf_map_lookup_elem(&rope, &key);
    __u32 ctx = 0;

    (void)skb;
    if (!work || !work->heads || work->heads > QWEN3_ROPE_HEADS)
        return 0;
    bpf_loop(work->heads, rotate_head, &ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
