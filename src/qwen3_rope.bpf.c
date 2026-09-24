#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_rope.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_rope_state);
} rope SEC(".maps");

SEC("socket")
int qwen3_rope_apply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_rope_state *work = bpf_map_lookup_elem(&rope, &key);
    int i;

    (void)skb;
    if (!work)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH / 2; i++) {
        __s64 first = work->input_q16[i];
        __s64 second = work->input_q16[i + QWEN3_TILE_WIDTH / 2];
        __s64 cosine = work->cosine_q20[i];
        __s64 sine = work->sine_q20[i];
        work->output_q16[i] = (__s32)((first * cosine - second * sine) >> 20);
        work->output_q16[i + QWEN3_TILE_WIDTH / 2] =
            (__s32)((second * cosine + first * sine) >> 20);
    }
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
