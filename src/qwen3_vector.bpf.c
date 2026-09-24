#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_vector.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_vector_state);
} vector SEC(".maps");

SEC("socket")
int qwen3_vector_add(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    int i;

    (void)skb;
    if (!work)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work->output_q16[i] = work->left_q16[i] + work->right_q16[i];
    return 0;
}

SEC("socket")
int qwen3_vector_multiply(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    int i;

    (void)skb;
    if (!work)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        work->output_q16[i] =
            (__s32)(((__s64)work->left_q16[i] * work->right_q16[i]) >> 16);
    return 0;
}

SEC("socket")
int qwen3_vector_argmax(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_vector_state *work = bpf_map_lookup_elem(&vector, &key);
    int i;

    (void)skb;
    if (!work)
        return 0;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
        if (work->left_q16[i] > work->best_q16) {
            work->best_q16 = work->left_q16[i];
            work->best_index = work->base_index + i;
        }
    }
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
