#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_tile.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_tile);
} tile SEC(".maps");

SEC("socket")
int qwen3_matvec_tile(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_tile *work = bpf_map_lookup_elem(&tile, &key);
    __s64 sum = 0;
    int i;

    (void)skb;
    if (!work)
        return 0;

#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_TILE_WIDTH; i++)
        sum += (__s64)work->activation[i] * (__s64)work->weight[i];

    work->accumulator += sum;
    work->completed_tiles++;
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
