#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_arena_bf16.h"

#define __arena __attribute__((address_space(1)))

struct {
    __uint(type, BPF_MAP_TYPE_ARENA);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 512);
#ifdef __TARGET_ARCH_arm64
    __ulong(map_extra, 0x1ull << 32);
#else
    __ulong(map_extra, 0x1ull << 44);
#endif
} arena SEC(".maps");

__u16 __arena weight_bf16[QWEN3_ARENA_BF16_ROWS][QWEN3_ARENA_BF16_COLS];
__s32 __arena q24_by_bf16[1 << 16];

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_arena_bf16_work);
} work SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_row = row & (QWEN3_ARENA_BF16_ROWS - 1);
    __s64 sum = 0;
    int i;

    (void)ctx;
    if (!state || row >= state->rows)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_ARENA_BF16_COLS; i++) {
        __u16 bits;

        if (i >= state->cols)
            break;
        bits = weight_bf16[bounded_row][i];
        sum += (__s64)state->input_q16[i] * q24_by_bf16[bits];
    }
    state->output_q16[bounded_row] = sum >> 24;
    if (state->track_argmax &&
        state->output_q16[bounded_row] > state->best_q16) {
        state->best_q16 = state->output_q16[bounded_row];
        state->best_index = state->base_index + row;
    }
    state->completed++;
    return 0;
}

SEC("socket")
int qwen3_arena_bf16_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS)
        return 0;
    bpf_loop(state->rows, compute_row, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
