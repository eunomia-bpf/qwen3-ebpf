#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_int4.h"

#define __arena __attribute__((address_space(1)))

struct {
    __uint(type, BPF_MAP_TYPE_ARENA);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 16);
#ifdef __TARGET_ARCH_arm64
    __ulong(map_extra, 0x1ull << 32);
#else
    __ulong(map_extra, 0x1ull << 44);
#endif
} arena SEC(".maps");

__u8 __arena weight_packed[QWEN3_INT4_ROWS][QWEN3_INT4_COLS / 2];
__s32 __arena scale_q24[QWEN3_INT4_ROWS][QWEN3_INT4_COLS /
                                               QWEN3_INT4_GROUP];

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_arena_int4_work);
} work SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_int4_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_row = row & (QWEN3_INT4_ROWS - 1);
    __s64 sum = 0;
    int i;

    (void)ctx;
    if (!state || row >= state->rows)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_INT4_COLS; i++) {
        __u8 packed, nibble;
        __s32 weight;

        if (i >= state->cols)
            break;
        packed = weight_packed[bounded_row][i / 2];
        nibble = (i & 1) ? packed >> 4 : packed & 15;
        weight = (nibble ^ 8) - 8;
        sum += (__s64)state->input_q16[i] * weight *
               scale_q24[bounded_row][i / QWEN3_INT4_GROUP];
    }
    state->output_q16[bounded_row] = sum >> 24;
    state->completed++;
    return 0;
}

SEC("socket")
int qwen3_arena_int4_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_arena_int4_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!state || !state->rows || state->rows > QWEN3_INT4_ROWS ||
        !state->cols || state->cols > QWEN3_INT4_COLS ||
        state->cols % QWEN3_INT4_GROUP)
        return 0;
    bpf_loop(state->rows, compute_row, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
