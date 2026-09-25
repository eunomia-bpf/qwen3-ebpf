#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_int4.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_int4_work);
} int4 SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_int4_work *work = bpf_map_lookup_elem(&int4, &key);
    __s64 sum = 0;
    __u32 bounded_row = row & (QWEN3_INT4_ROWS - 1);
    int i;

    (void)ctx;
    if (!work || row >= work->rows)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_INT4_COLS; i++) {
        __u8 packed, nibble;
        __s32 weight;

        if (i >= work->cols)
            break;
        packed = work->weight_packed[bounded_row][i / 2];
        nibble = (i & 1) ? packed >> 4 : packed & 15;
        weight = (nibble ^ 8) - 8;
        sum += (__s64)work->input_q16[i] * weight *
               work->scale_q24[bounded_row][i / QWEN3_INT4_GROUP];
    }
    work->output_q16[bounded_row] = sum >> 24;
    work->completed++;
    return 0;
}

SEC("socket")
int qwen3_int4_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_int4_work *work = bpf_map_lookup_elem(&int4, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->rows || work->rows > QWEN3_INT4_ROWS ||
        !work->cols || work->cols > QWEN3_INT4_COLS ||
        work->cols % QWEN3_INT4_GROUP)
        return 0;
    bpf_loop(work->rows, compute_row, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
