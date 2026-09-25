#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_int8.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_int8_work);
} int8 SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_int8_work *work = bpf_map_lookup_elem(&int8, &key);
    __u32 bounded_row = row & (QWEN3_INT8_ROWS - 1);
    __s64 sum = 0;
    int i;

    (void)ctx;
    if (!work || row >= work->rows)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_INT8_COLS; i++) {
        if (i >= work->cols)
            break;
        sum += (__s64)work->input_q16[i] *
               work->weight_q8[bounded_row][i] *
               work->scale_q32[bounded_row][i / QWEN3_INT8_GROUP];
    }
    work->output_q16[bounded_row] = sum >> 32;
    if (work->track_argmax && work->output_q16[bounded_row] > work->best_q16) {
        work->best_q16 = work->output_q16[bounded_row];
        work->best_index = work->base_index + row;
    }
    work->completed++;
    return 0;
}

SEC("socket")
int qwen3_int8_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_int8_work *work = bpf_map_lookup_elem(&int8, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->rows || work->rows > QWEN3_INT8_ROWS ||
        !work->cols || work->cols > QWEN3_INT8_COLS ||
        work->cols % QWEN3_INT8_GROUP)
        return 0;
    bpf_loop(work->rows, compute_row, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
