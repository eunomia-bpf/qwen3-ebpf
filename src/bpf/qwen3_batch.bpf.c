#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_batch.h"

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_batch_work);
} batch SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_batch_work *work = bpf_map_lookup_elem(&batch, &key);
    __s64 sum = 0;
    __u32 bounded_row = row & (QWEN3_BATCH_ROWS - 1);
    int i;

    (void)ctx;
    if (!work || row >= work->rows)
        return 1;
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_BATCH_COLS; i++) {
        if (i >= work->cols)
            break;
        sum += (__s64)work->input_q16[i] * work->weight_q24[bounded_row][i];
    }
    work->output_q16[bounded_row] = sum >> 24;
    if (work->track_argmax && work->output_q16[bounded_row] > work->best_q16) {
        work->best_q16 = work->output_q16[bounded_row];
        work->best_index = work->base_index + row;
    }
    work->completed++;
    return 0;
}

SEC("socket")
int qwen3_batch_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_batch_work *work = bpf_map_lookup_elem(&batch, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!work || !work->rows || work->rows > QWEN3_BATCH_ROWS ||
        !work->cols || work->cols > QWEN3_BATCH_COLS)
        return 0;
    bpf_loop(work->rows, compute_row, &callback_ctx, 0);
    return 0;
}

char LICENSE[] SEC("license") = "MIT";
