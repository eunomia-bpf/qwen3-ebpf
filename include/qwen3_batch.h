#ifndef QWEN3_BATCH_H
#define QWEN3_BATCH_H

#include <linux/types.h>

#define QWEN3_BATCH_ROWS 128
#define QWEN3_BATCH_COLS 3072

/* One BPF invocation computes up to 128 complete rows. */
struct qwen3_batch_work {
    __s32 input_q16[QWEN3_BATCH_COLS];
    __s32 weight_q24[QWEN3_BATCH_ROWS][QWEN3_BATCH_COLS];
    __s64 output_q16[QWEN3_BATCH_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
    __u32 track_argmax;
    __u32 base_index;
    __u32 best_index;
    __s64 best_q16;
};

#endif
