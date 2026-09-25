#ifndef QWEN3_BATCH_H
#define QWEN3_BATCH_H

#include <linux/types.h>

#define QWEN3_BATCH_ROWS 16
#define QWEN3_BATCH_COLS 3072

/* One BPF invocation computes up to 16 complete rows. */
struct qwen3_batch_work {
    __s32 input_q16[QWEN3_BATCH_COLS];
    __s32 weight_q24[QWEN3_BATCH_ROWS][QWEN3_BATCH_COLS];
    __s64 output_q16[QWEN3_BATCH_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
};

#endif
