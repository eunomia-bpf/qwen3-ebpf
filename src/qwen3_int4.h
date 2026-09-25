#ifndef QWEN3_INT4_H
#define QWEN3_INT4_H

#include <linux/types.h>

#define QWEN3_INT4_ROWS 16
#define QWEN3_INT4_COLS 3072
#define QWEN3_INT4_GROUP 128

struct qwen3_int4_work {
    __s32 input_q16[QWEN3_INT4_COLS];
    __u8 weight_packed[QWEN3_INT4_ROWS][QWEN3_INT4_COLS / 2];
    __s32 scale_q24[QWEN3_INT4_ROWS][QWEN3_INT4_COLS / QWEN3_INT4_GROUP];
    __s64 output_q16[QWEN3_INT4_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
};

/* The optional arena operator keeps only activations and results in this map. */
struct qwen3_arena_int4_work {
    __s32 input_q16[QWEN3_INT4_COLS];
    __s64 output_q16[QWEN3_INT4_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
};

#endif
