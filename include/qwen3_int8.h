#ifndef QWEN3_INT8_H
#define QWEN3_INT8_H

#include <linux/types.h>

#define QWEN3_INT8_ROWS 128
#define QWEN3_INT8_COLS 3072
#define QWEN3_INT8_GROUP 32

struct qwen3_int8_work {
    __s32 input_q16[QWEN3_INT8_COLS];
    __s8 weight_q8[QWEN3_INT8_ROWS][QWEN3_INT8_COLS];
    __s64 scale_q32[QWEN3_INT8_ROWS][QWEN3_INT8_COLS / QWEN3_INT8_GROUP];
    __s64 output_q16[QWEN3_INT8_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
    __u32 track_argmax;
    __u32 base_index;
    __u32 best_index;
    __s64 best_q16;
};

#endif
