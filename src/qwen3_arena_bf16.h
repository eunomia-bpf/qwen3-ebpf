#ifndef QWEN3_ARENA_BF16_H
#define QWEN3_ARENA_BF16_H

#include <linux/types.h>

#define QWEN3_ARENA_BF16_ROWS 16
#define QWEN3_ARENA_BF16_COLS 3072

struct qwen3_arena_bf16_work {
    __s32 input_q16[QWEN3_ARENA_BF16_COLS];
    __s64 output_q16[QWEN3_ARENA_BF16_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
};

#endif
