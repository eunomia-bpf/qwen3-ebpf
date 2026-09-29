#ifndef QWEN3_ARENA_BF16_H
#define QWEN3_ARENA_BF16_H

#include <linux/types.h>

#define QWEN3_ARENA_BF16_ROWS 128
#define QWEN3_ARENA_BF16_COLS 3072
#define QWEN3_ARENA_MODEL_MAX_BF16 (768u * 1024u * 1024u)

struct qwen3_arena_bf16_work {
    __s32 input_q16[QWEN3_ARENA_BF16_COLS];
    __s64 output_q16[QWEN3_ARENA_BF16_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
    __u32 track_argmax;
    __u32 base_index;
    __u32 best_index;
    __s64 best_q16;
    __u64 weight_first_bf16;
    __u64 model_elements;
    __u32 resident_weights;
};

#endif
