#ifndef QWEN3_VECTOR_H
#define QWEN3_VECTOR_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_vector_state {
    __s32 left_q16[QWEN3_TILE_WIDTH];
    __s32 right_q16[QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_TILE_WIDTH];
    __s32 best_q16;
    __u32 best_index;
    __u32 base_index;
};

#endif
