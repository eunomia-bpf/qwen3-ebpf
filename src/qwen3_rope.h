#ifndef QWEN3_ROPE_H
#define QWEN3_ROPE_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_rope_state {
    __s32 input_q16[QWEN3_TILE_WIDTH];
    __s32 cosine_q20[QWEN3_TILE_WIDTH / 2];
    __s32 sine_q20[QWEN3_TILE_WIDTH / 2];
    __s32 output_q16[QWEN3_TILE_WIDTH];
};

#endif
