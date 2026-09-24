#ifndef QWEN3_SILU_H
#define QWEN3_SILU_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_silu_state {
    __s32 input_q16[QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_TILE_WIDTH];
};

#endif
