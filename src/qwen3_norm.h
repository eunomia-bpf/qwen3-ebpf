#ifndef QWEN3_NORM_H
#define QWEN3_NORM_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_norm_state {
    __s32 activation_q16[QWEN3_TILE_WIDTH];
    __s32 weight_q20[QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_TILE_WIDTH];
    __u64 sum_sq_q32;
    __u64 inv_rms_q16;
    __u32 completed_tiles;
};

#endif
