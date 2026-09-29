#ifndef QWEN3_NORM_H
#define QWEN3_NORM_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_norm_state {
    __s32 activation_q16[QWEN3_HIDDEN_SIZE];
    __s32 weight_q20[QWEN3_HIDDEN_SIZE];
    __s32 output_q16[QWEN3_HIDDEN_SIZE];
    __u64 sum_sq_q32;
    __u64 inv_rms_q16;
    __u32 completed_tiles;
    __u32 total_tiles;
};

#define QWEN3_QK_HEADS (QWEN3_Q_HEADS + QWEN3_KV_HEADS)

struct qwen3_qk_norm_state {
    __s32 activation_q16[QWEN3_QK_HEADS * QWEN3_TILE_WIDTH];
    __s32 weight_q20[2][QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_QK_HEADS * QWEN3_TILE_WIDTH];
    __u32 completed_heads;
};

#endif
