#ifndef QWEN3_ATTENTION_H
#define QWEN3_ATTENTION_H

#include <linux/types.h>
#include "qwen3_tile.h"

struct qwen3_attention_state {
    __s32 query_q16[QWEN3_TILE_WIDTH];
    __s32 key_q16[QWEN3_TILE_WIDTH];
    __s32 value_q16[QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_TILE_WIDTH];
    __s32 score_q16;
    __s32 max_score_q16;
    __u64 mass_q16;
    __u32 seen;
};

#endif
