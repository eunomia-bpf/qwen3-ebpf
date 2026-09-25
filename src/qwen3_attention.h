#ifndef QWEN3_ATTENTION_H
#define QWEN3_ATTENTION_H

#include <linux/types.h>
#include "qwen3_tile.h"

#define QWEN3_ATTENTION_LAYERS 28
#define QWEN3_ATTENTION_KV_HEADS 8
#define QWEN3_ATTENTION_CONTEXT_LIMIT 40960
#define QWEN3_ATTENTION_CHUNK 256

struct qwen3_kv_pair {
    __s32 key_q16[QWEN3_TILE_WIDTH];
    __s32 value_q16[QWEN3_TILE_WIDTH];
};

struct qwen3_attention_state {
    __s32 query_q16[QWEN3_TILE_WIDTH];
    __s32 key_q16[QWEN3_TILE_WIDTH];
    __s32 value_q16[QWEN3_TILE_WIDTH];
    __s32 output_q16[QWEN3_TILE_WIDTH];
    __s32 score_q16;
    __s32 max_score_q16;
    __u64 mass_q16;
    __u32 seen;
    __u32 layer;
    __u32 kv_head;
    __u32 base_position;
    __u32 step_count;
};

struct qwen3_attention_heads_state {
    struct qwen3_attention_state heads[QWEN3_Q_HEADS];
    __u32 layer;
    __u32 base_position;
    __u32 step_count;
    __u32 completed_positions;
};

#endif
