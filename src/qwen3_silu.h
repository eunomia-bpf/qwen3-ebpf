#ifndef QWEN3_SILU_H
#define QWEN3_SILU_H

#include <linux/types.h>
#include "qwen3_tile.h"

#define QWEN3_SILU_MAX 3072

struct qwen3_silu_state {
    __s32 input_q16[QWEN3_SILU_MAX];
    __s32 output_q16[QWEN3_SILU_MAX];
    __u32 count;
    __u32 completed_tiles;
};

#endif
