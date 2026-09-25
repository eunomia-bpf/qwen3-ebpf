#ifndef QWEN3_VECTOR_H
#define QWEN3_VECTOR_H

#include <linux/types.h>
#include "qwen3_tile.h"

#define QWEN3_VECTOR_MAX 3072
/* Power-of-two backing keeps callback indices verifier-bounded. */
#define QWEN3_VECTOR_CAPACITY 4096

struct qwen3_vector_state {
    __s32 left_q16[QWEN3_VECTOR_CAPACITY];
    __s32 right_q16[QWEN3_VECTOR_CAPACITY];
    __s32 output_q16[QWEN3_VECTOR_CAPACITY];
    __u32 count;
    __u32 completed_tiles;
};

#endif
