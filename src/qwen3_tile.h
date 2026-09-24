#ifndef QWEN3_TILE_H
#define QWEN3_TILE_H

#include <linux/types.h>

#define QWEN3_TILE_WIDTH 128
#define QWEN3_HIDDEN_SIZE 1024
#define QWEN3_HIDDEN_TILES (QWEN3_HIDDEN_SIZE / QWEN3_TILE_WIDTH)

/* One Q8 x Q8 tile. The loader only supplies data; the dot product runs in BPF. */
struct qwen3_tile {
    __s8 activation[QWEN3_TILE_WIDTH];
    __s32 activation_q16[QWEN3_TILE_WIDTH];
    __s8 weight[QWEN3_TILE_WIDTH];
    __s32 weight_q24[QWEN3_TILE_WIDTH];
    __s64 accumulator;
    __s64 output_q16;
    __s32 weight_scale_q24;
    __u32 completed_tiles;
    __u32 fixed_point_mode;
};

#endif
