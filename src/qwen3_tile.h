#ifndef QWEN3_TILE_H
#define QWEN3_TILE_H

#include <linux/types.h>

#define QWEN3_TILE_WIDTH 128
#define QWEN3_HIDDEN_SIZE 1024
#define QWEN3_HIDDEN_TILES (QWEN3_HIDDEN_SIZE / QWEN3_TILE_WIDTH)

/* One Q8 x Q8 tile. The loader only supplies data; the dot product runs in BPF. */
struct qwen3_tile {
    __s8 activation[QWEN3_TILE_WIDTH];
    __s8 weight[QWEN3_TILE_WIDTH];
    __s64 accumulator;
    __u32 completed_tiles;
};

#endif
