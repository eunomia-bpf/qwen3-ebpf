#ifndef QWEN3_ARENA_BF16_H
#define QWEN3_ARENA_BF16_H

#include <linux/bpf.h>
#include <linux/types.h>

#define QWEN3_ARENA_BF16_ROWS 128
#define QWEN3_ARENA_BF16_COLS 3072
#define QWEN3_ARENA_EVENT_OUTPUTS 3072
#define QWEN3_ARENA_TOKEN_WIDTH 1024
#define QWEN3_ARENA_MODEL_MAX_BF16 (768u * 1024u * 1024u)

#define QWEN3_EVENT_READY 1u
#define QWEN3_EVENT_RUNNING 2u
#define QWEN3_EVENT_DONE 3u
#define QWEN3_EVENT_ERROR 4u

struct qwen3_event_state {
    struct bpf_wq work;
    __u64 status;
    __u32 requests;
    __u32 completed_requests;
    __u64 start_ns;
    __u64 finish_ns;
};

struct qwen3_arena_bf16_work {
    __s32 input_q16[QWEN3_ARENA_BF16_COLS];
    __s64 output_q16[QWEN3_ARENA_BF16_ROWS];
    __u32 rows;
    __u32 cols;
    __u32 completed;
    __u32 track_argmax;
    __u32 base_index;
    __u32 best_index;
    __s64 best_q16;
    __u64 weight_first_bf16;
    __u64 model_elements;
    __u32 resident_weights;
    __u32 matrix_total_rows;
    __u32 event_use_token;
    __u32 event_token_id;
    __u32 embedding_vocab;
    __u64 embedding_first_bf16;
    __u64 norm_first_bf16;
    __u64 norm_sum_sq_q32;
    __u64 norm_inv_rms_q16;
    __s32 embedding_q16[QWEN3_ARENA_TOKEN_WIDTH];
    __s32 matrix_output_q16[QWEN3_ARENA_EVENT_OUTPUTS];
};

#endif
