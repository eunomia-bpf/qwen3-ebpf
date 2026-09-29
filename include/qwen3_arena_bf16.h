#ifndef QWEN3_ARENA_BF16_H
#define QWEN3_ARENA_BF16_H

#include <linux/bpf.h>
#include <linux/types.h>

#define QWEN3_ARENA_BF16_ROWS 128
#define QWEN3_ARENA_BF16_COLS 3072
#define QWEN3_ARENA_EVENT_OUTPUTS 3072
#define QWEN3_ARENA_TOKEN_WIDTH 1024
#define QWEN3_EVENT_KV_LIMIT 256
#define QWEN3_ARENA_MODEL_MAX_BF16 (768u * 1024u * 1024u)

#define QWEN3_EVENT_READY 1u
#define QWEN3_EVENT_RUNNING 2u
#define QWEN3_EVENT_DONE 3u
#define QWEN3_EVENT_ERROR 4u
#define QWEN3_EVENT_STAGE_Q 1u
#define QWEN3_EVENT_STAGE_K 2u
#define QWEN3_EVENT_STAGE_V 3u
#define QWEN3_EVENT_STAGE_ATTENTION 4u
#define QWEN3_EVENT_STAGE_O 5u

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
    __u32 event_qkv;
    __u32 event_rope;
    __u32 event_attention;
    __u32 event_position;
    __u32 event_next_position;
    __u32 attention_past_cursor;
    __u32 event_stage;
    __u32 embedding_vocab;
    __u64 embedding_first_bf16;
    __u64 norm_first_bf16;
    __u64 q_first_bf16;
    __u64 k_first_bf16;
    __u64 v_first_bf16;
    __u64 o_first_bf16;
    __u64 post_norm_first_bf16;
    __u64 q_norm_first_bf16;
    __u64 k_norm_first_bf16;
    __u64 norm_sum_sq_q32;
    __u64 norm_inv_rms_q16;
    __u32 completed_qk_heads;
    __u32 completed_rope_heads;
    __u32 completed_attention_heads;
    __u32 attention_seen[16];
    __s32 attention_max_score_q16[16];
    __u64 attention_mass_q16[16];
    __s32 rope_cosine_q20[64];
    __s32 rope_sine_q20[64];
    __s32 embedding_q16[QWEN3_ARENA_TOKEN_WIDTH];
    __s32 hidden_q16[QWEN3_ARENA_TOKEN_WIDTH];
    __s32 matrix_output_q16[QWEN3_ARENA_EVENT_OUTPUTS];
    __s32 query_q16[2048];
    __s32 key_q16[QWEN3_ARENA_TOKEN_WIDTH];
    __s32 value_q16[QWEN3_ARENA_TOKEN_WIDTH];
    __s32 attention_output_q16[2048];
    __s32 projected_q16[QWEN3_ARENA_TOKEN_WIDTH];
};

#endif
