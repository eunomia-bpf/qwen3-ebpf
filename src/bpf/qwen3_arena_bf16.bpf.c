#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>
#include "qwen3_arena_bf16.h"

#define __arena __attribute__((address_space(1)))

struct {
    __uint(type, BPF_MAP_TYPE_ARENA);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, QWEN3_ARENA_MODEL_MAX_BF16 / 2048u + 512u);
#ifdef __TARGET_ARCH_arm64
    __ulong(map_extra, 0x1ull << 32);
#else
    __ulong(map_extra, 0x1ull << 44);
#endif
} arena SEC(".maps");

__u16 __arena weight_bf16[QWEN3_ARENA_BF16_ROWS][QWEN3_ARENA_BF16_COLS];
__s32 __arena q24_by_bf16[1 << 16];
__s32 __arena q16_by_bf16[1 << 16];
__s32 __arena q20_by_bf16[1 << 16];
__u16 __arena *model_bf16;

extern void *bpf_arena_alloc_pages(void *map, void *addr, __u32 page_cnt,
                                   int node_id, __u64 flags) __ksym;
extern int bpf_wq_init(struct bpf_wq *wq, void *map, unsigned int flags) __ksym;
extern int bpf_wq_start(struct bpf_wq *wq, unsigned int flags) __ksym;
extern int bpf_wq_set_callback_impl(struct bpf_wq *wq,
    int (*callback)(void *map, int *key, void *value),
    unsigned int flags__k, void *aux__ign) __ksym;

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(map_flags, BPF_F_MMAPABLE);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_arena_bf16_work);
} work SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct qwen3_event_state);
} event SEC(".maps");

static long compute_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_row = row & (QWEN3_ARENA_BF16_ROWS - 1);
    __u16 __arena *weights;
    __s64 sum = 0;
    int i;

    (void)ctx;
    if (!state || row >= state->rows)
        return 1;
    if (state->resident_weights) {
        __u64 first = state->weight_first_bf16 +
                      (__u64)(state->base_index + row) * state->cols;
        if (!model_bf16 || first > state->model_elements ||
            state->cols > state->model_elements - first)
            return 1;
        weights = model_bf16 + first;
    } else {
        weights = weight_bf16[bounded_row];
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < QWEN3_ARENA_BF16_COLS; i++) {
        __u16 bits;

        if (i >= state->cols)
            break;
        bits = weights[i];
        sum += (__s64)state->input_q16[i] * q24_by_bf16[bits];
    }
    state->output_q16[bounded_row] = sum >> 24;
    if (state->track_argmax &&
        state->output_q16[bounded_row] > state->best_q16) {
        state->best_q16 = state->output_q16[bounded_row];
        state->best_index = state->base_index + row;
    }
    state->completed++;
    return 0;
}

SEC("syscall")
int qwen3_arena_allocate_model(void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 pages;

    (void)ctx;
    if (!state || model_bf16 || !state->model_elements ||
        state->model_elements > QWEN3_ARENA_MODEL_MAX_BF16)
        return -1;
    pages = (state->model_elements + 2047u) / 2048u;
    model_bf16 = (__u16 __arena *)bpf_arena_alloc_pages(&arena, 0,
                                                        pages, -1, 0);
    return model_bf16 ? 0 : -1;
}

SEC("socket")
int qwen3_arena_bf16_rows(struct __sk_buff *skb)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 callback_ctx = 0;

    (void)skb;
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS)
        return 0;
    bpf_loop(state->rows, compute_row, &callback_ctx, 0);
    return 0;
}

static long store_output_row(__u32 row, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 index;
    __s64 output;

    if (!state || row >= state->rows) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    /* The mask makes the verifier retain a bounded 64-bit map offset. */
    index = (state->base_index + row) & 4095u;
    if (index >= QWEN3_ARENA_EVENT_OUTPUTS) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    output = state->output_q16[row & (QWEN3_ARENA_BF16_ROWS - 1)];
    if (output > 2147483647LL || output < -2147483648LL) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    if (!state->event_qkv)
        state->matrix_output_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_Q && index < 2048)
        state->query_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_K &&
             index < QWEN3_ARENA_TOKEN_WIDTH)
        state->key_q16[index] = (__s32)output;
    else if (state->event_stage == QWEN3_EVENT_STAGE_V &&
             index < QWEN3_ARENA_TOKEN_WIDTH)
        state->value_q16[index] = (__s32)output;
    else {
        *(__u32 *)ctx = 1;
        return 1;
    }
    return 0;
}

static long load_token_embedding(__u32 col, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u64 first;
    __u32 bounded_col = col & (QWEN3_ARENA_TOKEN_WIDTH - 1);
    __u16 bits;

    if (!state || !model_bf16 ||
        state->event_token_id >= state->embedding_vocab) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    first = state->embedding_first_bf16 +
            (__u64)state->event_token_id * QWEN3_ARENA_TOKEN_WIDTH + bounded_col;
    if (first >= state->model_elements) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    bits = model_bf16[first];
    state->embedding_q16[bounded_col] =
        q16_by_bf16[bits];
    return 0;
}

static __always_inline __u64 event_isqrt64(__u64 value)
{
    __u64 root = 1ULL << 20;
    int i;

#pragma clang loop unroll(disable)
    for (i = 0; i < 24; i++) {
        root = (root + value / root) >> 1;
        root |= 1;
    }
    return root;
}

static long event_norm_sum(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 7;
    __u64 sum = 0;
    int i;

    if (!state) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __s64 x = state->embedding_q16[bounded_tile * 128 + i];
        sum += (__u64)(x * x);
    }
    state->norm_sum_sq_q32 += sum;
    return 0;
}

static long event_norm_apply(__u32 tile, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_tile = tile & 7;
    int i;

    if (!state || !model_bf16) {
        *(__u32 *)ctx = 1;
        return 1;
    }
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u32 index = bounded_tile * 128 + i;
        __u64 first = state->norm_first_bf16 + index;
        __s64 normalized;
        __u16 bits;

        if (first >= state->model_elements) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        bits = model_bf16[first];
        normalized = ((__s64)state->embedding_q16[index] *
                      (__s64)state->norm_inv_rms_q16) >> 16;
        state->input_q16[index] =
            (__s32)((normalized * q20_by_bf16[bits]) >> 20);
    }
    return 0;
}

static long event_qk_norm_head(__u32 head, void *ctx)
{
    const __u32 key = 0;
    struct qwen3_arena_bf16_work *state = bpf_map_lookup_elem(&work, &key);
    __u32 bounded_head = head & 31;
    __u32 is_key, base;
    __u64 sum = 0, inv_rms;
    int i;

    if (!state || !model_bf16 || bounded_head >= 24) {
        *(__u32 *)ctx = 1;
        return 1;
    }
    is_key = bounded_head >= 16;
    base = (bounded_head & 15) * 128;
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __s64 x = is_key ? state->key_q16[base + i] :
                             state->query_q16[base + i];
        sum += (__u64)(x * x);
    }
    inv_rms = (1ULL << 32) / event_isqrt64(sum / 128 + 4295);
#pragma clang loop unroll(disable)
    for (i = 0; i < 128; i++) {
        __u64 first = (is_key ? state->k_norm_first_bf16 :
                               state->q_norm_first_bf16) + i;
        __u16 bits;
        __s64 x, normalized;

        if (first >= state->model_elements) {
            *(__u32 *)ctx = 1;
            return 1;
        }
        bits = model_bf16[first];
        x = is_key ? state->key_q16[base + i] :
                     state->query_q16[base + i];
        normalized = (x * (__s64)inv_rms) >> 16;
        if (is_key)
            state->key_q16[base + i] =
                (__s32)((normalized * q20_by_bf16[bits]) >> 20);
        else
            state->query_q16[base + i] =
                (__s32)((normalized * q20_by_bf16[bits]) >> 20);
    }
    state->completed_qk_heads++;
    return 0;
}

static int event_callback(void *map, int *key, void *value)
{
    const __u32 work_key = 0;
    struct qwen3_event_state *job = value;
    struct qwen3_arena_bf16_work *state =
        bpf_map_lookup_elem(&work, &work_key);
    __u32 callback_ctx = 0;
    __u32 total, base, rows;

    (void)map;
    (void)key;
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    total = state->matrix_total_rows ? state->matrix_total_rows : state->rows;
    base = state->base_index;
    rows = state->rows;
    if (total > QWEN3_ARENA_EVENT_OUTPUTS || base >= total ||
        rows > total - base ||
        (state->event_qkv &&
         (state->event_stage < QWEN3_EVENT_STAGE_Q ||
          state->event_stage > QWEN3_EVENT_STAGE_V))) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    if (state->event_use_token && base == 0 &&
        (!state->event_qkv || state->event_stage == QWEN3_EVENT_STAGE_Q)) {
        if (state->cols != QWEN3_ARENA_TOKEN_WIDTH ||
            state->event_token_id >= state->embedding_vocab ||
            state->embedding_first_bf16 > state->model_elements ||
            (__u64)state->embedding_vocab * QWEN3_ARENA_TOKEN_WIDTH >
                state->model_elements - state->embedding_first_bf16) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        bpf_loop(QWEN3_ARENA_TOKEN_WIDTH, load_token_embedding,
                 &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->norm_first_bf16 > state->model_elements ||
            QWEN3_ARENA_TOKEN_WIDTH >
                state->model_elements - state->norm_first_bf16) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->norm_sum_sq_q32 = 0;
        callback_ctx = 0;
        bpf_loop(8, event_norm_sum, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        state->norm_inv_rms_q16 =
            (1ULL << 32) / event_isqrt64(state->norm_sum_sq_q32 /
                                          QWEN3_ARENA_TOKEN_WIDTH + 4295);
        callback_ctx = 0;
        bpf_loop(8, event_norm_apply, &callback_ctx, 0);
        if (callback_ctx) {
            job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
    }
    state->completed = 0;
    bpf_loop(rows, compute_row, &callback_ctx, 0);
    if (state->completed != rows) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    callback_ctx = 0;
    bpf_loop(rows, store_output_row, &callback_ctx, 0);
    if (callback_ctx) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    base += rows;
    if (base == total) {
        if (state->event_qkv && state->event_stage < QWEN3_EVENT_STAGE_V) {
            if (state->event_stage == QWEN3_EVENT_STAGE_Q) {
                state->event_stage = QWEN3_EVENT_STAGE_K;
                state->weight_first_bf16 = state->k_first_bf16;
            } else if (state->event_stage == QWEN3_EVENT_STAGE_K) {
                state->event_stage = QWEN3_EVENT_STAGE_V;
                state->weight_first_bf16 = state->v_first_bf16;
            } else {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->matrix_total_rows = QWEN3_ARENA_TOKEN_WIDTH;
            state->base_index = 0;
            state->rows = QWEN3_ARENA_BF16_ROWS;
            if (bpf_wq_start(&job->work, 0))
                job->status = QWEN3_EVENT_ERROR;
            return 0;
        }
        if (state->event_qkv) {
            if (state->q_norm_first_bf16 > state->model_elements ||
                128 > state->model_elements - state->q_norm_first_bf16 ||
                state->k_norm_first_bf16 > state->model_elements ||
                128 > state->model_elements - state->k_norm_first_bf16) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
            state->completed_qk_heads = 0;
            callback_ctx = 0;
            bpf_loop(24, event_qk_norm_head, &callback_ctx, 0);
            if (callback_ctx || state->completed_qk_heads != 24) {
                job->status = QWEN3_EVENT_ERROR;
                return 0;
            }
        }
        state->base_index = base;
        job->finish_ns = bpf_ktime_get_ns();
        job->completed_requests++;
        job->status = QWEN3_EVENT_DONE;
        return 0;
    }
    state->base_index = base;
    state->rows = total - base < QWEN3_ARENA_BF16_ROWS ?
        total - base : QWEN3_ARENA_BF16_ROWS;
    if (bpf_wq_start(&job->work, 0))
        job->status = QWEN3_EVENT_ERROR;
    return 0;
}

SEC("syscall")
int qwen3_event_init(void *ctx)
{
    const __u32 key = 0;
    struct qwen3_event_state *job = bpf_map_lookup_elem(&event, &key);

    (void)ctx;
    if (!job || bpf_wq_init(&job->work, &event, 0) ||
        bpf_wq_set_callback_impl(&job->work, event_callback, 0, 0))
        return -1;
    job->status = QWEN3_EVENT_READY;
    return 0;
}

SEC("xdp")
int qwen3_event_xdp(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *end = (void *)(long)ctx->data_end;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    struct udphdr *udp;
    unsigned char *payload;
    const __u32 key = 0;
    struct qwen3_event_state *job;
    struct qwen3_arena_bf16_work *state;
    __u64 old_status;
    __u32 token_id = 0;

    if ((void *)(eth + 1) > end || eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;
    ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > end || ip->ihl != 5 || ip->protocol != IPPROTO_UDP)
        return XDP_PASS;
    udp = (void *)(ip + 1);
    if ((void *)(udp + 1) > end || udp->dest != bpf_htons(49002))
        return XDP_PASS;
    payload = (void *)(udp + 1);
    if (payload + 4 > (unsigned char *)end ||
        payload[0] != 'Q' || payload[1] != '3' ||
        payload[2] != 'B' || payload[3] != 'P')
        return XDP_PASS;
    state = bpf_map_lookup_elem(&work, &key);
    if (!state)
        return XDP_PASS;
    if (state->event_use_token) {
        if (payload + 8 > (unsigned char *)end)
            return XDP_PASS;
        token_id = ((__u32)payload[4] << 24) |
                   ((__u32)payload[5] << 16) |
                   ((__u32)payload[6] << 8) | payload[7];
        if (token_id >= state->embedding_vocab)
            return XDP_PASS;
    }
    job = bpf_map_lookup_elem(&event, &key);
    if (!job)
        return XDP_PASS;
    old_status = __sync_val_compare_and_swap(&job->status,
                                              QWEN3_EVENT_READY,
                                              QWEN3_EVENT_RUNNING);
    if (old_status != QWEN3_EVENT_READY &&
        (old_status != QWEN3_EVENT_DONE ||
         __sync_val_compare_and_swap(&job->status,
                                      QWEN3_EVENT_DONE,
                                      QWEN3_EVENT_RUNNING) !=
                                      QWEN3_EVENT_DONE))
        return XDP_PASS;
    state->base_index = 0;
    state->event_token_id = token_id;
    if (state->event_qkv) {
        state->event_stage = QWEN3_EVENT_STAGE_Q;
        state->weight_first_bf16 = state->q_first_bf16;
        state->matrix_total_rows = 2048;
    }
    if (state->matrix_total_rows)
        state->rows = state->matrix_total_rows < QWEN3_ARENA_BF16_ROWS ?
            state->matrix_total_rows : QWEN3_ARENA_BF16_ROWS;
    job->requests++;
    job->start_ns = bpf_ktime_get_ns();
    job->finish_ns = 0;
    if (bpf_wq_start(&job->work, 0))
        job->status = QWEN3_EVENT_ERROR;
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
