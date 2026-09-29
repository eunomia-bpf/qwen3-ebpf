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

static int event_callback(void *map, int *key, void *value)
{
    const __u32 work_key = 0;
    struct qwen3_event_state *job = value;
    struct qwen3_arena_bf16_work *state =
        bpf_map_lookup_elem(&work, &work_key);
    __u32 callback_ctx = 0;

    (void)map;
    (void)key;
    if (!state || !state->rows || state->rows > QWEN3_ARENA_BF16_ROWS ||
        !state->cols || state->cols > QWEN3_ARENA_BF16_COLS) {
        job->status = QWEN3_EVENT_ERROR;
        return 0;
    }
    state->completed = 0;
    bpf_loop(state->rows, compute_row, &callback_ctx, 0);
    job->status = state->completed == state->rows ?
        QWEN3_EVENT_DONE : QWEN3_EVENT_ERROR;
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
    job = bpf_map_lookup_elem(&event, &key);
    if (!job || __sync_val_compare_and_swap(&job->status,
                                             QWEN3_EVENT_READY,
                                             QWEN3_EVENT_RUNNING) !=
                                             QWEN3_EVENT_READY)
        return XDP_PASS;
    job->requests++;
    if (bpf_wq_start(&job->work, 0))
        job->status = QWEN3_EVENT_ERROR;
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
