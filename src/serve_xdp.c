#include <errno.h>
#include <math.h>
#include <net/if.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_arena_bf16.h"
#include "qwen3_arena_bf16.skel.h"
#include "safetensors.h"

static volatile sig_atomic_t stopping;
static uint8_t q24_valid[UINT16_MAX + 1];

static void stop_on_signal(int signum)
{
    (void)signum;
    stopping = 1;
}

static int tensor_offset(struct safetensors_file *file, const char *name,
                         uint64_t expected, __u64 *first)
{
    uint64_t byte, elements;

    if (safetensors_find_bf16(file, name, &byte, &elements) ||
        elements != expected || (byte & 1)) {
        fprintf(stderr, "missing or invalid BF16 tensor: %s\n", name);
        return -1;
    }
    if (file->mapping_size < 8 + file->header_length ||
        byte > file->mapping_size - 8 - file->header_length ||
        elements > (file->mapping_size - 8 - file->header_length - byte) / 2) {
        fprintf(stderr, "BF16 tensor outside model payload: %s\n", name);
        return -1;
    }
    *first = byte / 2;
    return 0;
}

static int layer_offsets(struct safetensors_file *file, unsigned int layer,
                         struct qwen3_arena_layer_weights *weights)
{
    char name[96];
    const char *projection[4] = {"q_proj", "k_proj", "v_proj", "o_proj"};
    __u64 *projection_first[4] = {
        &weights->q_first_bf16, &weights->k_first_bf16,
        &weights->v_first_bf16, &weights->o_first_bf16,
    };
    const char *mlp[3] = {"gate_proj", "up_proj", "down_proj"};
    __u64 *mlp_first[3] = {
        &weights->gate_first_bf16, &weights->up_first_bf16,
        &weights->down_first_bf16,
    };

    snprintf(name, sizeof(name), "model.layers.%u.input_layernorm.weight", layer);
    if (tensor_offset(file, name, 1024, &weights->input_norm_first_bf16))
        return -1;
    snprintf(name, sizeof(name),
             "model.layers.%u.post_attention_layernorm.weight", layer);
    if (tensor_offset(file, name, 1024, &weights->post_norm_first_bf16))
        return -1;
    snprintf(name, sizeof(name), "model.layers.%u.self_attn.q_norm.weight", layer);
    if (tensor_offset(file, name, 128, &weights->q_norm_first_bf16))
        return -1;
    snprintf(name, sizeof(name), "model.layers.%u.self_attn.k_norm.weight", layer);
    if (tensor_offset(file, name, 128, &weights->k_norm_first_bf16))
        return -1;
    for (unsigned int i = 0; i < 4; i++) {
        uint64_t elements = (uint64_t)(i == 0 ? 2048 : 1024) *
                            (i == 3 ? 2048 : 1024);

        snprintf(name, sizeof(name), "model.layers.%u.self_attn.%s.weight",
                 layer, projection[i]);
        if (tensor_offset(file, name, elements, projection_first[i]))
            return -1;
    }
    for (unsigned int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "model.layers.%u.mlp.%s.weight",
                 layer, mlp[i]);
        if (tensor_offset(file, name, 3072ULL * 1024, mlp_first[i]))
            return -1;
    }
    return 0;
}

static void load_lookup_tables(struct qwen3_arena_bf16_bpf *skel)
{
    for (uint32_t bits = 0; bits <= UINT16_MAX; bits++) {
        uint32_t float_bits = bits << 16;
        float value;
        double scaled;

        memcpy(&value, &float_bits, sizeof(value));
        scaled = (double)value * 16777216.0;
        q24_valid[bits] = isfinite(scaled) &&
                          scaled <= INT32_MAX && scaled >= INT32_MIN;
        skel->arena->q24_by_bf16[bits] = q24_valid[bits] ?
            (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5)) : 0;
        scaled = (double)value * 65536.0;
        skel->arena->q16_by_bf16[bits] = isfinite(scaled) &&
            scaled <= INT32_MAX && scaled >= INT32_MIN ?
            (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5)) : 0;
        scaled = (double)value * 1048576.0;
        skel->arena->q20_by_bf16[bits] = isfinite(scaled) &&
            scaled <= INT32_MAX && scaled >= INT32_MIN ?
            (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5)) : 0;
    }
}

static int validate_q24_matrix(const uint8_t *model, size_t bytes,
                               __u64 first, size_t elements)
{
    if (first > bytes / 2 || elements > bytes / 2 - first)
        return -1;
    for (size_t i = 0; i < elements; i++) {
        size_t offset = (size_t)(first + i) * 2;
        uint16_t bits = (uint16_t)model[offset] |
                        ((uint16_t)model[offset + 1] << 8);

        if (!q24_valid[bits]) {
            fprintf(stderr, "matrix BF16 weight out of Q24 range\n");
            return -1;
        }
    }
    return 0;
}

static int validate_layer_matrices(const uint8_t *model, size_t bytes,
                                   const struct qwen3_arena_layer_weights *weights)
{
    const __u64 offsets[7] = {
        weights->q_first_bf16, weights->k_first_bf16,
        weights->v_first_bf16, weights->o_first_bf16,
        weights->gate_first_bf16, weights->up_first_bf16,
        weights->down_first_bf16,
    };
    const size_t elements[7] = {
        2048 * 1024, 1024 * 1024, 1024 * 1024, 1024 * 2048,
        3072 * 1024, 3072 * 1024, 3072 * 1024,
    };

    for (size_t i = 0; i < 7; i++)
        if (validate_q24_matrix(model, bytes, offsets[i], elements[i]))
            return -1;
    return 0;
}

int main(int argc, char **argv)
{
    struct safetensors_file file = {0};
    struct qwen3_arena_bf16_bpf *skel = NULL;
    struct qwen3_arena_bf16_work *work = MAP_FAILED;
    struct bpf_link *link = NULL;
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    const uint8_t *model;
    size_t bytes, mapping_len, page;
    __u64 first;
    unsigned int ifindex;
    int rc = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: %s model.safetensors interface\n", argv[0]);
        return 2;
    }
    ifindex = if_nametoindex(argv[2]);
    if (!ifindex || safetensors_open(&file, argv[1])) {
        fprintf(stderr, "could not find interface or open model\n");
        return 1;
    }
    if (file.mapping_size < 8 + file.header_length)
        goto done;
    bytes = file.mapping_size - 8 - file.header_length;
    if ((bytes & 1) || bytes / 2 > QWEN3_ARENA_MODEL_MAX_BF16)
        goto done;
    model = file.mapping + 8 + file.header_length;
    skel = qwen3_arena_bf16_bpf__open_and_load();
    if (!skel)
        goto done;
    page = (size_t)sysconf(_SC_PAGESIZE);
    mapping_len = (sizeof(*work) + page - 1) / page * page;
    work = mmap(NULL, mapping_len, PROT_READ | PROT_WRITE, MAP_SHARED,
                bpf_map__fd(skel->maps.work), 0);
    if (work == MAP_FAILED)
        goto done;
    memset(work, 0, sizeof(*work));
    work->model_elements = bytes / 2;
    load_lookup_tables(skel);
    if (bpf_prog_test_run_opts(bpf_program__fd(
            skel->progs.qwen3_arena_allocate_model), &opts) ||
        opts.retval || !skel->bss->model_bf16)
        goto done;
    memcpy(skel->bss->model_bf16, model, bytes);
    if (tensor_offset(&file, "model.embed_tokens.weight",
                      (uint64_t)QWEN3_EVENT_VOCAB * QWEN3_ARENA_TOKEN_WIDTH,
                      &first))
        goto done;
    work->embedding_first_bf16 = first;
    if (tensor_offset(&file, "model.norm.weight",
                      QWEN3_ARENA_TOKEN_WIDTH, &first))
        goto done;
    work->final_norm_first_bf16 = first;
    for (uint32_t layer = 0; layer < QWEN3_EVENT_MAX_LAYERS; layer++) {
        struct qwen3_arena_layer_weights weights = {0};

        if (layer_offsets(&file, layer, &weights) ||
            validate_layer_matrices(model, bytes, &weights) ||
            bpf_map_update_elem(bpf_map__fd(skel->maps.event_layer_weights),
                                &layer, &weights, BPF_ANY))
            goto done;
    }
    work->resident_weights = 1;
    work->embedding_vocab = QWEN3_EVENT_VOCAB;
    work->event_use_token = 1;
    work->event_qkv = 1;
    work->event_rope = 1;
    work->event_attention = 1;
    work->event_layer_count = QWEN3_EVENT_MAX_LAYERS;
    work->event_final_logits = 1;
    opts.retval = 0;
    if (bpf_prog_test_run_opts(bpf_program__fd(skel->progs.qwen3_event_init),
                               &opts) || opts.retval)
        goto done;
    link = bpf_program__attach_xdp(skel->progs.qwen3_event_xdp, ifindex);
    if (!link || libbpf_get_error(link)) {
        link = NULL;
        goto done;
    }
    signal(SIGINT, stop_on_signal);
    signal(SIGTERM, stop_on_signal);
    fprintf(stderr, "XDP Qwen3 ready on %s UDP/49002; Ctrl-C detaches\n", argv[2]);
    while (!stopping)
        pause();
    rc = 0;
done:
    if (rc)
        fprintf(stderr, "XDP model loader failed: %s\n", strerror(errno));
    bpf_link__destroy(link);
    if (work != MAP_FAILED)
        munmap(work, mapping_len);
    qwen3_arena_bf16_bpf__destroy(skel);
    safetensors_close(&file);
    return rc;
}
