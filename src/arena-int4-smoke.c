#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_int4.h"
#include "safetensors.h"
#include "qwen3_arena_int4.skel.h"

static void put_weight(__u8 *packed, int col, int value)
{
    __u8 nibble = (__u8)value & 15;

    if (col & 1)
        packed[col / 2] = (packed[col / 2] & 15) | (nibble << 4);
    else
        packed[col / 2] = (packed[col / 2] & 240) | nibble;
}

static int run_kernel(struct qwen3_arena_int4_bpf *skel,
                      struct qwen3_arena_int4_work *work, int rows)
{
    const __u8 packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };

    work->rows = (__u32)rows;
    work->completed = 0;
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_int4_rows), &opts)) {
        perror("arena INT4 test-run");
        return -1;
    }
    return work->completed == (__u32)rows ? 0 : -1;
}

static int run_synthetic(struct qwen3_arena_int4_bpf *skel,
                         struct qwen3_arena_int4_work *work)
{
    int row, col;

    memset(work, 0, sizeof(*work));
    work->cols = QWEN3_INT4_COLS;
    for (col = 0; col < QWEN3_INT4_COLS; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_INT4_ROWS; row++) {
        for (col = 0; col < QWEN3_INT4_COLS / QWEN3_INT4_GROUP; col++)
            skel->arena->scale_q24[row][col] = 32768;
        for (col = 0; col < QWEN3_INT4_COLS; col++)
            put_weight(skel->arena->weight_packed[row], col,
                       (row + col) % 15 - 7);
    }
    if (run_kernel(skel, work, QWEN3_INT4_ROWS))
        return -1;
    for (row = 0; row < QWEN3_INT4_ROWS; row++) {
        __s64 expected = 0;
        for (col = 0; col < QWEN3_INT4_COLS; col++)
            expected += (__s64)work->input_q16[col] *
                        ((row + col) % 15 - 7) * 32768;
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "arena INT4 row %d mismatch\n", row);
            return -1;
        }
    }
    return 0;
}

static int run_model_row(struct qwen3_arena_int4_bpf *skel,
                         struct qwen3_arena_int4_work *work,
                         const char *path)
{
    struct safetensors_file file;
    uint64_t first_byte, elements;
    float weights[1024];
    double original = 0, quantized = 0;
    int group, col, rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (safetensors_find_bf16(&file,
            "model.layers.0.self_attn.q_proj.weight", &first_byte,
            &elements) || elements != 2048ULL * 1024 ||
        safetensors_read_bf16_at(&file, first_byte, 0, 1024, weights))
        goto done;
    memset(work, 0, sizeof(*work));
    memset(skel->arena->weight_packed[0], 0,
           sizeof(skel->arena->weight_packed[0]));
    work->cols = 1024;
    for (group = 0; group < 1024 / QWEN3_INT4_GROUP; group++) {
        float max_abs = 0;
        double scale;

        for (col = group * QWEN3_INT4_GROUP;
             col < (group + 1) * QWEN3_INT4_GROUP; col++) {
            float absolute = fabsf(weights[col]);
            if (absolute > max_abs)
                max_abs = absolute;
        }
        scale = max_abs / 7.0;
        if (scale * 16777216.0 > INT32_MAX)
            goto done;
        skel->arena->scale_q24[0][group] =
            (__s32)llround(scale * 16777216.0);
        for (col = group * QWEN3_INT4_GROUP;
             col < (group + 1) * QWEN3_INT4_GROUP; col++) {
            int value = scale ? (int)lround(weights[col] / scale) : 0;
            if (value < -7) value = -7;
            if (value > 7) value = 7;
            put_weight(skel->arena->weight_packed[0], col, value);
            work->input_q16[col] = ((col % 11) - 5) * 8192;
            original += (double)work->input_q16[col] / 65536.0 *
                        weights[col];
            quantized += (double)work->input_q16[col] / 65536.0 *
                         value * skel->arena->scale_q24[0][group] /
                         16777216.0;
        }
    }
    if (run_kernel(skel, work, 1) ||
        fabs((double)work->output_q16[0] / 65536.0 - quantized) >
            1.0 / 65536.0) {
        fprintf(stderr, "arena INT4 real row disagrees with C dot\n");
        goto done;
    }
    printf("arena INT4 model row: BF16=%.9f INT4=%.9f BPF=%.9f\n",
           original, quantized, (double)work->output_q16[0] / 65536.0);
    rc = 0;
done:
    safetensors_close(&file);
    return rc;
}

int main(int argc, char **argv)
{
    struct qwen3_arena_int4_bpf *skel;
    struct qwen3_arena_int4_work *work;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(*work) + page - 1) / page) * page;
    int rc = 1;

    if (argc > 2)
        return 2;
    skel = qwen3_arena_int4_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "arena INT4 BPF object load failed\n");
        return 1;
    }
    work = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED,
                bpf_map__fd(skel->maps.work), 0);
    if (work == MAP_FAILED) {
        perror("arena INT4 work mmap");
        goto done;
    }
    if (!run_synthetic(skel, work) &&
        (argc == 1 || !run_model_row(skel, work, argv[1]))) {
        puts("arena INT4 smoke passed");
        rc = 0;
    }
    munmap(work, length);
done:
    qwen3_arena_int4_bpf__destroy(skel);
    return rc;
}
