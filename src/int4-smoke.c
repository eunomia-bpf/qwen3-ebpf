#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_int4.h"
#include "safetensors.h"

static void put_weight(struct qwen3_int4_work *work, int row, int col,
                       int value)
{
    uint8_t *packed = &work->weight_packed[row][col / 2];
    uint8_t nibble = (uint8_t)value & 15;

    if (col & 1)
        *packed = (*packed & 15) | (nibble << 4);
    else
        *packed = (*packed & 240) | nibble;
}

static int run_case(struct qwen3_int4_work *work, int program_fd,
                    int rows, int cols)
{
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    int row, col;

    memset(work, 0, sizeof(*work));
    work->rows = rows;
    work->cols = cols;
    for (col = 0; col < cols; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols / QWEN3_INT4_GROUP; col++)
            work->scale_q24[row][col] = 32768;
        for (col = 0; col < cols; col++)
            put_weight(work, row, col, (row + col) % 15 - 7);
    }
    if (bpf_prog_test_run_opts(program_fd, &opts)) {
        perror("INT4 program test-run");
        return -1;
    }
    if (work->completed != (uint32_t)rows)
        return -1;
    for (row = 0; row < rows; row++) {
        int64_t expected = 0;
        for (col = 0; col < cols; col++)
            expected += (int64_t)work->input_q16[col] *
                        ((row + col) % 15 - 7) * 32768;
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "INT4 row %d got %lld, expected %lld\n",
                    row, (long long)work->output_q16[row],
                    (long long)(expected >> 24));
            return -1;
        }
    }
    return 0;
}

static int run_model_row(struct qwen3_int4_work *work, int program_fd,
                         const char *path)
{
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    struct safetensors_file file;
    uint64_t first_byte, elements;
    float weights[1024];
    double original = 0, quantized = 0;
    int group, col, rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (safetensors_find_bf16(&file,
            "model.layers.0.self_attn.q_proj.weight",
            &first_byte, &elements) || elements != 2048ULL * 1024 ||
        safetensors_read_bf16_at(&file, first_byte, 0, 1024, weights))
        goto done;
    memset(work, 0, sizeof(*work));
    work->rows = 1;
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
        work->scale_q24[0][group] = (int32_t)llround(scale * 16777216.0);
        for (col = group * QWEN3_INT4_GROUP;
             col < (group + 1) * QWEN3_INT4_GROUP; col++) {
            int value = scale ? (int)lround(weights[col] / scale) : 0;
            if (value < -7) value = -7;
            if (value > 7) value = 7;
            put_weight(work, 0, col, value);
            work->input_q16[col] = ((col % 11) - 5) * 8192;
            original += (double)work->input_q16[col] / 65536.0 *
                        weights[col];
            quantized += (double)work->input_q16[col] / 65536.0 *
                         value * work->scale_q24[0][group] / 16777216.0;
        }
    }
    if (bpf_prog_test_run_opts(program_fd, &opts)) {
        perror("INT4 model row test-run");
        goto done;
    }
    if (work->completed != 1 ||
        fabs((double)work->output_q16[0] / 65536.0 - quantized) >
            1.0 / 65536.0) {
        fprintf(stderr, "INT4 model row disagrees with C fixed-point dot\n");
        goto done;
    }
    printf("INT4 model row: BF16=%.9f INT4=%.9f BPF=%.9f\n",
           original, quantized, (double)work->output_q16[0] / 65536.0);
    rc = 0;
done:
    safetensors_close(&file);
    return rc;
}

int main(int argc, char **argv)
{
    struct bpf_object *object;
    struct bpf_program *program;
    struct bpf_map *map;
    struct qwen3_int4_work *work;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(*work) + page - 1) / page) * page;
    int rc;

    if (argc < 2 || argc > 3)
        return 2;
    object = bpf_object__open_file(argv[1], NULL);
    if (!object || libbpf_get_error(object))
        return 1;
    if (bpf_object__load(object)) {
        bpf_object__close(object);
        return 1;
    }
    program = bpf_object__find_program_by_name(object, "qwen3_int4_rows");
    map = bpf_object__find_map_by_name(object, "int4");
    if (!program || !map) {
        bpf_object__close(object);
        return 1;
    }
    work = mmap(NULL, length, PROT_READ | PROT_WRITE,
                MAP_SHARED, bpf_map__fd(map), 0);
    if (work == MAP_FAILED) {
        perror("INT4 map mmap");
        bpf_object__close(object);
        return 1;
    }
    rc = run_case(work, bpf_program__fd(program), QWEN3_INT4_ROWS, 3072) ||
         run_case(work, bpf_program__fd(program), 1, 128) ||
         (argc == 3 && run_model_row(work, bpf_program__fd(program), argv[2]));
    munmap(work, length);
    bpf_object__close(object);
    if (rc) {
        fprintf(stderr, "INT4 matvec smoke failed\n");
        return 1;
    }
    puts("INT4 matvec smoke passed");
    return 0;
}
