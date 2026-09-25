#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_int8.h"
#include "safetensors.h"

static int run_program(int fd)
{
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };

    return bpf_prog_test_run_opts(fd, &opts);
}

static int run_synthetic(struct qwen3_int8_work *work, int fd)
{
    int row, col;

    memset(work, 0, sizeof(*work));
    work->rows = QWEN3_INT8_ROWS;
    work->cols = QWEN3_INT8_COLS;
    for (col = 0; col < QWEN3_INT8_COLS; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_INT8_ROWS; row++) {
        for (col = 0; col < QWEN3_INT8_COLS / QWEN3_INT8_GROUP; col++)
            work->scale_q32[row][col] = 32768;
        for (col = 0; col < QWEN3_INT8_COLS; col++)
            work->weight_q8[row][col] = (row + col) % 255 - 127;
    }
    work->track_argmax = 1;
    work->best_q16 = INT32_MIN;
    if (run_program(fd) || work->completed != QWEN3_INT8_ROWS) {
        perror("INT8 synthetic test-run");
        return -1;
    }
    for (row = 0; row < QWEN3_INT8_ROWS; row++) {
        int64_t expected = 0;
        for (col = 0; col < QWEN3_INT8_COLS; col++)
            expected += (int64_t)work->input_q16[col] *
                        ((row + col) % 255 - 127) * 32768;
        if (work->output_q16[row] != expected >> 32) {
            fprintf(stderr, "INT8 synthetic row %d mismatch\n", row);
            return -1;
        }
    }
    if (work->output_q16[work->best_index] != work->best_q16)
        return -1;
    return 0;
}

static int run_model_row(struct qwen3_int8_work *work, int fd,
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
    work->rows = 1;
    work->cols = 1024;
    for (group = 0; group < 1024 / QWEN3_INT8_GROUP; group++) {
        float max_abs = 0;
        double scale;

        for (col = group * QWEN3_INT8_GROUP;
             col < (group + 1) * QWEN3_INT8_GROUP; col++) {
            float absolute = fabsf(weights[col]);
            if (absolute > max_abs)
                max_abs = absolute;
        }
        scale = max_abs / 127.0;
        if (scale * 4294967296.0 > INT64_MAX)
            goto done;
        work->scale_q32[0][group] =
            (int64_t)llround(scale * 4294967296.0);
        for (col = group * QWEN3_INT8_GROUP;
             col < (group + 1) * QWEN3_INT8_GROUP; col++) {
            int value = scale ? (int)lround(weights[col] / scale) : 0;
            if (value < -127) value = -127;
            if (value > 127) value = 127;
            work->weight_q8[0][col] = (int8_t)value;
            work->input_q16[col] = ((col % 11) - 5) * 8192;
            original += (double)work->input_q16[col] / 65536.0 *
                        weights[col];
            quantized += (double)work->input_q16[col] / 65536.0 *
                         value * work->scale_q32[0][group] / 4294967296.0;
        }
    }
    if (run_program(fd) || work->completed != 1 ||
        fabs((double)work->output_q16[0] / 65536.0 - quantized) >
            1.0 / 65536.0) {
        fprintf(stderr, "INT8 real row disagrees with C dot\n");
        goto done;
    }
    printf("INT8 model row: BF16=%.9f INT8=%.9f BPF=%.9f\n",
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
    struct qwen3_int8_work *work;
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
    program = bpf_object__find_program_by_name(object, "qwen3_int8_rows");
    map = bpf_object__find_map_by_name(object, "int8");
    if (!program || !map) {
        bpf_object__close(object);
        return 1;
    }
    work = mmap(NULL, length, PROT_READ | PROT_WRITE,
                MAP_SHARED, bpf_map__fd(map), 0);
    if (work == MAP_FAILED) {
        perror("INT8 map mmap");
        bpf_object__close(object);
        return 1;
    }
    rc = run_synthetic(work, bpf_program__fd(program)) ||
         (argc == 3 && run_model_row(work, bpf_program__fd(program), argv[2]));
    munmap(work, length);
    bpf_object__close(object);
    if (rc)
        return 1;
    puts("INT8 matvec smoke passed");
    return 0;
}
