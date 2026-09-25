#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_batch.h"

static int run_case(int map_fd, int program_fd, int rows, int cols)
{
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(struct qwen3_batch_work) + page - 1) / page) * page;
    struct bpf_map_info info = {0};
    uint32_t info_len = sizeof(info);
    void *mapping = mmap(NULL, length, PROT_READ | PROT_WRITE,
                         MAP_SHARED, map_fd, 0);
    struct qwen3_batch_work *work;
    int row, col, best_row = 0, rc = -1;
    int64_t best = INT32_MIN;

    if (mapping == MAP_FAILED) {
        int mmap_errno = errno;
        bpf_map_get_info_by_fd(map_fd, &info, &info_len);
        fprintf(stderr, "map value=%u entries=%u flags=%u page=%zu length=%zu errno=%d\n",
                info.value_size, info.max_entries, info.map_flags,
                page, length, mmap_errno);
        errno = mmap_errno;
        perror("batch map mmap");
        return -1;
    }
    work = mapping;
    memset(work, 0, sizeof(*work));
    work->rows = rows;
    work->cols = cols;
    work->track_argmax = 1;
    work->base_index = 256;
    work->best_q16 = INT32_MIN;
    for (col = 0; col < cols; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < rows; row++)
        for (col = 0; col < cols; col++)
            work->weight_q24[row][col] = ((row + col) % 13 - 6) * 32768;
    if (bpf_prog_test_run_opts(program_fd, &opts)) {
        perror("batch program test-run");
        goto done;
    }
    if (work->completed != (uint32_t)rows) {
        fprintf(stderr, "batch completed %u, expected %d\n",
                work->completed, rows);
        goto done;
    }
    for (row = 0; row < rows; row++) {
        int64_t expected = 0;
        for (col = 0; col < cols; col++)
            expected += (int64_t)work->input_q16[col] *
                        work->weight_q24[row][col];
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "batch row %d got %lld, expected %lld\n",
                    row, (long long)work->output_q16[row],
                    (long long)(expected >> 24));
            goto done;
        }
        if (work->output_q16[row] > best) {
            best = work->output_q16[row];
            best_row = row;
        }
    }
    if (work->best_q16 != best || work->best_index != (uint32_t)(256 + best_row)) {
        fprintf(stderr, "batch argmax mismatch\n");
        goto done;
    }
    work->completed = 0;
    work->base_index += (uint32_t)rows;
    if (bpf_prog_test_run_opts(program_fd, &opts) ||
        work->completed != (uint32_t)rows ||
        work->best_q16 != best || work->best_index != (uint32_t)(256 + best_row)) {
        fprintf(stderr, "batch argmax tie across batches failed\n");
        goto done;
    }
    rc = 0;
done:
    munmap(mapping, length);
    return rc;
}

int main(int argc, char **argv)
{
    struct bpf_object *object;
    struct bpf_program *program;
    struct bpf_map *map;
    int rc;

    if (argc != 2)
        return 2;
    object = bpf_object__open_file(argv[1], NULL);
    if (!object || libbpf_get_error(object))
        return 1;
    if (bpf_object__load(object)) {
        bpf_object__close(object);
        return 1;
    }
    program = bpf_object__find_program_by_name(object, "qwen3_batch_rows");
    map = bpf_object__find_map_by_name(object, "batch");
    rc = !program || !map ||
         run_case(bpf_map__fd(map), bpf_program__fd(program),
                  QWEN3_BATCH_ROWS, 3072) ||
         run_case(bpf_map__fd(map), bpf_program__fd(program), 1, 128);
    bpf_object__close(object);
    if (rc) {
        fprintf(stderr, "batch matvec smoke failed\n");
        return 1;
    }
    puts("batch matvec smoke passed");
    return 0;
}
