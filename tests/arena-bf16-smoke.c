#include <limits.h>
#include <math.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_arena_bf16.h"
#include "qwen3_arena_bf16.skel.h"
#include "safetensors.h"

static int run_kernel(struct qwen3_arena_bf16_bpf *skel,
                      struct qwen3_arena_bf16_work *work, int rows)
{
    const __u8 packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts), .data_in = packet, .data_size_in = sizeof(packet),
        .repeat = 1,
    };

    work->rows = (__u32)rows;
    work->completed = 0;
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_bf16_rows), &opts)) {
        perror("arena BF16 test-run");
        return -1;
    }
    return work->completed == (__u32)rows ? 0 : -1;
}

static void populate_lut(struct qwen3_arena_bf16_bpf *skel)
{
    uint32_t bits;

    for (bits = 0; bits <= UINT16_MAX; bits++) {
        uint32_t float_bits = bits << 16;
        float value;
        double scaled;

        memcpy(&value, &float_bits, sizeof(value));
        scaled = (double)value * 16777216.0;
        skel->arena->q24_by_bf16[bits] =
            !isfinite(scaled) || scaled > INT32_MAX || scaled < INT32_MIN
            ? 0 : (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
    }
}

static int run_synthetic(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work)
{
    static const __u16 samples[] = {0x3d00, 0xbd00, 0x3c00, 0};
    int row, col;

    memset(work, 0, sizeof(*work));
    work->cols = QWEN3_ARENA_BF16_COLS;
    for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++)
        for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++)
            skel->arena->weight_bf16[row][col] =
                samples[(row + col) % (sizeof(samples) / sizeof(samples[0]))];
    if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
        return -1;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        __s64 expected = 0;
        for (col = 0; col < QWEN3_ARENA_BF16_COLS; col++) {
            __u16 bits = skel->arena->weight_bf16[row][col];
            expected += (__s64)work->input_q16[col] *
                        skel->arena->q24_by_bf16[bits];
        }
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "arena BF16 row %d mismatch\n", row);
            return -1;
        }
    }
    return 0;
}

static int run_resident_synthetic(struct qwen3_arena_bf16_bpf *skel,
                                   struct qwen3_arena_bf16_work *work)
{
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    static const __u16 samples[] = {0x3d00, 0xbd00, 0x3c00, 0};
    int row, col;

    memset(work, 0, sizeof(*work));
    work->model_elements = 256;
    if (bpf_prog_test_run_opts(
            bpf_program__fd(skel->progs.qwen3_arena_allocate_model),
            &opts) || opts.retval || !skel->bss->model_bf16) {
        fprintf(stderr, "arena BF16 resident allocation failed\n");
        return -1;
    }
    work->cols = 128;
    work->resident_weights = 1;
    for (col = 0; col < 128; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < 2; row++)
        for (col = 0; col < 128; col++)
            skel->bss->model_bf16[row * 128 + col] =
                samples[(row + col) % 4];
    if (run_kernel(skel, work, 2))
        return -1;
    for (row = 0; row < 2; row++) {
        __s64 expected = 0;
        for (col = 0; col < 128; col++) {
            __u16 bits = skel->bss->model_bf16[row * 128 + col];
            expected += (__s64)work->input_q16[col] *
                        skel->arena->q24_by_bf16[bits];
        }
        if (work->output_q16[row] != expected >> 24) {
            fprintf(stderr, "arena BF16 resident row %d mismatch\n", row);
            return -1;
        }
    }
    return 0;
}

static int run_xdp_event(struct qwen3_arena_bf16_bpf *skel,
                         struct qwen3_arena_bf16_work *work)
{
    const __u32 key = 0;
    const char payload[] = "Q3BP";
    struct bpf_test_run_opts opts = {.sz = sizeof(opts)};
    struct qwen3_event_state job;
    struct bpf_link *link = NULL;
    struct sockaddr_in dst = {
        .sin_family = AF_INET,
        .sin_port = htons(49002),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    struct timespec begin, now;
    __s64 expected[2], elapsed_ns;
    unsigned int ifindex = if_nametoindex("lo");
    int fd = -1, rc = -1;

    if (!ifindex) {
        perror("loopback ifindex");
        return -1;
    }
    expected[0] = work->output_q16[0];
    expected[1] = work->output_q16[1];
    work->output_q16[0] = INT64_MIN;
    work->output_q16[1] = INT64_MIN;
    work->completed = 0;
    if (bpf_prog_test_run_opts(bpf_program__fd(skel->progs.qwen3_event_init),
                               &opts) || opts.retval) {
        fprintf(stderr, "BPF workqueue initialization failed\n");
        return -1;
    }
    link = bpf_program__attach_xdp(skel->progs.qwen3_event_xdp, ifindex);
    if (!link || libbpf_get_error(link)) {
        fprintf(stderr, "XDP attach to container loopback failed: %ld\n",
                link ? libbpf_get_error(link) : -1L);
        link = NULL;
        goto done;
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0 || clock_gettime(CLOCK_MONOTONIC, &begin)) {
        perror("XDP event socket/clock");
        goto done;
    }
    if (sendto(fd, payload, sizeof(payload) - 1, 0,
               (struct sockaddr *)&dst, sizeof(dst)) != sizeof(payload) - 1) {
        perror("XDP event sendto");
        goto done;
    }
    do {
        if (bpf_map_lookup_elem(bpf_map__fd(skel->maps.event), &key, &job)) {
            perror("XDP event lookup");
            goto done;
        }
        if (job.status == QWEN3_EVENT_ERROR) {
            fprintf(stderr, "XDP event callback/start failed\n");
            goto done;
        }
        if (job.status == QWEN3_EVENT_DONE)
            break;
        usleep(1000);
        if (clock_gettime(CLOCK_MONOTONIC, &now))
            goto done;
        elapsed_ns = (__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
                     now.tv_nsec - begin.tv_nsec;
    } while (elapsed_ns < 5000000000LL);
    if (job.status != QWEN3_EVENT_DONE || job.requests != 1 ||
        work->completed != 2 || work->output_q16[0] != expected[0] ||
        work->output_q16[1] != expected[1]) {
        fprintf(stderr, "XDP event result mismatch: status=%llu requests=%u completed=%u\n",
                (unsigned long long)job.status, job.requests, work->completed);
        goto done;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed_ns = (__s64)(now.tv_sec - begin.tv_sec) * 1000000000 +
                 now.tv_nsec - begin.tv_nsec;
    printf("XDP live packet -> BPF workqueue -> resident matvec: %.3f ms\n",
           (double)elapsed_ns / 1000000.0);
    rc = 0;
done:
    if (fd >= 0)
        close(fd);
    bpf_link__destroy(link);
    return rc;
}

static int run_model_rows(struct qwen3_arena_bf16_bpf *skel,
                          struct qwen3_arena_bf16_work *work,
                          const char *path)
{
    struct safetensors_file file;
    uint64_t first_byte, elements;
    int32_t q24[1024];
    const uint8_t *raw;
    __s64 expected[QWEN3_ARENA_BF16_ROWS] = {0};
    int row, col, rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (safetensors_find_bf16(&file,
            "model.layers.0.self_attn.q_proj.weight", &first_byte,
            &elements) || elements != 2048ULL * 1024)
        goto done;
    raw = file.mapping + 8 + file.header_length + first_byte;
    memset(work, 0, sizeof(*work));
    work->cols = 1024;
    for (col = 0; col < 1024; col++)
        work->input_q16[col] = ((col % 11) - 5) * 8192;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        if (safetensors_read_bf16_q24_at(&file, first_byte,
                (uint64_t)row * 1024, 1024, q24))
            goto done;
        for (col = 0; col < 1024; col++) {
            __u16 bits = (__u16)raw[2 * (row * 1024 + col)] |
                         ((__u16)raw[2 * (row * 1024 + col) + 1] << 8);
            skel->arena->weight_bf16[row][col] = bits;
            if (skel->arena->q24_by_bf16[bits] != q24[col]) {
                fprintf(stderr, "arena BF16 lookup differs at row %d col %d\n",
                        row, col);
                goto done;
            }
            expected[row] += (__s64)work->input_q16[col] * q24[col];
        }
    }
    if (run_kernel(skel, work, QWEN3_ARENA_BF16_ROWS))
        goto done;
    for (row = 0; row < QWEN3_ARENA_BF16_ROWS; row++) {
        if (work->output_q16[row] != expected[row] >> 24) {
            fprintf(stderr, "arena BF16 real row %d differs from Q24 input\n",
                    row);
            goto done;
        }
    }
    printf("arena BF16 model rows: %d exact Q24 outputs; first Q16=%lld\n",
           QWEN3_ARENA_BF16_ROWS, (long long)work->output_q16[0]);
    rc = 0;
done:
    safetensors_close(&file);
    return rc;
}

int main(int argc, char **argv)
{
    struct qwen3_arena_bf16_bpf *skel;
    struct qwen3_arena_bf16_work *work;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    size_t length = ((sizeof(*work) + page - 1) / page) * page;
    int rc = 1;

    if (argc > 2)
        return 2;
    skel = qwen3_arena_bf16_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "arena BF16 BPF object load failed\n");
        return 1;
    }
    work = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED,
                bpf_map__fd(skel->maps.work), 0);
    if (work == MAP_FAILED) {
        perror("arena BF16 work mmap");
        goto done;
    }
    populate_lut(skel);
    if (!run_synthetic(skel, work) &&
        !run_resident_synthetic(skel, work) &&
        (argc == 1 ||
         (strcmp(argv[1], "--xdp-loopback") == 0 ?
          !run_xdp_event(skel, work) : !run_model_rows(skel, work, argv[1])))) {
        puts("arena BF16 smoke passed");
        rc = 0;
    }
    munmap(work, length);
done:
    qwen3_arena_bf16_bpf__destroy(skel);
    return rc;
}
