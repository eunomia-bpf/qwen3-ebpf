#include <errno.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_tile.h"

/* Safetensors stores BF16 values after an 8-byte little-endian header size. */
static int load_qwen_q_proj_row(const char *path, int8_t *quantized, float *scale)
{
    static const char key[] = "\"model.layers.0.self_attn.q_proj.weight\"";
    uint8_t length_bytes[8], row[2 * QWEN3_HIDDEN_SIZE];
    uint64_t header_length = 0, start, end;
    char *header = NULL, *field, *offsets, *cursor;
    FILE *file = NULL;
    float values[QWEN3_HIDDEN_SIZE], max_abs = 0;
    int i, rc = -1;

    file = fopen(path, "rb");
    if (!file || fread(length_bytes, 1, sizeof(length_bytes), file) != sizeof(length_bytes))
        goto done;
    for (i = 0; i < 8; i++)
        header_length |= (uint64_t)length_bytes[i] << (8 * i);
    if (header_length == 0 || header_length > 1024 * 1024)
        goto done;
    header = calloc((size_t)header_length + 1, 1);
    if (!header || fread(header, 1, (size_t)header_length, file) != header_length)
        goto done;
    field = strstr(header, key);
    if (!field || !(offsets = strstr(field, "\"data_offsets\"")) ||
        !(cursor = strchr(offsets, '[')))
        goto done;
    start = strtoull(cursor + 1, &cursor, 10);
    if (!cursor || !(cursor = strchr(cursor, ',')))
        goto done;
    end = strtoull(cursor + 1, NULL, 10);
    if (end < start || end - start < sizeof(row) ||
        fseeko(file, (off_t)(8 + header_length + start), SEEK_SET) != 0 ||
        fread(row, 1, sizeof(row), file) != sizeof(row))
        goto done;
    for (i = 0; i < QWEN3_HIDDEN_SIZE; i++) {
        uint32_t bits = ((uint32_t)row[2 * i] |
                         ((uint32_t)row[2 * i + 1] << 8)) << 16;
        float magnitude;
        memcpy(&values[i], &bits, sizeof(bits));
        magnitude = values[i] < 0 ? -values[i] : values[i];
        if (magnitude > max_abs)
            max_abs = magnitude;
    }
    if (max_abs == 0 || max_abs > 1000)
        goto done;
    *scale = max_abs / 127.0f;
    for (i = 0; i < QWEN3_HIDDEN_SIZE; i++) {
        float value = values[i] / *scale;
        int rounded = (int)(value + (value >= 0 ? 0.5f : -0.5f));
        if (rounded > 127)
            rounded = 127;
        if (rounded < -127)
            rounded = -127;
        quantized[i] = (int8_t)rounded;
    }
    rc = 0;
done:
    free(header);
    if (file)
        fclose(file);
    return rc;
}

int main(int argc, char **argv)
{
    struct bpf_object *obj = NULL;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_tile work = {0};
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    int8_t model_row[QWEN3_HIDDEN_SIZE];
    float model_scale = 0;
    int64_t expected = 0;
    const uint32_t key = 0;
    int prog_fd, map_fd, tile_index, i, rc = 1;

    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s build/qwen3_matvec.bpf.o [model.safetensors]\n", argv[0]);
        return 2;
    }
    if (argc == 3 && load_qwen_q_proj_row(argv[2], model_row, &model_scale)) {
        fprintf(stderr, "could not load Qwen3 layer-0 Q projection row from %s\n", argv[2]);
        return 1;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj)) {
        fprintf(stderr, "opening BPF object failed\n");
        return 1;
    }
    if (bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF object failed: %s\n", strerror(errno));
        goto done;
    }
    prog = bpf_object__find_program_by_name(obj, "qwen3_matvec_tile");
    map = bpf_object__find_map_by_name(obj, "tile");
    if (!prog || !map) {
        fprintf(stderr, "BPF program or map missing\n");
        goto done;
    }
    prog_fd = bpf_program__fd(prog);
    map_fd = bpf_map__fd(map);
    for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
        for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
            work.activation[i] = (int8_t)((tile_index * 17 + i) % 21 - 10);
            work.weight[i] = argc == 3
                ? model_row[tile_index * QWEN3_TILE_WIDTH + i]
                : (int8_t)((tile_index * 7 + i * 3) % 19 - 9);
            expected += (int64_t)work.activation[i] * work.weight[i];
        }
        if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY)) {
            perror("bpf_map_update_elem");
            goto done;
        }
        if (bpf_prog_test_run_opts(prog_fd, &opts)) {
            perror("bpf_prog_test_run_opts");
            goto done;
        }
        if (bpf_map_lookup_elem(map_fd, &key, &work)) {
            perror("bpf_map_lookup_elem");
            goto done;
        }
    }
    if (work.accumulator != expected ||
        work.completed_tiles != QWEN3_HIDDEN_TILES) {
        fprintf(stderr, "kernel mismatch: got %lld/%u, expected %lld/%u\n",
                (long long)work.accumulator, work.completed_tiles,
                (long long)expected, QWEN3_HIDDEN_TILES);
        goto done;
    }
    printf("kernel Q8 matvec: %u MACs, sum=%lld, tiles=%u (matches C reference)%s\n",
           QWEN3_HIDDEN_SIZE, (long long)work.accumulator, work.completed_tiles,
           argc == 3 ? "; weights: actual Qwen3-0.6B layer-0 Q-projection row" : "");
    if (argc == 3)
        printf("weight Q8 scale: %.9g (BF16-to-Q8 preprocessing on host)\n", model_scale);
    rc = 0;
done:
    bpf_object__close(obj);
    return rc;
}
