#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "qwen3_tile.h"
#include "safetensors.h"

#define V_ROWS 1024

static double seconds_since(const struct timespec *start)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec - start->tv_sec +
           (now.tv_nsec - start->tv_nsec) / 1e9;
}

int main(int argc, char **argv)
{
    struct safetensors_file model;
    struct bpf_object *obj = NULL;
    struct bpf_program *prog;
    struct bpf_map *map;
    struct qwen3_tile work;
    const uint8_t packet[64] = {0};
    struct bpf_test_run_opts opts = {
        .sz = sizeof(opts),
        .data_in = packet,
        .data_size_in = sizeof(packet),
        .repeat = 1,
    };
    float weights[QWEN3_HIDDEN_SIZE];
    uint64_t tensor_first, tensor_elements;
    const uint32_t key = 0;
    struct timespec start;
    double max_error = 0;
    int prog_fd, map_fd, row, tile_index, i, rc = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: %s build/qwen3_matvec.bpf.o model.safetensors\n", argv[0]);
        return 2;
    }
    if (safetensors_open(&model, argv[2])) {
        fprintf(stderr, "opening model failed\n");
        return 1;
    }
    if (safetensors_find_bf16(&model,
            "model.layers.0.self_attn.v_proj.weight",
            &tensor_first, &tensor_elements) ||
        tensor_elements != (uint64_t)V_ROWS * QWEN3_HIDDEN_SIZE) {
        fprintf(stderr, "Qwen3 layer-0 V projection has unexpected shape\n");
        goto done;
    }
    obj = bpf_object__open_file(argv[1], NULL);
    if (!obj || libbpf_get_error(obj) || bpf_object__load(obj)) {
        fprintf(stderr, "loading BPF matvec failed\n");
        goto done;
    }
    prog = bpf_object__find_program_by_name(obj, "qwen3_matvec_tile");
    map = bpf_object__find_map_by_name(obj, "tile");
    if (!prog || !map) {
        fprintf(stderr, "BPF matvec program or map missing\n");
        goto done;
    }
    prog_fd = bpf_program__fd(prog);
    map_fd = bpf_map__fd(map);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (row = 0; row < V_ROWS; row++) {
        double reference = 0;
        if (safetensors_read_bf16_at(&model, tensor_first,
                (uint64_t)row * QWEN3_HIDDEN_SIZE,
                QWEN3_HIDDEN_SIZE, weights)) {
            fprintf(stderr, "reading V projection row %d failed\n", row);
            goto done;
        }
        memset(&work, 0, sizeof(work));
        work.fixed_point_mode = 2;
        work.total_tiles = QWEN3_HIDDEN_TILES;
        for (tile_index = 0; tile_index < QWEN3_HIDDEN_TILES; tile_index++) {
            for (i = 0; i < QWEN3_TILE_WIDTH; i++) {
                int index = tile_index * QWEN3_TILE_WIDTH + i;
                int x = (index * 7) % 21 - 10;
                float weight = weights[index];
                work.activation_q16[i] = x * (1 << 16);
                work.weight_q24[i] = (int32_t)(weight * (1 << 24) +
                    (weight >= 0 ? 0.5f : -0.5f));
                reference += x * weight;
            }
            if (bpf_map_update_elem(map_fd, &key, &work, BPF_ANY) ||
                bpf_prog_test_run_opts(prog_fd, &opts) ||
                bpf_map_lookup_elem(map_fd, &key, &work)) {
                perror("BPF V projection");
                goto done;
            }
        }
        if (work.completed_tiles != QWEN3_HIDDEN_TILES) {
            fprintf(stderr, "BPF V projection row %d incomplete\n", row);
            goto done;
        }
        {
            double error = fabs(work.output_q16 / 65536.0 - reference);
            if (error > max_error)
                max_error = error;
        }
    }
    printf("kernel layer-0 V projection: %u MACs, %d rows, %.3f s, max_abs_error=%.9g\n",
           (unsigned)(V_ROWS * QWEN3_HIDDEN_SIZE), V_ROWS,
           seconds_since(&start), max_error);
    if (max_error > 0.005) {
        fprintf(stderr, "kernel V projection exceeds one-matrix error budget\n");
        goto done;
    }
    rc = 0;
done:
    bpf_object__close(obj);
    safetensors_close(&model);
    return rc;
}
