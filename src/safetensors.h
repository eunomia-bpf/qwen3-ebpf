#ifndef QWEN3_SAFETENSORS_H
#define QWEN3_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

struct safetensors_file {
    const uint8_t *mapping;
    size_t mapping_size;
    char *header;
    uint64_t header_length;
};

int safetensors_open(struct safetensors_file *file, const char *path);
void safetensors_close(struct safetensors_file *file);
int safetensors_find_bf16(const struct safetensors_file *file,
                          const char *tensor, uint64_t *first_byte,
                          uint64_t *elements);
int safetensors_read_bf16_at(struct safetensors_file *file,
                             uint64_t first_byte, uint64_t first,
                             size_t count, float *out);

/* Read a contiguous BF16 slice from a named tensor into host floats. */
int safetensors_read_bf16(const char *path, const char *tensor,
                          uint64_t first, size_t count, float *out);

#endif
