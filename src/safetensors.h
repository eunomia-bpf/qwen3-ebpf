#ifndef QWEN3_SAFETENSORS_H
#define QWEN3_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

/* Read a contiguous BF16 slice from a named tensor into host floats. */
int safetensors_read_bf16(const char *path, const char *tensor,
                          uint64_t first, size_t count, float *out);

#endif
