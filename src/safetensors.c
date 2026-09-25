#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "safetensors.h"

int safetensors_open(struct safetensors_file *file, const char *path)
{
    struct stat st;
    int fd;
    size_t i;

    if (!file || !path)
        return -1;
    memset(file, 0, sizeof(*file));
    fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    if (fstat(fd, &st) || st.st_size < 8 ||
        (uint64_t)st.st_size > SIZE_MAX) {
        close(fd);
        return -1;
    }
    file->mapping_size = (size_t)st.st_size;
    file->mapping = mmap(NULL, file->mapping_size, PROT_READ, MAP_PRIVATE,
                         fd, 0);
    close(fd);
    if (file->mapping == MAP_FAILED) {
        file->mapping = NULL;
        goto fail;
    }
    for (i = 0; i < 8; i++)
        file->header_length |= (uint64_t)file->mapping[i] << (8 * i);
    if (!file->header_length || file->header_length > 1024 * 1024 ||
        file->header_length > file->mapping_size - 8)
        goto fail;
    file->header = calloc((size_t)file->header_length + 1, 1);
    if (!file->header)
        goto fail;
    memcpy(file->header, file->mapping + 8, (size_t)file->header_length);
    return 0;
fail:
    safetensors_close(file);
    return -1;
}

void safetensors_close(struct safetensors_file *file)
{
    if (!file)
        return;
    if (file->mapping)
        munmap((void *)file->mapping, file->mapping_size);
    free(file->header);
    memset(file, 0, sizeof(*file));
}

int safetensors_find_bf16(const struct safetensors_file *file,
                          const char *tensor, uint64_t *first_byte,
                          uint64_t *elements)
{
    char key[256], *field, *field_end, *offsets, *cursor;
    uint64_t start, end;

    if (!file || !file->header || !tensor || !first_byte || !elements ||
        snprintf(key, sizeof(key), "\"%s\"", tensor) >= (int)sizeof(key))
        return -1;
    field = strstr(file->header, key);
    if (!field || !(field_end = strchr(field, '}')) ||
        !(cursor = strstr(field, "\"BF16\"")) || cursor > field_end ||
        !(offsets = strstr(field, "\"data_offsets\"")) || offsets > field_end ||
        !(cursor = strchr(offsets, '[')) || cursor > field_end)
        return -1;
    start = strtoull(cursor + 1, &cursor, 10);
    if (!(cursor = strchr(cursor, ',')) || cursor > field_end)
        return -1;
    end = strtoull(cursor + 1, NULL, 10);
    if (end < start || (end - start) % 2)
        return -1;
    *first_byte = start;
    *elements = (end - start) / 2;
    return 0;
}

int safetensors_read_bf16_at(struct safetensors_file *file,
                             uint64_t first_byte, uint64_t first,
                             size_t count, float *out)
{
    const uint8_t *raw;
    size_t data_offset, payload_size;
    size_t i;

    if (!file || !file->mapping || !out || !count || count > SIZE_MAX / 2)
        return -1;
    data_offset = 8 + (size_t)file->header_length;
    payload_size = file->mapping_size - data_offset;
    if (first_byte > payload_size ||
        first > (payload_size - (size_t)first_byte) / 2 ||
        count > (payload_size - (size_t)first_byte) / 2 - (size_t)first)
        return -1;
    raw = file->mapping + data_offset + (size_t)first_byte + (size_t)first * 2;
    for (i = 0; i < count; i++) {
        uint32_t bits = ((uint32_t)raw[2 * i] |
                         ((uint32_t)raw[2 * i + 1] << 8)) << 16;
        memcpy(&out[i], &bits, sizeof(bits));
    }
    return 0;
}

int safetensors_read_bf16_q24_at(struct safetensors_file *file,
                                 uint64_t first_byte, uint64_t first,
                                 size_t count, int32_t *out)
{
    static int64_t q24_by_bf16[UINT16_MAX + 1];
    static atomic_int lookup_state = ATOMIC_VAR_INIT(0);
    const uint8_t *raw;
    size_t data_offset, payload_size, i;

    if (!file || !file->mapping || !out || !count || count > SIZE_MAX / 2)
        return -1;
    data_offset = 8 + (size_t)file->header_length;
    payload_size = file->mapping_size - data_offset;
    if (first_byte > payload_size ||
        first > (payload_size - (size_t)first_byte) / 2 ||
        count > (payload_size - (size_t)first_byte) / 2 - (size_t)first)
        return -1;
    raw = file->mapping + data_offset + (size_t)first_byte + (size_t)first * 2;
    if (atomic_load_explicit(&lookup_state, memory_order_acquire) != 2) {
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(&lookup_state, &expected,
                1, memory_order_acq_rel, memory_order_acquire)) {
            uint32_t bits;
            for (bits = 0; bits <= UINT16_MAX; bits++) {
                uint32_t float_bits = bits << 16;
                float value;
                double scaled;
                memcpy(&value, &float_bits, sizeof(value));
                scaled = (double)value * 16777216.0;
                q24_by_bf16[bits] = !isfinite(scaled) ||
                    scaled > INT32_MAX || scaled < INT32_MIN
                    ? INT64_MIN
                    : (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
            }
            atomic_store_explicit(&lookup_state, 2, memory_order_release);
        } else {
            while (atomic_load_explicit(&lookup_state, memory_order_acquire) != 2)
                ;
        }
    }
    for (i = 0; i < count; i++) {
        uint16_t bits = (uint16_t)raw[2 * i] |
                        (uint16_t)raw[2 * i + 1] << 8;
        int64_t value = q24_by_bf16[bits];
        if (value == INT64_MIN)
            return -1;
        out[i] = (int32_t)value;
    }
    return 0;
}

int safetensors_read_bf16(const char *path, const char *tensor,
                          uint64_t first, size_t count, float *out)
{
    struct safetensors_file file;
    uint64_t first_byte, elements;
    int rc = -1;

    if (safetensors_open(&file, path))
        return -1;
    if (!safetensors_find_bf16(&file, tensor, &first_byte, &elements) &&
        first <= elements && count <= elements - first)
        rc = safetensors_read_bf16_at(&file, first_byte, first, count, out);
    safetensors_close(&file);
    return rc;
}
