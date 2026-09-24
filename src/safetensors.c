#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "safetensors.h"

int safetensors_open(struct safetensors_file *file, const char *path)
{
    uint8_t length_bytes[8];
    size_t i;

    if (!file || !path)
        return -1;
    memset(file, 0, sizeof(*file));
    file->stream = fopen(path, "rb");
    if (!file->stream ||
        fread(length_bytes, 1, sizeof(length_bytes), file->stream) != sizeof(length_bytes))
        goto fail;
    for (i = 0; i < 8; i++)
        file->header_length |= (uint64_t)length_bytes[i] << (8 * i);
    if (!file->header_length || file->header_length > 1024 * 1024)
        goto fail;
    file->header = calloc((size_t)file->header_length + 1, 1);
    if (!file->header ||
        fread(file->header, 1, (size_t)file->header_length, file->stream) !=
            file->header_length)
        goto fail;
    return 0;
fail:
    safetensors_close(file);
    return -1;
}

void safetensors_close(struct safetensors_file *file)
{
    if (!file)
        return;
    if (file->stream)
        fclose(file->stream);
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
    uint8_t *raw;
    size_t i;

    if (!file || !file->stream || !out || !count || count > SIZE_MAX / 2)
        return -1;
    raw = malloc(count * 2);
    if (!raw)
        return -1;
    if (fseeko(file->stream,
               (off_t)(8 + file->header_length + first_byte + first * 2),
               SEEK_SET) || fread(raw, 2, count, file->stream) != count) {
        free(raw);
        return -1;
    }
    for (i = 0; i < count; i++) {
        uint32_t bits = ((uint32_t)raw[2 * i] |
                         ((uint32_t)raw[2 * i + 1] << 8)) << 16;
        memcpy(&out[i], &bits, sizeof(bits));
    }
    free(raw);
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
