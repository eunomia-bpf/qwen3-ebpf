#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "safetensors.h"

int safetensors_read_bf16(const char *path, const char *tensor,
                          uint64_t first, size_t count, float *out)
{
    uint8_t length_bytes[8], *raw = NULL;
    uint64_t header_length = 0, start, end;
    char key[256], *header = NULL, *field, *field_end, *offsets, *cursor;
    FILE *file = NULL;
    size_t i;
    int rc = -1;

    if (!path || !tensor || !out || !count ||
        snprintf(key, sizeof(key), "\"%s\"", tensor) >= (int)sizeof(key))
        return -1;
    file = fopen(path, "rb");
    if (!file || fread(length_bytes, 1, sizeof(length_bytes), file) != sizeof(length_bytes))
        goto done;
    for (i = 0; i < 8; i++)
        header_length |= (uint64_t)length_bytes[i] << (8 * i);
    if (!header_length || header_length > 1024 * 1024)
        goto done;
    header = calloc((size_t)header_length + 1, 1);
    if (!header || fread(header, 1, (size_t)header_length, file) != header_length)
        goto done;
    field = strstr(header, key);
    if (!field || !(field_end = strchr(field, '}')) ||
        !(cursor = strstr(field, "\"BF16\"")) || cursor > field_end ||
        !(offsets = strstr(field, "\"data_offsets\"")) || offsets > field_end ||
        !(cursor = strchr(offsets, '[')) || cursor > field_end)
        goto done;
    start = strtoull(cursor + 1, &cursor, 10);
    if (!(cursor = strchr(cursor, ',')) || cursor > field_end)
        goto done;
    end = strtoull(cursor + 1, NULL, 10);
    if (end < start || first > (end - start) / 2 ||
        count > ((end - start) / 2 - first) || count > SIZE_MAX / 2)
        goto done;
    raw = malloc(count * 2);
    if (!raw || fseeko(file, (off_t)(8 + header_length + start + first * 2), SEEK_SET) ||
        fread(raw, 2, count, file) != count)
        goto done;
    for (i = 0; i < count; i++) {
        uint32_t bits = ((uint32_t)raw[2 * i] |
                         ((uint32_t)raw[2 * i + 1] << 8)) << 16;
        memcpy(&out[i], &bits, sizeof(bits));
    }
    rc = 0;
done:
    free(raw);
    free(header);
    if (file)
        fclose(file);
    return rc;
}
