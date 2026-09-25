#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "safetensors.h"

int main(void)
{
    uint8_t mapping[10] = {0};
    struct safetensors_file file = {
        .mapping = mapping, .mapping_size = sizeof(mapping)
    };
    unsigned bits;

    for (bits = 0; bits <= UINT16_MAX; bits++) {
        uint32_t float_bits = bits << 16;
        float value;
        double scaled;
        int32_t actual, expected;
        int invalid, rc;

        mapping[8] = (uint8_t)bits;
        mapping[9] = (uint8_t)(bits >> 8);
        memcpy(&value, &float_bits, sizeof(value));
        scaled = (double)value * 16777216.0;
        invalid = !isfinite(scaled) ||
                  scaled > INT32_MAX || scaled < INT32_MIN;
        rc = safetensors_read_bf16_q24_at(&file, 0, 0, 1, &actual);
        if (invalid) {
            if (!rc)
                goto fail;
            continue;
        }
        expected = (int32_t)(scaled + (scaled >= 0 ? 0.5 : -0.5));
        if (rc || actual != expected)
            goto fail;
    }
    puts("BF16 to Q24: all 65,536 bit patterns matched");
    return 0;
fail:
    fprintf(stderr, "BF16 to Q24 mismatch at 0x%04x\n", bits);
    return 1;
}
