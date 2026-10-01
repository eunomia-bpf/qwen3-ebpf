#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen3_tokenizer.h"

static int check(struct qwen3_tokenizer *tokenizer, const char *text,
                 const uint32_t *expected, size_t expected_count)
{
    uint32_t *actual = NULL;
    size_t count = 0, i, used = 0, capacity = strlen(text) + 1;
    unsigned char *reconstructed = malloc(capacity);
    int rc = -1;

    if (!reconstructed ||
        qwen3_tokenizer_encode(tokenizer, text, &actual, &count))
        goto done;
    if (count != expected_count) {
        fprintf(stderr, "token count mismatch for %s: %zu != %zu\n",
                text, count, expected_count);
        goto done;
    }
    for (i = 0; i < count; i++) {
        unsigned char *piece = NULL;
        size_t length;
        if (actual[i] != expected[i] ||
            qwen3_tokenizer_decode(tokenizer, actual[i], &piece, &length)) {
            fprintf(stderr, "token mismatch at %zu for %s: %u != %u\n",
                    i, text, actual[i], expected[i]);
            goto done;
        }
        if (used + length >= capacity) {
            free(piece);
            goto done;
        }
        memcpy(reconstructed + used, piece, length);
        used += length;
        free(piece);
    }
    reconstructed[used] = 0;
    if (strcmp((const char *)reconstructed, text) != 0) {
        fprintf(stderr, "roundtrip mismatch for %s\n", text);
        goto done;
    }
    printf("tokenizer: %s -> %zu IDs (official reference and roundtrip match)\n",
           text, count);
    rc = 0;
done:
    free(actual);
    free(reconstructed);
    return rc;
}

int main(int argc, char **argv)
{
    static const uint32_t english[] = {9707, 11, 1879, 0};
    static const uint32_t chinese[] = {108386, 3837, 99489, 6313};
    static const uint32_t chat[] = {151644, 872, 198, 13048, 151645};
    static const uint32_t spaces[] = {32, 220, 425};
    static const uint32_t emoji[] = {92731, 26525, 232, 323, 356, 22890};
    static const uint32_t mixed[] = {
        104811, 6364, 220, 16, 17, 18, 608, 384, 33, 19701
    };
    static const uint32_t lines[] = {271, 13048, 197, 18532};
    static const uint32_t long_context[] = {
        2082, 384, 33, 19701, 2025, 646, 22986, 3922, 27035, 1573, 279, 4622,
        7575, 5611, 11, 714, 264, 3460, 4128, 1614, 3880, 6172, 12557, 10709,
        11, 6529, 11, 323, 264, 6500, 315, 6788, 6894, 323, 2750, 13,
        362, 24052, 646, 6718, 279, 34447, 1119, 61115, 975, 4584, 7354, 1393,
        10282, 14324, 304, 10001, 4938, 13, 576, 1887, 4755, 525, 3425, 279,
        88737, 26344, 279, 2025, 11, 3425, 11504, 27035, 21129, 279, 4396, 6500,
        1584, 11, 323, 3425, 835, 311, 835, 39270, 33327, 264, 20692, 1196,
        3550, 1882, 13, 362, 6849, 825, 3950, 29716, 374, 5390, 11, 3602,
        432, 4157, 5695, 13403, 3941, 1293, 50932, 476, 6092, 7709, 1212, 34035,
        9442, 13, 76817, 1265, 990, 279, 1852, 1614, 14324, 11, 3950, 2022,
        11, 17654, 50452, 11, 323, 18405, 22711, 369, 2176, 12716, 624
    };
    struct qwen3_tokenizer *tokenizer;
    int rc;

    if (argc != 2) {
        fprintf(stderr, "usage: %s tokenizer.json\n", argv[0]);
        return 2;
    }
    tokenizer = qwen3_tokenizer_open(argv[1]);
    if (!tokenizer) {
        fprintf(stderr, "could not load Qwen3 tokenizer\n");
        return 1;
    }
    rc = check(tokenizer, "Hello, world!", english,
               sizeof(english) / sizeof(english[0])) ||
         check(tokenizer, "你好，世界！", chinese,
               sizeof(chinese) / sizeof(chinese[0])) ||
         check(tokenizer, "<|im_start|>user\nHi<|im_end|>", chat,
               sizeof(chat) / sizeof(chat[0])) ||
         check(tokenizer, "A  B", spaces,
               sizeof(spaces) / sizeof(spaces[0])) ||
         check(tokenizer, "Emoji 😊 and C++.", emoji,
               sizeof(emoji) / sizeof(emoji[0])) ||
         check(tokenizer, "中文 English 123 / eBPF", mixed,
               sizeof(mixed) / sizeof(mixed[0])) ||
         check(tokenizer, "\n\nHi\tthere", lines,
               sizeof(lines) / sizeof(lines[0]));
    if (!rc) {
        char text[1024];
        FILE *fixture = fopen("tests/long-context.txt", "rb");
        size_t bytes = fixture ? fread(text, 1, sizeof(text) - 1, fixture) : 0;
        int complete = fixture && bytes < sizeof(text) - 1 && feof(fixture) &&
                       !ferror(fixture);

        if (fixture)
            fclose(fixture);
        if (!complete) {
            fprintf(stderr, "could not read complete long-context fixture\n");
            rc = 1;
        } else {
            text[bytes] = 0;
            rc = check(tokenizer, text, long_context,
                       sizeof(long_context) / sizeof(long_context[0]));
        }
    }
    qwen3_tokenizer_close(tokenizer);
    return rc ? 1 : 0;
}
