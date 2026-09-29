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
    qwen3_tokenizer_close(tokenizer);
    return rc ? 1 : 0;
}
