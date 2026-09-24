#ifndef QWEN3_TOKENIZER_H
#define QWEN3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

struct qwen3_tokenizer;

struct qwen3_tokenizer *qwen3_tokenizer_open(const char *path);
void qwen3_tokenizer_close(struct qwen3_tokenizer *tokenizer);
int qwen3_tokenizer_encode(struct qwen3_tokenizer *tokenizer, const char *text,
                           uint32_t **ids, size_t *count);
int qwen3_tokenizer_decode(struct qwen3_tokenizer *tokenizer, uint32_t id,
                           unsigned char **bytes, size_t *length);

#endif
