#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>
#include <oniguruma.h>
#include "qwen3_tokenizer.h"

#define QWEN3_TOKENIZER_IDS 151936

struct token_slot {
    const char *text;
    uint32_t id;
};

struct merge_slot {
    uint64_t pair;
    uint32_t result;
    uint32_t rank;
    int used;
};

struct special_token {
    const char *text;
    uint32_t id;
};

struct qwen3_tokenizer {
    struct json_object *root;
    OnigRegex pattern;
    struct token_slot *vocab;
    size_t vocab_slots;
    struct merge_slot *merges;
    size_t merge_slots;
    const char **by_id;
    struct special_token specials[64];
    size_t special_count;
    uint32_t byte_id[256];
    uint32_t byte_cp[256];
    int16_t cp_byte[512];
};

struct id_buffer {
    uint32_t *data;
    size_t count;
    size_t capacity;
};

static uint64_t text_hash(const char *text)
{
    uint64_t hash = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        hash ^= *p++;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static size_t pair_hash(uint64_t pair, size_t slots)
{
    pair ^= pair >> 33;
    pair *= 0xff51afd7ed558ccdULL;
    pair ^= pair >> 33;
    return (size_t)pair & (slots - 1);
}

static uint32_t vocab_lookup(const struct qwen3_tokenizer *t,
                             const char *text)
{
    size_t i = (size_t)text_hash(text) & (t->vocab_slots - 1);
    while (t->vocab[i].text) {
        if (strcmp(t->vocab[i].text, text) == 0)
            return t->vocab[i].id;
        i = (i + 1) & (t->vocab_slots - 1);
    }
    return UINT32_MAX;
}

static const struct merge_slot *merge_lookup(const struct qwen3_tokenizer *t,
                                              uint32_t left, uint32_t right)
{
    uint64_t pair = ((uint64_t)left << 32) | right;
    size_t i = pair_hash(pair, t->merge_slots);
    while (t->merges[i].used) {
        if (t->merges[i].pair == pair)
            return &t->merges[i];
        i = (i + 1) & (t->merge_slots - 1);
    }
    return NULL;
}

static size_t utf8_from_cp(uint32_t cp, char out[4])
{
    if (cp < 128) {
        out[0] = (char)cp;
        out[1] = 0;
        return 1;
    }
    out[0] = (char)(0xc0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3f));
    out[2] = 0;
    return 2;
}

static int build_alphabet(struct qwen3_tokenizer *t)
{
    uint32_t next = 256;
    int byte;

    for (byte = 0; byte < 512; byte++)
        t->cp_byte[byte] = -1;
    for (byte = 0; byte < 256; byte++) {
        char symbol[4];
        uint32_t cp = (byte >= 33 && byte <= 126) ||
                      (byte >= 161 && byte <= 172) ||
                      (byte >= 174 && byte <= 255)
                      ? (uint32_t)byte : next++;
        t->byte_cp[byte] = cp;
        t->cp_byte[cp] = (int16_t)byte;
        utf8_from_cp(cp, symbol);
        t->byte_id[byte] = vocab_lookup(t, symbol);
        if (t->byte_id[byte] == UINT32_MAX)
            return -1;
    }
    return 0;
}

static int load_vocab(struct qwen3_tokenizer *t, struct json_object *model)
{
    struct json_object *vocab;
    size_t count;

    if (!json_object_object_get_ex(model, "vocab", &vocab) ||
        !json_object_is_type(vocab, json_type_object))
        return -1;
    count = (size_t)json_object_object_length(vocab);
    t->vocab_slots = 1;
    while (t->vocab_slots < count * 2)
        t->vocab_slots *= 2;
    t->vocab = calloc(t->vocab_slots, sizeof(*t->vocab));
    t->by_id = calloc(QWEN3_TOKENIZER_IDS, sizeof(*t->by_id));
    if (!t->vocab || !t->by_id)
        return -1;
    json_object_object_foreach(vocab, text, value) {
        int64_t id = json_object_get_int64(value);
        size_t i = (size_t)text_hash(text) & (t->vocab_slots - 1);
        if (id < 0 || id >= QWEN3_TOKENIZER_IDS)
            return -1;
        while (t->vocab[i].text)
            i = (i + 1) & (t->vocab_slots - 1);
        t->vocab[i].text = text;
        t->vocab[i].id = (uint32_t)id;
        t->by_id[id] = text;
    }
    return build_alphabet(t);
}

static int load_merges(struct qwen3_tokenizer *t, struct json_object *model)
{
    struct json_object *merges;
    size_t count, rank;

    if (!json_object_object_get_ex(model, "merges", &merges) ||
        !json_object_is_type(merges, json_type_array))
        return -1;
    count = json_object_array_length(merges);
    t->merge_slots = 1;
    while (t->merge_slots < count * 2)
        t->merge_slots *= 2;
    t->merges = calloc(t->merge_slots, sizeof(*t->merges));
    if (!t->merges)
        return -1;
    for (rank = 0; rank < count; rank++) {
        struct json_object *entry = json_object_array_get_idx(merges, rank);
        const char *left, *right;
        uint32_t left_id, right_id, result_id;
        size_t left_len, right_len, i;
        char *joined;
        uint64_t pair;

        if (!entry || json_object_array_length(entry) != 2)
            return -1;
        left = json_object_get_string(json_object_array_get_idx(entry, 0));
        right = json_object_get_string(json_object_array_get_idx(entry, 1));
        if (!left || !right)
            return -1;
        left_id = vocab_lookup(t, left);
        right_id = vocab_lookup(t, right);
        if (left_id == UINT32_MAX || right_id == UINT32_MAX)
            return -1;
        left_len = strlen(left);
        right_len = strlen(right);
        joined = malloc(left_len + right_len + 1);
        if (!joined)
            return -1;
        memcpy(joined, left, left_len);
        memcpy(joined + left_len, right, right_len + 1);
        result_id = vocab_lookup(t, joined);
        free(joined);
        if (result_id == UINT32_MAX)
            return -1;
        pair = ((uint64_t)left_id << 32) | right_id;
        i = pair_hash(pair, t->merge_slots);
        while (t->merges[i].used)
            i = (i + 1) & (t->merge_slots - 1);
        t->merges[i].pair = pair;
        t->merges[i].result = result_id;
        t->merges[i].rank = (uint32_t)rank;
        t->merges[i].used = 1;
    }
    return 0;
}

static int load_specials(struct qwen3_tokenizer *t)
{
    struct json_object *added;
    size_t i, count;

    if (!json_object_object_get_ex(t->root, "added_tokens", &added) ||
        !json_object_is_type(added, json_type_array))
        return -1;
    count = json_object_array_length(added);
    if (count > sizeof(t->specials) / sizeof(t->specials[0]))
        return -1;
    for (i = 0; i < count; i++) {
        struct json_object *entry = json_object_array_get_idx(added, i);
        struct json_object *content, *id_object;
        int64_t id;
        if (!entry || !json_object_object_get_ex(entry, "content", &content) ||
            !json_object_object_get_ex(entry, "id", &id_object))
            return -1;
        id = json_object_get_int64(id_object);
        if (id < 0 || id >= QWEN3_TOKENIZER_IDS)
            return -1;
        t->specials[i].text = json_object_get_string(content);
        t->specials[i].id = (uint32_t)id;
        t->by_id[id] = t->specials[i].text;
    }
    t->special_count = count;
    return 0;
}

struct qwen3_tokenizer *qwen3_tokenizer_open(const char *path)
{
    struct qwen3_tokenizer *t = calloc(1, sizeof(*t));
    struct json_object *model, *pre, *pretokenizers, *split, *pattern, *regex;
    OnigErrorInfo error;
    const char *expression;
    int onig_rc;

    if (!t)
        return NULL;
    t->root = json_object_from_file(path);
    if (!t->root || !json_object_object_get_ex(t->root, "model", &model) ||
        !json_object_object_get_ex(t->root, "pre_tokenizer", &pre) ||
        !json_object_object_get_ex(pre, "pretokenizers", &pretokenizers) ||
        !(split = json_object_array_get_idx(pretokenizers, 0)) ||
        !json_object_object_get_ex(split, "pattern", &pattern) ||
        !json_object_object_get_ex(pattern, "Regex", &regex) ||
        load_vocab(t, model) || load_merges(t, model) || load_specials(t))
        goto fail;
    expression = json_object_get_string(regex);
    if (!expression)
        goto fail;
    onig_rc = onig_new(&t->pattern, (const UChar *)expression,
                       (const UChar *)expression + strlen(expression),
                       ONIG_OPTION_DEFAULT, ONIG_ENCODING_UTF8,
                       ONIG_SYNTAX_DEFAULT, &error);
    if (onig_rc != ONIG_NORMAL)
        goto fail;
    return t;
fail:
    qwen3_tokenizer_close(t);
    return NULL;
}

void qwen3_tokenizer_close(struct qwen3_tokenizer *t)
{
    if (!t)
        return;
    if (t->pattern)
        onig_free(t->pattern);
    if (t->root)
        json_object_put(t->root);
    free(t->vocab);
    free(t->merges);
    free(t->by_id);
    free(t);
}

static int append_id(struct id_buffer *out, uint32_t id)
{
    if (out->count == out->capacity) {
        size_t capacity = out->capacity ? out->capacity * 2 : 32;
        uint32_t *resized = realloc(out->data, capacity * sizeof(*resized));
        if (!resized)
            return -1;
        out->data = resized;
        out->capacity = capacity;
    }
    out->data[out->count++] = id;
    return 0;
}

static int encode_piece(const struct qwen3_tokenizer *t, const char *text,
                        size_t length, struct id_buffer *out)
{
    uint32_t *parts;
    size_t count = length, i;

    if (!length)
        return 0;
    parts = malloc(count * sizeof(*parts));
    if (!parts)
        return -1;
    for (i = 0; i < count; i++)
        parts[i] = t->byte_id[(unsigned char)text[i]];
    while (count > 1) {
        const struct merge_slot *best = NULL;
        size_t best_at = 0;
        for (i = 0; i + 1 < count; i++) {
            const struct merge_slot *candidate =
                merge_lookup(t, parts[i], parts[i + 1]);
            if (candidate && (!best || candidate->rank < best->rank)) {
                best = candidate;
                best_at = i;
            }
        }
        if (!best)
            break;
        parts[best_at] = best->result;
        memmove(parts + best_at + 1, parts + best_at + 2,
                (count - best_at - 2) * sizeof(*parts));
        count--;
    }
    for (i = 0; i < count; i++)
        if (append_id(out, parts[i])) {
            free(parts);
            return -1;
        }
    free(parts);
    return 0;
}

static int encode_normal(struct qwen3_tokenizer *t, const char *text,
                         size_t length, struct id_buffer *out)
{
    OnigRegion *region = onig_region_new();
    const UChar *start = (const UChar *)text;
    const UChar *end = start + length;
    size_t cursor = 0;
    int rc = -1;

    if (!region)
        return -1;
    while (cursor < length) {
        int found = onig_search(t->pattern, start, end, start + cursor,
                                end, region, ONIG_OPTION_NONE);
        size_t begin, finish;
        if (found == ONIG_MISMATCH)
            break;
        if (found < 0)
            goto done;
        begin = (size_t)region->beg[0];
        finish = (size_t)region->end[0];
        if (begin < cursor || finish <= begin || finish > length ||
            encode_piece(t, text + cursor, begin - cursor, out) ||
            encode_piece(t, text + begin, finish - begin, out))
            goto done;
        cursor = finish;
    }
    if (encode_piece(t, text + cursor, length - cursor, out))
        goto done;
    rc = 0;
done:
    onig_region_free(region, 1);
    return rc;
}

int qwen3_tokenizer_encode(struct qwen3_tokenizer *t, const char *text,
                           uint32_t **ids, size_t *count)
{
    struct id_buffer out = {0};
    size_t cursor = 0, length;

    if (!t || !text || !ids || !count)
        return -1;
    length = strlen(text);
    while (cursor < length) {
        const char *first = NULL;
        size_t special = 0, i;
        for (i = 0; i < t->special_count; i++) {
            const char *candidate = strstr(text + cursor, t->specials[i].text);
            if (candidate && (!first || candidate < first ||
                (candidate == first &&
                 strlen(t->specials[i].text) > strlen(t->specials[special].text)))) {
                first = candidate;
                special = i;
            }
        }
        if (!first) {
            if (encode_normal(t, text + cursor, length - cursor, &out))
                goto fail;
            break;
        }
        if (encode_normal(t, text + cursor, (size_t)(first - text) - cursor,
                          &out) || append_id(&out, t->specials[special].id))
            goto fail;
        cursor = (size_t)(first - text) + strlen(t->specials[special].text);
    }
    *ids = out.data;
    *count = out.count;
    return 0;
fail:
    free(out.data);
    return -1;
}

int qwen3_tokenizer_decode(struct qwen3_tokenizer *t, uint32_t id,
                           unsigned char **bytes, size_t *length)
{
    const unsigned char *p;
    unsigned char *result;
    size_t used = 0;

    if (!t || !bytes || !length || id >= QWEN3_TOKENIZER_IDS ||
        !t->by_id[id])
        return -1;
    p = (const unsigned char *)t->by_id[id];
    if (id >= 151643) {
        *length = strlen((const char *)p);
        result = malloc(*length + 1);
        if (!result)
            return -1;
        memcpy(result, p, *length + 1);
        *bytes = result;
        return 0;
    }
    result = malloc(strlen((const char *)p) + 1);
    if (!result)
        return -1;
    while (*p) {
        uint32_t cp;
        if (*p < 128) {
            cp = *p++;
        } else if ((*p & 0xe0) == 0xc0 && p[1]) {
            cp = ((uint32_t)(p[0] & 0x1f) << 6) | (p[1] & 0x3f);
            p += 2;
        } else {
            free(result);
            return -1;
        }
        if (cp >= 512 || t->cp_byte[cp] < 0) {
            free(result);
            return -1;
        }
        result[used++] = (unsigned char)t->cp_byte[cp];
    }
    result[used] = 0;
    *bytes = result;
    *length = used;
    return 0;
}
