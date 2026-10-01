#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "qwen3_tokenizer.h"

#define QWEN3_0_6B_VOCAB 151936UL
#define XDP_CLIENT_MAX_TOKENS 256

static uint64_t monotonic_ns(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now))
        return 0;
    return (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
}

static void put_u32(uint8_t *out, uint32_t value)
{
    out[0] = value >> 24;
    out[1] = value >> 16;
    out[2] = value >> 8;
    out[3] = value;
}

static uint32_t get_u32(const uint8_t *in)
{
    return ((uint32_t)in[0] << 24) | ((uint32_t)in[1] << 16) |
           ((uint32_t)in[2] << 8) | in[3];
}

static int request_token(int fd, const struct sockaddr_in *server,
                         uint32_t token, uint32_t position, int prefill)
{
    uint8_t request[12] = {'Q', '3', 'B', 'P'};
    uint8_t poll[20] = {'Q', '3', 'B', 'R'};
    uint8_t reply[20];
    uint64_t start = monotonic_ns();

    if (!start)
        return -1;
    if (prefill)
        request[3] = 'F';
    put_u32(request + 4, token);
    put_u32(request + 8, position);
    put_u32(poll + 4, position + 1);
    if (sendto(fd, request, sizeof(request), 0,
               (const struct sockaddr *)server, sizeof(*server)) !=
        sizeof(request))
        return -1;
    while (monotonic_ns() - start < 30000000000ULL) {
        struct sockaddr_in sender;
        socklen_t sender_len = sizeof(sender);
        ssize_t size;

        if (sendto(fd, poll, sizeof(poll), 0,
                   (const struct sockaddr *)server, sizeof(*server)) !=
            sizeof(poll))
            return -1;
        size = recvfrom(fd, reply, sizeof(reply), 0,
                        (struct sockaddr *)&sender, &sender_len);
        if (size == sizeof(reply) && sender_len == sizeof(sender) &&
            sender.sin_addr.s_addr == server->sin_addr.s_addr &&
            sender.sin_port == server->sin_port &&
            !memcmp(reply, prefill ? "Q3BK" : "Q3BA", 4) &&
            get_u32(reply + 16) == position + 1) {
            uint64_t bits = 0;

            if (prefill) {
                printf("position=%u input=%u prefill_done elapsed_ms=%.3f\n",
                       position, token,
                       (monotonic_ns() - start) / 1000000.0);
                return 0;
            }
            for (int i = 0; i < 8; i++)
                bits = (bits << 8) | reply[8 + i];
            printf("position=%u input=%u next_token=%u logit_q16=%" PRId64
                   " elapsed_ms=%.3f\n", position, token,
                   get_u32(reply + 4), (int64_t)bits,
                   (monotonic_ns() - start) / 1000000.0);
            return 0;
        }
        if (size < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
            errno != ECONNREFUSED && errno != EINTR)
            return -1;
        usleep(1000);
    }
    fprintf(stderr, "no XDP result for position %u\n", position);
    return -1;
}

int main(int argc, char **argv)
{
    struct sockaddr_in server = {
        .sin_family = AF_INET,
        .sin_port = htons(49002),
    };
    struct timeval timeout = {.tv_usec = 5000};
    struct qwen3_tokenizer *tokenizer = NULL;
    uint32_t *tokens = NULL;
    size_t count = 0;
    uint64_t start;
    int first = 2, prefill = 0, fd, rc = 1;

    if (argc < 3 || inet_pton(AF_INET, argv[1], &server.sin_addr) != 1) {
        fprintf(stderr, "usage: %s IPv4 [--prefill] token_id [token_id ...]\n"
                        "   or: %s IPv4 [--prefill] --tokenizer tokenizer.json --prompt text\n",
                argv[0], argv[0]);
        return 2;
    }
    if (!strcmp(argv[2], "--prefill")) {
        prefill = 1;
        first++;
    }
    if (argc == first + 4 && !strcmp(argv[first], "--tokenizer") &&
        !strcmp(argv[first + 2], "--prompt")) {
        tokenizer = qwen3_tokenizer_open(argv[first + 1]);
        if (!tokenizer ||
            qwen3_tokenizer_encode(tokenizer, argv[first + 3],
                                   &tokens, &count) ||
            !count || count > XDP_CLIENT_MAX_TOKENS) {
            fprintf(stderr, "could not encode prompt within XDP KV limit\n");
            goto done;
        }
        fprintf(stderr, "input_tokens=%zu\n", count);
    } else {
        count = (size_t)argc - first;
        if (!count || count > XDP_CLIENT_MAX_TOKENS) {
            fprintf(stderr, "too many token IDs for XDP KV cache\n");
            goto done;
        }
        tokens = calloc(count, sizeof(*tokens));
        if (!tokens)
            goto done;
        for (size_t i = 0; i < count; i++) {
            char *end;
            unsigned long value;

            errno = 0;
            value = strtoul(argv[i + first], &end, 10);
            if (errno || !argv[i + first][0] || *end ||
                value >= QWEN3_0_6B_VOCAB) {
                fprintf(stderr, "invalid token ID: %s\n", argv[i + first]);
                goto done;
            }
            tokens[i] = (uint32_t)value;
        }
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0 || setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                            &timeout, sizeof(timeout))) {
        if (fd >= 0)
            close(fd);
        goto done;
    }
    start = monotonic_ns();
    for (size_t i = 0; i < count; i++)
        if (request_token(fd, &server, tokens[i], (uint32_t)i,
                          prefill && i + 1 < count))
            goto close_socket;
    printf("tokens=%zu total_ms=%.3f\n", count,
           (monotonic_ns() - start) / 1000000.0);
    rc = 0;
close_socket:
    close(fd);
done:
    free(tokens);
    qwen3_tokenizer_close(tokenizer);
    return rc;
}
