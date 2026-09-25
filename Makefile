CLANG ?= clang
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror
ARCH_INCLUDE := /usr/include/$(shell uname -m)-linux-gnu

.PHONY: all test test-model test-tokenizer clean

all: build/qwen3_matvec.bpf.o build/matvec-smoke build/matrix-smoke build/qwen3_batch.bpf.o build/batch-smoke build/qwen3_int4.bpf.o build/int4-smoke build/qwen3_norm.bpf.o build/norm-smoke build/qwen3_silu.bpf.o build/silu-smoke build/qwen3_vector.bpf.o build/vector-smoke build/qwen3_rope.bpf.o build/rope-smoke build/qwen3_attention.bpf.o build/attention-smoke build/infer build/safetensors-smoke

build:
	mkdir -p build

build/qwen3_matvec.bpf.o: src/qwen3_matvec.bpf.c src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_batch.bpf.o: src/qwen3_batch.bpf.c src/qwen3_batch.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/batch-smoke: src/batch-smoke.c src/qwen3_batch.h | build
	$(CC) $(CFLAGS) -Isrc src/batch-smoke.c -o $@ -lbpf -lelf -lz

build/qwen3_int4.bpf.o: src/qwen3_int4.bpf.c src/qwen3_int4.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/int4-smoke: src/int4-smoke.c src/qwen3_int4.h src/safetensors.c src/safetensors.h | build
	$(CC) $(CFLAGS) -Isrc src/int4-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz -lm

build/qwen3_norm.bpf.o: src/qwen3_norm.bpf.c src/qwen3_norm.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_silu.bpf.o: src/qwen3_silu.bpf.c src/qwen3_silu.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_vector.bpf.o: src/qwen3_vector.bpf.c src/qwen3_vector.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_rope.bpf.o: src/qwen3_rope.bpf.c src/qwen3_rope.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_attention.bpf.o: src/qwen3_attention.bpf.c src/qwen3_attention.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/matvec-smoke: src/matvec-smoke.c src/safetensors.c src/safetensors.h src/qwen3_tile.h | build
	$(CC) $(CFLAGS) -Isrc src/matvec-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz

build/matrix-smoke: src/matrix-smoke.c src/safetensors.c src/safetensors.h src/qwen3_tile.h | build
	$(CC) $(CFLAGS) -Isrc src/matrix-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz -lm

build/norm-smoke: src/norm-smoke.c src/safetensors.c src/safetensors.h src/qwen3_norm.h | build
	$(CC) $(CFLAGS) -Isrc src/norm-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz -lm

build/silu-smoke: src/silu-smoke.c src/qwen3_silu.h | build
	$(CC) $(CFLAGS) -Isrc src/silu-smoke.c -o $@ -lbpf -lelf -lz -lm

build/vector-smoke: src/vector-smoke.c src/qwen3_vector.h | build
	$(CC) $(CFLAGS) -Isrc src/vector-smoke.c -o $@ -lbpf -lelf -lz

build/rope-smoke: src/rope-smoke.c src/qwen3_rope.h | build
	$(CC) $(CFLAGS) -Isrc src/rope-smoke.c -o $@ -lbpf -lelf -lz -lm

build/attention-smoke: src/attention-smoke.c src/qwen3_attention.h | build
	$(CC) $(CFLAGS) -Isrc src/attention-smoke.c -o $@ -lbpf -lelf -lz -lm

build/tokenizer-smoke: src/tokenizer-smoke.c src/qwen3_tokenizer.c src/qwen3_tokenizer.h | build
	$(CC) $(CFLAGS) -Isrc src/tokenizer-smoke.c src/qwen3_tokenizer.c -o $@ -ljson-c -lonig

build/infer: src/infer.c src/safetensors.c src/safetensors.h src/qwen3_tokenizer.c src/qwen3_tokenizer.h src/qwen3_tile.h src/qwen3_batch.h src/qwen3_norm.h src/qwen3_silu.h src/qwen3_vector.h src/qwen3_rope.h src/qwen3_attention.h | build
	$(CC) $(CFLAGS) -Isrc src/infer.c src/safetensors.c src/qwen3_tokenizer.c -o $@ -lbpf -lelf -lz -lm -ljson-c -lonig

build/safetensors-smoke: src/safetensors-smoke.c src/safetensors.c src/safetensors.h | build
	$(CC) $(CFLAGS) -Isrc src/safetensors-smoke.c src/safetensors.c -o $@ -lm

test: all
	./build/safetensors-smoke
	./build/matvec-smoke build/qwen3_matvec.bpf.o
	./build/batch-smoke build/qwen3_batch.bpf.o
	./build/int4-smoke build/qwen3_int4.bpf.o
	./build/silu-smoke build/qwen3_silu.bpf.o
	./build/vector-smoke build/qwen3_vector.bpf.o
	./build/rope-smoke build/qwen3_rope.bpf.o
	./build/attention-smoke build/qwen3_attention.bpf.o

test-model: all
	test -n "$(MODEL)"
	./build/matvec-smoke build/qwen3_matvec.bpf.o "$(MODEL)" --q24
	./build/int4-smoke build/qwen3_int4.bpf.o "$(MODEL)"
	./build/norm-smoke build/qwen3_norm.bpf.o "$(MODEL)"

test-tokenizer: build/tokenizer-smoke
	test -n "$(TOKENIZER)"
	./build/tokenizer-smoke "$(TOKENIZER)"

clean:
	rm -r build
