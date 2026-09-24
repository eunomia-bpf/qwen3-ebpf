CLANG ?= clang
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror
ARCH_INCLUDE := /usr/include/$(shell uname -m)-linux-gnu

.PHONY: all test test-model clean

all: build/qwen3_matvec.bpf.o build/matvec-smoke build/qwen3_norm.bpf.o build/norm-smoke build/qwen3_silu.bpf.o build/silu-smoke

build:
	mkdir -p build

build/qwen3_matvec.bpf.o: src/qwen3_matvec.bpf.c src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_norm.bpf.o: src/qwen3_norm.bpf.c src/qwen3_norm.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/qwen3_silu.bpf.o: src/qwen3_silu.bpf.c src/qwen3_silu.h src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/matvec-smoke: src/matvec-smoke.c src/safetensors.c src/safetensors.h src/qwen3_tile.h | build
	$(CC) $(CFLAGS) -Isrc src/matvec-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz

build/norm-smoke: src/norm-smoke.c src/safetensors.c src/safetensors.h src/qwen3_norm.h | build
	$(CC) $(CFLAGS) -Isrc src/norm-smoke.c src/safetensors.c -o $@ -lbpf -lelf -lz -lm

build/silu-smoke: src/silu-smoke.c src/qwen3_silu.h | build
	$(CC) $(CFLAGS) -Isrc src/silu-smoke.c -o $@ -lbpf -lelf -lz -lm

test: all
	./build/matvec-smoke build/qwen3_matvec.bpf.o
	./build/silu-smoke build/qwen3_silu.bpf.o

test-model: all
	test -n "$(MODEL)"
	./build/matvec-smoke build/qwen3_matvec.bpf.o "$(MODEL)" --q24
	./build/norm-smoke build/qwen3_norm.bpf.o "$(MODEL)"

clean:
	rm -r build
