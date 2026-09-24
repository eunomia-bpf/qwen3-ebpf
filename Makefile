CLANG ?= clang
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror
ARCH_INCLUDE := /usr/include/$(shell uname -m)-linux-gnu

.PHONY: all test clean

all: build/qwen3_matvec.bpf.o build/matvec-smoke

build:
	mkdir -p build

build/qwen3_matvec.bpf.o: src/qwen3_matvec.bpf.c src/qwen3_tile.h | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) -Isrc -c $< -o $@

build/matvec-smoke: src/matvec-smoke.c src/qwen3_tile.h | build
	$(CC) $(CFLAGS) -Isrc $< -o $@ -lbpf -lelf -lz

test: all
	./build/matvec-smoke build/qwen3_matvec.bpf.o

clean:
	rm -r build
