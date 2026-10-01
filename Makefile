CLANG ?= clang
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror
ARCH_INCLUDE := /usr/include/$(shell uname -m)-linux-gnu
ARENA_CLANG ?= clang-19
ARENA_BPFTOOL ?= bpftool
ARENA_ARCH ?= $(shell uname -m | sed 's/aarch64/arm64/')
ARENA_LIBBPF_INCLUDE ?= /usr/include
ARENA_UAPI_INCLUDE ?= /usr/include
ARENA_LIBBPF ?= -lbpf

INCLUDES := -Iinclude
HEADERS := $(wildcard include/*.h)
BPF_NAMES := batch norm silu vector rope attention
BPF_OBJECTS := $(addprefix build/qwen3_,$(addsuffix .bpf.o,$(BPF_NAMES)))
EXPERIMENT_NAMES := matvec int4 int8
EXPERIMENT_OBJECTS := $(addprefix build/qwen3_,$(addsuffix .bpf.o,$(EXPERIMENT_NAMES)))
SMOKE_NAMES := matvec matrix batch int4 int8 norm silu vector rope attention safetensors
SMOKE_BINS := $(addprefix build/,$(addsuffix -smoke,$(SMOKE_NAMES)))
HOST_SOURCES := src/infer.c src/safetensors.c src/qwen3_tokenizer.c
HOST_LIBS := -lbpf -lelf -lz -lm -ljson-c -lonig
ARENA_INCLUDES := $(INCLUDES) -Ibuild -I$(ARENA_LIBBPF_INCLUDE) -I$(ARENA_UAPI_INCLUDE)
ARENA_STEMS := build/qwen3_arena_int4 build/qwen3_arena_bf16

.DEFAULT_GOAL := all
.PHONY: all help experiments test test-model test-tokenizer test-arena-int4 test-arena-bf16 test-arena-xdp test-arena-xdp-model test-arena-xdp-attention clean

# The default build is the working inference path, not the experiment suite.
all: build/infer $(BPF_OBJECTS)

help:
	@printf '%s\n' 'make                 Build inference and its BPF operators' \
	  'make test            Run synthetic operator tests (BPF privileges required)' \
	  'make test-model MODEL=...        Run official-weight operator checks' \
	  'make test-tokenizer TOKENIZER=... Run tokenizer checks (no BPF privileges)' \
	  'make experiments     Build standalone matvec / INT4 / INT8 experiments' \
	  'make build/infer-arena-bf16      Build optional arena inference' \
	  'make test-arena-bf16 [MODEL=...] Run optional arena BF16 checks' \
	  'make test-arena-xdp   Live XDP/workqueue smoke test on this network namespace loopback' \
	  'make test-arena-xdp-model MODEL=... Run token-driven real-weight XDP matrix check' \
	  'make test-arena-xdp-attention MODEL=... Check two-token attention and 28-layer XDP prefix' \
	  'make test-arena-int4 [MODEL=...] Run optional arena INT4 checks' \
	  'make clean           Remove build outputs'

build:
	mkdir -p $@

vpath %.bpf.c src/bpf experiments
build/%.bpf.o: %.bpf.c $(HEADERS) | build
	$(CLANG) -O2 -g -target bpf -I$(ARCH_INCLUDE) $(INCLUDES) -c $< -o $@

build/infer: $(HOST_SOURCES) $(HEADERS) | build
	$(CC) $(CFLAGS) $(INCLUDES) $(HOST_SOURCES) -o $@ $(HOST_LIBS)

# Model-backed checks share the Safetensors reader; other tests need only libbpf.
build/matvec-smoke build/matrix-smoke build/int4-smoke build/int8-smoke build/norm-smoke: src/safetensors.c
build/%-smoke: tests/%-smoke.c $(HEADERS) | build
	$(CC) $(CFLAGS) $(INCLUDES) $(filter %.c,$^) -o $@ -lbpf -lelf -lz -lm

build/safetensors-smoke: tests/safetensors-smoke.c src/safetensors.c include/safetensors.h | build
	$(CC) $(CFLAGS) $(INCLUDES) $(filter %.c,$^) -o $@ -lm

build/tokenizer-smoke: tests/tokenizer-smoke.c src/qwen3_tokenizer.c include/qwen3_tokenizer.h | build
	$(CC) $(CFLAGS) $(INCLUDES) $(filter %.c,$^) -o $@ -ljson-c -lonig

experiments: $(EXPERIMENT_OBJECTS) build/matvec-smoke build/matrix-smoke build/int4-smoke build/int8-smoke

# Arena globals need Clang 19 and arena-capable libbpf, UAPI headers and bpftool.
$(addsuffix .tmp.bpf.o,$(ARENA_STEMS)): build/qwen3_arena_%.tmp.bpf.o: qwen3_arena_%.bpf.c $(HEADERS) | build
	$(ARENA_CLANG) -O2 -g -target bpf -D__TARGET_ARCH_$(ARENA_ARCH) -I$(ARCH_INCLUDE) $(ARENA_INCLUDES) -c $< -o $@

$(addsuffix .bpf.o,$(ARENA_STEMS)): build/qwen3_arena_%.bpf.o: build/qwen3_arena_%.tmp.bpf.o
	$(ARENA_BPFTOOL) gen object $@ $<

$(addsuffix .skel.h,$(ARENA_STEMS)): build/qwen3_arena_%.skel.h: build/qwen3_arena_%.bpf.o
	$(ARENA_BPFTOOL) gen skeleton $< > $@

build/arena-int4-smoke build/arena-bf16-smoke: build/arena-%-smoke: tests/arena-%-smoke.c src/safetensors.c build/qwen3_arena_%.skel.h $(HEADERS) | build
	$(CC) $(CFLAGS) $(ARENA_INCLUDES) $(filter %.c,$^) -o $@ $(ARENA_LIBBPF) -lelf -lz -lm

build/infer-arena-bf16: $(HOST_SOURCES) $(HEADERS) build/qwen3_arena_bf16.skel.h $(BPF_OBJECTS) | build
	$(CC) $(CFLAGS) -DQWEN3_USE_ARENA_BF16 $(ARENA_INCLUDES) $(HOST_SOURCES) -o $@ $(ARENA_LIBBPF) -lelf -lz -lm -ljson-c -lonig

test: all $(EXPERIMENT_OBJECTS) $(SMOKE_BINS)
	./build/safetensors-smoke
	./build/matvec-smoke build/qwen3_matvec.bpf.o
	./build/batch-smoke build/qwen3_batch.bpf.o
	./build/int4-smoke build/qwen3_int4.bpf.o
	./build/int8-smoke build/qwen3_int8.bpf.o
	./build/silu-smoke build/qwen3_silu.bpf.o
	./build/vector-smoke build/qwen3_vector.bpf.o
	./build/rope-smoke build/qwen3_rope.bpf.o
	./build/attention-smoke build/qwen3_attention.bpf.o

test-model: all $(EXPERIMENT_OBJECTS) build/matvec-smoke build/int4-smoke build/int8-smoke build/norm-smoke
	test -n "$(MODEL)"
	./build/matvec-smoke build/qwen3_matvec.bpf.o "$(MODEL)" --q24
	./build/int4-smoke build/qwen3_int4.bpf.o "$(MODEL)"
	./build/int8-smoke build/qwen3_int8.bpf.o "$(MODEL)"
	./build/norm-smoke build/qwen3_norm.bpf.o "$(MODEL)"

test-tokenizer: build/tokenizer-smoke
	test -n "$(TOKENIZER)"
	./build/tokenizer-smoke "$(TOKENIZER)"

test-arena-int4: build/arena-int4-smoke
	./build/arena-int4-smoke $(if $(MODEL),"$(MODEL)")

test-arena-bf16: build/arena-bf16-smoke
	./build/arena-bf16-smoke $(if $(MODEL),"$(MODEL)")

test-arena-xdp: build/arena-bf16-smoke
	./build/arena-bf16-smoke --xdp-loopback

test-arena-xdp-model: build/arena-bf16-smoke
	test -n "$(MODEL)"
	./build/arena-bf16-smoke --xdp-model "$(MODEL)"

test-arena-xdp-attention: build/arena-bf16-smoke
	test -n "$(MODEL)"
	./build/arena-bf16-smoke --xdp-attention "$(MODEL)"

clean:
	$(RM) -r build
