# Usage and development

[Back to the project](../README.md) · [Architecture](architecture.md)

Run commands from the repository root: the inference driver resolves its BPF
objects under `build/`. Compile as an ordinary user; loading BPF programs needs
appropriate privileges (the examples below can be run with `sudo` on a test host).
Use a development machine, not a production kernel.

## Build dependencies

The recorded validation environment is Linux 6.17 arm64 with an Ubuntu 24.04
build container. Other kernels and architectures require validation; a successful
compile alone does not establish verifier compatibility.

For Ubuntu 24.04, the default path uses:

```sh
sudo apt-get install build-essential clang libbpf-dev libelf-dev zlib1g-dev \
  libjson-c-dev libonig-dev
make
```

Download `model.safetensors` and `tokenizer.json` from the
[official Qwen3-0.6B repository](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main).
Keep these external assets outside Git. The model file is about 1.5 GB; leave
additional memory for the mapped weights and the context-dependent BPF KV cache.

## Build and run

On a Linux host with clang's BPF target, libbpf, libelf, zlib, json-c, and
Oniguruma development headers, make, and BPF loading privileges:

```sh
make test
make test-tokenizer TOKENIZER=/path/to/tokenizer.json
```

If Clang 19, libbpf with arena support, and bpftool are available, the
separate arena operator checks are `make test-arena-int4` and
`make test-arena-bf16`, each optionally with
`MODEL=/path/to/model.safetensors`. Without `MODEL`, each runs the synthetic
check only. `ARENA_LIBBPF_INCLUDE`, `ARENA_UAPI_INCLUDE`, and
`ARENA_LIBBPF` can point to an external recent libbpf build; the normal
build does not need these dependencies.

To build and run full-model inference with resident BF16 weights on that newer
toolchain, use `make build/infer-arena-bf16` and invoke it with the same
arguments as `build/infer`. The loader copies the complete BF16 model payload
into dynamically allocated BPF arena pages once; BPF reads matrix rows there
and does the BF16-to-Q24 lookup and dot. Budget roughly another 1.5 GB of
kernel memory for the official model, in addition to its file mapping and KV
cache. C still selects operators, supplies embeddings and other small weights,
and reads results, so this is not yet a user-space-free inference service.

The ordinary arena test loads an ephemeral BPF program and map; it attaches
to no network interface. To exercise the separate packet-triggered operator
on a disposable test network namespace, run `make test-arena-xdp`. That test
temporarily attaches XDP to the namespace's loopback interface, sends ten
UDP requests, waits for the BPF workqueue to calculate two resident BF16 rows
per request, checks the outputs against C, and detaches the BPF link before
exit. `make test-arena-xdp-model MODEL=/path/to/model.safetensors` instead
uses the official layer-0 Q-projection matrix: the kernel workqueue advances
through all 2,048 rows in 128-row batches after each packet. Packets alternate
two token IDs; the workqueue reads their embeddings from resident weights, and
the test compares both embeddings and all outputs with C references. Do not
run either XDP test in a production network namespace. They are not full-model
inference: the loader still supplies the resident weights and reads results,
and no normalization or subsequent decoder layers run in this event path.

To test against an actual Qwen3-0.6B tensor, obtain the official
`model.safetensors` separately and run:

```sh
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors --q24
./build/norm-smoke build/qwen3_norm.bpf.o /path/to/model.safetensors
# or run all model-backed operator checks:
make test-model MODEL=/path/to/model.safetensors
```

This reads the first 1,024 BF16 weights from layer 0's Q-projection matrix,
quantizes that row to Q8 or Q24 in the loader, and performs its dot product
against a deterministic synthetic activation in eBPF. The loader's C dot
product is used only as an independent assertion. This is still **not** a
transformer forward pass by itself; the driver below performs that path.

Run the complete forward path with input token IDs or text (up to the model's
40,960 position limit):

```sh
./build/infer /path/to/model.safetensors 0
./build/infer /path/to/model.safetensors 0 1 2 3
./build/infer /path/to/model.safetensors 0 1 --generate 2
./build/infer /path/to/model.safetensors \
  --tokenizer /path/to/tokenizer.json --prompt "Hello, world!" --generate 2
# Optional: write all 151,936 Q16 logits for an external comparison.
./build/infer /path/to/model.safetensors 0 --dump-logits logits.i32
```

The token-ID mode reports generated IDs. Text mode tokenizes the prompt in C,
prints decoded bytes to stdout, and sends progress/ID diagnostics to stderr.
For a chat prompt, include the Qwen3 control tokens and role delimiters in the
text supplied to `--prompt`; the CLI does not invent a chat template.
`--generate` defaults to 1 and stops early on Qwen's end-of-turn ID `151645`.
Set `QWEN3_TRACE=1` for intermediate range diagnostics. The tokenizer is
loaded from the official `tokenizer.json` at runtime; it is not vendored.

To score every next token in a UTF-8 excerpt, rather than only generating from
its final position, use:

```sh
./build/infer /path/to/model.safetensors \
  --tokenizer /path/to/tokenizer.json --score-file excerpt.txt
```

This reports the number of scored tokens, summed negative log-likelihood,
perplexity, and elapsed time. The first token supplies context and is not
scored. The score is for the exact supplied token sequence, not for an entire
benchmark corpus. To compare against the official BF16 model, place its
`config.json` and `tokenizer_config.json` alongside the weights and tokenizer
and run `python tests/reference_score.py MODEL_DIR excerpt.txt` in an
environment with PyTorch and Transformers. Both commands use the same text;
the reference runs on CPU, disables automatic special-token insertion, and
scores the same next-token positions. Neither command attaches BPF to a live
network or kernel event.

## Build and test targets

| Target | Purpose |
| --- | --- |
| `make` | Inference executable and the six default BPF operators. |
| `make help` | List build and test entrypoints. |
| `make test` | Existing synthetic checks, including standalone quantized operators; requires BPF loading privileges. |
| `make test-model MODEL=...` | Official-weight checks for matvec, INT4, INT8, and RMSNorm. |
| `make test-tokenizer TOKENIZER=...` | Tokenization/decoding vectors; no BPF loading. |
| `make experiments` | Build standalone matvec, INT4, INT8 operators and their test drivers. |
| `make build/infer-arena-bf16` | Build the optional full-model arena path and its operator dependencies. |
| `make test-arena-bf16` / `make test-arena-int4` | Optional arena operator checks, with `MODEL=...` for real weights. |
| `make test-arena-xdp` | Live loopback XDP → BPF workqueue → resident BF16 operator check in the current network namespace. |
| `make test-arena-xdp-model MODEL=...` | Ten alternating token-ID requests using resident embeddings and the first-layer Q-projection matrix, checked against C. |
| `make clean` | Remove only generated files under `build/`. |

Output names and CLI arguments are unchanged. Tests live in `tests/`, shared
headers in `include/`, and standalone BPF experiments in `experiments/`.
The arena targets are opt-in and require the newer toolchain described above.

## Contributing

Keep the default fixed-point inference path reproducible. For changes to an
operator, run its smoke test and `make test`. For tokenizer changes, run
`make test-tokenizer`. For inference changes, also compare token IDs and dumped
logits on the same model and prompts; an operator check is not whole-model
accuracy evidence. Describe the kernel/toolchain used and which checks were not
run. Keep new experimental backends separate until their scope is clear.
