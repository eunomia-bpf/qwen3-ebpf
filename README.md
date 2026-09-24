# Qwen3-0.6B in Linux eBPF (experimental)

An experimental C/libbpf implementation of Qwen3-0.6B forward
computation in Linux eBPF. The current milestone runs **all 28 decoder layers
for multi-token contexts**, projects to the full vocabulary, and produces the
next token ID in the kernel. Greedy decoding can reuse the cache to produce
additional token IDs. Its attention operator processes one historical
KV pair per BPF invocation, with a C-managed cache sized to the input length.
The host loads and quantizes official BF16 weights, dispatches bounded
BPF tiles, and reads results; model arithmetic and argmax run in eBPF. A C
ByteLevel/BPE tokenizer handles text at the edge. No model weights or tokenizer
data are distributed here.

## Current experiment

`src/qwen3_matvec.bpf.c` runs integer multiply-accumulate in a socket-filter
BPF program, with Q8×Q8 and Q16-activation×Q24-weight paths.
`src/matvec-smoke.c` invokes it with `bpf_prog_test_run_opts`,
checks the map result after bounded 128-element tiles, and compares against a C
reference. That operator test alone does not establish model inference; the
separate inference driver below connects all decoder layers.
The same BPF program also accepts a caller-supplied tile count; `make test`
now checks a 3,072-element path (24 tiles), matching Qwen3's MLP intermediate
width, while model-backed Q-projection uses eight tiles.

`src/qwen3_norm.bpf.c` implements RMSNorm as separate accumulate, finalize,
and apply BPF programs. The split matters: a combined accumulation and
branching integer-square-root program exceeded the verifier's one-million
instruction processing budget on the test kernel. The finalized version keeps
the RMS computation in eBPF while bounding each verification unit. RMSNorm
weights use Q20 rather than Q24 because the official model contains weights
above Q24's representable range.
`src/qwen3_silu.bpf.c` approximates SiLU entirely with integer operations in
eBPF, without a user-space lookup or per-input host computation.
`src/qwen3_vector.bpf.c` implements residual addition, MLP gating multiply,
and output-logit argmax in eBPF. The host only supplies and retrieves tiles.
`src/qwen3_rope.bpf.c` rotates paired half-head dimensions in eBPF, and
`src/qwen3_attention.bpf.c` computes Q·K scores and an online, stable
softmax/V reduction for each prior position. `src/infer.c` composes these operators with all model
tensors, including Q/K projection, QK normalization, RoPE, causal attention,
and grouped-query head sharing. The C-managed KV cache is sized to the prompt;
the cached vectors and attention arithmetic are produced in eBPF.
For a one-token context, attention softmax has exactly one entry and is exactly
1. Longer contexts exercise the actual Q/K and attention path.

On a Linux host with clang's BPF target, libbpf, libelf, zlib, json-c, and
Oniguruma development headers, make, and BPF loading privileges:

```sh
make test
make test-tokenizer TOKENIZER=/path/to/tokenizer.json
```

The test loads an ephemeral BPF program and map; it attaches to no network
interface and installs nothing persistently.

To test against an actual Qwen3-0.6B tensor, obtain the official
`model.safetensors` separately and run:

```sh
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors --q24
./build/norm-smoke build/qwen3_norm.bpf.o /path/to/model.safetensors
# or run both model-backed checks:
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

First measured run (2026-09-24): Linux 6.17.0 arm64, Ubuntu 24.04 build
container, BPF program accepted by the kernel verifier. The synthetic row
returned `-1345`. For the official layer-0 Q-projection row, Q8 gave
`-0.0366821289` and Q24 gave `-0.0083770752`, versus the original BF16 C
reference `-0.00837016106`. Both integer accumulators matched independent C
assertions across eight BPF invocations. This one-row result shows why Q8 is
insufficient for cancellation-heavy operations; it does not bound whole-model
error. The downloaded model file's SHA-256 was
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
it is not included in this repository.
The official tokenizer used for validation had SHA-256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

The layer-0 input RMSNorm check with its actual BF16 scale weights passed on
the same kernel, with maximum absolute error `7.4e-05` across 1,024 elements
against a C floating-point reference for a deterministic input vector. This
is a one-vector operator test, not a complete-layer accuracy guarantee.
The SiLU check passed with maximum absolute error `0.000634` across 1,024
inputs in `[-8, 8]` against a C floating-point reference. The inference
driver combines it with the MLP gate and up projections.
The 128-wide Q/K RMSNorm check passed with maximum absolute error `0.000597`;
RoPE's test at positions 0, 1, 7, and 63 had maximum error `1.56e-05`.
The online attention operator passed synthetic one-head tests with 1, 2, 4,
8, and 16 positions; the worst maximum absolute error was `0.000905`.

The full layer-0 V-projection matrix (1,024 rows, 1,048,576 MACs) then ran
through the BPF matvec with official weights and deterministic activations.
Its maximum per-row absolute error against the independent BF16 C reference
was `1.53e-05`; the measured operator test took `0.018 s` on the test host.
The model file is opened once and its tensor offset resolved once for this
matrix. This timing excludes model download, compilation, and the rest of a
decoder layer; it is not an LLM throughput claim.

The end-to-end one-token run used the same official weights on Linux
6.17.0 arm64. Input token IDs `0` and `1` produced IDs `9` and `14582`,
respectively, matching Hugging Face Transformers 5.14.1 BF16 reference runs.
Across all 151,936 logits, the kernel Q16 output versus that reference had
mean absolute error `0.052` / `0.030` and RMSE `0.065` / `0.038` for the
two inputs. The top three IDs agreed for both. Measured forward times were
8.47 s and 8.57 s, excluding model download, compilation, and container
startup. These are two-input experimental checks, not a general accuracy or
performance guarantee.

The end-to-end `[0, 1]` context returned next token ID `220`, matching the
official Transformers 5.14.1 BF16 model. Across its 151,936 logits, mean
absolute error was `0.024` and RMSE `0.030`; the top three token IDs agreed.
The measured two-position forward pass took `23.38 s`, excluding model
download and compilation. This is one two-token validation input, not a
general-context accuracy guarantee.

The end-to-end `[0, 1, 2, 3]` context returned next token ID `2`, also
matching the official BF16 model. Across all 151,936 logits, mean absolute
error was `0.019` and RMSE `0.024`; the top four IDs agreed. The measured
four-position forward pass took `29.43 s`. These checks establish a real
multi-position path, not accuracy for all possible prompts or lengths.

With input `[0, 1]` and `--generate 2`, greedy cache-reusing decoding returned
`220, 16`. The official BF16 model agreed on both IDs; for the second output
(context `[0, 1, 220]`), full-vocabulary MAE was `0.025` and RMSE `0.031`.
This verifies one incremental step, not long-form generation quality.

The C tokenizer matched the official tokenizer on English, Chinese, emoji,
whitespace, mixed text, and chat control-token vectors, including byte-level
decode roundtrips. The text prompt `Hello, world!` became the official IDs
`[9707, 11, 1879, 0]` and generated ` This` (ID `1096`). The official BF16
model agreed on that ID and all top-five IDs; full-vocabulary MAE was `0.035`
and RMSE `0.044`. This is one text-prompt accuracy check.

## General-context target and hard problems

The target is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), not a
toy transformer: 28 decoder layers, 1,024 hidden width, grouped-query
attention, QK normalization, RoPE, RMSNorm, SiLU, KV cache, and a 151,936-token
vocabulary. The multi-position path uses the full model's weights and all
28 layers. Remaining research includes broader numerical validation, long
contexts, generation quality, verifier portability, and throughput. The
four-token result must not
be extrapolated to arbitrary contexts. The user-space driver may load weights,
tokenize, invoke BPF, and read output, but cannot
substitute user-space model math for kernel forward computation.

This is a research prototype. It is not intended for production kernels or
performance-sensitive traffic. The project code is MIT licensed; Qwen model
weights, if obtained separately, retain their own Apache-2.0 license.

## Related work and novelty boundary

CPU-only C Qwen3 implementations already exist, as do CUDA Qwen3-0.6B
implementations. An existing eunomia-bpf tutorial profiles a CUDA Qwen3
engine with eBPF; its model arithmetic runs on the GPU, not in eBPF. KernelX
puts an eBPF signal path in front of a user-space LLM, while published eBPF
work has run much smaller neural networks in the kernel. As of September 2026,
our search did not find a public full
Qwen3-0.6B Linux-eBPF forward-pass implementation. That is a search result,
not proof of absolute novelty.

- [Qwen3-0.6B model and configuration](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3-0.6B tokenizer.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer.json)
- [Hugging Face ByteLevel mapping source](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs)
- [qwen3.c, CPU-only C](https://github.com/adriancable/qwen3.c)
- [qwen3.cu, CUDA](https://github.com/gigit0000/qwen3.cu)
- [eunomia-bpf CUDA Qwen3 profiling tutorial](https://github.com/eunomia-bpf/bpf-developer-tutorial/blob/main/src/xpu/flamegraph/README.md)
- [KernelX, eBPF/user-space LLM bridge](https://github.com/pie-314/KernelX)
- [Linux BPF verifier documentation](https://docs.kernel.org/bpf/verifier.html)
