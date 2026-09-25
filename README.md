# Qwen3-0.6B in Linux eBPF (experimental)

An experimental C/libbpf implementation of Qwen3-0.6B forward
computation in Linux eBPF. The current milestone runs **all 28 decoder layers
for multi-token contexts**, projects to the full vocabulary, and produces the
next token ID in the kernel. Greedy decoding can reuse the cache to produce
additional token IDs. The KV cache is shared through a memory-mapped BPF map;
the attention operator scans up to 256 prior positions per BPF invocation
using `bpf_loop`.
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
width. The original tile program remains an operator test. Full inference now
uses `src/qwen3_batch.bpf.c`: `bpf_loop` computes up to 16 complete matrix
rows per invocation. Its 204 KiB work map is memory-mapped into C, so the
driver writes weights and activations and reads results without per-batch map
update/lookup syscalls. The BF16 model file is mapped read-only once, avoiding
per-row file seeks, reads, and allocations; active rows are still converted to
Q24 in C for each forward pass. This is not yet a resident quantized-weight
or arena implementation.

`src/qwen3_int4.bpf.c` separately tests a 16-row, group-128 signed-INT4
matrix operator with packed nibbles and Q24 group scales. Its operator smoke
test checks the BPF result against the same packed calculation in C; it is not
used by the complete inference driver.

`src/qwen3_norm.bpf.c` implements RMSNorm in one BPF invocation. Two bounded
`bpf_loop` callbacks accumulate and apply up to eight 128-element tiles around
the integer-square-root step. Its work map is memory-mapped, so C can supply
the full vector and read the result without per-tile map syscalls. An earlier
monolithic implementation exceeded the verifier's one-million-instruction
processing budget on the test kernel; the callback-based version passed on
that kernel. RMSNorm weights use Q20 rather than Q24 because the official
model contains weights above Q24's representable range.
`src/qwen3_silu.bpf.c` approximates SiLU entirely with integer operations in
eBPF, without a user-space lookup or per-input host computation.
`src/qwen3_vector.bpf.c` implements residual addition, MLP gating multiply,
and output-logit argmax in eBPF. The host only supplies and retrieves tiles.
`src/qwen3_rope.bpf.c` rotates paired half-head dimensions in eBPF, and
`src/qwen3_attention.bpf.c` computes Q·K scores and an online, stable
softmax/V reduction for each prior position. It reads KV pairs from a
memory-mapped BPF map and traverses history in bounded `bpf_loop` chunks;
C writes each newly generated K/V pair directly to that map. `src/infer.c`
composes these operators with all model
tensors, including Q/K projection, QK normalization, RoPE, causal attention,
and grouped-query head sharing. The KV map is sized to the requested input
and generation length; the cached vectors and attention arithmetic are
produced in eBPF.
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
8, 16, and 257 positions; the worst maximum absolute error was `0.000905`.
The 257-position case crosses a 256-item `bpf_loop` chunk boundary and matches
the former one-KV-pair-per-invocation path byte for byte. It does not validate
a full 257-token model forward pass.

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

After the batched, mmap-backed matrix change, a same-host run in the same
Ubuntu 24.04 container measured 8.526 s for the previous tile driver and
3.695 s for the new driver on input token `0`. Their 151,936 Q16 logits
matched byte for byte. The new driver also produced ` This` for the
`Hello, world!` text prompt in 8.817 s, and generated `220, 16` from
`[0, 1]` in 6.866 s; both corresponding logit dumps matched the prior
kernel runs byte for byte. Each duration is a single run, not a throughput
distribution or proof of a general speedup. The inference workload remains
far slower than conventional optimized model inference.

With the mmap-backed KV cache and batched attention path, a further same-host
single run measured 2.790 s for input token `0` and 7.502 s for
`Hello, world!`. The `[0, 1]` two-token generation run took 5.736 s.
Their full-vocabulary logits matched the preceding kernel implementation
byte for byte. These isolated timings are not statistically stable speedup
estimates, and the cache still uses a dense map allocation proportional to
the requested context length. The kernel scans history in chunks rather
than one unbounded invocation; large-context capacity and latency have not
been validated.

Mapping the BF16 model file read-only removed the repeated file I/O syscalls.
In one-token `strace -c` runs, the earlier driver made 496,730 `lseek`,
291,467 `read`, and 143,667 `bpf` calls; the mapped driver made 143,667
`bpf` calls and only 20 `read` calls. A separate warm-cache one-token run
measured 2.912 s before and 2.695 s after this change, with byte-identical
full-vocabulary logits. These are individual observations, not a controlled
benchmark or a speedup guarantee; many operator dispatches remain.

Increasing the matrix batch from 4 to 16 rows reduced one-token `bpf`
syscalls from 143,667 to 50,667 in the same test container. Individual
one-token runs measured 2.695 s at 4 rows and 1.605 s at 16 rows. An
experimental 32-row version loaded and returned identical logits, but its
single measured 1.626 s did not establish a further benefit; the smaller
16-row work map is retained. The final 16-row path also reproduced the
earlier byte-identical logits for `Hello, world!` and `[0, 1] --generate 2`.
These timings are exploratory, not a controlled performance distribution.

Weight preparation now maps each of the 65,536 possible BF16 bit patterns to
its Q24 value once per process, then converts active matrix rows by lookup.
An exhaustive conversion test checks representable finite patterns against
the previous floating-point formula and rejects non-finite or out-of-range
values. Three interleaved one-token runs on the same host measured
1.425/1.576/1.476 s for the previous conversion and 1.154/1.258/1.049 s
for the lookup path. The latter produced byte-identical full-vocabulary Q16
logits for token `0`, `Hello, world!`, and `[0, 1] --generate 2`. These are
small exploratory samples, not a stable throughput or cross-host speed claim.
The lookup table occupies about 512 KiB; weight conversion still happens in C
for each active matrix row, not in BPF or in a resident weight arena.

Collapsing each RMSNorm from separate tile calls and three stages into a
single BPF invocation reduced one-token `bpf` syscalls from 50,667 to
43,171 on the same host. The 1,024-element and 128-element operator checks
passed; token `0`, `Hello, world!`, and `[0, 1] --generate 2` retained
byte-identical full-vocabulary logits. Three interleaved one-token runs
measured 1.291/1.190/1.204 s before and 1.289/1.282/1.246 s after the
change; two interleaved `[0, 1] --generate 2` runs measured 3.282/3.183 s
before and 3.198/3.115 s after. The call reduction is verified, but these
small, variable timings do not establish a latency improvement.

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

### Current boundary and limits

The model's dense arithmetic, attention reduction, and argmax run in BPF.
Tokenization and text decoding remain in C: the exact tokenizer loads a large
JSON vocabulary, uses Oniguruma regex splitting and dynamic BPE data
structures, and turns token IDs back into bytes. Moving that I/O path into
verified BPF would require a separate bounded implementation and would not
remove the dense matrix cost. C also loads BF16 tensors, converts each active
row to Q24, calculates RoPE trigonometric inputs, and dispatches operators.
These are real host-side responsibilities, not hidden kernel inference.

`bpf_loop` now batches 16 matrix rows, up to eight RMSNorm tiles, and up to
256 attention-history items per invocation. It does not turn a 28-layer model
into one BPF invocation: matrix batches, normalization calls, and
token-by-token generation still cross the user/kernel boundary. Bounded units
keep verifier complexity and per-invocation runtime manageable. The attention
smoke test crosses the 256-item boundary, but a full long-context model run
has not been validated.

The mmap-backed array is sufficient for the current shared working buffers;
an arena is not yet used for resident model weights. Arena allocation alone
would not make 0.6B parameters fit cheaply or remove their conversion cost.
The current KV layout reserves 1,024 bytes for each position/layer/KV-head
pair, about 224 KiB per position and 8.75 GiB at 40,960 positions, before
weights and other buffers. It may fail under real memory limits. Q24 is the
current arithmetic quantization, converted from the official BF16 file each
forward pass. One Q8 row test showed substantial error, so the code does not
advertise Q8 as a drop-in replacement.
Two exploratory variants were not retained. Converting BF16 to Q24 inside
the 3,072-column BPF loop failed verifier loading on the test kernel with
`The sequence of 8193 jumps is too complex`. Per-row scaled int16 weights
loaded and kept the same top token for input `0`, but the 151,936 logits had
mean absolute difference `0.00517` from the Q24 path, while two warm-cache
single-token runs took 3.30 s and 3.17 s versus 1.61 s for the Q24 path.
These are exploratory measurements. A separate self-contained Safetensors
variant preconverting all rank-2 BF16 tensors to I32 Q24 was also tested and
not retained. Its one-token logits matched the existing path byte for byte,
but its file grew from 1,503,300,328 to 3,006,434,093 bytes. Interleaved
same-host runs on input `0` took 5.18/3.93 s with the preconverted file
versus 3.22/2.37 s with the original BF16 file. The host had only about
4.6 GiB available memory during this test, so file size and cache pressure
may explain part of the regression; no isolated cause or general speed claim
is established. Conversion itself took about 8 s. This tests prepacked Q24,
not an efficient lower-bit representation; a reusable, smaller quantized
weight layout with whole-model accuracy and speed evidence remains open.

A further exploratory full-model INT4 path was tested and not retained as an
inference option. It prepacked the matrices used by the driver into
316,616,704 bytes of signed 4-bit weights plus group-128 Q24 scales, taking
1.928 s in one run before forward timing started. The INT4 BPF operator
passed the verifier and synthetic and real-weight row smoke checks; on one
official Q-projection row, the deterministic BF16 dot was `0.125738472`,
the quantized C dot `0.119238757`, and BPF `0.119232178`. However, the
complete model's first generated IDs changed for inputs `0`, `1`, and
`[9707, 11, 1879, 0]`: the Q24 path returned `9`, `14582`, and `1096`,
while this naive INT4 path returned `284`, `284`, and `21927`. Full-vocabulary
logit MAE versus Q24 was `1.837`, `1.396`, and `0.981`, respectively. The
corresponding single-run forward times were `0.956/0.995/3.197 s` for INT4
versus `1.243/1.229/3.546 s` for Q24, excluding INT4 prepacking. These
small samples show a possible kernel forward-speed benefit, not a usable
quantized model or end-to-end speedup. Group-32 scales and retaining the
BF16/Q24 vocabulary projection alone did not recover token `9` for input
`0`; quantizing only MLP matrices still changed it to `284`. A no-INT4
control with the same experimental driver returned `9`, so the mismatch was
not caused merely by loading the extra BPF object. A calibrated or mixed-
precision scheme needs full-model accuracy validation before integration.

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
