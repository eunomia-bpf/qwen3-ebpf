# Experiments and recorded validation

[Back to the project](../README.md) · [Architecture](architecture.md) · [Usage](usage.md)

These are the original development measurements, preserved for reproducibility.
They cover multiple revisions and limited inputs, not a single benchmark of the
current tree. No new performance or accuracy measurements are implied by this
repository reorganization.

## Measurements and validation

### Fixed-excerpt next-token scoring (2026-09-29)

The current full-model driver scored two text excerpts from the
WikiText-2 *raw* test split, using the official Qwen3-0.6B BF16 weights and
tokenizer. The source Parquet file was
[`Salesforce/wikitext` test-00000-of-00001.parquet](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/wikitext-2-raw-v1/test-00000-of-00001.parquet),
SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
One excerpt concatenated rows 3 and 4; the other used row 1011, the first row
at or after 1000 with 500–750 characters. The UTF-8 excerpt file hashes were
`f52e5b6da11cbe8bd5b077ec943d0932823c47df841de85d80582a172d93f5ef`
and `b53e6aa42d41d932cae2a7e62422c24c59d52e5682709634d3a5cfbf948479b4`.
The excerpts were written in Windows text mode, so line endings in the files
are CRLF; both implementations received the same bytes.
The first input token was context, not a scored target; no BOS or chat template
was inserted.

| Test excerpt | Scored tokens | eBPF Q16 NLL / perplexity / seconds | Official BF16 CPU NLL / perplexity / seconds |
| --- | ---: | ---: | ---: |
| Rows 3–4 | 423 | 1358.512 / 24.819 / 401.580 | 1344.427 / 24.006 / 6.407 |
| Row 1011 | 124 | 352.885 / 17.216 / 114.904 | 342.837 / 15.876 / 3.674 |

Runs used Linux `6.17.0-1029-nvidia` on arm64. The BF16 reference was
`tests/reference_score.py` with PyTorch `2.11.0` and Transformers `5.14.1`,
limited to four CPU cores with `OMP_NUM_THREADS=4`; the eBPF driver used its
normal single-threaded operator dispatch. Both timings start *after* model
loading. They are single runs, not a controlled throughput benchmark, and the
CPU parallelism differs: the BF16 reference processes all positions in one
teacher-forced forward pass while eBPF advances one position at a time. The
timings therefore do not form a matched online-generation comparison. The
eBPF path is much slower in this scoring setup; there is no measured speed
benefit. The eBPF perplexities are higher than BF16 on both
excerpts. This is a repeatable fixed-excerpt check, **not** whole-test-set
perplexity, generation quality, or evidence that a live kernel event is useful.
The model and tokenizer SHA-256 values are recorded below.

### Resident BF16 arena (2026-09-29)

On the same Linux 6.17 arm64 test host, the optional arena backend dynamically
allocated pages for the official model's 1.5 GB tensor payload and copied it
once during loading. The arena BPF operator then read matrix rows by their
Safetensors offsets rather than receiving a new weight batch from C for each
call. A two-row synthetic resident test and a 128-row official-weight operator
test passed. Complete 28-layer runs for token `0`, a two-token `Hello`
generation, and input IDs `0` through `9` produced the same generated IDs
and byte-identical final 151,936-entry Q16 logit files as the default path.

Single-run wall-clock measurements, including model loading, were 1.412 s
(default) versus 2.176 s (resident) for one input token, and 7.573 s versus
7.852 s for ten input tokens. The corresponding one-token forward-only times
were 1.213 s and 1.089 s. These are exploratory runs under a shared-host
load, not evidence of an end-to-end speedup: the resident path pays for arena
allocation and copying up front, while C still dispatches every operator.
The full-model driver is not attached to a live socket or XDP hook, and the
28-layer schedule remains in user space. A separate live-loopback test on the
same kernel sent one UDP packet through XDP, queued a BPF work item, and
checked a two-row, 128-column resident BF16 matvec against independent C
results. Two packet-to-result observations were 0.120 and 1.180 ms; these
are only operator-path measurements, not full-model latency or speedup
claim. The test loader supplied the input and read the map result.

The next event-path check used the official layer-0 Q-projection tensor
(2,048 × 1,024 BF16 weights) and one deterministic Q16 activation. Ten
consecutive loopback UDP requests each advanced all 16 BPF workqueue batches
without a user-space dispatch between batches. All 2,048 outputs per request
matched the independent C Q24 dot products. On one shared-host run, median
wall time was 2.124 ms for ten sets of 16 `BPF_PROG_RUN` dispatches and
2.211 ms from packet send to completion observed by the test; the XDP-to-BPF
completion timestamp median was 2.125 ms. The paths have different timing
boundaries and the host was not isolated. This establishes neither a speedup
nor complete model inference; the loader still supplies activations, and the
28-layer schedule is in C.

A follow-up test moved the activation source into BPF: each packet supplied
token ID `0` or the final valid ID `151935`, and the work item read the
corresponding 1,024-element embedding from a resident copy of the official
embedding tensor before the same Q-projection. Ten alternating requests passed
exact Q16 embedding and
2,048-output Q24 comparisons with C; invalid magic and out-of-range token IDs
did not start work. One shared-host run gave median 2.085 ms for the
user-dispatched matrix batches and 2.158 ms for packet-to-result observation
(2.092 ms by BPF timestamps). These are small, non-isolated samples with
different timing boundaries, not evidence of a reliable speedup. That
revision did not apply RMSNorm or execute the remaining decoder layers.

The following revision added the first layer's input RMSNorm to the workqueue
path. The event now reads the official BF16 norm weights from the arena,
converts them to Q20, and normalizes the token embedding before Q projection.
For ten alternating requests over the first and last valid token IDs, the
1,024 raw embeddings, 1,024 normalized values, and all 2,048 projection
outputs per request matched a separate C implementation of the same fixed-point
arithmetic. Invalid token IDs and bad request markers did not start work.
This remains only the beginning of one decoder layer; the previous direct
matrix timings do not include the embedding and RMSNorm work now present in
the event path, so they are not a matched performance comparison.

The event state machine now carries that normalized vector through the first
layer's Q, K, and V projections without a user-space dispatch between the
32 matrix batches. Ten alternating real-token requests matched separate C
Q24 reference results for all 4,096 projection rows per request. In one
shared-host run, the 32 direct `BPF_PROG_RUN` matrix calls had a median of
4.034 ms; packet-to-result observation had a median of 4.200 ms (4.126 ms
from XDP to completion by BPF timestamps). The direct timing excludes
embedding and normalization, so these are not matched full-request timings
and do not establish a speedup. Q/K norm, RoPE, attention, MLP, later layers,
and final-token delivery remain outside this event path.

The next increment added head-wise Q/K RMSNorm after the same 4,096 Q/K/V
projection rows. The XDP work item reads the official first-layer Q/K norm
weights from the BF16 arena and normalizes all 16 query and 8 key heads in
kernel space; V remains the raw projection. Ten alternating real-token
requests matched separate C fixed-point references for both the unnormalized
projection rows and all normalized Q/K outputs. The previously reported direct
matrix timing excludes this additional work and is not a matched comparison.
RoPE, attention, output projection, residual, MLP, later layers, and final
token selection are still absent from the event path.

The following XDP increment sends a position alongside the token ID. BPF
generates RoPE sine/cosine coefficients from its resident inverse-frequency
constants and rotates all first-layer Q/K heads after normalization. Ten
requests at positions 0, 1, 128, and 40,959 (two alternating real token IDs)
matched the C reference coefficient arrays and all 4,096 Q/K/V outputs
exactly in Q16/Q20 fixed-point arithmetic. An out-of-range position did not
start work. The observed packet-to-result timings remain operator-prefix
measurements, not a full-model speed comparison. Attention, output projection,
residual, MLP, later layers, and final token selection remain outside the
XDP-triggered path.

The first-layer attention increment stores each packet's rotated K and raw V
in a kernel BPF map, then has the workqueue process prior positions for all
16 query heads. On the same Linux 6.17 arm64 test container, ten packets
repeating positions 0 and 1 matched a separate C fixed-point reference for
all 2,048 attention outputs per packet; the independent floating-point
two-token softmax comparison had maximum absolute error below `1.3e-4`.
The test also checked each cached K/V element and rejected a skipped position.
Observed packet-to-result times were about 4.1–4.2 ms for this prefix,
including projection, normalization, RoPE, and attention; the direct matrix
timing excludes most of that work and is not a speedup comparison. The
event path still lacks output projection, residual/MLP, later layers,
vocabulary argmax, and result delivery to the network client.

The next increment loaded the official first-layer output-projection weights
into the same BPF arena. After the attention work item finishes, the kernel
copies its 2,048 outputs into the resident matrix input, schedules all 1,024
output-projection rows, and adds those results to the original hidden vector.
Ten alternating two-token requests matched the C Q24 projection and Q16
residual element by element. The shared-host packet-to-result observations
were about 6.1–6.3 ms for this longer prefix; the direct matrix-only timing
does not cover equivalent work and provides no speedup evidence. At that
point the event path still lacked post-attention normalization, MLP, later
layers, vocabulary argmax, and network result delivery.

The following increment loaded the official post-attention RMSNorm weights
into the arena. The XDP-triggered workqueue normalizes the post-projection
residual entirely in BPF; ten two-token requests matched an independent C
fixed-point reference element by element. This extends the first-layer prefix
but still does not include MLP, later layers, final token selection, or
network result delivery. The observations remain unmatched to a full-request
user-space baseline and do not establish a speedup.

The next step kept the normalized residual in the work map and scheduled the
official first-layer MLP gate and up projections, 3,072 rows each, entirely
from the BPF workqueue. Ten two-token packets matched independent C Q24
matrix calculations for every row. Packet-to-result observations varied from
about 12 to 26 ms on the shared test host; these include the longer prefix and
are not a matched full-request comparison. SiLU, the gate/up product, down
projection, later layers, argmax, and network result delivery remain outside
the event path.

The first-layer path now also applies SiLU to the gate projection, multiplies
it by the up projection, schedules the 1,024-row down projection from the
3,072-element product, and adds the result to the hidden vector. The ten
packet-triggered two-token requests matched C fixed-point references for the
MLP product, down projection, and final layer hidden state element by element.
This 12,288-row prefix took about 15.1–15.3 ms per packet on the shared
container in this run. It is not a full-model result or a matched user-space
comparison; the remaining 27 layers, vocabulary argmax, and network return
path are not implemented in the event scheduler.

The next state-machine increment added resident per-layer tensor offsets and
layer-indexed KV slots. A packet at position zero now drives 28 consecutive
decoder layers without user-space operator scheduling. The official BF16 model
payload is preloaded into the BPF arena by the test loader. Each 2–28-layer
prefix matched an independent C fixed-point reference for its final 1,024
hidden values and last-layer K/V; layer zero and last-layer cache slots were
checked separately. The reference initially contained an unsigned/signed
multiplication error, exposed at layer four, and passed only after that error
was corrected. This does not yet establish multi-token full-model correctness,
vocabulary argmax, a network reply, or a matched performance benefit.

The full event continuation now computes final RMSNorm and all 151,936
vocabulary logits in BPF, records the argmax, and returns it on a subsequent
XDP UDP poll. Three repeated full-model packet requests returned token `9`
and Q16 logit `739605`, matching both an independent C fixed-point reference
and the existing full-model arena CLI. The XDP result was received over UDP,
not merely read from a map. On the same Linux 6.17 arm64 host, with the process
and workqueue on Cortex-X925 CPU 5, packet-send-to-result was 580.090–580.711
ms, versus 0.813–0.817 s for the existing arena CLI pinned to CPU 5 (about
29% lower latency). On Cortex-A725 CPU 0, the corresponding ranges were
1221.591–1222.465 ms and 1.858–1.860 s (about 34% lower latency). Both clocks
start after model loading; the XDP number includes the result poll, while the
CLI number ends when the token is calculated. These are single-token,
single-session observations, not a claim about multi-token quality, sustained
throughput, or production network performance. Unpinned workqueue placement
changed latency by roughly 2× on this heterogeneous CPU, so CPU placement
must be controlled in comparisons.

The standalone XDP loader was then built with the same official BF16 model
and attached to loopback in the isolated test container. A live UDP request
for token `0` at position zero returned token `9` and Q16 logit `739605`;
SIGINT detached XDP. In a separate loader session, four consecutive packets
for token IDs `[0, 1, 2, 3]` at positions zero through three each completed
and returned a UDP result. At position one, the event path returned token
`220`, Q16 logit `746817`, exactly matching `infer-arena-bf16` for `[0, 1]`.
At position three, it returned token `2`, Q16 logit `806121`, again exactly
matching the same CLI for `[0, 1, 2, 3]`. This validates KV reuse and final
argmax for these short multi-token sequences through the standalone live XDP
path; it does not establish long-context quality, full-vocabulary logit
agreement for every position, concurrent sessions, or generation quality.

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
The SiLU check passed with maximum absolute error `0.000634` across 128 and
3,072 inputs in `[-8, 8]` against a C floating-point reference. The inference
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

With the then-mmap-backed KV cache and batched attention path, a same-host
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
single measured 1.626 s did not establish a further benefit at that point.
The 16-row path also reproduced the earlier byte-identical logits for
`Hello, world!` and `[0, 1] --generate 2`.
These timings are exploratory, not a controlled performance distribution.

On 2026-09-25, larger `bpf_loop` batches were tested on the same host with
the accurate Q24 path. The 128-row version loaded, passed the matrix smoke
test, and produced byte-identical full-vocabulary logits to 16 and 256 rows
for `Hello, world!` and `[0, 1] --generate 2`; the latter also generated the
same two token IDs. One-token `bpf` calls fell from 43,168 at 16 rows to
16,043 at 128 rows. Interleaved `Hello, world!` timings were 3.624/3.733 s
at 16 rows and 3.276/3.208 s at 128 rows. Additional interleaved 128- and
256-row timings overlapped, so 128 rows is retained with roughly half the
work-map size of 256 rows. These are small, noisy same-host samples, not a
general throughput claim; the one-token timing did not show a stable gain.

The vocabulary projection now accumulates argmax inside the same BPF matrix
callbacks. On the same host, this reduced one-token `bpf` calls from 16,043
to 12,482 without changing any full-vocabulary logits or generated IDs for
token `0`, `Hello, world!`, or `[0, 1] --generate 2`. Six interleaved
`Hello, world!` runs measured 3.029/3.184/3.212 s before and
3.380/3.074/3.116 s after; the syscall reduction is clear, but these
samples do not establish a latency improvement.

The residual-add and MLP-multiply operators now process a complete 1,024-
or 3,072-element vector per invocation rather than making a map update,
test-run, and lookup for every 128-element tile. On the same host, one-token
`bpf` calls fell from 12,482 to 9,205. The 128- and 3,072-element operator
tests passed, and token `0`, `Hello, world!`, and `[0, 1] --generate 2`
retained byte-identical full-vocabulary logits. This establishes fewer
syscalls, not a stable latency gain.

RoPE now batches all 24 query/key heads per layer in one BPF invocation,
using a memory-mapped work map and one shared set of trigonometric inputs.
One-token `bpf` calls fell from 9,205 to 7,217. Operator tests passed for
one and 24 heads at positions 0, 1, 7, and 63; token `0`, `Hello, world!`,
and `[0, 1] --generate 2` retained byte-identical full-vocabulary logits.
Six interleaved `Hello, world!` runs measured 3.195/3.256/3.064 s before
and 3.130/3.168/3.176 s after. The syscall reduction is clear, but the
latency samples overlap.

SiLU now computes all 24 MLP tiles in one BPF invocation instead of making
one map update, test-run, and lookup per tile. The 128- and 3,072-element
operator checks passed on the same kernel. Token `0` and `Hello, world!`
retained byte-identical full-vocabulary logits, and `[0, 1] --generate 2`
still produced IDs `220, 16`. One-token `bpf` calls fell from 7,217 to
5,229. This is a measured syscall reduction, not a controlled wall-clock
speedup claim.

Q/K RMSNorm now processes all 16 query and eight key heads in one bounded
`bpf_loop` invocation per layer. An official-weight operator check matched
the previous 24 separate BPF calls element for element; complete-vocabulary
logits for token `0`, `Hello, world!`, and a two-token generation check were
byte-identical. With both versions loading the same BPF object, one-token
`bpf` calls fell from 5,231 to 4,587, exactly 644 fewer calls across 28
layers. Three interleaved warm one-token runs took 1.204/1.242/1.199 s
before and 1.221/1.138/1.209 s after. The calls fell, but these timings
do not establish a latency gain.

Cached attention now processes all 16 query heads in one BPF invocation per
256-position chunk, reading the same BPF KV array as before and reusing
each KV lookup for its two query heads. An operator check across all heads
and 257 positions matched the per-head BPF path element for element. Token
`0`, `Hello, world!`, and two-token generation
retained byte-identical full-vocabulary logits. One-token `bpf` calls fell
from 4,587 to 4,169 across the two builds; the new object adds two setup
calls, while the execution path saves 15 calls per layer, or 420 per token.
Three warm one-token runs of the new path took 1.186/1.017/1.016 s; these
small, non-interleaved samples do not establish a latency improvement.

The attention program now stores each newly projected K/V pair in the BPF
KV array before scanning history. The host supplies the new vectors through
the small attention work map but no longer maps or writes the cache directly.
An eight-KV-head operator test verified the stored values and the same-call
attention output; token `0`, `Hello, world!`, and two-token generation kept
byte-identical full-vocabulary logits. The one-token `bpf` call count stayed
at 4,169. This moves cache maintenance into BPF, not the whole scheduling
loop, and has not established a latency gain.

Weight preparation now maps each of the 65,536 possible BF16 bit patterns to
its Q24 value once per process, then converts active matrix rows by lookup.
An exhaustive conversion test checks representable finite patterns against
the previous floating-point formula and rejects non-finite or out-of-range
values. Three interleaved one-token runs on the same host measured
1.425/1.576/1.476 s for the previous conversion and 1.154/1.258/1.049 s
for the lookup path. The latter produced byte-identical full-vocabulary Q16
logits for token `0`, `Hello, world!`, and `[0, 1] --generate 2`. These are
small exploratory samples, not a stable throughput or cross-host speed claim.
The lookup table occupies about 512 KiB. At that revision, weight conversion
still happened in C for each active matrix row; the later resident arena path
is measured separately above.

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

## Alternatives not integrated into inference

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

A separate group-32 INT8 experiment also passed the verifier and its
128-row synthetic operator test. On one official Q-projection row, BF16, C
INT8, and BPF INT8 dots were `0.125738472`, `0.125476453`, and
`0.125473022`. A temporary full-model driver, not retained in `infer`,
generated the same first IDs as Q24 for input tokens `0` and `1` (`9` and
`14582`), but for the exact text `Hello, world!` it generated `20166`
instead of Q24's `1096`. On that four-token prompt, measured end-to-end
times were `8.027 s` for on-the-fly INT8 quantization and `3.310 s` for
Q24 on the same host; the respective one-token `0` times were `2.630 s`
and `1.183 s`. These are single runs, not a throughput benchmark. Keeping
MLP matrices at Q24 still changed the four-token prompt's first ID to
`20166`; using finer 16-weight INT8 groups did not establish fidelity.
An INT8 format may be useful with calibration, better mixed precision, and
reusable prepacking, but this experiment is not a drop-in replacement.

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
