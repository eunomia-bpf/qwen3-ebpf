# Qwen3-0.6B in Linux eBPF (experimental)

An experiment toward running the **model's forward computation inside Linux
eBPF**, with a pure-C loader. This is not a finished Qwen3-0.6B inference
engine. The current milestone implements and checks one genuine in-kernel
integer dot-product primitive spanning Qwen3-0.6B's 1,024-wide hidden vector.
The host supplies tiles and invokes the BPF program; it does not compute the
tested dot product. No model weights are distributed here.

## Current experiment

`src/qwen3_matvec.bpf.c` runs integer multiply-accumulate in a socket-filter
BPF program, with Q8×Q8 and Q16-activation×Q24-weight paths.
`src/matvec-smoke.c` invokes it with `bpf_prog_test_run_opts`,
checks the map result after eight 128-element tiles, and compares against a C
reference. This test establishes that the arithmetic ran in the kernel BPF VM;
it does **not** establish that an LLM token can yet be generated.

On a Linux host with clang's BPF target, libbpf development headers, make,
and BPF loading privileges:

```sh
make test
```

The test loads an ephemeral BPF program and map; it attaches to no network
interface and installs nothing persistently.

To test against an actual Qwen3-0.6B tensor, obtain the official
`model.safetensors` separately and run:

```sh
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors
./build/matvec-smoke build/qwen3_matvec.bpf.o /path/to/model.safetensors --q24
```

This reads the first 1,024 BF16 weights from layer 0's Q-projection matrix,
quantizes that row to Q8 or Q24 in the loader, and performs its dot product
against a deterministic synthetic activation in eBPF. The loader's C dot
product is used only as an independent assertion. This is still **not** a
transformer forward pass or a generated token.

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

## Full-model target and hard problems

The target is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), not a
toy transformer: 28 decoder layers, 1,024 hidden width, grouped-query
attention, QK normalization, RoPE, RMSNorm, SiLU, KV cache, and a 151,936-token
vocabulary. The next implementation steps are to ingest all required model
tensors, define a whole-model quantization/error budget, add the remaining
operators in eBPF, and
orchestrate bounded BPF invocations for complete token generation. Integer
approximations of nonlinear operations and verifier/runtime limits require
measurement. The user-space driver may load weights, tokenize, invoke BPF,
and read output, but cannot substitute user-space model math for the kernel
forward pass. A full-model claim requires an actual generated token checked
against a known Qwen3 reference, with kernel verifier and runtime evidence.

This is a research prototype. It is not intended for production kernels or
performance-sensitive traffic. The project code is MIT licensed; Qwen model
weights, if obtained separately, retain their own Apache-2.0 license.

## Related work and novelty boundary

CPU-only C Qwen3 implementations already exist, as do CUDA Qwen3-0.6B
implementations. KernelX puts an eBPF signal path in front of a user-space
LLM, while published eBPF work has run much smaller neural networks in the
kernel. As of September 2026, our search did not find a public full
Qwen3-0.6B Linux-eBPF forward-pass implementation. That is a search result,
not proof of absolute novelty.

- [Qwen3-0.6B model and configuration](https://huggingface.co/Qwen/Qwen3-0.6B)
- [qwen3.c, CPU-only C](https://github.com/adriancable/qwen3.c)
- [qwen3.cu, CUDA](https://github.com/gigit0000/qwen3.cu)
- [KernelX, eBPF/user-space LLM bridge](https://github.com/pie-314/KernelX)
- [Linux BPF verifier documentation](https://docs.kernel.org/bpf/verifier.html)
