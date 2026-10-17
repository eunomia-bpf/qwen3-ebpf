# Paper

[bpfllm.pdf](bpfllm.pdf) is a four-page workshop-style report on this
project: *Can eBPF Run a Real Language Model? Lessons from a Kernel-Resident
Qwen3 Forward Pass*. The report calls the system **bpfllm**. It covers the
operator and XDP event modes, the techniques that got the model through the
verifier, fidelity against the official BF16 model, and why the kernel path
is slower than one user-space CPU core.

Every number in the report comes from the recorded measurements in
[docs/experiments.md](../docs/experiments.md). Multiply-accumulate counts and
rates are derived from the Qwen3-0.6B configuration and those timings, and
the report labels them as such.

## Build

The source is plain LaTeX with the ACM `acmart` class and TikZ; it has no
external figure or bibliography files.

```sh
cd paper
latexmk -pdf bpfllm.tex
```

On Debian or Ubuntu, `texlive-publishers` (acmart), `texlive-fonts-extra`
(the Libertine and newtx fonts acmart uses), `texlive-pictures` (TikZ),
`texlive-latex-recommended`, `texlive-latex-extra` and `latexmk` provide
everything it needs.

## Related write-ups

- Blog: [English](https://eunomia.dev/blog/2026/09/24/qwen3-ebpf-in-kernel-inference/)
  · [中文](https://eunomia.dev/zh/blog/2026/09/24/qwen3-ebpf-in-kernel-inference/)
