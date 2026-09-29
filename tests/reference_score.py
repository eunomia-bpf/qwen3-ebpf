"""Score the same UTF-8 excerpt with the official Qwen3 model on CPU.

Usage: python tests/reference_score.py MODEL_DIR TEXT_FILE
MODEL_DIR must contain the official model.safetensors, config.json,
tokenizer.json, and tokenizer_config.json. Nothing is downloaded.
"""

import argparse
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("text_file")
    args = parser.parse_args()

    with open(args.text_file, encoding="utf-8") as sample:
        text = sample.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if inputs.shape[1] < 2:
        parser.error("at least two tokens are needed")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs).logits[:, :-1, :].float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            inputs[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
    elapsed = time.perf_counter() - started
    scored = inputs.shape[1] - 1
    print(
        f"scored_tokens={scored} nll_sum={nll:.9f} "
        f"perplexity={math.exp(nll / scored):.9f} elapsed={elapsed:.3f} s"
    )


if __name__ == "__main__":
    main()
