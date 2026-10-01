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
    parser.add_argument("--print-token-ids", action="store_true")
    args = parser.parse_args()

    with open(args.text_file, encoding="utf-8") as sample:
        text = sample.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if args.print_token_ids:
        print("token_ids=" + ",".join(map(str, inputs[0].tolist())))
    if inputs.shape[1] < 2:
        parser.error("at least two tokens are needed")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, local_files_only=True
    ).eval()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            inputs[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        final_logits = logits[0, -1]
        next_token = final_logits.argmax().item()
        top_logit = final_logits[next_token].item()
        top_values, top_ids = final_logits.topk(5)
        top_five = ",".join(
            f"{token_id.item()}:{value.item():.6f}"
            for token_id, value in zip(top_ids, top_values)
        )
    elapsed = time.perf_counter() - started
    scored = inputs.shape[1] - 1
    print(
        f"scored_tokens={scored} nll_sum={nll:.9f} "
        f"perplexity={math.exp(nll / scored):.9f} "
        f"last_input_token={inputs[0, -1].item()} "
        f"next_token={next_token} top_logit={top_logit:.9f} "
        f"top_five={top_five} "
        f"elapsed={elapsed:.3f} s"
    )


if __name__ == "__main__":
    main()
