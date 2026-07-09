#!/usr/bin/env python
"""
Generate DNA sequences from a trained expression-conditioned model checkpoint.

Reproduces the inference stage of `tutorial/tutorial_3cl_human_enhancer.ipynb`
as a single non-interactive command. Reads a CSV of prompts, samples one
sequence per prompt in batches, and writes the results to a CSV.

The input CSV must contain an `instruction` column with the natural-language
prompts (for example produced by `scripts/train_expression_model.py`'s prompt
template). Any additional columns are carried through to the output.

Example
-------
    python scripts/generate_sequences.py \
        --checkpoint runs/three_cell_line/.../best.ckpt \
        --prompts data/three_cell_line_test_prompts.csv \
        --llm Qwen/Qwen2.5-3B-Instruct --latent-size 2048 \
        --max-new-tokens 200 --top-k 4 --temperature 1.0 --seed 2024 \
        --output generated.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from llama_text_pred import LightningModel, resolve_device, set_seed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt file")
    parser.add_argument("--prompts", required=True,
                        help="CSV with an 'instruction' column")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--latent-size", type=int, default=2048)
    parser.add_argument("--max-token-len", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature; >1 diversifies, <1 sharpens")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None,
                        help="torch device string, e.g. 'cuda', 'cuda:0', 'cpu' "
                             "(default: auto / $LAMDNA_DEVICE)")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    model = LightningModel.load_from_checkpoint(
        args.checkpoint,
        llm_name=args.llm,
        latent_size=args.latent_size,
        max_token_len=args.max_token_len,
    )
    model = model.to(device)
    model.eval()

    prompts_df = pd.read_csv(args.prompts)
    prompts = list(prompts_df["instruction"].values)

    generated = []
    for start in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[start:start + args.batch_size]
        # A per-batch seed keeps the run reproducible while still varying
        # sampling across batches (matching the tutorial).
        generated += model.generate(
            batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + start,
        )

    out = prompts_df.copy()
    out["generated_seq"] = generated
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(generated)} sequences to {args.output}")


if __name__ == "__main__":
    main()
