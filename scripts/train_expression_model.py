#!/usr/bin/env python
"""
End-to-end training script for the expression-conditioned DNA design model
(the `llama_text_pred.py` pipeline described in the README).

This reproduces the training procedure used in
`tutorial/tutorial_3cl_human_enhancer.ipynb` as a single, non-interactive,
device-agnostic command. The frozen instruction encoder (Llama/Qwen), device
selection, and all RNG seeds are handled inside the model / helpers so that the
run is reproducible on both GPU and CPU-only machines.

Expected input CSVs (train and validation) must contain at least these columns:

    chrom   : chromosome name (used only to build the instruction text)
    seq     : the DNA sequence (A/C/G/T/N) to be modeled
    hepg2   : expression value in the HepG2 cell line
    k562    : expression value in the K562 cell line
    sknsh   : expression value in the SK-N-SH cell line

One (instruction, sequence, value) example is emitted per cell line, matching
the tutorial. Adapt `build_examples` if your atlas uses different cell types.

Example
-------
    python scripts/train_expression_model.py \
        --train data/three_cell_line_train.csv \
        --val   data/three_cell_line_val.csv \
        --hyenadna-model hyenadna-tiny-1k-seqlen-d256 \
        --llm Qwen/Qwen2.5-3B-Instruct --latent-size 2048 \
        --save-dir runs/three_cell_line --max-epochs 1000 --batch-size 1024 \
        --seed 2024
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make the repository importable when running from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from llama_text_pred import CharExpLanguageDataset, LightningModel, set_seed

# Avoid tokenizer fork warnings / nondeterminism from parallel tokenization.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CELL_LINES = ["hepg2", "k562", "sknsh"]


def build_examples(df, cell_lines):
    """
    Expand each row into one (instruction, sequence, value) example per cell
    line, mirroring the tutorial's prompt template.
    """
    instructions, sequences, values = [], [], []
    for _, row in df.iterrows():
        for cell in cell_lines:
            value = round(float(row[cell]), 3)
            instructions.append(
                f"Please generate a cis-regulatory element from chromosome "
                f"{row['chrom']} with expression level {value} in {cell}: "
            )
            sequences.append(row["seq"])
            values.append(value)
    return (
        np.array(sequences),
        np.array(instructions),
        np.array(values, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val", required=True, help="Path to validation CSV")
    parser.add_argument("--hyenadna-model", default="hyenadna-tiny-1k-seqlen-d256",
                        help="HyenaDNA checkpoint name (see HYENADNA_MODEL_NAMES)")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Directory for HyenaDNA checkpoints "
                             "(default: ./checkpoints/<hyenadna-model>)")
    parser.add_argument("--hyenadna-path", default=None,
                        help="Path to the cloned hyena-dna repo "
                             "(default: $HYENADNA_PATH or ./hyena-dna)")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF name of the frozen instruction encoder")
    parser.add_argument("--latent-size", type=int, default=2048,
                        help="Hidden size of the LLM (2048 for Qwen2.5-3B, "
                             "3072 for Llama-3.2-3B)")
    parser.add_argument("--max-token-len", type=int, default=30)
    parser.add_argument("--align-head", default="relu", choices=["relu", "linear"])
    parser.add_argument("--save-dir", default="runs/expression_model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-check-interval", type=int, default=700)
    parser.add_argument("--device", type=int, default=0,
                        help="GPU index passed to the Lightning trainer")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    # Seed the *entire* pipeline (Python, NumPy, torch, Lightning + workers).
    set_seed(args.seed)

    ckpt_dir = args.ckpt_dir or f"./checkpoints/{args.hyenadna_model}"

    # Shuffle the training rows deterministically before expansion.
    train_df = pd.read_csv(args.train).sample(frac=1, random_state=args.seed)
    val_df = pd.read_csv(args.val)

    train_seqs, train_inst, train_vals = build_examples(train_df, CELL_LINES)
    val_seqs, val_inst, val_vals = build_examples(val_df, CELL_LINES)

    train_ds = CharExpLanguageDataset(
        train_seqs, train_inst, train_vals,
        llm_name=args.llm, max_token_len=args.max_token_len,
    )
    val_ds = CharExpLanguageDataset(
        val_seqs, val_inst, val_vals,
        llm_name=args.llm, max_token_len=args.max_token_len,
    )

    model_kwargs = dict(
        ckpt_dir=ckpt_dir,
        save_dir=args.save_dir,
        lr=args.lr,
        label_len=1,
        llm_name=args.llm,
        latent_size=args.latent_size,
        align_head=args.align_head,
        max_token_len=args.max_token_len,
    )
    if args.hyenadna_path is not None:
        model_kwargs["hyenadna_path"] = args.hyenadna_path

    # The Llama/Qwen encoder is frozen inside LightningModel.__init__ and left
    # out of the optimizer; no manual requires_grad toggling is needed.
    model = LightningModel(**model_kwargs)

    trainer = model.train_on_dataset(
        train_ds, val_ds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        val_check_interval=args.val_check_interval,
    )

    print("Best checkpoint:", trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
