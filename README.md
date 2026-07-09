# Multi-Modal Agent Facilitates Atlas-Level DNA Sequence Design

Code for the paper *Multi-Modal Agent Facilitates Atlas-Level DNA Sequence
Design*. The model couples a frozen instruction LLM (Llama-3.2 / Qwen-2.5) as a
text tower with a HyenaDNA sequence backbone to generate cis-regulatory
elements conditioned on natural-language prompts (chromosome, cell line, target
expression), with an auxiliary head that regresses the expression value.

- **`llama_text_pred.py`** — default pipeline (text conditioning **+** expression
  predictor head). Start here.
- **`llama_text.py`** — same model without the expression predictor.
- **`lightning_llama.py`** — additional variants (embedding-conditioned model).
- **`mix_prompt_training/`** — train from multiple prompts jointly.
- **`agent/`** — optional Biomni-based agentic pipeline (see "Agent workflow").
- **`tutorial/`** — notebooks demonstrating training, generation, and explainability.

## 1. Installation

### 1.1 Pinned Python environment

Tested with **Python 3.11**. Install the pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key constraints (see `requirements.txt` for the full list and rationale):
`pytorch-lightning < 2.0` (the code uses the `validation_epoch_end` hook),
`torch < 2.0` (regLM / pinned HyenaDNA compatibility), and
`transformers >= 4.45` (Llama-3.2 / Qwen-2.5 support).

### 1.2 HyenaDNA backbone

Clone the HyenaDNA repository; it provides the `ConvLMHeadModel` backbone:

```bash
git clone https://github.com/HazyResearch/hyena-dna
```

Point the code at it with **either** the `HYENADNA_PATH` environment variable
or the `hyenadna_path=` / `--hyenadna-path` argument (default: `./hyena-dna`):

```bash
export HYENADNA_PATH=/abs/path/to/hyena-dna
```

Pretrained HyenaDNA checkpoints are downloaded automatically from
`https://huggingface.co/LongSafari/<model>` into `--ckpt-dir` on first use
(see `HYENADNA_MODEL_NAMES` for the available names).

### 1.3 regLM

Install regLM using the version vendored in the `regLM/` folder:

```bash
pip install -e regLM
```

### 1.4 Instruction LLM access

The frozen text tower is loaded from Hugging Face (e.g.
`Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`). For gated
models, authenticate first and (optionally) redirect the cache:

```bash
export HF_TOKEN=<your_hf_token>
export HF_HOME=/path/to/hf_cache   # optional
```

## 2. Data

The models train on a table of sequences with per-cell-line expression. Each
CSV row provides:

| column | meaning |
| --- | --- |
| `chrom` | chromosome name (used to build the prompt) |
| `seq` | DNA sequence (`A/C/G/T/N`) |
| `hepg2`, `k562`, `sknsh` | expression value per cell line |

The three-cell-line enhancer dataset used in the paper is derived from the
Gosai et al. massively-parallel reporter assay. Place your `train` / `val` /
`test` CSVs anywhere and pass their paths on the command line.

## 3. Reproducing the main results

Two self-contained scripts wrap the training and generation stages shown in
`tutorial/tutorial_3cl_human_enhancer.ipynb`. Both select the device
automatically (GPU if available, else CPU; override with `LAMDNA_DEVICE` or
`--device`) and seed the full pipeline (Python / NumPy / torch / Lightning).

### 3.1 Train

```bash
python scripts/train_expression_model.py \
    --train data/three_cell_line_train.csv \
    --val   data/three_cell_line_val.csv \
    --hyenadna-model hyenadna-tiny-1k-seqlen-d256 \
    --llm Qwen/Qwen2.5-3B-Instruct --latent-size 2048 --max-token-len 30 \
    --save-dir runs/three_cell_line \
    --max-epochs 1000 --batch-size 1024 --lr 1e-3 \
    --val-check-interval 700 --seed 2024
```

`--latent-size` must match the LLM hidden size (`2048` for Qwen2.5-3B, `3072`
for Llama-3.2-3B). The best checkpoint path is printed at the end.

### 3.2 Generate

```bash
python scripts/generate_sequences.py \
    --checkpoint runs/three_cell_line/<...>/best.ckpt \
    --prompts data/three_cell_line_test_prompts.csv \
    --llm Qwen/Qwen2.5-3B-Instruct --latent-size 2048 \
    --max-new-tokens 200 --top-k 4 --temperature 1.0 \
    --seed 2024 --output generated.csv
```

The `--prompts` CSV needs an `instruction` column; the training script's prompt
template is `"Please generate a cis-regulatory element from chromosome {chrom}
with expression level {value} in {cell_line}: "`.

## 4. Model & inference notes

- **Frozen text tower.** The instruction LLM is used purely as an encoder: its
  parameters are set to `requires_grad=False`, it is kept in `eval()` mode, and
  its forward pass runs under `torch.no_grad()`, so no gradients are built
  through the language model. Only the HyenaDNA backbone, the alignment head,
  and the expression head are optimized. (`llama_text.py` is the exception —
  there the LLM is intentionally fine-tuned and included in the optimizer.)
- **Device handling.** No hard-coded CUDA. `resolve_device()` picks the device
  from `--device` / `LAMDNA_DEVICE` / CUDA availability / CPU.
- **Reproducibility.** Call `set_seed(seed)` (done by both scripts) to seed
  Python, NumPy, torch, and Lightning workers. Generation additionally accepts a
  `seed` argument.
- **Sampling.** `generate(..., temperature, top_k, top_p)` scales the logits by
  `temperature` before the softmax (`>1` diversifies, `<1` sharpens), then
  applies optional top-k / top-p filtering.

## 5. Agent workflow (optional)

`agent/` contains an optional Biomni-based pipeline for autonomous DNA design
and analysis. **The agent executes LLM-generated code**, so run it in an
isolated, sandboxed environment (dedicated conda env / container / VM), provide
the API key via the `ANTHROPIC_API_KEY` environment variable (never hard-code
it), and point the data path at a scratch directory. See `agent/README.md`,
`agent/DETAILS.md`, and `agent/test_run.py` for a minimal, scoped example.

## Acknowledgement

We thank the developers of [regLM](https://github.com/Genentech/regLM/tree/main/src/reglm),
[gReLU](https://genentech.github.io/gReLU/),
[HyenaDNA](https://github.com/HazyResearch/hyena-dna),
[Biomni](https://github.com/snap-stanford/Biomni), and
[Hugging Face](https://huggingface.co/) for their open-source work.

## Citation

```
TBD
```
