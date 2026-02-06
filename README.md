# This is code repo for paper: Multi-Modal Agent Facilitates Atlas-Level DNA Sequence Design

## Installation

Step1: Get HyenaDNA

```
git clone https://github.com/HazyResearch/hyena-dna
```

Step2: Prepare the environment

Install HyenaDNA

Install regLM (by referring the version in this folder)

Update transformers to the updated version (for testing the updated LLMs), e.g.:

```
pip install transformers --upgrade
```

Step3 (optional): Setup agent pipeline

Please refer the folder "agent" to set up Biomni for building an AI agentic pipeline of DNA sequence design and analysis.

## Training

Please refer the **llama_text_pred.py** as the default model pipeline, and we also provide **mix_prompt_training** folder to train a model based on multiple prompts.

## Tutorials

Please refer the files under folder **tutorial** for an example of model training and inference.

## Acknowledgement

We thank developers of [regLM](https://github.com/Genentech/regLM/tree/main/src/reglm), [gReLU](https://genentech.github.io/gReLU/), and [Huggingface](https://huggingface.co/) for their great work in open-source software design.

## Citation

```
TBD
```

