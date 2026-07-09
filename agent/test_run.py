"""
Minimal example of driving the Biomni-based DNA-design agent.

Scope & sandboxing
------------------
The Biomni A1 agent executes LLM-generated Python/bash code to accomplish a
task. Treat it as an untrusted-code executor:

  * Run it inside an isolated environment (dedicated conda env / container /
    VM) with no access to credentials or data you are not willing to expose.
  * Point ``path`` at a scratch directory; the data lake (~11GB) is downloaded
    there on first run.
  * Provide the API key via the ``ANTHROPIC_API_KEY`` environment variable
    rather than hard-coding it in source (never commit real tokens).

This script only demonstrates the single `create_dna_sequence` tool used in the
paper; see agent/DETAILS.md for the full workflow.
"""
import os

# Read the key from the environment; fail early with a clear message instead of
# shipping a placeholder token in the code.
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "Set the ANTHROPIC_API_KEY environment variable before running "
        "(e.g. `export ANTHROPIC_API_KEY=sk-...`)."
    )

from biomni.agent import A1
from biomni.tool.systems_biology import create_dna_sequence

# Use a scratch/sandbox directory for the (large) data lake download.
data_path = os.environ.get("BIOMNI_DATA_PATH", "./data")

# Initialize the agent. The data lake is downloaded on first run (~11GB).
agent = A1(path=data_path, llm="claude-sonnet-4-20250514")

# # Execute biomedical tasks using natural language
# agent.go("Please analyze the function of AACCTTGG based on ChatNT.")

# Direct tool call: generate a promoter conditioned on chromosome / expression.
chrinfo = 1
value = 3.000
cellline = "hepg2"
print(create_dna_sequence(
    f"Please generate a promoter from chromosome {chrinfo} "
    f"with log2FoldChange {round(value, 3)} in {cellline}: "
))
