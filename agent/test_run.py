import os
os.environ["ANTHROPIC_API_KEY"]="your token"

from biomni.agent import A1
from biomni.tool.systems_biology import create_dna_sequence

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(path='./data', llm='claude-sonnet-4-20250514')

# # Execute biomedical tasks using natural language
# agent.go("Please analyze the function of AACCTTGG based on ChatNT.")

from biomni.utils import function_to_api_schema
from biomni.llm import get_llm

# llm = get_llm('claude-sonnet-4-20250514')
# desc = function_to_api_schema(chatnt_call, llm)
# print(desc)
chrinfo = 1
value = 3.000
cellline = 'hepg2'
print(create_dna_sequence(f"Using GPU, please generate a promoter from chromosome {chrinfo} with log2FoldChange {round(value,3)} in {cellline}: "))