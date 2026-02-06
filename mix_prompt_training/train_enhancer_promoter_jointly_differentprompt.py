import numpy as np
import pandas as pd
import torch
from plotnine import *

## Download yeast native promoter expression data

import numpy as np
import pandas as pd
import torch
from plotnine import *

import numpy as np
import pandas as pd
import torch
from plotnine import *

import numpy as np
import pandas as pd
import torch

k562_prompts = [
    "Generate a DNA promoter sequence located on chromosome {chr} that achieves a log2 fold change of {K562_log2FoldChange} in K562 cells.",
    "Design a promoter from chromosome {chr} optimized for expression corresponding to log2FoldChange {K562_log2FoldChange} in K562.",
    "Create a functional promoter sequence on chromosome {chr} with transcriptional activity matching log2FoldChange {K562_log2FoldChange} in K562 cells.",
    "Synthesize a promoter element derived from chromosome {chr} that drives expression at log2FoldChange {K562_log2FoldChange} in K562.",
    "Propose a biologically plausible promoter on chromosome {chr} that reproduces log2FoldChange {K562_log2FoldChange} in K562.",
    "Generate a candidate promoter sequence from chromosome {chr} whose regulatory strength matches log2FoldChange {K562_log2FoldChange} in K562 cells.",
    "Design a transcriptionally active promoter on chromosome {chr} consistent with log2FoldChange {K562_log2FoldChange} measured in K562.",
    "Construct a promoter DNA sequence from chromosome {chr} targeting expression change {K562_log2FoldChange} (log2 scale) in K562.",
    "Create a regulatory promoter sequence on chromosome {chr} calibrated to log2FoldChange {K562_log2FoldChange} in K562 cells.",
    "Generate a promoter architecture from chromosome {chr} that yields log2FoldChange {K562_log2FoldChange} in the K562 cell line."
]
hepg2_prompts = [
    "Generate a promoter from chromosome {chr} that induces a log2 fold change of {HepG2_log2FoldChange} in HepG2 cells.",
    "Design a promoter sequence on chromosome {chr} optimized for log2FoldChange {HepG2_log2FoldChange} in HepG2.",
    "Create a chromosome {chr} promoter whose regulatory output matches log2FoldChange {HepG2_log2FoldChange} in HepG2.",
    "Synthesize a functional promoter derived from chromosome {chr} producing log2FoldChange {HepG2_log2FoldChange} in HepG2 cells.",
    "Propose a biologically realistic promoter on chromosome {chr} with expression level {HepG2_log2FoldChange} (log2) in HepG2.",
    "Generate a promoter DNA sequence from chromosome {chr} that achieves log2FoldChange {HepG2_log2FoldChange} in HepG2.",
    "Design a transcriptionally active promoter from chromosome {chr} consistent with HepG2 log2FoldChange {HepG2_log2FoldChange}.",
    "Construct a regulatory promoter sequence on chromosome {chr} tuned to log2FoldChange {HepG2_log2FoldChange} in HepG2 cells.",
    "Create a promoter element from chromosome {chr} that recapitulates log2FoldChange {HepG2_log2FoldChange} in HepG2.",
    "Generate a promoter architecture on chromosome {chr} aligned with log2FoldChange {HepG2_log2FoldChange} observed in HepG2."
]
gm12878_prompts = [
    "Generate a promoter sequence from chromosome {chr} with log2FoldChange {GM12878_log2FoldChange} in GM12878 cells.",
    "Design a chromosome {chr} promoter optimized for expression change {GM12878_log2FoldChange} (log2) in GM12878.",
    "Create a functional promoter on chromosome {chr} matching log2FoldChange {GM12878_log2FoldChange} in GM12878.",
    "Synthesize a promoter element derived from chromosome {chr} that drives {GM12878_log2FoldChange} log2 expression in GM12878 cells.",
    "Propose a biologically plausible promoter from chromosome {chr} achieving log2FoldChange {GM12878_log2FoldChange} in GM12878.",
    "Generate a promoter DNA sequence on chromosome {chr} consistent with GM12878 log2FoldChange {GM12878_log2FoldChange}.",
    "Design a transcriptionally active promoter from chromosome {chr} tuned to {GM12878_log2FoldChange} in GM12878 cells.",
    "Construct a regulatory promoter sequence from chromosome {chr} producing log2FoldChange {GM12878_log2FoldChange} in GM12878.",
    "Create a promoter on chromosome {chr} whose activity reflects log2FoldChange {GM12878_log2FoldChange} in GM12878.",
    "Generate a promoter architecture derived from chromosome {chr} yielding log2FoldChange {GM12878_log2FoldChange} in GM12878 cells."
]

sknsh_prompts = [
    "Generate a promoter from chromosome {chr} with log2FoldChange {SKNSH_log2FoldChange} in sknsh cells.",
    "Design a promoter sequence on chromosome {chr} optimized for {SKNSH_log2FoldChange} (log2) in sknsh.",
    "Create a chromosome {chr} promoter whose expression output matches log2FoldChange {SKNSH_log2FoldChange} in sknsh cells.",
    "Synthesize a functional promoter derived from chromosome {chr} driving {SKNSH_log2FoldChange} log2 expression in sknsh.",
    "Propose a biologically realistic promoter on chromosome {chr} with log2FoldChange {SKNSH_log2FoldChange} in sknsh.",
    "Generate a promoter DNA sequence from chromosome {chr} calibrated to {SKNSH_log2FoldChange} in sknsh cells.",
    "Design a transcriptionally active promoter on chromosome {chr} consistent with sknsh log2FoldChange {SKNSH_log2FoldChange}.",
    "Construct a regulatory promoter from chromosome {chr} yielding log2FoldChange {SKNSH_log2FoldChange} in sknsh.",
    "Create a promoter element on chromosome {chr} reproducing log2FoldChange {SKNSH_log2FoldChange} in sknsh cells.",
    "Generate a promoter architecture from chromosome {chr} aligned with log2FoldChange {SKNSH_log2FoldChange} observed in sknsh."
]

a549_prompts = [
    "Generate a promoter sequence from chromosome {chr} with log2FoldChange {A549_log2FoldChange} in A549 cells.",
    "Design a chromosome {chr} promoter optimized for {A549_log2FoldChange} (log2) expression in A549.",
    "Create a functional promoter on chromosome {chr} matching log2FoldChange {A549_log2FoldChange} in A549 cells.",
    "Synthesize a promoter element derived from chromosome {chr} that drives {A549_log2FoldChange} log2 expression in A549.",
    "Propose a biologically plausible promoter from chromosome {chr} achieving log2FoldChange {A549_log2FoldChange} in A549 cells.",
    "Generate a promoter DNA sequence on chromosome {chr} consistent with A549 log2FoldChange {A549_log2FoldChange}.",
    "Design a transcriptionally active promoter on chromosome {chr} tuned to {A549_log2FoldChange} in A549 cells.",
    "Construct a regulatory promoter from chromosome {chr} producing log2FoldChange {A549_log2FoldChange} in A549.",
    "Create a promoter element on chromosome {chr} that recapitulates log2FoldChange {A549_log2FoldChange} in A549 cells.",
    "Generate a promoter architecture derived from chromosome {chr} yielding log2FoldChange {A549_log2FoldChange} in A549."
]

np.random.seed(2024)

## Download yeast native promoter expression data

yeast_data_train = pd.read_csv("./dnadesign/promoter_data/dataset_fivecellline_train_crosschr.csv")
yeast_data_val = pd.read_csv("./dnadesign/promoter_data/dataset_fivecellline_val_crosschr.csv")



yeast_data_val

train = yeast_data_train
val = yeast_data_val

train_new = pd.DataFrame()
val_new = pd.DataFrame()

## Bin the sequences and create labels

inst = []
seq = []
value_l = []
for item in train.index:
    row = train.loc[item]
    out1 = str(np.random.choice(k562_prompts))
    out2 = str(np.random.choice(hepg2_prompts))
    out3 = str(np.random.choice(gm12878_prompts))
    out4 = str(np.random.choice(sknsh_prompts))
    out5 = str(np.random.choice(a549_prompts))
    inst.append(out1.format(chr = row['chr'], K562_log2FoldChange=round(row['K562_log2FoldChange'],3)))
    inst.append(out2.format(chr = row['chr'], HepG2_log2FoldChange=round(row['HepG2_log2FoldChange'],3)))
    inst.append(out3.format(chr = row['chr'], GM12878_log2FoldChange=round(row['GM12878_log2FoldChange'],3)))
    inst.append(out4.format(chr = row['chr'], SKNSH_log2FoldChange=round(row['SK-N-SH_log2FoldChange'],3)))
    inst.append(out5.format(chr = row['chr'], A549_log2FoldChange=round(row['A549_log2FoldChange'],3)))
    
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    
    value_l.append(round(row['K562_log2FoldChange'],3))
    value_l.append(round(row['HepG2_log2FoldChange'],3))
    value_l.append(round(row['GM12878_log2FoldChange'],3))
    value_l.append(round(row['SK-N-SH_log2FoldChange'],3))
    value_l.append(round(row['A549_log2FoldChange'],3))
train_new['instruction'] = inst
train_new['output'] = seq
train_new['exp'] = value_l

inst = []
seq = []
value_l = []
for item in val.index:
    row = val.loc[item]
    out1 = str(np.random.choice(k562_prompts))
    out2 = str(np.random.choice(hepg2_prompts))
    out3 = str(np.random.choice(gm12878_prompts))
    out4 = str(np.random.choice(sknsh_prompts))
    out5 = str(np.random.choice(a549_prompts))
    inst.append(out1.format(chr = row['chr'], K562_log2FoldChange=round(row['K562_log2FoldChange'],3)))
    inst.append(out2.format(chr = row['chr'], HepG2_log2FoldChange=round(row['HepG2_log2FoldChange'],3)))
    inst.append(out3.format(chr = row['chr'], GM12878_log2FoldChange=round(row['GM12878_log2FoldChange'],3)))
    inst.append(out4.format(chr = row['chr'], SKNSH_log2FoldChange=round(row['SK-N-SH_log2FoldChange'],3)))
    inst.append(out5.format(chr = row['chr'], A549_log2FoldChange=round(row['A549_log2FoldChange'],3)))
    
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    
    value_l.append(round(row['K562_log2FoldChange'],3))
    value_l.append(round(row['HepG2_log2FoldChange'],3))
    value_l.append(round(row['GM12878_log2FoldChange'],3))
    value_l.append(round(row['SK-N-SH_log2FoldChange'],3))
    value_l.append(round(row['A549_log2FoldChange'],3))
val_new['instruction'] = inst
val_new['output'] = seq
val_new['exp'] = value_l

yeast_data_train = pd.read_csv("./dnadesign/data_and_model/mdlm/gosai_data/dataset_threecellline_train_crosschr.csv")
yeast_data_val = pd.read_csv("./dnadesign/data_and_model/mdlm/gosai_data/dataset_threecellline_val_crosschr.csv")


hepg2_enhancer_prompts = [
    "Please generate a cis-regulatory element from chromosome {chrom} with expression level {hepg2} in hepg2.",
    "Design an enhancer sequence located on chromosome {chrom} that drives expression level {hepg2} in hepg2 cells.",
    "Create a functional cis-regulatory element from chromosome {chrom} consistent with expression level {hepg2} in hepg2.",
    "Synthesize an enhancer derived from chromosome {chrom} producing expression level {hepg2} in hepg2 cells.",
    "Propose a biologically plausible enhancer on chromosome {chrom} with expression level {hepg2} in hepg2.",
    "Generate a candidate enhancer sequence from chromosome {chrom} calibrated to expression level {hepg2} in hepg2 cells.",
    "Design a transcriptionally active cis-regulatory element on chromosome {chrom} matching expression level {hepg2} in hepg2.",
    "Construct an enhancer DNA sequence from chromosome {chrom} targeting expression level {hepg2} in hepg2 cells.",
    "Create a regulatory enhancer element from chromosome {chrom} that recapitulates expression level {hepg2} in hepg2.",
    "Generate an enhancer architecture from chromosome {chrom} aligned with expression level {hepg2} observed in hepg2 cells."
]


k562_enhancer_prompts = [
    "Please generate a cis-regulatory element from chromosome {chrom} with expression level {k562} in k562.",
    "Design an enhancer sequence located on chromosome {chrom} that drives expression level {k562} in k562 cells.",
    "Create a functional cis-regulatory element from chromosome {chrom} consistent with expression level {k562} in k562.",
    "Synthesize an enhancer derived from chromosome {chrom} producing expression level {k562} in k562 cells.",
    "Propose a biologically plausible enhancer on chromosome {chrom} with expression level {k562} in k562.",
    "Generate a candidate enhancer sequence from chromosome {chrom} calibrated to expression level {k562} in k562 cells.",
    "Design a transcriptionally active cis-regulatory element on chromosome {chrom} matching expression level {k562} in k562.",
    "Construct an enhancer DNA sequence from chromosome {chrom} targeting expression level {k562} in k562 cells.",
    "Create a regulatory enhancer element from chromosome {chrom} that recapitulates expression level {k562} in k562.",
    "Generate an enhancer architecture from chromosome {chrom} aligned with expression level {k562} observed in k562 cells."
]

sknsh_enhancer_prompts = [
    "Please generate a cis-regulatory element from chromosome {chrom} with expression level {sknsh} in sknsh.",
    "Design an enhancer sequence located on chromosome {chrom} that drives expression level {sknsh} in sknsh cells.",
    "Create a functional cis-regulatory element from chromosome {chrom} consistent with expression level {sknsh} in sknsh.",
    "Synthesize an enhancer derived from chromosome {chrom} producing expression level {sknsh} in sknsh cells.",
    "Propose a biologically plausible enhancer on chromosome {chrom} with expression level {sknsh} in sknsh.",
    "Generate a candidate enhancer sequence from chromosome {chrom} calibrated to expression level {sknsh} in sknsh cells.",
    "Design a transcriptionally active cis-regulatory element on chromosome {chrom} matching expression level {sknsh} in sknsh.",
    "Construct an enhancer DNA sequence from chromosome {chrom} targeting expression level {sknsh} in sknsh cells.",
    "Create a regulatory enhancer element from chromosome {chrom} that recapitulates expression level {sknsh} in sknsh.",
    "Generate an enhancer architecture from chromosome {chrom} aligned with expression level {sknsh} observed in sknsh cells."
]


train = yeast_data_train
val = yeast_data_val

train_new_e = pd.DataFrame()
val_new_e = pd.DataFrame()

## Bin the sequences and create labels

inst = []
seq = []
value_l = []
for item in train.index:
    row = train.loc[item]

    out1 = str(np.random.choice(hepg2_enhancer_prompts ))
    out2 = str(np.random.choice(k562_enhancer_prompts ))
    out3 = str(np.random.choice(sknsh_enhancer_prompts))
    inst.append(out1.format(chrom = row['chrom'], hepg2=round(row['hepg2'],3)))
    inst.append(out2.format(chrom = row['chrom'], k562=round(row['k562'],3)))
    inst.append(out3.format(chrom = row['chrom'], sknsh=round(row['sknsh'],3)))
    
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    
    value_l.append(round(row['hepg2'],3))
    value_l.append(round(row['k562'],3))
    value_l.append(round(row['sknsh'],3))
train_new_e['instruction'] = inst
train_new_e['output'] = seq
train_new_e['exp'] = value_l

inst = []
seq = []
value_l = []
for item in val.index:
    row = val.loc[item]
    out1 = str(np.random.choice(hepg2_enhancer_prompts ))
    out2 = str(np.random.choice(k562_enhancer_prompts ))
    out3 = str(np.random.choice(sknsh_enhancer_prompts))
    inst.append(out1.format(chrom = row['chrom'], hepg2=round(row['hepg2'],3)))
    inst.append(out2.format(chrom = row['chrom'], k562=round(row['k562'],3)))
    inst.append(out3.format(chrom = row['chrom'], sknsh=round(row['sknsh'],3)))
    seq.append(row['seq'])
    seq.append(row['seq'])
    seq.append(row['seq'])
    
    value_l.append(round(row['hepg2'],3))
    value_l.append(round(row['k562'],3))
    value_l.append(round(row['sknsh'],3))
val_new_e['instruction'] = inst
val_new_e['output'] = seq
val_new_e['exp'] = value_l
# train_new = train_new.sample(frac=1, random_state=2024)

train_new = pd.concat((train_new, train_new_e))
val_new = pd.concat((val_new, val_new_e))



import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("./")
sys.path.append("./hyena-dna/src")
sys.path.append("./hyena-dna/")

import pandas as pd


import reglm

# import PyPDF2
import os
import torch
os.environ['HF_TOKEN'] = 'your_token'
from transformers import AutoTokenizer, AutoModelForCausalLM

instruct_len = 30
from llama_text_pred import CharExpLanguageDataset

# train_ds = reglm.lightning_llama_edit.CharLanguageDataset(train['output'].values, train['instruction'].values, llm_name="Qwen/Qwen2.5-7B-Instruct")
# val_ds = reglm.lightning_llama_edit.CharLanguageDataset(val['output'].values, val['instruction'].values, llm_name="Qwen/Qwen2.5-7B-Instruct")

train_ds = CharExpLanguageDataset(train_new['output'].values, train_new['instruction'].values, train_new['exp'].values, llm_name="Qwen/Qwen2.5-3B-Instruct", max_token_len=instruct_len)
val_ds = CharExpLanguageDataset(val_new['output'].values, val_new['instruction'].values, train_new['exp'].values, llm_name="Qwen/Qwen2.5-3B-Instruct", max_token_len=instruct_len)

import reglm.lightning
import reglm.dataset

# train_ds = reglm.dataset.CharDataset(df_comb['output'].values,['0' for i in range(len(df_comb))])
# val_ds = reglm.dataset.CharDataset(df_comb_new['output'].values, ['0' for i in range(len(df_comb_new))])


# import hyenadna

config = {
 'd_model': 32,
 'n_layer': 2,
 'd_inner': 32,
 'vocab_size': 12,
 'pad_vocab_size_multiple': 8,
 'return_hidden_state': True,
 # 'layer': {
 #     'emb_dim': 5,
 #     'filter_order': 64,
 #     'l_max': train_ds.seq_len + train_ds.label_len + 1,
 #     '_name_': 'hyena'
 # }
}

HYENADNA_MODEL_NAMES = [
    "hyenadna-tiny-16k-seqlen-d128",
    "hyenadna-large-1m-seqlen",
    "hyenadna-medium-160k-seqlen",
    "hyenadna-medium-450k-seqlen",
    "hyenadna-small-32k-seqlen",
    "hyenadna-tiny-1k-seqlen",
    "hyenadna-tiny-1k-seqlen-d256",
]

model_name = "hyenadna-tiny-1k-seqlen-d256"
# model_name = "hyenadna-tiny-16k-seqlen-d128"
# from llama_text_pred import LightningModel
from llama_text_pred import LightningModel
model = LightningModel(
    config=config, lr=1e-3, label_len=1,  ckpt_dir=f"./checkpoints/{model_name}", save_dir='./generation_more_test_value_align_qwen2.53b100epoch_35cl_mixprompt/',llm_name="Qwen/Qwen2.5-3B-Instruct",latent_size=2048,align_head='relu', max_token_len=instruct_len)

import torch

# # len(train)

# # ## Train and validate model

trainer = model.train_on_dataset(train_ds, val_ds, max_epochs=100,
            batch_size=64, num_workers=1, device=0, val_check_interval=50,
        )

print(trainer.checkpoint_callback.best_model_path)
