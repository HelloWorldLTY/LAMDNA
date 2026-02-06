import itertools
import json
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import Accuracy
from transformers import AddedToken, AutoTokenizer

HYENADNA_MODEL_NAMES = [
    "hyenadna-tiny-16k-seqlen-d128",
    "hyenadna-large-1m-seqlen",
    "hyenadna-medium-160k-seqlen",
    "hyenadna-medium-450k-seqlen",
    "hyenadna-small-32k-seqlen",
    "hyenadna-tiny-1k-seqlen",
    "hyenadna-tiny-1k-seqlen-d256",
]


import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class CharLanguageDataset(Dataset):
    def __init__(self, seqs, labels, seq_len=None, llm_name = "meta-llama/Llama-3.2-3B-Instruct", max_token_len=15):
        """
        A dataset class to produce tokenized sequences for training regLM.

        Each sequence will be represented as 0<LABEL><SEQ>1; hence 0 is the start
        token and 1 is the end token.

        Args:
            seqs (list): List of sequences.
            labels (list): List of labels as strings
            seq_len (int): Maximum sequence length
        """
        # Check
        assert len(seqs) == len(labels), "seqs and labels should have equal length"
        # assert (
        #     len(set([len(x) for x in labels])) == 1
        # ), "All labels should be of equal length"

        # Store data
        self.seqs = seqs
        self.labels = labels
        self.max_token_len = max_token_len

        # maximum sequence length
        self.seq_len = seq_len or np.max([len(seq) for seq in self.seqs])
        # self.label_len = len(self.labels[0])
        self.label_len = 1
        self.unique_labels = set(
            np.concatenate([[tok for tok in lab] for lab in self.labels])
        )
        # assert (
        #     len(self.unique_labels) <= 10
        # ), ">10 label classes are currently not supported"

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}
        
        device_map = 'cuda'
        # model_name="meta-llama/Llama-3.1-8B-Instruct"
        model_name = llm_name
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer 
        
        

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, seq):
        """
        Encode a sequence as a torch tensor of tokens

        Args:
            seq (str): DNA sequence

        Returns:
            torch.LongTensor of shape (seq_len,)
        """
        return torch.LongTensor([self.base_stoi[tok] for tok in seq])

    def encode_label(self, label):
        """
        Encode a label as a torch tensor of tokens

        Args:
            label (str): label token sequence

        Returns:
            torch.LongTensor of shape (label_len,)
        """
        return torch.IntTensor(self.tokenizer(label,padding='max_length', truncation=True, max_length=self.max_token_len, padding_side='left')['input_ids'])

    def decode(self, idxs, is_labeled=False):
        """
        Given a torch tensor of tokens, return the decoded sequence as a string.

        Args:
            idxs (list, torch.LongTensor): list or 1-D tensor
            is_labeled (bool): Whether labels are included

        Returns:
            labeled sequence as a string
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().tolist()
        if is_labeled:
            # Split the input into sequence and label
            label = idxs[: self.label_len]
            seq = idxs[self.label_len :]
            # Decode them separately and rejoin
            return "".join(
                [self.label_itos[i] for i in label] + [self.base_itos[i] for i in seq]
            )
        else:
            # Only a sequence is provided
            return "".join([self.base_itos[i] for i in idxs])

    def __getitem__(self, idx):
        """
        Return a single labeled example as a tensor of tokens
        x = 0<LABEL><SEQ>
        y = <SEQ>1

        Args:
            idx: Index of example to return

        Returns:
            x (torch.LongTensor): tensor of shape (1 + self.label_len + self.seq_len)
            y (torch.LongTensor): tensor of shape (self.seq_len + 1, )
        """
        # Get sequence
        seq = self.seqs[idx]

        # Encode sequence
        seq = self.encode_seq(seq)

        # Get label
        label = self.labels[idx]

        # Encode label
        # label = self.encode_label(label)
        instruct = torch.IntTensor(self.tokenizer(label,padding='max_length', truncation=True, max_length=self.max_token_len, padding_side='right')['input_ids'])

        # Generate empty tensors
        x = torch.zeros(self.seq_len + self.label_len + 1, dtype=torch.long)
        y = torch.zeros(self.seq_len + 1, dtype=torch.long)

        # Input: START(0) + label + sequence + trailing zeros (will be ignored)
        x[1 : 1 + self.label_len] = -1
        x[1 + self.label_len : 1 + self.label_len + len(seq)] = seq

        # Output: sequence + END (1) + trailing zeros (will be ignored)
        y[: len(seq)] = seq
        y[len(seq)] = 1

        return instruct, x, y


def load_pretrained_model(
    ckpt_dir="./checkpoints/",
    model="hyenadna-medium-160k-seqlen",
    hyenadna_path="/code/hyena-dna",
):
    """
    Load a pretrained hyenaDNA foundation model.

    Args:
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        model (str): Name of model to load
        hyenadna_path (str): Path to cloned hyenaDNA repository

    Returns:
        model (nn.Module): pre-trained HyenaDNA foundation model
    """
    sys.path.append(hyenadna_path)
    from src.models.sequence.long_conv_lm import ConvLMHeadModel

    # Check model name
    assert model in HYENADNA_MODEL_NAMES

    # Make directory if needed
    if not os.path.exists(ckpt_dir):
        print("Making checkpoint directory")
        os.makedirs(ckpt_dir)

    # Download model if not already downloaded
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print("Downloading model")
        config = f"https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        ckpt = f"https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        os.system(f"wget -P {ckpt_dir} {config}")
        os.system(f"wget -P {ckpt_dir} {ckpt}")

    # Load config
    config = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))

    # Generate model
    model = ConvLMHeadModel(**config)

    # Load weights
    state_dict = torch.load(
        os.path.join(ckpt_dir, "weights.ckpt"), map_location=torch.device("cpu")
    )
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["state_dict"], "model."
    )
    model_state_dict = state_dict["state_dict"]
    for key in list(model_state_dict.keys()):
        if "torchmetrics" in key:
            model_state_dict.pop(key)

    model.load_state_dict(model_state_dict)
    return model


class LightningModel(pl.LightningModule):
    """
    LightningModule class to train and use autoregressive token-conditioned
    regLM language models.

    Args:
        config (dict): Config dictionary containing model parameters
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        hyenadna_path (str): Path to cloned hyenaDNA repository
        save_dir (str): Directory to save model checkpoints and logs
        lr (float): Learning rate
        label_len (int): Number of label tokens preceding each DNA sequence
    """

    def __init__(
        self,
        config=None,
        ckpt_dir="./checkpoints/hyenadna-medium-160k-seqlen",
        hyenadna_path="/code/hyena-dna",
        save_dir=".",
        lr=1e-4,
        label_len=None,
        llm_name = "meta-llama/Llama-3.2-3B-Instruct",
        latent_size = 3072,
        align_head = 'relu',
        max_token_len = 15
    ):
        super().__init__()

        self.save_dir = save_dir
        self.label_len = label_len
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.latent_size = latent_size
        self.max_token_len = max_token_len

        # Build model
        if ckpt_dir is not None:
            self.model = load_pretrained_model(
                ckpt_dir=ckpt_dir, hyenadna_path=hyenadna_path
            )
        elif config is not None:
            sys.path.append(hyenadna_path)
            from src.models.sequence.long_conv_lm import ConvLMHeadModel

            self.model = ConvLMHeadModel(**config)
        else:
            raise ValueError("either config or ckpt_dir must be provided.")

        # Print number of model parameters
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (self.n_params / 1e6,))

        # Metrics: accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)
        self.val_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

        # Loss function
        # Trailing zeros in the label will be ignored in calculating the loss
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=0)
        
        device_map = 'cuda'
        # model_name="meta-llama/Llama-3.1-8B-Instruct"
        model_name = llm_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer 
        
        model_llama = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map)
        print("Successfully create Llama")
        
        if align_head == 'relu':
            self.emb_align = nn.Sequential(nn.Linear(self.latent_size,self.latent_size), nn.GELU(), nn.Linear(self.latent_size,256))
            
        if align_head == 'linear':
            self.emb_align = nn.Linear(self.latent_size,256)
        print("Successfully create aligner")
        
        self.Llama_encoder = model_llama.cuda()
        self.model = self.model.cuda()
        self.emb_align = self.emb_align.cuda()
        
        print(self.Llama_encoder.device)

    def forward(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        
        if isinstance(x, list):
            instruct = x[0]
            # print(instruct.shape)
            x = x[1]
            # print(x.shape)
            
            instruct_emb = self.Llama_encoder(instruct,return_dict=True, output_hidden_states=True)['hidden_states'][-1]
            # print(instruct_emb.shape)
            # print(instruct_emb.shape)
            instruct_emb = self.emb_align(instruct_emb)
            # print(instruct_emb.shape)
            # print(x[:,2:].shape)
#             seq_enc = self.model.backbone.forward(x[:,2:])
#             # print(seq_enc.shape)
#             final_emb = torch.cat((instruct_emb,seq_enc), dim=1)

            seq_enc = self.model.backbone.forward_multimodal(x[:,2:],instruct_emb)
            final_emb = seq_enc
            
            # print(final_emb.shape)
            # print(final_emb.device)
            logits = self.model.lm_head(final_emb).swapaxes(
                1, 2
            )  # N, label + seq + end + trailing zeros
            
            # print(logits.shape)
    
            # Drop the label probabilities
            if drop_label:
                # print(logits.shape)
                logits = logits[:, :, instruct_emb.shape[1]-1:]  # N, seq + end + trailing
    
            # Return logits or normalized probabilities
            if return_logits:
                return logits
            else:
                return logits.softmax(1)
        else:
            instruct = x.cuda()
            # print(len(instruct))
            instruct_emb = self.Llama_encoder(instruct,return_dict=True, output_hidden_states=True)['hidden_states'][-1]
            instruct_emb = self.emb_align(instruct_emb)
            instruct_emb = self.model.backbone.forward_multimodal(None,instruct_emb)
#             instruct_emb = seq_enc
            
            logits = self.model.lm_head(instruct_emb).swapaxes(
                1, 2
            )  # N, label + seq + end + trailing zeros
            
            # print(logits.shape)
    
            # Drop the label probabilities
            if drop_label:
                logits = logits[:, :,  instruct_emb.shape[1]-1:]  # N, seq + end + trailing
                # print(logits.shape)
    
            # Return logits or normalized probabilities
            if return_logits:
                return logits
            else:
                return logits.softmax(1)
                
    def forward_joint(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        # instruct = x[0:x.shape[0]-1]
        # # print(instruct.shape)
        # x = x[x.shape[0]-1:]
        # # print(x.shape)
#         print(x)
#         print(x.shape)
        instruct = x[:,0:self.max_token_len]
        x = x[:,self.max_token_len:]
        instruct_emb = self.Llama_encoder(instruct,return_dict=True, output_hidden_states=True)['hidden_states'][-1]
        # print(instruct_emb.shape)
        # print(instruct_emb.shape)
        instruct_emb = self.emb_align(instruct_emb)
        # print(instruct_emb.shape)
        # print(x[:,2:].shape)
#         seq_enc = self.model.backbone(x)
#         # print(seq_enc.shape)
#         final_emb = torch.cat((instruct_emb,seq_enc), dim=1)

        seq_enc = self.model.backbone.forward_multimodal(x[:,2:],instruct_emb)
        final_emb = seq_enc
        
        # print(final_emb.shape)
        # print(final_emb.device)
        logits = self.model.lm_head(final_emb).swapaxes(
            1, 2
        )  # N, label + seq + end + trailing zeros
        
        # print(logits.shape)

        # Drop the label probabilities
        if drop_label:
            # print(logits.shape)
            logits = logits[:, :, final_emb.shape[1]-1:]  # N, seq + end + trailing

        # Return logits or normalized probabilities
        if return_logits:
            return logits
        else:
            return logits.softmax(1)

    def forward_emb(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        
        if isinstance(x, list):
            instruct = x[0]
            # print(instruct.shape)
            x = x[1]
            # print(x.shape)
            
            instruct_emb = self.Llama_encoder(instruct,return_dict=True, output_hidden_states=True)['hidden_states'][-1]
            # print(instruct_emb.shape)
            # print(instruct_emb.shape)
            instruct_emb = self.emb_align(instruct_emb)
            # print(instruct_emb.shape)
            # print(x[:,2:].shape)
            seq_enc = self.model.backbone.embeddings(x[:,2:])
            
            return instruct_emb.mean(axis=1), seq_enc.mean(axis=1)
        
    def training_step(self, batch, batch_idx):
        inst, x, y = batch
        logits = self.forward(
            [inst,x], drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
#         ins_emb,seq_emb = self.forward_emb(
#             [inst,x], drop_label=True, return_logits=True
#         )  # N, seq + end + trailing
#         loss+= F.mse_loss(ins_emb,seq_emb)
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inst, x, y = batch
        logits = self.forward(
            [inst,x], drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
#         ins_emb,seq_emb = self.forward_emb(
#             [inst,x], drop_label=True, return_logits=True
#         )  # N, seq + end + trailing
#         loss+= F.mse_loss(ins_emb,seq_emb)
        self.val_acc.update(logits.argmax(1), y)
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(self, output):
        val_acc = self.val_acc.compute()
        self.log("val_acc", val_acc)
        val_loss = torch.mean(torch.Tensor(output))
        print(f"\nVal loss: {val_loss}, val acc: {val_acc}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.model.parameters() ) + list(self.emb_align.parameters()), lr=float(self.lr))
        return optimizer

    def train_on_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size=128,
        num_workers=8,
        device=0,
        max_epochs=3,
        val_check_interval=5000,
        weights=None,
        save_all=False,
    ):
        """
        Train regLM model.

        Args:
            train_dataset (CharDataset): Training dataset
            val_dataset (CharDataset): Validation dataset
            batch_size (int): Batch size
            num_workers (int): Number of workers for training
            device (int): GPU index
            max_epochs (int): Number of epochs to train
            val_check_interval (int): Number of steps after which to
                check validation loss

        Returns:
            pl.Trainer object
        """
        torch.set_float32_matmul_precision("medium")

        # Save dataset params
        self.seq_len = train_dataset.seq_len
        self.label_len = train_dataset.label_len
        self.unique_labels = train_dataset.unique_labels

        # Logger
        logger = CSVLogger(self.save_dir)

        # Create callback
        callbacks = [
            ModelCheckpoint(monitor="val_acc", mode="max"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ]
        if save_all:
            callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1))

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[device],
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.save_dir,
            val_check_interval=val_check_interval,
        )

        # Make dataloaders
        if weights is None:
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
            )

        val_data = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # First validation pass
        trainer.validate(model=self, dataloaders=val_data)

        # Training
        trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)

        return trainer

    def compute_accuracy_on_dataset(
        self,
        dataset,
        batch_size=64,
        num_workers=8,
    ):
        """
        Perform inference on a dataset and return per-example accuracy
        Note: this will include the accuracy of predicting the END token (1)

        Args:
            dataset (CharDataset): Inference dataset
            batch_size (int): Batch size for inference
            num_workers (int): Number of workers for inference

        Returns: List of booleans indicating whether the predicted base at each position
        was equal to the true label or not.
        """
        # Make dataloader
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Empty lists to hold predictions
        y_hat = []
        y = []

        self.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(iter(dl)):
                ins = batch[0].to(self.device)
                x = batch[1].to(self.device)
                probs = self.forward(
                    [ins,x], drop_label=True
                ).squeeze()  # N, 16, seq+end+trailing
                y_hat.append(probs.argmax(1).cpu().detach())  # N, seq+end+trailing
                y.append(batch[2].detach().squeeze())  # N, seq+end+trailing zeros

        # Stack batched predictions
        y_hat = torch.vstack(y_hat).numpy()  # N, L
        y = torch.vstack(y).numpy()  # N, L

        # Compare predicted base and true label where the true label is not 0.
        # Hence, this computation will include the END token (1) but not any
        # hypothetical bases / trailing zeros.
        is_non_zero = y != 0
        is_equal = y_hat == y

        # Return booleans
        return [e[n] for e, n in zip(is_equal, is_non_zero)]

    def encode_labels(self, labels, add_start=False):
        """
        Encode labels as a list of indices for inference

        Args:
            labels (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        if isinstance(labels, str):
            labels = [labels]

        # # Convert label to indices
        # idxs = torch.LongTensor(
        #     [[self.label_stoi[tok] for tok in label] for label in labels]
        # )  # N, label_len

        # # Add start token
        # if add_start:
        #     idxs = torch.cat(
        #         (torch.LongTensor([[0]] * idxs.shape[0]), idxs), axis=1
        #     )  # N, 1 + label_len
        # return idxs
        idxs = torch.IntTensor(self.tokenizer(labels,padding='max_length', truncation=True, max_length=self.max_token_len, padding_side='left')['input_ids'])
        idxs = idxs.to(self.device)
        # print(idxs.shape)
        return idxs

    def encode_seqs(self, seqs, add_stop=False):
        """
        Encode sequences as lists of indices for inference

        Args:
            seqs (list, str): DNA sequence(s) as string or list of strings
            add_stop (bool): Whether to add an end token (1) after each sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) if add_stop is False
            or (N, L+1) if add_stop is True.
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        # Convert sequences to indices
        idxs = torch.LongTensor(
            [[self.base_stoi[tok] for tok in seq] for seq in seqs]
        )  # N, seq_len

        # Add END token
        if add_stop:
            idxs = torch.cat(
                (idxs, torch.LongTensor([[1]] * idxs.shape[0])), axis=1
            )  # N, seq_len+1

        return idxs

    def encode(self, seqs, labels, add_start=False, add_stop=False):
        """
        Encode sequences and labels as indices for model inference.

        Args:
            seqs (list, str): Strings of base tokens
            label (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)
            add_stop (bool): Add an end token (1) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        return torch.cat(
            [
                self.encode_labels(labels, add_start=add_start),
                self.encode_seqs(seqs, add_stop=add_stop),
            ],
            axis=1,
        )

    def decode(self, idxs):
        """
        Decodes indices into DNA sequences

        Args:
            idxs (torch.LongTensor): tensor or array of shape (N, L)

        Returns:
            seqs (list): list of strings
        """
        if idxs.dim() == 1:  # L
            idxs = idxs.unsqueeze(0)  # 1, L

        idxs = idxs.cpu().detach().numpy()

        # Create empty list
        seqs = []

        # Fill list
        for x in idxs:
            curr_seq = []

            # Decode
            for pos, ix in enumerate(x):
                # Ignore start token
                if (pos == 0) and (ix == 0):
                    continue

                # Terminate at end token
                if ix == 1:
                    break

                # Replace non-bases with N
                elif (ix < 7) or (ix > 11):
                    ix = 11

                curr_seq.append(self.base_itos[ix])

            seqs.append("".join(curr_seq))
        return seqs

    def probs_to_likelihood(self, probs, idxs):
        """
        Compute the likelihood of each base in a sequence given model predictions
        on the sequence.

        Args:
            probs (torch.FloatTensor): tensor of shape (N, 16, L)
            idxs (torch.LongTensor): tensor of shape (N, L)

        Returns:
            tensor of shape (N, L) containing the probabilities of real bases
        """
        # Check shapes
        assert probs.dim() == 3, probs.shape
        assert probs.shape[1] == 16, probs.shape
        assert idxs.dim() == 2, idxs.shape
        assert idxs.shape[0] == probs.shape[0], (idxs.shape, probs.shape)
        assert idxs.shape[1] == probs.shape[2], (idxs.shape, probs.shape)

        # Construct mask indicating positions of actual bases
        mask = F.one_hot(idxs, num_classes=16).type(torch.bool)

        # Return probabilities at real bases
        return torch.masked_select(probs.swapaxes(1, 2).cpu().detach(), mask).reshape(
            idxs.shape
        )

    def P_seqs(self, seqs, labels, per_pos=False, log=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=True
        )  # N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=False)[
            :, :, :-1
        ]  # N, 16, label+seq+1

        # Compute likelihood
        idxs = idxs[:, 1:]  # N, label+seq+1
        L = self.probs_to_likelihood(probs, idxs).numpy()  # N, label+seq+1

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_seqs_given_labels(self, seqs, labels, per_pos=False, log=True, add_stop=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        # Encode sequence with labels
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=add_stop
        )  # N, 0+label+seq OR N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=True)[
            :, :, :-1
        ]  # N, 16, seq OR N, 16, seq+1
        idxs = idxs[:, 1 + self.label_len :]  # N, seq

        # Compute likelihoods
        L = self.probs_to_likelihood(probs=probs, idxs=idxs).numpy()  # N, seq

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_labels_given_seqs(self, seqs, labels, per_pos=True, log=True):
        # List all possible labels
        possible_labels = [
            "".join(x)
            for x in itertools.product(self.unique_labels, repeat=self.label_len)
        ]
        # Compute likelihood for each possible label
        likelihoods = np.stack(
            [
                self.P_seqs(
                    seqs,
                    [label] * len(seqs),
                    per_pos=True,
                    log=False,
                )
                for label in possible_labels
            ],
            axis=-1,
        )  # N, L, possible_labels
        # Calculate numerator and denominator for posterior
        denominators = np.sum(likelihoods, -1)  # N, L
        numerators = np.vstack(
            [
                likelihoods[i, :, np.array(possible_labels) == labels[i]]
                for i in range(len(seqs))
            ]
        )

        # Compute posterior
        if log:
            P = np.log(numerators) - np.log(denominators)  # N, L
        else:
            P = np.divide(numerators, denominators)  # N, L
        if per_pos:
            return P
        else:
            if log:
                return np.sum(P, 1)  # N
            else:
                return np.product(P, 1)  # N

    def normalize_filtered_probs(self, filtered_probs):
        """
        Normalize probabilities at each position to sum to 1.

        Args:
            filtered_probs (torch.floatTensor): Tensor of shape (N, 16, L) or (N, 16)

        Returns:
            Normalized tensor of the same shape
        """
        return filtered_probs / filtered_probs.sum(dim=1, keepdim=True)

    def filter_base_probs(self, probs, normalize=True):
        """
        Return probabilities for valid bases only

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            normalize (bool): Whether to re-normalize the probabilities at each
            position to sum to 1.

        Returns:
            filtered_probs (torch.FloatTensor): tensor of shape (N, 4)
        """
        # Check shape
        assert probs.dim() == 2
        assert probs.shape[1] == 16

        # Filter probabilities
        filtered_probs = probs[:, [7, 8, 9, 10]]

        # Normalize
        if normalize:
            filtered_probs = self.normalize_filtered_probs(filtered_probs)
        return filtered_probs

    def threshold_probs(self, filtered_probs, top_k=None, top_p=None):
        """
        Threshold the filtered probabilities for valid bases

        Args:
            filtered_probs (torch.tensor, dtype torch.float32): tensor of shape (N, 4)
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.

        Returns:
            tensor of shape (N, 4)
        """
        # Check shape
        assert filtered_probs.dim() == 2
        assert filtered_probs.shape[1] == 4

        # Top K sampling
        if top_k is not None:
            p_idxs = filtered_probs.argsort(1, descending=True)
            for seq_idx in range(filtered_probs.shape[0]):
                filtered_probs[seq_idx, p_idxs[seq_idx, top_k:]] = 0

        # Top P (nucleus) sampling
        if top_p is not None:
            p_sorted, p_idxs = filtered_probs.sort(1, descending=True)
            cut = (p_sorted.cumsum(1) > top_p).cpu().detach().numpy().argmax(1).tolist()
            for seq_idx, cut_idx in enumerate(cut):
                if cut_idx < 3:
                    filtered_probs[seq_idx, p_idxs[seq_idx, cut_idx + 1 :]] = 0

        return filtered_probs

    def sample_idxs(
        self, probs, random_state=None, top_k=None, top_p=None, normalize_filtered=True
    ):
        """
        Sample from model predictions at a single position to return a single
        base per example

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            random_state (torch.Generator): torch.Generator object
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering

        Returns:
            idxs (torch.LongTensor): tensor of shape (N)
        """
        # Check
        assert probs.dim() == 2, probs.shape
        assert probs.shape[1] == 16, probs.shape

        # Subset to valid bases
        probs = self.filter_base_probs(probs, normalize=normalize_filtered)  # N, 4
        # print(probs)
        # print(probs.shape)

        # Threshold probabilities for sampling
        probs = self.threshold_probs(probs, top_k=top_k, top_p=top_p)  # N, 4
        # print(probs)
        # print(probs.shape)
        # Re-normalize
        probs = self.normalize_filtered_probs(probs)
        
        # print(probs)
        # print(probs.shape)

        # Sample
        if random_state is None:
            idxs = probs.multinomial(1).squeeze() + 7
        else:
            idxs = probs.multinomial(1, generator=random_state).squeeze() + 7

        # Send to device
        return idxs.to(probs.device)

    @torch.no_grad()
    def generate(
        self,
        labels,
        max_new_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        normalize_filtered=True,
        seed=None,
        label_len = 1,
    ):
        """
        Args:
            labels (str, list): Strings of label tokens
            max_new_tokens (int): Maximum number of tokens to add
            temperature (float): Temperature
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering
            seed (int): Random seed for sampling

        Returns:
            seqs (list): List of strings
        """
        # Check labels
        if isinstance(labels, str):
            labels = [labels]
        # assert len(labels[0]) == self.label_len

        # bases to add
        if max_new_tokens is None:
            max_new_tokens = self.seq_len
        
        # Encode labels
        idxs = self.encode_labels(labels, add_start=False).to(
            self.device
        )  # N, label_len+1
        self.label_len = idxs.shape[1] - 1
        # idxs = self.encode_labels(labels, add_start=False).to(
        #     self.device
        # )  # N, label_len
        # print("idx shape", idxs.shape)
        # Get random state
        rng = torch.Generator(device=self.device)
        
        # idxs = self.encode_labels(labels, add_start=False).to(
        #     self.device
        # )  # N, label_len+1
        # print("idx shape", idxs.shape)
        # # Get random state
        # rng = torch.Generator(device='cpu')
        if seed is not None:
            rng.manual_seed(seed)

        # Add bases
        
        with torch.no_grad():
            for idx_c in range(max_new_tokens):
                self.eval()
                if idx_c ==0:
                    # Predict next base probabilities
                    probs_next = self.forward(idxs, return_logits=False)[:, :, -1]  # N, 16
                else:
                    probs_next = self.forward_joint(idxs, return_logits=False)[:, :, -1]  # N, 16
                # print(probs_next.shape)
    
                # Get next indices
                idxs_next = self.sample_idxs(
                    probs_next,
                    random_state=rng,
                    top_k=top_k,
                    top_p=top_p,
                    normalize_filtered=normalize_filtered,
                )  # N
                # print(idxs_next)
                # Add new indices
                if idxs_next.shape == None:
                    idxs = torch.cat((idxs, idxs_next.unsqueeze(1)), dim=1)
                else:
                    idxs = torch.cat((idxs, idxs_next.reshape(-1,1)), dim=1)
                del probs_next
        
        # print(idxs.shape)
        return self.decode(idxs[:, self.label_len + 1 :])

    def beam_search(
        self, beam_width, batch_size, label, random_state=None, sample=False
    ):
        bases = self.encode_seqs(["ACGT"]).T
        idxs = self.encode_labels([label], add_start=True)
        self.eval()

        for i in tqdm.tqdm(range(self.seq_len)):
            # Construct all possible sequences
            idxs = idxs.repeat_interleave(4, 0)
            bases_ = bases.tile(idxs.shape[0] // 4, 1)
            possibilities = torch.hstack((idxs, bases_))
            probs = []

            # Compute likelihood of each sequence
            with torch.no_grad():
                for st in range(0, possibilities.shape[0], batch_size):
                    en = min(st + batch_size, possibilities.shape[0])
                    batch = possibilities[st:en].to(self.device)
                    batch_probs = self.forward(batch, drop_label=True)[:, :, :-1].cpu()
                    probs.append(batch_probs)

            probs = torch.cat(probs)
            L = self.probs_to_likelihood(
                probs=probs, idxs=possibilities[:, self.label_len + 1 :]
            )
            L = torch.sum(torch.log(L), 1)

            # Sample sequences for next iteration
            curr_beam_width = min(beam_width, len(L))
            if sample:
                L = torch.exp(L)
                if random_state is None:
                    choice = L.multinomial(curr_beam_width).squeeze()
                else:
                    choice = L.multinomial(
                        curr_beam_width, generator=random_state
                    ).squeeze()
            else:
                choice = L.numpy().argsort()[-curr_beam_width:]
            idxs = possibilities[choice, :]

        # Decode
        seqs = self.decode(idxs[:, self.label_len + 1 :])
        return seqs

    def on_save_checkpoint(self, checkpoint):
        """
        Save data relevant parameters to the model checkpoint on training.
        """
        checkpoint["hyper_parameters"]["seq_len"] = self.seq_len
        checkpoint["hyper_parameters"]["label_len"] = self.label_len
        checkpoint["hyper_parameters"]["unique_labels"] = self.unique_labels
        
        
class EmbLanguageDataset(Dataset):
    def __init__(self, seqs, labels, seq_len=None, llm_name = "meta-llama/Llama-3.2-3B-Instruct", max_token_len=15):
        """
        A dataset class to produce tokenized sequences for training regLM.

        Each sequence will be represented as 0<LABEL><SEQ>1; hence 0 is the start
        token and 1 is the end token.

        Args:
            seqs (list): List of sequences.
            labels (list): List of labels as strings
            seq_len (int): Maximum sequence length
        """
        # Check
        assert len(seqs) == len(labels), "seqs and labels should have equal length"
        # assert (
        #     len(set([len(x) for x in labels])) == 1
        # ), "All labels should be of equal length"

        # Store data
        self.seqs = seqs
        self.labels = labels
        self.max_token_len = max_token_len

        # maximum sequence length
        self.seq_len = seq_len or np.max([len(seq) for seq in self.seqs])
        # self.label_len = len(self.labels[0])
        self.label_len = 1
        self.unique_labels = set(
            np.concatenate([[tok for tok in lab] for lab in self.labels])
        )
        # assert (
        #     len(self.unique_labels) <= 10
        # ), ">10 label classes are currently not supported"

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}
        
        device_map = 'cuda'
        # model_name="meta-llama/Llama-3.1-8B-Instruct"
        model_name = llm_name
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer 
        
        

    def __len__(self):
        return len(self.seqs)

    def encode_seq(self, seq):
        """
        Encode a sequence as a torch tensor of tokens

        Args:
            seq (str): DNA sequence

        Returns:
            torch.LongTensor of shape (seq_len,)
        """
        return torch.LongTensor([self.base_stoi[tok] for tok in seq])

    def encode_label(self, label):
        """
        Encode a label as a torch tensor of tokens

        Args:
            label (str): label token sequence

        Returns:
            torch.LongTensor of shape (label_len,)
        """
        return torch.IntTensor(self.tokenizer(label,padding='max_length', truncation=True, max_length=self.max_token_len, padding_side='left')['input_ids'])

    def decode(self, idxs, is_labeled=False):
        """
        Given a torch tensor of tokens, return the decoded sequence as a string.

        Args:
            idxs (list, torch.LongTensor): list or 1-D tensor
            is_labeled (bool): Whether labels are included

        Returns:
            labeled sequence as a string
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().tolist()
        if is_labeled:
            # Split the input into sequence and label
            label = idxs[: self.label_len]
            seq = idxs[self.label_len :]
            # Decode them separately and rejoin
            return "".join(
                [self.label_itos[i] for i in label] + [self.base_itos[i] for i in seq]
            )
        else:
            # Only a sequence is provided
            return "".join([self.base_itos[i] for i in idxs])

    def __getitem__(self, idx):
        """
        Return a single labeled example as a tensor of tokens
        x = 0<LABEL><SEQ>
        y = <SEQ>1

        Args:
            idx: Index of example to return

        Returns:
            x (torch.LongTensor): tensor of shape (1 + self.label_len + self.seq_len)
            y (torch.LongTensor): tensor of shape (self.seq_len + 1, )
        """
        # Get sequence
        seq = self.seqs[idx]

        # Encode sequence
        seq = self.encode_seq(seq)

        # Get label
        label = self.labels[idx]

        # Encode label
        # label = self.encode_label(label)
        instruct = label

        # Generate empty tensors
        x = torch.zeros(self.seq_len + self.label_len + 1, dtype=torch.long)
        y = torch.zeros(self.seq_len + 1, dtype=torch.long)

        # Input: START(0) + label + sequence + trailing zeros (will be ignored)
        x[1 : 1 + self.label_len] = -1
        x[1 + self.label_len : 1 + self.label_len + len(seq)] = seq

        # Output: sequence + END (1) + trailing zeros (will be ignored)
        y[: len(seq)] = seq
        y[len(seq)] = 1

        return instruct, x, y


def load_pretrained_model(
    ckpt_dir="./checkpoints/",
    model="hyenadna-medium-160k-seqlen",
    hyenadna_path="/code/hyena-dna",
):
    """
    Load a pretrained hyenaDNA foundation model.

    Args:
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        model (str): Name of model to load
        hyenadna_path (str): Path to cloned hyenaDNA repository

    Returns:
        model (nn.Module): pre-trained HyenaDNA foundation model
    """
    sys.path.append(hyenadna_path)
    from src.models.sequence.long_conv_lm import ConvLMHeadModel

    # Check model name
    assert model in HYENADNA_MODEL_NAMES

    # Make directory if needed
    if not os.path.exists(ckpt_dir):
        print("Making checkpoint directory")
        os.makedirs(ckpt_dir)

    # Download model if not already downloaded
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print("Downloading model")
        config = f"https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        ckpt = f"https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        os.system(f"wget -P {ckpt_dir} {config}")
        os.system(f"wget -P {ckpt_dir} {ckpt}")

    # Load config
    config = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))

    # Generate model
    model = ConvLMHeadModel(**config)

    # Load weights
    state_dict = torch.load(
        os.path.join(ckpt_dir, "weights.ckpt"), map_location=torch.device("cpu")
    )
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["state_dict"], "model."
    )
    model_state_dict = state_dict["state_dict"]
    for key in list(model_state_dict.keys()):
        if "torchmetrics" in key:
            model_state_dict.pop(key)

    model.load_state_dict(model_state_dict)
    return model


class EmbLightningModel(pl.LightningModule):
    """
    LightningModule class to train and use autoregressive token-conditioned
    regLM language models.

    Args:
        config (dict): Config dictionary containing model parameters
        ckpt_dir (str): Path to directory containing downloaded model checkpoints,
            or in which they should be downloaded
        hyenadna_path (str): Path to cloned hyenaDNA repository
        save_dir (str): Directory to save model checkpoints and logs
        lr (float): Learning rate
        label_len (int): Number of label tokens preceding each DNA sequence
    """

    def __init__(
        self,
        config=None,
        ckpt_dir="./checkpoints/hyenadna-medium-160k-seqlen",
        hyenadna_path="/code/hyena-dna",
        save_dir=".",
        lr=1e-4,
        label_len=None,
        llm_name = "meta-llama/Llama-3.2-3B-Instruct",
        latent_size = 3072,
        align_head = 'relu',
        max_token_len = 15
    ):
        super().__init__()

        self.save_dir = save_dir
        self.label_len = label_len
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.latent_size = latent_size
        self.max_token_len = max_token_len

        # Build model
        if ckpt_dir is not None:
            self.model = load_pretrained_model(
                ckpt_dir=ckpt_dir, hyenadna_path=hyenadna_path
            )
        elif config is not None:
            sys.path.append(hyenadna_path)
            from src.models.sequence.long_conv_lm import ConvLMHeadModel

            self.model = ConvLMHeadModel(**config)
        else:
            raise ValueError("either config or ckpt_dir must be provided.")

        # Print number of model parameters
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print("number of parameters: %.2fM" % (self.n_params / 1e6,))

        # Metrics: accuracy
        self.train_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)
        self.val_acc = Accuracy(task="multiclass", num_classes=16, ignore_index=0)

        # Encoding
        self.label_stoi = {
            "0": 2,
            "1": 3,
            "2": 4,
            "3": 5,
            "4": 6,
            "5": 7,
            "6": 8,
            "7": 9,
            "8": 10,
            "9": 11,
        }
        self.base_stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }
        self.label_itos = {v: k for k, v in self.label_stoi.items()}
        self.base_itos = {v: k for k, v in self.base_stoi.items()}

        # Loss function
        # Trailing zeros in the label will be ignored in calculating the loss
        self.loss = lambda logits, y: F.cross_entropy(logits, y, ignore_index=0)
        
        device_map = 'cuda'
        # model_name="meta-llama/Llama-3.1-8B-Instruct"
        model_name = llm_name
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = device_map)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer 
        
        model_llama = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map)
        print("Successfully create Llama")
        
        if align_head == 'relu':
            self.emb_align = nn.Sequential(nn.Linear(self.latent_size,self.latent_size), nn.GELU(), nn.Linear(self.latent_size,256))
            
        if align_head == 'linear':
            self.emb_align = nn.Linear(self.latent_size,256)
        print("Successfully create aligner")
        
        # self.Llama_encoder = model_llama.cuda()
        self.model = self.model.cuda()
        self.emb_align = self.emb_align.cuda()
        
        # print(self.Llama_encoder.device)

    def forward(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        
        if isinstance(x, list):
            instruct = x[0]
            # print(instruct.shape)
            x = x[1]
            # print(x.shape)
            
            instruct_emb = instruct
            # print(instruct_emb.shape)
            # print(instruct_emb.shape)
            instruct_emb = self.emb_align(instruct_emb)
            # print(instruct_emb.shape)
            # print(x[:,2:].shape)
#             seq_enc = self.model.backbone.forward(x[:,2:])
#             # print(seq_enc.shape)
#             final_emb = torch.cat((instruct_emb,seq_enc), dim=1)

            seq_enc = self.model.backbone.forward_multimodal(x[:,2:],instruct_emb)
            final_emb = seq_enc
            
            # print(final_emb.shape)
            # print(final_emb.device)
            logits = self.model.lm_head(final_emb).swapaxes(
                1, 2
            )  # N, label + seq + end + trailing zeros
            
            # print(logits.shape)
    
            # Drop the label probabilities
            if drop_label:
                # print(logits.shape)
                logits = logits[:, :, instruct_emb.shape[1]-1:]  # N, seq + end + trailing
    
            # Return logits or normalized probabilities
            if return_logits:
                return logits
            else:
                return logits.softmax(1)
        else:
            instruct = x.cuda()
            # print(len(instruct))
            instruct_emb = instruct
            instruct_emb = self.emb_align(instruct_emb)
            instruct_emb = self.model.backbone.forward_multimodal(None,instruct_emb)
#             instruct_emb = seq_enc
            
            logits = self.model.lm_head(instruct_emb).swapaxes(
                1, 2
            )  # N, label + seq + end + trailing zeros
            
            # print(logits.shape)
    
            # Drop the label probabilities
            if drop_label:
                logits = logits[:, :,  instruct_emb.shape[1]-1:]  # N, seq + end + trailing
                # print(logits.shape)
    
            # Return logits or normalized probabilities
            if return_logits:
                return logits
            else:
                return logits.softmax(1)
                
    def forward_joint(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        # instruct = x[0:x.shape[0]-1]
        # # print(instruct.shape)
        # x = x[x.shape[0]-1:]
        # # print(x.shape)
#         print(x)
#         print(x.shape)
        instruct = x[:,0:self.max_token_len]
        x = x[:,self.max_token_len:]
        instruct_emb = instruct
        # print(instruct_emb.shape)
        # print(instruct_emb.shape)
        instruct_emb = self.emb_align(instruct_emb)
        # print(instruct_emb.shape)
        # print(x[:,2:].shape)
#         seq_enc = self.model.backbone(x)
#         # print(seq_enc.shape)
#         final_emb = torch.cat((instruct_emb,seq_enc), dim=1)

        seq_enc = self.model.backbone.forward_multimodal(x[:,2:],instruct_emb)
        final_emb = seq_enc
        
        # print(final_emb.shape)
        # print(final_emb.device)
        logits = self.model.lm_head(final_emb).swapaxes(
            1, 2
        )  # N, label + seq + end + trailing zeros
        
        # print(logits.shape)

        # Drop the label probabilities
        if drop_label:
            # print(logits.shape)
            logits = logits[:, :, final_emb.shape[1]-1:]  # N, seq + end + trailing

        # Return logits or normalized probabilities
        if return_logits:
            return logits
        else:
            return logits.softmax(1)

    def forward_emb(self, x, drop_label=True, return_logits=False):
        """
        Args:
            x (torch.tensor, dtype torch.float32): tensor of shape (N, L)
            drop_label (bool): Whether to drop the predictions for the
                positions corresponding to label tokens
            return_logits (bool): If true, return logits. Otherwise, return
                probabilities

        Returns:
            logits (torch.tensor, dtype torch.float32): tensor of shape
                (N, 16, L - label_len) if drop_label is True,
                or (N, 16, L) if drop_label is False.
                Note that the prediction for the END token (1) as well as the
                hypothetical position after it will be included.
        """
        
        if isinstance(x, list):
            instruct = x[0]
            # print(instruct.shape)
            x = x[1]
            # print(x.shape)
            
            instruct_emb = instruct
            # print(instruct_emb.shape)
            # print(instruct_emb.shape)
            instruct_emb = self.emb_align(instruct_emb)
            # print(instruct_emb.shape)
            # print(x[:,2:].shape)
            seq_enc = self.model.backbone.embeddings(x[:,2:])
            
            return instruct_emb.mean(axis=1), seq_enc.mean(axis=1)
        
    def training_step(self, batch, batch_idx):
        inst, x, y = batch
        logits = self.forward(
            [inst,x], drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
#         ins_emb,seq_emb = self.forward_emb(
#             [inst,x], drop_label=True, return_logits=True
#         )  # N, seq + end + trailing
#         loss+= F.mse_loss(ins_emb,seq_emb)
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inst, x, y = batch
        logits = self.forward(
            [inst,x], drop_label=True, return_logits=True
        )  # N, seq + end + trailing
        loss = self.loss(logits, y)  # Loss will be calculated over seq + end positions
#         ins_emb,seq_emb = self.forward_emb(
#             [inst,x], drop_label=True, return_logits=True
#         )  # N, seq + end + trailing
#         loss+= F.mse_loss(ins_emb,seq_emb)
        self.val_acc.update(logits.argmax(1), y)
        self.log(
            "val_loss",
            loss,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_epoch_end(self, output):
        val_acc = self.val_acc.compute()
        self.log("val_acc", val_acc)
        val_loss = torch.mean(torch.Tensor(output))
        print(f"\nVal loss: {val_loss}, val acc: {val_acc}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.model.parameters() ) + list(self.emb_align.parameters()), lr=float(self.lr))
        return optimizer

    def train_on_dataset(
        self,
        train_dataset,
        val_dataset,
        batch_size=128,
        num_workers=8,
        device=0,
        max_epochs=3,
        val_check_interval=5000,
        weights=None,
        save_all=False,
    ):
        """
        Train regLM model.

        Args:
            train_dataset (CharDataset): Training dataset
            val_dataset (CharDataset): Validation dataset
            batch_size (int): Batch size
            num_workers (int): Number of workers for training
            device (int): GPU index
            max_epochs (int): Number of epochs to train
            val_check_interval (int): Number of steps after which to
                check validation loss

        Returns:
            pl.Trainer object
        """
        torch.set_float32_matmul_precision("medium")

        # Save dataset params
        self.seq_len = train_dataset.seq_len
        self.label_len = train_dataset.label_len
        self.unique_labels = train_dataset.unique_labels

        # Logger
        logger = CSVLogger(self.save_dir)

        # Create callback
        callbacks = [
            ModelCheckpoint(monitor="val_acc", mode="max"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ]
        if save_all:
            callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=-1))

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu",
            devices=[device],
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.save_dir,
            val_check_interval=val_check_interval,
        )

        # Make dataloaders
        if weights is None:
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            train_data = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                sampler=sampler,
            )

        val_data = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # First validation pass
        trainer.validate(model=self, dataloaders=val_data)

        # Training
        trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)

        return trainer

    def compute_accuracy_on_dataset(
        self,
        dataset,
        batch_size=64,
        num_workers=8,
    ):
        """
        Perform inference on a dataset and return per-example accuracy
        Note: this will include the accuracy of predicting the END token (1)

        Args:
            dataset (CharDataset): Inference dataset
            batch_size (int): Batch size for inference
            num_workers (int): Number of workers for inference

        Returns: List of booleans indicating whether the predicted base at each position
        was equal to the true label or not.
        """
        # Make dataloader
        dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Empty lists to hold predictions
        y_hat = []
        y = []

        self.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(iter(dl)):
                ins = batch[0].to(self.device)
                x = batch[1].to(self.device)
                probs = self.forward(
                    [ins,x], drop_label=True
                ).squeeze()  # N, 16, seq+end+trailing
                y_hat.append(probs.argmax(1).cpu().detach())  # N, seq+end+trailing
                y.append(batch[2].detach().squeeze())  # N, seq+end+trailing zeros

        # Stack batched predictions
        y_hat = torch.vstack(y_hat).numpy()  # N, L
        y = torch.vstack(y).numpy()  # N, L

        # Compare predicted base and true label where the true label is not 0.
        # Hence, this computation will include the END token (1) but not any
        # hypothetical bases / trailing zeros.
        is_non_zero = y != 0
        is_equal = y_hat == y

        # Return booleans
        return [e[n] for e, n in zip(is_equal, is_non_zero)]

    def encode_labels(self, labels, add_start=False):
        """
        Encode labels as a list of indices for inference

        Args:
            labels (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        if isinstance(labels, str):
            labels = [labels]

        # # Convert label to indices
        # idxs = torch.LongTensor(
        #     [[self.label_stoi[tok] for tok in label] for label in labels]
        # )  # N, label_len

        # # Add start token
        # if add_start:
        #     idxs = torch.cat(
        #         (torch.LongTensor([[0]] * idxs.shape[0]), idxs), axis=1
        #     )  # N, 1 + label_len
        # return idxs
        idxs = torch.IntTensor(self.tokenizer(labels,padding='max_length', truncation=True, max_length=self.max_token_len, padding_side='left')['input_ids'])
        idxs = idxs.to(self.device)
        # print(idxs.shape)
        return idxs

    def encode_seqs(self, seqs, add_stop=False):
        """
        Encode sequences as lists of indices for inference

        Args:
            seqs (list, str): DNA sequence(s) as string or list of strings
            add_stop (bool): Whether to add an end token (1) after each sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L) if add_stop is False
            or (N, L+1) if add_stop is True.
        """
        if isinstance(seqs, str):
            seqs = [seqs]

        # Convert sequences to indices
        idxs = torch.LongTensor(
            [[self.base_stoi[tok] for tok in seq] for seq in seqs]
        )  # N, seq_len

        # Add END token
        if add_stop:
            idxs = torch.cat(
                (idxs, torch.LongTensor([[1]] * idxs.shape[0])), axis=1
            )  # N, seq_len+1

        return idxs

    def encode(self, seqs, labels, add_start=False, add_stop=False):
        """
        Encode sequences and labels as indices for model inference.

        Args:
            seqs (list, str): Strings of base tokens
            label (list, str): Strings of label tokens
            add_start (bool): Whether to add the start token (0)
            add_stop (bool): Add an end token (1) after the sequence

        Returns:
            idxs (torch.LongTensor): tensor of shape (N, L)
        """
        return torch.cat(
            [
                self.encode_labels(labels, add_start=add_start),
                self.encode_seqs(seqs, add_stop=add_stop),
            ],
            axis=1,
        )

    def decode(self, idxs):
        """
        Decodes indices into DNA sequences

        Args:
            idxs (torch.LongTensor): tensor or array of shape (N, L)

        Returns:
            seqs (list): list of strings
        """
        if idxs.dim() == 1:  # L
            idxs = idxs.unsqueeze(0)  # 1, L

        idxs = idxs.cpu().detach().numpy()

        # Create empty list
        seqs = []

        # Fill list
        for x in idxs:
            curr_seq = []

            # Decode
            for pos, ix in enumerate(x):
                # Ignore start token
                if (pos == 0) and (ix == 0):
                    continue

                # Terminate at end token
                if ix == 1:
                    break

                # Replace non-bases with N
                elif (ix < 7) or (ix > 11):
                    ix = 11

                curr_seq.append(self.base_itos[ix])

            seqs.append("".join(curr_seq))
        return seqs

    def probs_to_likelihood(self, probs, idxs):
        """
        Compute the likelihood of each base in a sequence given model predictions
        on the sequence.

        Args:
            probs (torch.FloatTensor): tensor of shape (N, 16, L)
            idxs (torch.LongTensor): tensor of shape (N, L)

        Returns:
            tensor of shape (N, L) containing the probabilities of real bases
        """
        # Check shapes
        assert probs.dim() == 3, probs.shape
        assert probs.shape[1] == 16, probs.shape
        assert idxs.dim() == 2, idxs.shape
        assert idxs.shape[0] == probs.shape[0], (idxs.shape, probs.shape)
        assert idxs.shape[1] == probs.shape[2], (idxs.shape, probs.shape)

        # Construct mask indicating positions of actual bases
        mask = F.one_hot(idxs, num_classes=16).type(torch.bool)

        # Return probabilities at real bases
        return torch.masked_select(probs.swapaxes(1, 2).cpu().detach(), mask).reshape(
            idxs.shape
        )

    def P_seqs(self, seqs, labels, per_pos=False, log=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=True
        )  # N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=False)[
            :, :, :-1
        ]  # N, 16, label+seq+1

        # Compute likelihood
        idxs = idxs[:, 1:]  # N, label+seq+1
        L = self.probs_to_likelihood(probs, idxs).numpy()  # N, label+seq+1

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_seqs_given_labels(self, seqs, labels, per_pos=False, log=True, add_stop=True):
        """
        Args:
            seqs (list, str): Sequences as strings
            labels(list, str): Labels as strings
            log (bool): Return log likelihood
            include_end (bool): Include the end token

        Returns:
            np.array of shape (N)
        """
        # Encode sequence with labels
        idxs = self.encode(
            seqs, labels, add_start=True, add_stop=add_stop
        )  # N, 0+label+seq OR N, 0+label+seq+1

        # Compute probabilities
        self.eval()
        probs = self.forward(idxs.to(self.device), drop_label=True)[
            :, :, :-1
        ]  # N, 16, seq OR N, 16, seq+1
        idxs = idxs[:, 1 + self.label_len :]  # N, seq

        # Compute likelihoods
        L = self.probs_to_likelihood(probs=probs, idxs=idxs).numpy()  # N, seq

        # Log and sum
        if log:
            L = np.log(L)
        if per_pos:
            return L
        else:
            if log:
                return np.sum(L, 1)  # N
            else:
                return np.product(L, 1)  # N

    def P_labels_given_seqs(self, seqs, labels, per_pos=True, log=True):
        # List all possible labels
        possible_labels = [
            "".join(x)
            for x in itertools.product(self.unique_labels, repeat=self.label_len)
        ]
        # Compute likelihood for each possible label
        likelihoods = np.stack(
            [
                self.P_seqs(
                    seqs,
                    [label] * len(seqs),
                    per_pos=True,
                    log=False,
                )
                for label in possible_labels
            ],
            axis=-1,
        )  # N, L, possible_labels
        # Calculate numerator and denominator for posterior
        denominators = np.sum(likelihoods, -1)  # N, L
        numerators = np.vstack(
            [
                likelihoods[i, :, np.array(possible_labels) == labels[i]]
                for i in range(len(seqs))
            ]
        )

        # Compute posterior
        if log:
            P = np.log(numerators) - np.log(denominators)  # N, L
        else:
            P = np.divide(numerators, denominators)  # N, L
        if per_pos:
            return P
        else:
            if log:
                return np.sum(P, 1)  # N
            else:
                return np.product(P, 1)  # N

    def normalize_filtered_probs(self, filtered_probs):
        """
        Normalize probabilities at each position to sum to 1.

        Args:
            filtered_probs (torch.floatTensor): Tensor of shape (N, 16, L) or (N, 16)

        Returns:
            Normalized tensor of the same shape
        """
        return filtered_probs / filtered_probs.sum(dim=1, keepdim=True)

    def filter_base_probs(self, probs, normalize=True):
        """
        Return probabilities for valid bases only

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            normalize (bool): Whether to re-normalize the probabilities at each
            position to sum to 1.

        Returns:
            filtered_probs (torch.FloatTensor): tensor of shape (N, 4)
        """
        # Check shape
        assert probs.dim() == 2
        assert probs.shape[1] == 16

        # Filter probabilities
        filtered_probs = probs[:, [7, 8, 9, 10]]

        # Normalize
        if normalize:
            filtered_probs = self.normalize_filtered_probs(filtered_probs)
        return filtered_probs

    def threshold_probs(self, filtered_probs, top_k=None, top_p=None):
        """
        Threshold the filtered probabilities for valid bases

        Args:
            filtered_probs (torch.tensor, dtype torch.float32): tensor of shape (N, 4)
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.

        Returns:
            tensor of shape (N, 4)
        """
        # Check shape
        assert filtered_probs.dim() == 2
        assert filtered_probs.shape[1] == 4

        # Top K sampling
        if top_k is not None:
            p_idxs = filtered_probs.argsort(1, descending=True)
            for seq_idx in range(filtered_probs.shape[0]):
                filtered_probs[seq_idx, p_idxs[seq_idx, top_k:]] = 0

        # Top P (nucleus) sampling
        if top_p is not None:
            p_sorted, p_idxs = filtered_probs.sort(1, descending=True)
            cut = (p_sorted.cumsum(1) > top_p).cpu().detach().numpy().argmax(1).tolist()
            for seq_idx, cut_idx in enumerate(cut):
                if cut_idx < 3:
                    filtered_probs[seq_idx, p_idxs[seq_idx, cut_idx + 1 :]] = 0

        return filtered_probs

    def sample_idxs(
        self, probs, random_state=None, top_k=None, top_p=None, normalize_filtered=True
    ):
        """
        Sample from model predictions at a single position to return a single
        base per example

        Args:
            probs (torch.tensor, dtype torch.float32): tensor of shape (N, 16)
            random_state (torch.Generator): torch.Generator object
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering

        Returns:
            idxs (torch.LongTensor): tensor of shape (N)
        """
        # Check
        assert probs.dim() == 2, probs.shape
        assert probs.shape[1] == 16, probs.shape

        # Subset to valid bases
        probs = self.filter_base_probs(probs, normalize=normalize_filtered)  # N, 4
        # print(probs)
        # print(probs.shape)

        # Threshold probabilities for sampling
        probs = self.threshold_probs(probs, top_k=top_k, top_p=top_p)  # N, 4
        # print(probs)
        # print(probs.shape)
        # Re-normalize
        probs = self.normalize_filtered_probs(probs)
        
        # print(probs)
        # print(probs.shape)

        # Sample
        if random_state is None:
            idxs = probs.multinomial(1).squeeze() + 7
        else:
            idxs = probs.multinomial(1, generator=random_state).squeeze() + 7

        # Send to device
        return idxs.to(probs.device)

    @torch.no_grad()
    def generate(
        self,
        labels,
        max_new_tokens=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        normalize_filtered=True,
        seed=None,
        label_len = 1,
    ):
        """
        Args:
            labels (str, list): Strings of label tokens
            max_new_tokens (int): Maximum number of tokens to add
            temperature (float): Temperature
            top_k (int): Select the top k bases at each position. Set probabilites
                of other bases to 0.
            top_p (float): Select the top bases at each position until their cumulative
                probability reaches this value. Set probabilites of other bases to 0.
            normalize_filtered (bool): Normalize probabilities to sum to 1
                after filtering
            seed (int): Random seed for sampling

        Returns:
            seqs (list): List of strings
        """
        # Check labels
        if isinstance(labels, str):
            labels = [labels]
        # assert len(labels[0]) == self.label_len

        # bases to add
        if max_new_tokens is None:
            max_new_tokens = self.seq_len
        
        # Encode labels
        idxs = self.encode_labels(labels, add_start=False).to(
            self.device
        )  # N, label_len+1
        self.label_len = idxs.shape[1] - 1
        # idxs = self.encode_labels(labels, add_start=False).to(
        #     self.device
        # )  # N, label_len
        # print("idx shape", idxs.shape)
        # Get random state
        rng = torch.Generator(device=self.device)
        
        # idxs = self.encode_labels(labels, add_start=False).to(
        #     self.device
        # )  # N, label_len+1
        # print("idx shape", idxs.shape)
        # # Get random state
        # rng = torch.Generator(device='cpu')
        if seed is not None:
            rng.manual_seed(seed)

        # Add bases
        
        with torch.no_grad():
            for idx_c in range(max_new_tokens):
                self.eval()
                if idx_c ==0:
                    # Predict next base probabilities
                    probs_next = self.forward(idxs, return_logits=False)[:, :, -1]  # N, 16
                else:
                    probs_next = self.forward_joint(idxs, return_logits=False)[:, :, -1]  # N, 16
                # print(probs_next.shape)
    
                # Get next indices
                idxs_next = self.sample_idxs(
                    probs_next,
                    random_state=rng,
                    top_k=top_k,
                    top_p=top_p,
                    normalize_filtered=normalize_filtered,
                )  # N
                # print(idxs_next)
                # Add new indices
                if idxs_next.shape == None:
                    idxs = torch.cat((idxs, idxs_next.unsqueeze(1)), dim=1)
                else:
                    idxs = torch.cat((idxs, idxs_next.reshape(-1,1)), dim=1)
                del probs_next
        
        # print(idxs.shape)
        return self.decode(idxs[:, self.label_len + 1 :])

    def beam_search(
        self, beam_width, batch_size, label, random_state=None, sample=False
    ):
        bases = self.encode_seqs(["ACGT"]).T
        idxs = self.encode_labels([label], add_start=True)
        self.eval()

        for i in tqdm.tqdm(range(self.seq_len)):
            # Construct all possible sequences
            idxs = idxs.repeat_interleave(4, 0)
            bases_ = bases.tile(idxs.shape[0] // 4, 1)
            possibilities = torch.hstack((idxs, bases_))
            probs = []

            # Compute likelihood of each sequence
            with torch.no_grad():
                for st in range(0, possibilities.shape[0], batch_size):
                    en = min(st + batch_size, possibilities.shape[0])
                    batch = possibilities[st:en].to(self.device)
                    batch_probs = self.forward(batch, drop_label=True)[:, :, :-1].cpu()
                    probs.append(batch_probs)

            probs = torch.cat(probs)
            L = self.probs_to_likelihood(
                probs=probs, idxs=possibilities[:, self.label_len + 1 :]
            )
            L = torch.sum(torch.log(L), 1)

            # Sample sequences for next iteration
            curr_beam_width = min(beam_width, len(L))
            if sample:
                L = torch.exp(L)
                if random_state is None:
                    choice = L.multinomial(curr_beam_width).squeeze()
                else:
                    choice = L.multinomial(
                        curr_beam_width, generator=random_state
                    ).squeeze()
            else:
                choice = L.numpy().argsort()[-curr_beam_width:]
            idxs = possibilities[choice, :]

        # Decode
        seqs = self.decode(idxs[:, self.label_len + 1 :])
        return seqs

    def on_save_checkpoint(self, checkpoint):
        """
        Save data relevant parameters to the model checkpoint on training.
        """
        checkpoint["hyper_parameters"]["seq_len"] = self.seq_len
        checkpoint["hyper_parameters"]["label_len"] = self.label_len
        checkpoint["hyper_parameters"]["unique_labels"] = self.unique_labels
