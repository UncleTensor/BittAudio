import torch
import torchaudio
import numpy as np
import os
import sys

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)


# from bark_extras import HuBERTManager, CustomHubert, CustomTokenizer, load_codec_model, generate_text_semantic, preload_models, codec_decode, generate_coarse, generate_fine 
from encodec.utils import convert_audio

import gc
import os
import re
import json
import numpy
import math
import tqdm
import torch
import funcy
import shutil
import logging
import os.path
import os.path
import fairseq
import logging
import hashlib
import contextlib
import numpy as np
import urllib.request
import torch.nn as nn
import huggingface_hub
from pathlib import Path
from zipfile import ZipFile
from torch import nn, optim
import torch.nn.functional as F
from einops import pack, unpack
from encodec import EncodecModel
from dataclasses import dataclass
from scipy.special import softmax
from torch.nn import functional as F
from transformers import BertTokenizer
from torchaudio.functional import resample
from huggingface_hub import hf_hub_download
from einops import rearrange, repeat, reduce
from torch.serialization import MAP_LOCATION
from audiolm_pytorch.utils import curtail_to_multiple
logging.root.setLevel(logging.ERROR)



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        FULL_T = k.shape[-2]

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            if past_kv is not None:
                # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
                # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
                # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so 
                # to work around this we set is_causal=False.
                is_causal = False
            else:
                is_causal = True

            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,FULL_T-T:FULL_T,:FULL_T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.input_vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False, training=False):
        device = idx.device
        b, t = idx.size()
        if past_kv is not None:
            assert t == 1
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        else:
            if merge_context:
                assert(idx.shape[1] >= 256+256+1)
                t = idx.shape[1] - 256
            else:
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the GPT model itself
            if merge_context:
                tok_emb = torch.cat([
                    self.transformer.wte(idx[:,:256]) + self.transformer.wte(idx[:,256:256+256]),
                    self.transformer.wte(idx[:,256+256:])
                ], dim=1)
            else:
                tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_kv[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # shape (1, t)
            assert position_ids.shape == (1, t)

        pos_emb = self.transformer.wpe(position_ids) # position embeddings of shape (1, t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.transformer.h, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv = new_kv + (kv,)

        x = self.transformer.ln_f(x)

        
        if training:
            logits = self.lm_head(x)
            return logits

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return (logits, new_kv)


"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""



class NonCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") and self.dropout == 0.0
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FineGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        del self.lm_head
        self.config = config
        self.n_codes_total = config.n_codes_total
        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList(
                    [
                        nn.Embedding(config.input_vocab_size, config.n_embd)
                        for _ in range(config.n_codes_total)
                    ]
                ),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, self.n_codes_total)
            ]
        )
        for i in range(self.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx, idx):
        device = idx.device
        b, t, codes = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, (b, t, codes)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.transformer.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1

class HuBERTManager:
    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', file_name: str = 'hubert.pt'):
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, file_name)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')
        return install_file


    @staticmethod
    def make_sure_tokenizer_installed(model: str = 'quantifier_hubert_base_ls960_14.pth', repo: str = 'GitMylo/bark-voice-cloning', local_file: str = 'tokenizer.pth'):
        install_dir = os.path.join('data', 'models', 'hubert')
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, local_file)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, model, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, model), install_file)
            print('Downloaded tokenizer')
        return install_file
    
"""
Modified HuBERT model without kmeans.
Original author: https://github.com/lucidrains/
Modified by: https://www.github.com/gitmylo/
License: MIT
"""

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class CustomHubert(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        target_sample_hz=16000,
        seq_len_multiple_of=None,
        output_layer=9,
        device=None
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        if device is not None:
            self.to(device)

        model_path = Path(checkpoint_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        if device is not None:
            model[0].to(device)

        self.model = model[0]
        self.model.eval()

    @property
    def groups(self):
        return 1

    @torch.no_grad()
    def forward(
        self,
        wav_input,
        flatten=True,
        input_sample_hz=None
    ):
        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(
            wav_input,
            features_only=True,
            mask=False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
            output_layer=self.output_layer
        )

        embed, packed_shape = pack([embed['x']], '* d')

        # codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        codebook_indices = torch.from_numpy(embed.cpu().detach().numpy()).to(device)  # .long()

        if flatten:
            return codebook_indices

        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        return codebook_indices
    
"""
Custom tokenizer model.
Author: https://www.github.com/gitmylo/
License: MIT
"""

class CustomTokenizer(nn.Module):
    def __init__(self, hidden_size=1024, input_size=768, output_size=10000, version=0):
        super(CustomTokenizer, self).__init__()
        next_size = input_size
        if version == 0:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            next_size = hidden_size
        if version == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
            self.intermediate = nn.Linear(hidden_size, 4096)
            next_size = 4096

        self.fc = nn.Linear(next_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer: optim.Optimizer = None
        self.lossfunc = nn.CrossEntropyLoss()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    def forward(self, x):
        x, _ = self.lstm(x)
        if self.version == 1:
            x = self.intermediate(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    @torch.no_grad()
    def get_token(self, x):
        """
        Used to get the token for the first
        :param x: An array with shape (N, input_size) where N is a whole number greater or equal to 1, and input_size is the input size used when creating the model.
        :return: An array with shape (N,) where N is the same as N from the input. Every number in the array is a whole number in range 0...output_size - 1 where output_size is the output size used when creating the model.
        """
        return torch.argmax(self(x), dim=1)

    def prepare_training(self):
        self.optimizer = optim.Adam(self.parameters(), 0.001)

    def train_step(self, x_train, y_train, log_loss=False):
        # y_train = y_train[:-1]
        # y_train = y_train[1:]

        optimizer = self.optimizer
        lossfunc = self.lossfunc
        # Zero the gradients
        self.zero_grad()

        # Forward pass
        y_pred = self(x_train)

        y_train_len = len(y_train)
        y_pred_len = y_pred.shape[0]

        if y_train_len > y_pred_len:
            diff = y_train_len - y_pred_len
            y_train = y_train[diff:]
        elif y_train_len < y_pred_len:
            diff = y_pred_len - y_train_len
            y_pred = y_pred[:-diff, :]

        y_train_hot = torch.zeros(len(y_train), self.output_size)
        y_train_hot[range(len(y_train)), y_train] = 1
        y_train_hot = y_train_hot.to('cuda')

        # Calculate the loss
        loss = lossfunc(y_pred, y_train_hot)

        # Print loss
        if log_loss:
            print('Loss', loss.item())

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    def save(self, path):
        info_path = '.'.join(os.path.basename(path).split('.')[:-1]) + '/.info'
        torch.save(self.state_dict(), path)
        data_from_model = Data(self.input_size, self.hidden_size, self.output_size, self.version)
        with ZipFile(path, 'a') as model_zip:
            model_zip.writestr(info_path, data_from_model.save())
            model_zip.close()

    @staticmethod
    def load_from_checkpoint(path, map_location: MAP_LOCATION = None):
        old = True
        with ZipFile(path) as model_zip:
            filesMatch = [file for file in model_zip.namelist() if file.endswith('/.info')]
            file = filesMatch[0] if filesMatch else None
            if file:
                old = False
                data_from_model = Data.load(model_zip.read(file).decode('utf-8'))
            model_zip.close()
        if old:
            model = CustomTokenizer()
        else:
            model = CustomTokenizer(data_from_model.hidden_size, data_from_model.input_size, data_from_model.output_size, data_from_model.version)
        model.load_state_dict(torch.load(path))
        if map_location:
            model = model.to(map_location)
        return model



class Data:
    input_size: int
    hidden_size: int
    output_size: int
    version: int

    def __init__(self, input_size=768, hidden_size=1024, output_size=10000, version=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.version = version

    @staticmethod
    def load(string):
        data = json.loads(string)
        return Data(data['input_size'], data['hidden_size'], data['output_size'], data['version'])

    def save(self):
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'version': self.version,
        }
        return json.dumps(data)


def auto_train(data_path, save_path='model.pth', load_model: str | None = None, save_epochs=1):
    data_x, data_y = [], []

    if load_model and os.path.isfile(load_model):
        print('Loading model from', load_model)
        model_training = CustomTokenizer.load_from_checkpoint(load_model, 'cuda')
    else:
        print('Creating new model.')
        model_training = CustomTokenizer(version=1).to('cuda')  # Settings for the model to run without lstm
    save_path = os.path.join(data_path, save_path)
    base_save_path = '.'.join(save_path.split('.')[:-1])

    sem_string = '_semantic.npy'
    feat_string = '_semantic_features.npy'

    ready = os.path.join(data_path, 'ready')
    for input_file in os.listdir(ready):
        full_path = os.path.join(ready, input_file)
        if input_file.endswith(sem_string):
            data_y.append(numpy.load(full_path))
        elif input_file.endswith(feat_string):
            data_x.append(numpy.load(full_path))
    model_training.prepare_training()

    epoch = 1

    while 1:
        for i in range(save_epochs):
            j = 0
            for x, y in zip(data_x, data_y):
                model_training.train_step(torch.tensor(x).to('cuda'), torch.tensor(y).to('cuda'), j % 50 == 0)  # Print loss every 50 steps
                j += 1
        save_p = save_path
        save_p_2 = f'{base_save_path}_epoch_{epoch}.pth'
        model_training.save(save_p)
        model_training.save(save_p_2)
        print(f'Epoch {epoch} completed')
        epoch += 1

# bark.generation 
        
if (
    torch.cuda.is_available() and
    hasattr(torch.cuda, "amp") and
    hasattr(torch.cuda.amp, "autocast") and
    hasattr(torch.cuda, "is_bf16_supported") and
    torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:
    @contextlib.contextmanager
    def autocast():
        yield


# hold models in global scope to lazy load
global models
models = {}

global models_devices
models_devices = {}


CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000


logger = logging.getLogger(__name__)


CUR_PATH = os.path.dirname(os.path.abspath(__file__))


default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "serp", "bark_v0")


USE_SMALL_MODELS = os.environ.get("SERP_USE_SMALL_MODELS", False)
GLOBAL_ENABLE_MPS = os.environ.get("SERP_ENABLE_MPS", False)
OFFLOAD_CPU = os.environ.get("SERP_OFFLOAD_CPU", False)


REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
        "checksum": "b3e42bcbab23b688355cd44128c4cdd3",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
        "checksum": "5fe964825e3b0321f9d5f3857b89194d",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
        "checksum": "5428d1befe05be2ba32195496e58dc90",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
        "checksum": "54afa89d65e318d4f5f80e8e8799026a",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
        "checksum": "8a98094e5e3a255a5c9c0ab7efe8fd28",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
        "checksum": "59d184ed44e3650774a2f0503a48a97b",
    },
}


if not hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available():
    logger.warning(
        "torch version does not support flash attention. You will get faster" +
        " inference speed by upgrade torch to newest nightly version."
    )


def _string_md5(s):
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _get_ckpt_path(model_type, use_small=False, path=None):
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_name = REMOTE_MODEL_PATHS[model_key]["file_name"]
    if path is None:
        path = CACHE_DIR
    return os.path.join(path, f"{model_name}")


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and use_gpu and GLOBAL_ENABLE_MPS:
        device = "mps"
    else:
        device = "cpu"
    return device


def _download(from_hf_path, file_name, to_local_path):
    to_local_path = to_local_path.replace("\\", "/")
    path = '/'.join(to_local_path.split("/")[:-1])
    os.makedirs(path, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=path)
    os.replace(os.path.join(path, file_name), to_local_path)

class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_models(model_key=None):
    global models
    model_keys = [model_key] if model_key is not None else models.keys()
    for k in model_keys:
        if k in models:
            del models[k]
    _clear_cuda_cache()
    gc.collect()


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    # if (
    #     os.path.exists(ckpt_path) and
    #     _md5(ckpt_path) != model_info["checksum"]
    # ):
    #     logger.warning(f"found outdated {model_type} model, removing.")
    #     os.remove(ckpt_path)
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        _download(model_info["repo_id"], model_info["file_name"], ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    # check if config.json is in the same directory as the checkpoint
    # if so, load it
    # otherwise, assume it's in the checkpoint
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_args = json.load(f)
    else:
        model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**model_args)
    model = ModelClass(gptconf)
    if checkpoint.get("model", None) is not None:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    unwanted_suffixes = [
        "lora_right_weight",
        "lora_left_weight",
        "lora_right_bias",
        "lora_left_bias",
    ]
    for k, v in list(state_dict.items()):
        for suffix in unwanted_suffixes:
            if k.endswith(suffix):
                state_dict.pop(k)
    # super hacky - should probably refactor this
    if state_dict.get('lm_head.0.weight', None) is not None:
        state_dict['lm_head.weight'] = state_dict.pop('lm_head.0.weight')
    if state_dict.get('lm_heads.0.0.weight', None) is not None:
        state_dict['lm_heads.0.weight'] = state_dict.pop('lm_heads.0.0.weight')
    if state_dict.get('lm_heads.1.0.weight', None) is not None:
        state_dict['lm_heads.1.weight'] = state_dict.pop('lm_heads.1.0.weight')
    if state_dict.get('lm_heads.2.0.weight', None) is not None:
        state_dict['lm_heads.2.weight'] = state_dict.pop('lm_heads.2.0.weight')
    if state_dict.get('lm_heads.3.0.weight', None) is not None:
        state_dict['lm_heads.3.weight'] = state_dict.pop('lm_heads.3.0.weight')
    if state_dict.get('lm_heads.4.0.weight', None) is not None:
        state_dict['lm_heads.4.weight'] = state_dict.pop('lm_heads.4.0.weight')
    if state_dict.get('lm_heads.5.0.weight', None) is not None:
        state_dict['lm_heads.5.weight'] = state_dict.pop('lm_heads.5.0.weight')
    if state_dict.get('lm_heads.6.0.weight', None) is not None:
        state_dict['lm_heads.6.weight'] = state_dict.pop('lm_heads.6.0.weight')
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        print(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    if checkpoint.get("best_val_loss", None) is not None:
        val_loss = checkpoint["best_val_loss"].item()
        logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict
    _clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return {
            "model": model,
            "tokenizer": tokenizer,
        }
    return model


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    _clear_cuda_cache()
    return model


def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text", path=None):
    _load_model_f = funcy.partial(_load_model, model_type=model_type, use_small=use_small)
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        if path.endswith(".ckpt") or path.endswith(".pt") or path.endswith(".bin"):
            ckpt_path = path
        else:
            ckpt_path = _get_ckpt_path(model_type, use_small=use_small, path=path)
        # clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        models[model_key]["model"].to(device)
    else:
        models[model_key].to(device)
    return models[model_key]


def load_codec_model(use_gpu=True, force_reload=False):
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = _load_codec_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]


def preload_models(
    text_use_gpu=True,
    text_use_small=False,
    text_model_path=None,
    coarse_use_gpu=True,
    coarse_use_small=False,
    coarse_model_path=None,
    fine_use_gpu=True,
    fine_use_small=False,
    fine_model_path=None,
    codec_use_gpu=True,
    force_reload=False,
    path=None,
):
    """Load all the necessary models for the pipeline."""
    if _grab_best_device() == "cpu" and (
        text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        logger.warning("No GPU being used. Careful, inference might be very slow!")
    _ = load_model(
        model_type="text", use_gpu=text_use_gpu, use_small=text_use_small, force_reload=force_reload, path=path if text_model_path is None else text_model_path
    )
    _ = load_model(
        model_type="coarse",
        use_gpu=coarse_use_gpu,
        use_small=coarse_use_small,
        force_reload=force_reload,
        path=path if coarse_model_path is None else coarse_model_path,
    )
    _ = load_model(
        model_type="fine", use_gpu=fine_use_gpu, use_small=fine_use_small, force_reload=force_reload, path=path if fine_model_path is None else fine_model_path
    )
    _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)


####
# Generation Functionality
####

def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


def generate_text_semantic(
    text,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    use_kv_caching=False,
):
    """Generate semantic tokens from text."""
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if history_prompt is not None:
        if history_prompt.endswith(".npz"):
            try:
                semantic_history = np.load(history_prompt)["semantic_prompt"]
            except:
                semantic_history = np.load(history_prompt)["semantic"]
        else:
            semantic_history = np.load(
                os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
            )["semantic_prompt"]
        assert (
            isinstance(semantic_history, np.ndarray)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        preload_models()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([
            encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
        ]).astype(np.int64)
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with _inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=100)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(relevant_logits / temp, dim=-1)
            # multinomial bugged on mps: shuttle to cpu if necessary
            inf_device = probs.device
            if probs.device.type == "mps":
                probs = probs.to("cpu")
            item_next = torch.multinomial(probs, num_samples=1)
            probs = probs.to(inf_device)
            item_next = item_next.to(inf_device)
            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(100 - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(100 - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(100 - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
            if req_pbar_state > pbar_state:
                pbar.update(req_pbar_state - pbar_state)
            pbar_state = req_pbar_state
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()
    return out


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def generate_coarse(
    x_semantic,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    use_kv_caching=False,
):
    """Generate coarse audio codes from semantic tokens."""
    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        if history_prompt.endswith(".npz"):
            x_history = np.load(history_prompt)
        else:
            x_history = np.load(
                os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
            )
        try:
            x_semantic_history = x_history["semantic_prompt"]
            x_coarse_history = x_history["coarse_prompt"]
        except:
            x_semantic_history = x_history["semantic"]
            x_coarse_history = x_history["coarse"]
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        preload_models()
    model = models["coarse"]
    if OFFLOAD_CPU:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with _inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = (
                    SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                    SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    # faster to convert to numpy
                    original_device = relevant_logits.device
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(original_device)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                # multinomial bugged on mps: shuttle to cpu if necessary
                inf_device = probs.device
                if probs.device.type == "mps":
                    probs = probs.to("cpu")
                item_next = torch.multinomial(probs, num_samples=1)
                probs = probs.to(inf_device)
                item_next = item_next.to(inf_device)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history) :]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    _clear_cuda_cache()
    return gen_coarse_audio_arr


def generate_fine(
    x_coarse_gen,
    history_prompt=None,
    temp=0.5,
    silent=True,
):
    """Generate full audio codes from coarse audio codes."""
    assert (
        isinstance(x_coarse_gen, np.ndarray)
        and len(x_coarse_gen.shape) == 2
        and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
        and x_coarse_gen.shape[1] > 0
        and x_coarse_gen.min() >= 0
        and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
    )
    if history_prompt is not None:
        if history_prompt.endswith(".npz"):
            try:
                x_fine_history = np.load(history_prompt)["fine_prompt"]
            except:
                x_fine_history = np.load(history_prompt)["fine"]
        else:
            x_fine_history = np.load(
                os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
            )["fine_prompt"]
        assert (
            isinstance(x_fine_history, np.ndarray)
            and len(x_fine_history.shape) == 2
            and x_fine_history.shape[0] == N_FINE_CODEBOOKS
            and x_fine_history.shape[1] >= 0
            and x_fine_history.min() >= 0
            and x_fine_history.max() <= CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # load models if not yet exist
    global models
    global models_devices
    if "fine" not in models:
        preload_models()
    model = models["fine"]
    if OFFLOAD_CPU:
        model.to(models_devices["fine"])
    device = next(model.parameters()).device
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with _inference_mode():
        in_arr = torch.tensor(in_arr.T).to(device)
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    # multinomial bugged on mps: shuttle to cpu if necessary
                    inf_device = probs.device
                    if probs.device.type == "mps":
                        probs = probs.to("cpu")
                    codebook_preds = torch.hstack(
                        [
                            torch.multinomial(probs[nnn], num_samples=1).to(inf_device)
                            for nnn in range(rel_start_fill_idx, 1024)
                        ]
                    )
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                    start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    _clear_cuda_cache()
    return gen_fine_arr


def codec_decode(fine_tokens):
    """Turn quantized audio codes into audio array using encodec."""
    # load models if not yet exist
    global models
    global models_devices
    if "codec" not in models:
        preload_models()
    model = models["codec"]
    if OFFLOAD_CPU:
        model.to(models_devices["codec"])
    device = next(model.parameters()).device
    arr = torch.from_numpy(fine_tokens)[None]
    arr = arr.to(device)
    arr = arr.transpose(0, 1)
    emb = model.quantizer.decode(arr)
    out = model.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()
    del arr, emb, out
    if OFFLOAD_CPU:
        model.to("cpu")
    return audio_arr



class ModelLoader:
   def __init__(self):
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.model = self.load_codec_model()
       self.hubert_manager = self.load_hubert_manager()
       self.hubert_model = self.load_hubert_model()
       self.tokenizer = self.load_tokenizer()

   def load_codec_model(self):
       return load_codec_model(use_gpu=True if self.device == 'cuda' else False)

   def load_hubert_manager(self):
       hubert_manager = HuBERTManager()
       hubert_manager.make_sure_hubert_installed()
       hubert_manager.make_sure_tokenizer_installed()
       return hubert_manager

   def load_hubert_model(self):
       return CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(self.device)

   def load_tokenizer(self):
       return CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(self.device)
   

class AudioProcessor:
  def __init__(self, filepath, model, device):
      self.filepath = filepath
      self.model = model
      self.device = device
      self.wav, self.sr = torchaudio.load(self.filepath)

  def process_audio(self):
      self.wav = convert_audio(self.wav, self.sr, self.model.sample_rate, self.model.channels)
      self.wav = self.wav.to(self.device)
      return self.wav


class SemanticGenerator:
   def __init__(self, hubert_model, tokenizer, wav, model):
       self.hubert_model = hubert_model
       self.tokenizer = tokenizer
       self.wav = wav
       self.model = model

   def generate_semantic_tokens(self):
       semantic_vectors = self.hubert_model.forward(self.wav, input_sample_hz=self.model.sample_rate)
       semantic_tokens = self.tokenizer.get_token(semantic_vectors)
       return semantic_tokens

class Encoder:
   def __init__(self, model, wav):
       self.model = model
       self.wav = wav

   def encode(self):
       with torch.no_grad():
           encoded_frames = self.model.encode(self.wav.unsqueeze(0))
       codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
       return codes


class AudioGenerator:
  def __init__(self, text_prompt, voice_name):
      self.text_prompt = text_prompt
      self.voice_name = voice_name

  def generate_audio(self):
      preload_models(
          text_use_gpu=True,
          text_use_small=False,
          coarse_use_gpu=True,
          coarse_use_small=False,
          fine_use_gpu=True,
          fine_use_small=False,
          codec_use_gpu=True,
          force_reload=False,
          path="models"
      )

      # Generation with more control
      x_semantic = generate_text_semantic(
          self.text_prompt,
          history_prompt=self.voice_name,
          temp=0.7,
          top_k=50,
          top_p=0.95,
      )

      x_coarse_gen = generate_coarse(
          x_semantic,
          history_prompt=self.voice_name,
          temp=0.7,
          top_k=50,
          top_p=0.95,
      )

      x_fine_gen = generate_fine(
          x_coarse_gen,
          history_prompt=self.voice_name,
          temp=0.5,
      )

      audio_array = codec_decode(x_fine_gen)

      return audio_array


class BarkVoiceCloning:
    def __init__(self) -> None:
        pass

    def clone_voice(self, prompt, voice_name, input_audio_file, model_loader=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model_loader = ModelLoader(device)
        # Process audio
        try:
            audio_processor = AudioProcessor(input_audio_file, model_loader.model, device)
            processed_audio = audio_processor.process_audio()
        except:
            raise Exception(f"Audio file not found or model issue: {model_loader.model}")
        # Generate semantic tokens
        semantic_generator = SemanticGenerator(model_loader.hubert_model, model_loader.tokenizer, processed_audio, model_loader.model)
        semantic_tokens = semantic_generator.generate_semantic_tokens()

        # Encode audio
        encoder = Encoder(model_loader.model, processed_audio)
        codes = encoder.encode()

        # Get the current file location
        current_file_location = os.path.dirname(os.path.abspath(__file__))

        # Create 'assets' and 'prompts' directories
        assets_directory = os.path.join(current_file_location, 'assets')
        prompts_directory = os.path.join(assets_directory, 'prompts')

        os.makedirs(prompts_directory, exist_ok=True)

        # Save prompts
        output_path = os.path.join(prompts_directory, voice_name + '.npz')
        np.savez(output_path, fine_prompt=codes.cpu(), coarse_prompt=codes[:2, :].cpu(), semantic_prompt=semantic_tokens.cpu())
        audio_generator = AudioGenerator(prompt, voice_name)
        audio_array = audio_generator.generate_audio()

        return audio_array