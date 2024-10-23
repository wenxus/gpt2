#%%
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import json
import os
import re
import requests
import tensorflow as tf
from tqdm import tqdm
import math
from transformers import AutoTokenizer

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
#%%
def layernorm(x, g, b, eps=1e-5): # x has shape [posn d_model]
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1,keepdims=True)
    return (g * (x - mean)/np.sqrt(var + eps)) + b

def embed(wembed, tokens): # "d_vocab, d_model", "position" -> "position d_model"
    return wembed[tokens]

def pos_embed(wpembed, tokens): # "n_ctx, d_model", "position" -> "position d_model"
    return wpembed[range(len(tokens))]

def unembed(w_u, x):
    return np.einsum('se,ev->sv', x, w_u)

# %%
def apply_casual_mask(attn_scores): # "n_heads query_pos key_pos" -> "n_heads query_pos key_pos"
    mask = 1-np.tri(attn_scores.shape[-1])
    result = np.where(mask, float('-inf'),attn_scores)
    return result

def softmax(x):
    return np.exp(x) / np.exp(x).sum()
#   exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#   return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# %%

def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def attention(Q, K, V, w, b): # x "seq, d_model" -> "seq, d_model"
    # w_k, w_q, w_v, "num_head, d_model, d_head"; w_o "num_head, d_head, d_model"
    # K = w_k@x; Q = w_q@x; A = softmax(apply_casual_mask(QK.T)); 
    # V = w_v@x; z = AV; result = w_o z; atten_out = result.sum(dim=-2)
    s,n,h = Q.shape
    attn_scores = apply_casual_mask(np.einsum('snh,hnx->nsx', Q, K.T)/np.sqrt(h)) # n s s
    score_list = np.split(attn_scores, n, axis = 0)
    attn_list = [softmax(s[0]) for s in score_list]
    A = np.stack(attn_list)
    z = np.einsum('nsx,xnh->snh', A, V)
    z = z.reshape(s, n*h)
    output = np.einsum('sc,ce->se', z, w) + b
    return output

# %%
def parse_attn_input(x, c_attn, c_proj, n_heads): # w (768x2304), b 2304; c_proj 768x768, 768
    s, dmodel = x.shape
    x = np.einsum('se,ec->sc', x, c_attn['w']) + c_attn['b'] # seq, d_model * 3

    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_heads, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    Q = np.array(qkv_heads[0])
    n,s,h = Q.shape
    Q = Q.reshape(s,n,h)
    K = np.array(qkv_heads[1])
    K = K.reshape(s,n,h)
    V = np.array(qkv_heads[2])
    V = V.reshape(s,n,h)
    return attention(Q, K, V, **c_proj)

def gelu(x):
    tanh_val = np.sqrt(2/math.pi) * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1 + np.tanh(tanh_val))

def mlp_block(x, c_fc, c_proj):
    pre = np.einsum('se,em->sm', x, c_fc['w']) + c_fc['b']
    post = gelu(pre)
    mlp_out = np.einsum('sm,me->se', post, c_proj['w']) + c_proj['b']
    return mlp_out

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params

def transformer_block(x, attn, ln_1, ln_2, mlp, n_heads):
    mid = parse_attn_input(layernorm(x, **ln_1), **attn, n_heads=n_heads) + x
    post = mlp_block(layernorm(mid, **ln_2), **mlp) + mid
    return post

#%%

def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_hparams_and_params(model_size, models_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return hparams, params

def gpt2(tokens, wte, wpe, blocks, ln_f, hparams):
    x = embed(wte, tokens) + pos_embed(wpe, tokens)
    for b in blocks:
        x = transformer_block(x, **b, n_heads=hparams["n_head"])

    logits = unembed(wte.T, layernorm(x, **ln_f))
    return logits
    
def run_gpt2(input):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hparams, params = load_hparams_and_params("124M", "models")
    
    tokens = tokenizer(input).input_ids
    result_t = []
    for i in tqdm(range(30)):
        logits = gpt2(tokens, **params, hparams=hparams)
        next_token_id = logits[-1,:].argmax(-1)
        result_t.append(next_token_id)
        tokens.append(next_token_id)
    
    return ''.join(tokenizer.batch_decode(result_t))

import json
def params_to_json():
    def ndarray_to_list(arr):
        """Converts a NumPy ndarray to a list."""
        return arr.tolist()
    with open('result.json', 'w') as fp:
        json.dump(p, fp, default=ndarray_to_list)

# %%
print(run_gpt2('A bluffy blue animal named Daisy'))
# %%
