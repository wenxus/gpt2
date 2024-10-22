#%%
import torch as t
from torch import nn, Tensor
# from torch.distributions.categorical import Categorical
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
# from encoder import get_encoder

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
print(cfg)
#%%
def layernorm(x, g, b, eps=1e-5): # x has shape [batch posn d_model]
    mean = x.mean(-1, keepdims=True)
    var = x.std(-1,keepdims=True)
    return (g * (x - mean)/np.sqrt(var + eps)) + b

def embed(wembed, tokens): # "d_vocab, d_model", "position" -> "position d_model"
    return wembed[tokens]

def pos_embed(wpembed, tokens): # "n_ctx, d_model", "position" -> "position d_model"
    return wpembed[range(len(tokens))]

def unembed(w_u, x):
    return np.einsum('se,ev->sv', x, w_u)

def rand_float_test(input, fn):
    print("Input shape:", random_input.shape)
    output = fn(random_input, 1, 0)
    # if isinstance(layernorm, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

#%%
random_input = np.random.rand(2, 4, 10)
rand_float_test(random_input, layernorm)
print("Output shape:", pos_embed(np.random.rand(5,6), np.ones((2,3), dtype=np.integer)).shape)
# %%
def apply_casual_mask(attn_scores): # "n_heads query_pos key_pos" -> "n_heads query_pos key_pos"
    mask = 1-np.tri(attn_scores.shape[-1])
    return np.where(mask, attn_scores, float('-inf'))

def softmax(x):
    return np.exp(x) / np.exp(x).sum()
    
# %%
m = nn.Softmax(dim=1)
input = t.tensor([[1,2],[3,4]], dtype=t.float32)
print(m(input))

# %%
def attention(w_q, w_k, w_v, b_q, b_k, b_v, w_o, b_o, x): # x "seq, d_model"
    # w_k, w_q, w_v, "num_head, d_model, d_head"; w_o "num_head, d_head, d_model"
    # K = w_k@x; Q = w_q@x; A = softmax(apply_casual_mask(QK.T)); 
    # V = w_v@x; z = AV; result = w_o z; atten_out = result.sum(dim=-2)
    e = x.shape[-1]
    n, e, h = w_k.shape
    K = np.einsum('se,enh->snh', x, w_k.reshape(e, n, h)) + b_k # s n h 
    Q = np.einsum('se,enh->snh', x, w_q.reshape(e, n, h)) + b_q # s n h 
    attn_scores = np.einsum('snh,hnx->nsx', Q, K.T)/np.sqrt(h) # n s s
    A = softmax(apply_casual_mask(attn_scores))
    #TODO
    V = np.einsum('se,enh->snh', x, w_v.reshape(e, n, h)) + b_v # s n h
    output = np.einsum('snh,hne->se', V, w_o.reshape(h, n, e)) + b_o
    print(f"attention {output.shape=}")
    return output

# %%
def parse_attn_input(x, c_attn, c_proj, n_heads): # w (768x2304), b 2304; c_proj 768x768, 768
    e, qkvheads = c_attn['w'].shape
    c_attn_rw = c_attn['w'].reshape(qkvheads, e)
    d_head = (qkvheads // 3)//n_heads
    qkv_w = c_attn_rw.reshape(3, n_heads, d_head, e)
    qkv_w = qkv_w.reshape(3, n_heads, e, d_head)
    qkv_b = c_attn['b'].reshape(3, n_heads, d_head)

    a, e = c_proj['w'].shape
    w_o = c_proj['w'].reshape(n_heads, a//n_heads, e)
    print(f"{qkv_w.shape=}")
    print(f"{qkv_b.shape=}")
    print(f"{w_o.shape=}")
    print(f"{x.shape=}")

    return attention(qkv_w[0], qkv_w[1], qkv_w[2], qkv_b[0], qkv_b[1], qkv_b[2], w_o, c_proj['b'], x)

def gelu(x):
    tanh_val = np.sqrt(2/math.pi) * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1 + (np.exp(2*tanh_val) - 1)/(np.exp(2*tanh_val) + 1))

def mlp_block(x, c_fc, c_proj):
    pre = np.einsum('se,em->sm', x, c_fc['w']) + c_fc['b']
    post = gelu(pre)
    mlp_out = np.einsum('sm,me->se', post, c_proj['w']) + c_proj['b']
    return mlp_out

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

# %%
p = None
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    hparams, params = load_hparams_and_params(model_size, models_dir)

    return params, hparams
    print(f"{params=}")
    # # encode the input string using the BPE tokenizer
    # input_ids = encoder.encode(prompt)

    # # make sure we are not surpassing the max sequence length of our model
    # assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # # generate output ids
    # output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # # decode the ids back into a string
    # output_text = encoder.decode(output_ids)

    # return output_text
p, hparams = main('today im testing this')
# %%
# print((p['blocks'])) # dict_keys(['blocks', 'ln_f', 'wpe', 'wte']) 
# dict_keys(['attn', 'ln_1', 'ln_2', 'mlp']) # in a list of 12

print(p['blocks'][0]['mlp']['c_fc'].keys()) # dict_keys(['c_attn', 'c_proj'])
print(p['blocks'][0]['mlp']['c_proj'].keys())
# print(p['blocks'][0]['attn']['c_attn'].keys()) # dict_keys(['b', 'w'])
# print(p['blocks'][0]['attn']['c_proj'].keys()) # dict_keys(['b', 'w'])

# # blocks [{'attn': {'c_attn': , 'c_proj'}}, {'ln_1': }, {l_2: }]
# # print(p['wte'].shape) # (50257, 768)
# print(len(p['blocks'][0]['attn']['c_proj']['w'])) # 768
# print(len(p['blocks'][0]['attn']['c_proj']['b'])) # 2304
# print(p['blocks'][0]['attn']['c_proj']['w'][0].shape) # (768x2304)
# print(p['blocks'][0]['attn']['c_proj']['b'][0]) # 2304x1
# %%
qkv = p['blocks'][0]['attn']['c_attn']['w']
np.split(qkv, 3, axis=-1)[0].shape
#%%
# parsing_attn_input(np.random.rand(2,768), **p['blocks'][0]['attn'], n_heads=hparams["n_head"])
#%%

def gpt2(tokens, wte, wpe, blocks, ln_f, hparams):
    print(f"{wte.shape=}")
    print(f"{wpe.shape=}")
    x = embed(wte, tokens) + pos_embed(wpe, tokens)
    print(f"{x.shape=}")
    for b in blocks:
        x = transformer_block(x, **b, n_heads=hparams["n_head"])

    logits = unembed(wte.T, layernorm(x, **ln_f))
    return logits
    
def run_gpt2(input):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    hparams, params = load_hparams_and_params("124M", "models")
    
    tokens = tokenizer(input).input_ids
    print(f"{tokens=}")
    result_t = []
    for i in tqdm(range(20)):
        logits = gpt2(tokens, **params, hparams=hparams)
        next_token_id = logits[-1,:].argmax(-1)
        print(f"{next_token_id=}")
        result_t.append(next_token_id)
    
    return tokenizer.batch_decode(result_t)


import json
def params_to_json():
    def ndarray_to_list(arr):
        """Converts a NumPy ndarray to a list."""
        return arr.tolist()
    with open('result.json', 'w') as fp:
        json.dump(p, fp, default=ndarray_to_list)

# %%
print(run_gpt2('i am testing'))
# %%
