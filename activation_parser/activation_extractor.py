#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text-only activation extractor for Hugging Face causal language models.

Key features
- load_model_and_tokenizer()
- batch extraction (efficient) with output_hidden_states=True
- layer selection
- pooling strategies:
    - mean_all_tokens
    - last_token
    - mean_last_k_tokens
    - all_tokens  (returns per-token vectors)

Layer indexing
- Uses HF convention: hidden_states is tuple length (num_layers+1)
  hidden_states[0] = embeddings output
  hidden_states[i] = after block i (1..num_layers)

So layer_id=0 means embedding output, and layer_id=num_layers means final block output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any, Tuple, Union

import torch


def get_num_hidden_layers(model) -> int:
    cfg = model.config
    # common config fields
    for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "language_config", None)):
        if c is None:
            continue
        v = getattr(c, "num_hidden_layers", None)
        if v is not None:
            return int(v)
    # fallback
    v = getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    if v is None:
        raise ValueError("Cannot resolve num_hidden_layers from config.")
    return int(v)


def resolve_layer_list(model, layer_list: Optional[Sequence[int]]) -> List[int]:
    L = get_num_hidden_layers(model)
    if layer_list is None:
        return list(range(L + 1))  # 0..L
    out = [int(x) for x in layer_list]
    # basic sanity
    for x in out:
        if x < 0 or x > L:
            raise ValueError(f"Layer id {x} out of range [0, {L}].")
    return out


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    num_layers: int


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    device_map: str = "auto",
) -> ModelBundle:
    """
    dtype: "auto" | "float16" | "bfloat16" | "float32"
    device_map: "auto" recommended for big models on GPU
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    dev = next(iter(model.parameters())).device
    L = get_num_hidden_layers(model)
    return ModelBundle(model=model, tokenizer=tokenizer, device=dev, num_layers=L)


# ----------------------------
# Pooling / selection
# ----------------------------
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, H)
    mask: (B, T) with 1 for valid, 0 for pad
    returns (B, H)
    """
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


def pool_hidden_states(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str,
    last_k: int = 3,
) -> torch.Tensor:
    """
    hidden: (B, T, H)
    attention_mask: (B, T)
    returns:
      - (B, H) for mean_all_tokens / last_token / mean_last_k_tokens
      - (B, T, H) for all_tokens
    """
    strategy = strategy.lower()

    if strategy == "all_tokens":
        return hidden

    if strategy == "mean_all_tokens":
        return _masked_mean(hidden, attention_mask)

    # find last valid index per sample
    # lengths = number of valid tokens
    lengths = attention_mask.sum(dim=1)  # (B,)
    last_idx = (lengths - 1).clamp_min(0)  # (B,)

    if strategy == "last_token":
        # gather last token
        B, T, H = hidden.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, H)  # (B,1,H)
        return hidden.gather(dim=1, index=idx).squeeze(1)  # (B,H)

    if strategy == "mean_last_k_tokens":
        B, T, H = hidden.shape
        out = []
        for b in range(B):
            lb = int(lengths[b].item())
            if lb <= 0:
                out.append(hidden[b, 0, :])
                continue
            start = max(0, lb - last_k)
            seg = hidden[b, start:lb, :]  # (k',H)
            out.append(seg.mean(dim=0))
        return torch.stack(out, dim=0)

    raise ValueError(f"Unknown pooling strategy: {strategy}")


@torch.inference_mode()
def extract_activations_batch(
    bundle: ModelBundle,
    texts: List[str],
    layer_list: Optional[Sequence[int]] = None,
    pooling: str = "mean_all_tokens",
    last_k: int = 3,
    max_length: Optional[int] = None,
    add_special_tokens: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Returns dict: {layer_id: tensor}
      - if pooling != all_tokens: tensor is (B, H)
      - if pooling == all_tokens: tensor is (B, T, H)
    """
    model, tok, device = bundle.model, bundle.tokenizer, bundle.device
    layers = resolve_layer_list(model, layer_list)

    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: (L+1) of (B,T,H)

    attn = enc["attention_mask"]
    result: Dict[int, torch.Tensor] = {}
    for lid in layers:
        pooled = pool_hidden_states(hs[lid], attn, strategy=pooling, last_k=last_k)
        result[int(lid)] = pooled.detach().to("cpu").float()  # keep CPU float32
    return result
