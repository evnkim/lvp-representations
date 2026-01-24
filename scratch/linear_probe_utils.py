"""
Helpers for WAN linear probing.
"""

from dataclasses import dataclass
from typing import List

import torch

from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer
from algorithms.wan.modules.vae import video_vae_factory


@dataclass
class TextConfig:
    name: str
    text_len: int
    ckpt_path: str | None
    device: str = "cpu"


@dataclass
class VaeConfig:
    ckpt_path: str
    z_dim: int
    mean: List[float]
    std: List[float]
    device: str = "cuda"


def setup_text_encoder(cfg: TextConfig, dtype: torch.dtype):
    text_encoder = (
        umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        .eval()
        .requires_grad_(False)
    )
    if cfg.ckpt_path:
        text_encoder.load_state_dict(
            torch.load(cfg.ckpt_path, map_location="cpu", weights_only=True)
        )
    text_encoder = text_encoder.to(cfg.device)

    tokenizer = HuggingfaceTokenizer(
        name=cfg.name,
        seq_len=cfg.text_len,
        clean="whitespace",
    )
    return text_encoder, tokenizer


@torch.no_grad()
def encode_texts(
    text_encoder, tokenizer, texts: List[str], device: str, out_device: str | None = None
):
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    if out_device is None:
        out_device = device
    return [u[:v].to(out_device) for u, v in zip(context, seq_lens)]


def setup_vae(cfg: VaeConfig, dtype: torch.dtype):
    vae = (
        video_vae_factory(
            pretrained_path=cfg.ckpt_path,
            z_dim=cfg.z_dim,
        )
        .eval()
        .requires_grad_(False)
        .to(cfg.device)
    )
    mean = torch.tensor(cfg.mean, dtype=dtype, device=cfg.device)
    std = torch.tensor(cfg.std, dtype=dtype, device=cfg.device)
    vae_scale = [mean, 1.0 / std]
    return vae, vae_scale


@torch.no_grad()
def encode_images_to_wan_latents(vae, vae_scale, images: torch.Tensor, dtype):
    """
    images: [B, 3, H, W] in [0, 1]
    returns: [B, C, T, H, W] latents (T=1)
    """
    images = images.to(dtype)
    images = images.mul(2.0).sub(1.0)  # [0,1] -> [-1,1]
    videos = images.unsqueeze(2)  # [B, 3, 1, H, W]
    return vae.encode(videos, vae_scale)


def compute_seq_len(video_lat: torch.Tensor, patch_size: tuple[int, int, int]) -> int:
    _, _, f, h, w = video_lat.shape
    pt, ph, pw = patch_size
    return (f // pt) * (h // ph) * (w // pw)


def build_timesteps(batch_size: int, t_value: float, device: str):
    return torch.full((batch_size,), float(t_value), device=device)
