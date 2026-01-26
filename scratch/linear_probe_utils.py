"""
Helpers for WAN linear probing.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from algorithms.wan.modules.t5 import umt5_xxl
from algorithms.wan.modules.clip import clip_xlm_roberta_vit_h_14
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
    text_encoder,
    tokenizer,
    texts: List[str],
    device: str,
    out_device: str | None = None,
    out_dtype: torch.dtype | None = None,
):
    ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    if out_device is None:
        out_device = device
    if out_dtype is None:
        out_dtype = context[0].dtype if isinstance(context, (list, tuple)) else context.dtype
    return [u[:v].to(out_device, dtype=out_dtype) for u, v in zip(context, seq_lens)]


def setup_vae(cfg: VaeConfig, dtype: torch.dtype):
    vae = (
        video_vae_factory(
            pretrained_path=cfg.ckpt_path,
            z_dim=cfg.z_dim,
        )
        .eval()
        .requires_grad_(False)
        .to(cfg.device, dtype=dtype)
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
    # Keep inputs in the same dtype as the VAE parameters to avoid conv dtype mismatch
    # (e.g. bf16 activations with fp32 bias).
    vae_dtype = next(iter(vae.parameters())).dtype
    images = images.to(dtype=vae_dtype)
    images = images.mul(2.0).sub(1.0)  # [0,1] -> [-1,1]
    videos = images.unsqueeze(2)  # [B, 3, 1, H, W]
    return vae.encode(videos, vae_scale)


def compute_seq_len(video_lat: torch.Tensor, patch_size: tuple[int, int, int]) -> int:
    _, _, f, h, w = video_lat.shape
    pt, ph, pw = patch_size
    return (f // pt) * (h // ph) * (w // pw)


def build_timesteps(batch_size: int, t_value: float, device: str):
    return torch.full((batch_size,), float(t_value), device=device)


@torch.no_grad()
def add_training_noise(
    video_lat: torch.Tensor,
    *,
    diffusion_type: str = "continuous",
    num_train_timesteps: int = 1000,
    sample_shift: float = 3.0,
    t_value: float | torch.Tensor | None = None,
):
    """
    Match WAN training noise for continuous diffusion (wan_t2v.add_training_noise).
    Returns (noisy_lat, t) where t is scaled to [0, num_train_timesteps].
    """
    if diffusion_type != "continuous":
        raise NotImplementedError("Only continuous diffusion is supported in linear probe.")

    device = video_lat.device
    noise = torch.randn_like(video_lat)
    if t_value is None:
        dist = torch.distributions.uniform.Uniform(0, 1)
        t = dist.sample((video_lat.size(0),)).to(device)
        t = t * sample_shift / (1 + (sample_shift - 1) * t)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
        t = t * num_train_timesteps
    else:
        if not torch.is_tensor(t_value):
            t_value = torch.tensor(float(t_value), device=device)
        t = t_value.to(device=device, dtype=video_lat.dtype)
        t_scaled = t / float(num_train_timesteps)
        t_expanded = t_scaled.view(1, 1, 1, 1, 1).expand(video_lat.size(0), 1, 1, 1, 1)
        noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
        t = t.expand(video_lat.size(0))
    return noisy_lat, t

def setup_clip_for_i2v(ckpt_path: str | None, device: str, dtype: torch.dtype):
    """
    Build WAN's CLIP vision encoder and its normalize transform.
    Returns (clip_model, clip_normalize).
    """
    clip, clip_transform = clip_xlm_roberta_vit_h_14(
        pretrained=False,
        return_transforms=True,
        return_tokenizer=False,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    clip = clip.eval().requires_grad_(False)
    if ckpt_path:
        clip.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    else:
        print(
            "[linear_probe_wan] WARNING: --clip-ckpt-path not provided; CLIP features will be from random weights."
        )
    clip = clip.to(torch.device(device), dtype=dtype)
    clip_normalize = clip_transform.transforms[-1]
    return clip, clip_normalize


@torch.no_grad()
def compute_i2v_conditioning_from_images(
    images: torch.Tensor,
    *,
    vae,
    vae_scale,
    clip_model,
    clip_normalize,
    device: str,
    dtype: torch.dtype,
    wan_in_dim: int,
):
    """
    Build i2v conditioning from a batch of ImageNet images in [0, 1].
    - x: VAE latents for the image (acts as the "denoising frame" input)
    - y: [mask_channels, x] concatenated along channels (image_embeds)
    - clip_fea: CLIP vision tokens from the same image
    """
    # x: [B, z_dim, 1, H', W']
    x = encode_images_to_wan_latents(vae, vae_scale, images, dtype)

    # y: [B, (mask_ch + z_dim), 1, H', W']
    b, z_dim, f, h, w = x.shape
    mask_ch = max(int(wan_in_dim) - 2 * int(z_dim), 0)
    mask = torch.zeros(b, mask_ch, f, h, w, device=device, dtype=x.dtype)
    y = torch.cat([mask, x], dim=1)

    # clip_fea: [B, 257, 1280] (expected by i2v cross-attn path)
    # IMPORTANT: run CLIP on the CLIP model's own device/dtype (may differ from WAN device)
    if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "patch_embedding"):
        clip_param = clip_model.visual.patch_embedding.weight
    else:
        clip_param = next(iter(clip_model.parameters()))
    clip_device = clip_param.device
    clip_dtype = clip_param.dtype
    if clip_dtype != torch.bfloat16 and clip_device.type == "cuda":
        clip_model = clip_model.to(device=clip_device, dtype=torch.bfloat16)
        clip_dtype = torch.bfloat16
    clip_in = images.to(device=clip_device, dtype=torch.float32)
    clip_in = F.interpolate(
        clip_in,
        size=(clip_model.image_size, clip_model.image_size),
        mode="bicubic",
        align_corners=False,
    )
    clip_in = clip_normalize(clip_in)  # images already in [0,1]
    clip_in = clip_in.to(dtype=clip_dtype)
    if clip_device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clip_fea = clip_model.visual(clip_in, use_31_block=True)
    else:
        clip_fea = clip_model.visual(clip_in, use_31_block=True)
    clip_fea = clip_fea.to(device=device, dtype=x.dtype)

    return x, y, clip_fea
