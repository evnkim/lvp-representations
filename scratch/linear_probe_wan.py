"""
Minimal WAN linear probing scaffold for ImageNet-style classification.

Provide VAE and text encoder checkpoints via CLI flags.
"""

import argparse
import gc
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Repo-local imports
from algorithms.wan.modules.model import WanModel
from data_classify.imagenet import ImageNetSubset
from scratch.linear_probe_utils import (
    TextConfig,
    VaeConfig,
    add_training_noise,
    build_timesteps,
    compute_seq_len,
    compute_i2v_conditioning_from_images,
    encode_images_to_wan_latents,
    encode_texts,
    setup_clip_for_i2v,
    setup_text_encoder,
    setup_vae,
)
from scratch.wandb_utils import WandbConfig, init_wandb, log_wandb


DEFAULT_VAE_MEAN = [
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
]
DEFAULT_VAE_STD = [
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
]


@dataclass
class ProbeConfig:
    ckpt_path: str
    layer_idx: int
    use_mlp: bool
    tuned_ckpt_path: str | None = None
    pipeline_test: bool = False
    tiny_model: bool = False
    use_random_inputs: bool = False
    num_classes: int = 1000
    pool: str = "mean"  # "mean" or "max"
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8
    subset_name: str = "imagenet-1k"
    data_root: str = "data/datasets"
    image_height: int = 480
    image_width: int = 832
    patch_size: tuple[int, int, int] = (1, 2, 2)
    t_value: float = 500.0
    text_prompt: str = "a photo"
    text_encoder_name: str = "google/umt5-xxl"
    text_len: int = 512
    text_ckpt_path: str | None = None
    text_device: str = "cuda"
    vae_ckpt_path: str | None = None
    vae_z_dim: int = 16
    vae_mean: list[float] | None = None
    vae_std: list[float] | None = None
    clip_ckpt_path: str | None = None
    clip_device: str = "cuda"
    wandb: WandbConfig | None = None
    lr: float = 1e-3
    epochs: int = 10
    log_every: int = 50
    log_samples: int = 8
    ckpt_dir: str = "checkpoints/linear_probe"
    resume_path: str | None = None


class WanFeatureModel(nn.Module):
    """
    Wraps WAN backbone and exposes residual stream at a chosen block.
    """

    def __init__(self, wan_model: WanModel, layer_idx: int, pool: str = "mean"):
        super().__init__()
        self.wan = wan_model
        self.layer_idx = layer_idx
        self.pool = pool
        self._feature = None

        def save_post(module, inputs, output):
            # output: [B, L, C]
            self._feature = output

        self._hook = self.wan.blocks[layer_idx].register_forward_hook(save_post)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        # Some WAN checkpoints are image-to-video ("i2v") and require extra conditioning.
        # For linear probing we may not have a CLIP encoder / bbox mask handy, so we pass
        # safe dummy tensors that satisfy the interface and preserve tensor shapes.
        if getattr(self.wan, "model_type", None) == "i2v":
            b, c, f, h, w = x.shape
            if y is None:
                # i2v expects channels: noisy_latents (c) + image_embeds (4 + c) = 4 + 2c
                # We approximate image_embeds as [zeros(mask=4ch), x].
                mask = torch.zeros(b, 4, f, h, w, device=x.device, dtype=x.dtype)
                y = torch.cat([mask, x], dim=1)
            if clip_fea is None:
                # MLPProj expects last dim 1280; token length is flexible for concat with text.
                clip_fea = torch.zeros(b, 1, 1280, device=x.device, dtype=x.dtype)

        _ = self.wan(x, t, context, seq_len, clip_fea=clip_fea, y=y)
        feats = self._feature  # [B, L, C]

        # TODO: unpad tokens using true seq_lens before pooling.
        if self.pool == "mean":
            feats = feats.mean(dim=1)
        elif self.pool == "max":
            feats = feats.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool {self.pool}")
        return feats


class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes, use_mlp=False):
        super().__init__()
        if use_mlp:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def build_dataloaders(cfg: ProbeConfig):
    """
    Build ImageNet train/val loaders using ImageNetSubset.
    """
    train_ds = ImageNetSubset(
        split="train",
        res=(cfg.image_height, cfg.image_width),
        subset_name=cfg.subset_name,
        crop_res=(cfg.image_height, cfg.image_width),
        data_root=cfg.data_root,
        crop_mode="random",
        preserve_aspect=True,
    )
    val_ds = ImageNetSubset(
        split="val",
        res=(cfg.image_height, cfg.image_width),
        subset_name=cfg.subset_name,
        crop_res=(cfg.image_height, cfg.image_width),
        data_root=cfg.data_root,
        crop_mode="center",
        preserve_aspect=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # NOTE: ImageNet has no public test labels. Use val as test if needed.
    test_loader = val_loader
    return train_loader, val_loader, test_loader


def compute_top1_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.numel()


def build_random_inputs(
    batch_size,
    in_dim,
    text_dim,
    text_len,
    height,
    width,
    device,
    dtype,
):
    video_lat = torch.randn(batch_size, in_dim, 1, height, width, device=device, dtype=dtype)
    context = [
        torch.randn(text_len, text_dim, device=device, dtype=dtype) for _ in range(batch_size)
    ]
    return video_lat, context, video_lat


@torch.no_grad()
def evaluate(
    feature_model,
    probe,
    data_loader,
    device,
    text_encoder,
    tokenizer,
    text_device,
    cached_prompt_context,
    vae,
    vae_scale,
    clip_model,
    clip_normalize,
    wan_in_dim,
    patch_size,
    t_value,
    text_prompt,
    dtype,
    use_random_inputs,
    in_dim,
    text_dim,
    text_len,
):
    feature_model.eval()
    probe.eval()

    total_correct = 0
    total_count = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

    if use_random_inputs:
        x, context, video_lat = build_random_inputs(
            images.size(0),
            in_dim,
            text_dim,
            text_len,
            images.shape[2],
            images.shape[3],
            device,
            dtype,
        )
        y = None
        clip_fea = None
        t = build_timesteps(images.size(0), t_value, device)
    else:
        # i2v: build clip_fea and y from the same input image.
        if getattr(feature_model.wan, "model_type", None) == "i2v":
            x, y, clip_fea = compute_i2v_conditioning_from_images(
                images,
                vae=vae,
                vae_scale=vae_scale,
                clip_model=clip_model,
                clip_normalize=clip_normalize,
                device=device,
                dtype=dtype,
                wan_in_dim=wan_in_dim,
            )
            video_lat = x
        else:
            video_lat = encode_images_to_wan_latents(vae, vae_scale, images, dtype)
            x = video_lat
            y = None
            clip_fea = None
        x, t = add_training_noise(video_lat, t_value=t_value)
        if cached_prompt_context is not None:
            context = [cached_prompt_context for _ in range(images.size(0))]
        else:
            context = encode_texts(
                text_encoder,
                tokenizer,
                [text_prompt] * images.size(0),
                text_device,
                out_device=device,
                out_dtype=dtype,
            )
        seq_len = compute_seq_len(video_lat, patch_size)
        feats = feature_model(x, t, context, seq_len, clip_fea=clip_fea, y=y)
        logits = probe(feats)

        correct, count = compute_top1_accuracy(logits, labels)
        total_correct += correct
        total_count += count

    acc = 100.0 * total_correct / max(total_count, 1)
    return acc


def train_linear(
    feature_model,
    probe,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    text_encoder,
    tokenizer,
    text_device,
    cached_prompt_context,
    vae,
    vae_scale,
    clip_model,
    clip_normalize,
    wan_in_dim,
    patch_size,
    t_value,
    text_prompt,
    dtype,
    wandb_run,
    log_every,
    use_random_inputs,
    in_dim,
    text_dim,
    text_len,
    log_samples,
    ckpt_dir,
    resume_path,
):
    criterion = nn.CrossEntropyLoss()
    feature_model.eval()
    probe.train()

    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = 0.0
    global_step = 0
    start_epoch = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        probe.load_state_dict(ckpt["probe"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val = float(ckpt.get("best_val", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        running_count = 0
        logged_samples = False
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_random_inputs:
                x, context, video_lat = build_random_inputs(
                    images.size(0),
                    in_dim,
                    text_dim,
                    text_len,
                    images.shape[2],
                    images.shape[3],
                    device,
                    dtype,
                )
                y = None
                clip_fea = None
                t = build_timesteps(images.size(0), t_value, device)
            else:
                if getattr(feature_model.wan, "model_type", None) == "i2v":
                    x, y, clip_fea = compute_i2v_conditioning_from_images(
                        images,
                        vae=vae,
                        vae_scale=vae_scale,
                        clip_model=clip_model,
                        clip_normalize=clip_normalize,
                        device=device,
                        dtype=dtype,
                        wan_in_dim=wan_in_dim,
                    )
                    video_lat = x
                else:
                    video_lat = encode_images_to_wan_latents(vae, vae_scale, images, dtype)
                    x = video_lat
                    y = None
                    clip_fea = None
                x, t = add_training_noise(video_lat, t_value=t_value)
                if cached_prompt_context is not None:
                    context = [cached_prompt_context for _ in range(images.size(0))]
                else:
                    context = encode_texts(
                        text_encoder,
                        tokenizer,
                        [text_prompt] * images.size(0),
                        text_device,
                        out_device=device,
                        out_dtype=dtype,
                    )
            seq_len = compute_seq_len(video_lat, patch_size)
            with torch.no_grad():
                feats = feature_model(x, t, context, seq_len, clip_fea=clip_fea, y=y)
            feats = feats.detach()
            logits = probe(feats)

            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if not logged_samples and wandb_run is not None and log_samples > 0:
                try:
                    import wandb

                    class_names = getattr(train_loader.dataset, "class_names", None)
                    num = min(log_samples, images.size(0))
                    samples = []
                    for idx in range(num):
                        label = int(labels[idx].item())
                        name = class_names[label] if class_names else str(label)
                        samples.append(
                            wandb.Image(
                                images[idx].detach().float().cpu(),
                                caption=f"label={label} name={name}",
                            )
                        )
                    log_wandb(
                        wandb_run,
                        {"train/samples": samples, "epoch": epoch},
                        step=global_step,
                    )
                except Exception as exc:
                    print(f"[linear_probe_wan] WARNING: failed to log samples: {exc}")
                logged_samples = True

            running_loss += loss.item() * labels.size(0)
            running_count += labels.size(0)
            if global_step < 100 or (log_every > 0 and (global_step % log_every) == 0):
                log_wandb(
                    wandb_run,
                    {
                        "train/step_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": global_step,
                    },
                    step=global_step,
                )
            global_step += 1

        scheduler.step()
        train_loss = running_loss / max(running_count, 1)
        val_acc = evaluate(
            feature_model,
            probe,
            val_loader,
            device,
            text_encoder,
            tokenizer,
            text_device,
            cached_prompt_context,
            vae,
            vae_scale,
            clip_model,
            clip_normalize,
            wan_in_dim,
            patch_size,
            t_value,
            text_prompt,
            dtype,
            use_random_inputs,
            in_dim,
            text_dim,
            text_len,
        )
        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        print(f"epoch {epoch}: val_top1={val_acc:.2f} best={best_val:.2f}")
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
        log_wandb(
            wandb_run,
            {
                "train/loss": train_loss,
                "val/top1": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            },
            step=epoch,
        )

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--tuned-ckpt-path", type=str, default=None)
    parser.add_argument("--pipeline-test", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--random-inputs", action="store_true")
    parser.add_argument("--layer-idx", type=int, default=20)
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument("--subset-name", type=str, default="imagenet-1k")
    parser.add_argument("--data-root", type=str, default="/data/scene-rep/ImageNet1K")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=832)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[1, 2, 2])
    parser.add_argument("--t-value", type=float, default=500.0)
    parser.add_argument("--text-prompt", type=str, default="a photo")
    parser.add_argument("--text-encoder-name", type=str, default="google/umt5-xxl")
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument(
        "--text-ckpt-path",
        type=str,
        default="data/ckpts/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument("--text-device", type=str, default="cpu")
    parser.add_argument("--vae-ckpt-path", type=str, default=None)
    parser.add_argument("--vae-z-dim", type=int, default=16)
    parser.add_argument("--vae-mean", type=float, nargs="*", default=None)
    parser.add_argument("--vae-std", type=float, nargs="*", default=None)
    parser.add_argument(
        "--clip-ckpt-path",
        type=str,
        default="data/ckpts/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    parser.add_argument("--clip-device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lvp-representation")
    parser.add_argument(
        "--wandb-entity", type=str, default="evnkim-massachusetts-institute-of-technology-org"
    )
    parser.add_argument("--wandb-name", type=str, default="wan-linear-probe")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-samples", type=int, default=8)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/linear_probe")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = ProbeConfig(
        ckpt_path=args.ckpt_path,
        tuned_ckpt_path=args.tuned_ckpt_path,
        pipeline_test=args.pipeline_test,
        tiny_model=args.tiny_model,
        use_random_inputs=args.random_inputs,
        layer_idx=args.layer_idx,
        use_mlp=args.use_mlp,
        subset_name=args.subset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=tuple(args.patch_size),
        t_value=args.t_value,
        text_prompt=args.text_prompt,
        text_encoder_name=args.text_encoder_name,
        text_len=args.text_len,
        text_ckpt_path=args.text_ckpt_path,
        text_device=args.text_device,
        vae_ckpt_path=args.vae_ckpt_path,
        vae_z_dim=args.vae_z_dim,
        vae_mean=args.vae_mean,
        vae_std=args.vae_std,
        clip_ckpt_path=args.clip_ckpt_path,
        clip_device=args.clip_device,
        wandb=WandbConfig(
            enabled=args.wandb,
            project=args.wandb_project,
            # entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
        ),
        log_every=args.log_every,
        log_samples=args.log_samples,
        ckpt_dir=args.ckpt_dir,
        resume_path=args.resume,
    )

    wandb_run = init_wandb(
        cfg.wandb,
        {
            "ckpt_path": cfg.ckpt_path,
            "layer_idx": cfg.layer_idx,
            "subset_name": cfg.subset_name,
            "batch_size": cfg.batch_size,
            "t_value": cfg.t_value,
            "text_prompt": cfg.text_prompt,
        },
    )

    if cfg.pipeline_test:
        cfg.tiny_model = True
        cfg.use_random_inputs = True

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if cfg.tiny_model:
        wan = WanModel(
            model_type="t2v",
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=8,
            dim=256,
            ffn_dim=512,
            freq_dim=64,
            text_dim=128,
            out_dim=8,
            num_heads=4,
            num_layers=2,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
    elif cfg.tuned_ckpt_path:
        if not cfg.ckpt_path:
            raise ValueError("Please provide --ckpt-path for tuned checkpoint loading.")
        wan = WanModel.from_config(WanModel._dict_from_json_file(cfg.ckpt_path + "/config.json"))
        ckpt = torch.load(cfg.tuned_ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            raw_state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            raw_state_dict = ckpt
        else:
            raise ValueError("Unsupported checkpoint format for --tuned-ckpt-path.")

        state_dict = {}
        for k, v in raw_state_dict.items():
            new_key = k[len("model.") :] if k.startswith("model.") else k
            state_dict[new_key] = v
        wan.load_state_dict(state_dict, assign=True)
    else:
        if not cfg.ckpt_path:
            raise ValueError("Please provide --ckpt-path (or use --tiny-model/--pipeline-test).")
        wan = WanModel.from_pretrained(cfg.ckpt_path)
    wan = wan.eval().to(cfg.device, dtype=dtype)

    for p in wan.parameters():
        p.requires_grad_(False)

    feature_model = WanFeatureModel(wan, layer_idx=cfg.layer_idx, pool=cfg.pool).to(cfg.device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    cfg.num_classes = train_loader.dataset.num_classes
    text_encoder = None
    tokenizer = None
    vae = None
    vae_scale = None
    cached_prompt_context = None
    clip_model = None
    clip_normalize = None

    if not cfg.use_random_inputs:
        if cfg.vae_ckpt_path is None:
            raise ValueError("Please provide --vae-ckpt-path")
        if cfg.vae_mean is None:
            cfg.vae_mean = DEFAULT_VAE_MEAN
        if cfg.vae_std is None:
            cfg.vae_std = DEFAULT_VAE_STD

        text_dtype = torch.float32 if cfg.text_device == "cpu" else dtype
        text_cfg = TextConfig(
            name=cfg.text_encoder_name,
            text_len=cfg.text_len,
            ckpt_path=cfg.text_ckpt_path,
            device=cfg.text_device,
        )
        vae_cfg = VaeConfig(
            ckpt_path=cfg.vae_ckpt_path,
            z_dim=cfg.vae_z_dim,
            mean=cfg.vae_mean,
            std=cfg.vae_std,
            device=cfg.device,
        )
        text_encoder, tokenizer = setup_text_encoder(text_cfg, dtype=text_dtype)
        vae, vae_scale = setup_vae(vae_cfg, dtype=dtype)

        # The prompt is constant, so we can encode it once and unload the text model.
        cached_prompt_context = encode_texts(
            text_encoder,
            tokenizer,
            [cfg.text_prompt],
            cfg.text_device,
            out_device=cfg.device,
            out_dtype=dtype,
        )[0].detach()
        del text_encoder
        del tokenizer
        text_encoder = None
        tokenizer = None
        gc.collect()
        if str(cfg.text_device).startswith("cuda"):
            torch.cuda.empty_cache()

        # For i2v checkpoints, also compute CLIP features from the same image batch.
        if getattr(wan, "model_type", None) == "i2v":
            clip_dtype = (
                torch.bfloat16 if str(cfg.clip_device).startswith("cuda") else torch.float32
            )
            clip_model, clip_normalize = setup_clip_for_i2v(
                cfg.clip_ckpt_path, device=cfg.clip_device, dtype=clip_dtype
            )

    # NOTE: in_dim should be verified by running a real forward pass.
    in_dim = wan.dim
    probe = LinearProbe(in_dim, cfg.num_classes, use_mlp=cfg.use_mlp).to(cfg.device, dtype=dtype)
    optimizer = torch.optim.SGD(probe.parameters(), lr=cfg.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)

    best_val = train_linear(
        feature_model,
        probe,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        cfg.device,
        cfg.epochs,
        text_encoder,
        tokenizer,
        cfg.text_device,
        cached_prompt_context,
        vae,
        vae_scale,
        clip_model,
        clip_normalize,
        wan.in_dim,
        cfg.patch_size,
        cfg.t_value,
        cfg.text_prompt,
        dtype,
        wandb_run,
        cfg.log_every,
        cfg.use_random_inputs,
        wan.in_dim,
        wan.text_dim,
        cfg.text_len,
        cfg.log_samples,
        cfg.ckpt_dir,
        cfg.resume_path,
    )

    test_acc = evaluate(
        feature_model,
        probe,
        test_loader,
        cfg.device,
        text_encoder,
        tokenizer,
        cfg.text_device,
        cached_prompt_context,
        vae,
        vae_scale,
        clip_model,
        clip_normalize,
        wan.in_dim,
        cfg.patch_size,
        cfg.t_value,
        cfg.text_prompt,
        dtype,
        cfg.use_random_inputs,
        wan.in_dim,
        wan.text_dim,
        cfg.text_len,
    )
    print(f"final test_top1={test_acc:.2f} best_val={best_val:.2f}")


if __name__ == "__main__":
    main()
