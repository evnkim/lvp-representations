"""
Minimal WAN linear probing scaffold for ImageNet-style classification.

This is intentionally incomplete: data loading, VAE encode, and text context
construction are left as TODOs to wire into your local setup.
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Repo-local imports
from algorithms.wan.modules.model import WanModel
from data_classify.imagenet import ImageNetSubset


@dataclass
class ProbeConfig:
    ckpt_path: str
    layer_idx: int
    use_mlp: bool
    num_classes: int = 1000
    pool: str = "mean"  # "mean" or "max"
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8
    subset_name: str = "imagenet-1k"
    data_root: str = "data/datasets"
    lr: float = 1e-3
    epochs: int = 10


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
    TODO:
    - build ImageNet train/val DataLoader
    - produce raw images (B, 3, H, W) and labels
    """
    train_ds = ImageNetSubset(
        split="train",
        subset_name=cfg.subset_name,
        data_root=cfg.data_root,
        crop_mode="random",
    )
    val_ds = ImageNetSubset(
        split="val",
        subset_name=cfg.subset_name,
        data_root=cfg.data_root,
        crop_mode="center",
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


@torch.no_grad()
def encode_images_to_wan_latents(vae, images, vae_mean, vae_inv_std):
    """
    TODO:
    - WAN expects x as list of tensors [C_in, F, H, W] per sample
    - For ImageNet, F=1 frame
    - Use VAE encoder to get z, then normalize with mean/std
    """
    raise NotImplementedError


@torch.no_grad()
def make_context(text_encoder, tokenizer, texts, device):
    """
    TODO:
    - Use WAN's text encoder + tokenizer to produce text embeddings
    - WAN expects `context` as List[Tensor], each [L, C]
    - Keep prompt fixed, e.g. "a photo"
    """
    raise NotImplementedError


def compute_top1_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.numel()


@torch.no_grad()
def evaluate(feature_model, probe, data_loader, device):
    feature_model.eval()
    probe.eval()

    total_correct = 0
    total_count = 0

    for images, labels in data_loader:
        # TODO: move to device and create WAN inputs.
        # images = images.to(device, non_blocking=True)
        # labels = labels.to(device, non_blocking=True)
        #
        # x = encode_images_to_wan_latents(...)
        # context = make_context(...)
        # t = ...
        # seq_len = ...
        # feats = feature_model(x, t, context, seq_len)
        # logits = probe(feats)
        #
        # correct, count = compute_top1_accuracy(logits, labels)
        # total_correct += correct
        # total_count += count
        pass

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
):
    criterion = nn.CrossEntropyLoss()
    feature_model.eval()
    probe.train()

    best_val = 0.0
    for epoch in range(epochs):
        for images, labels in train_loader:
            # TODO: move to device and create WAN inputs.
            # images = images.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            #
            # x = encode_images_to_wan_latents(...)
            # context = make_context(...)
            # t = ...
            # seq_len = ...
            # feats = feature_model(x, t, context, seq_len)
            # logits = probe(feats)
            #
            # loss = criterion(logits, labels)
            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # optimizer.step()
            pass

        scheduler.step()
        val_acc = evaluate(feature_model, probe, val_loader, device)
        best_val = max(best_val, val_acc)
        print(f"epoch {epoch}: val_top1={val_acc:.2f} best={best_val:.2f}")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--use-mlp", action="store_true")
    parser.add_argument("--subset-name", type=str, default="imagenet-1k")
    parser.add_argument("--data-root", type=str, default="data/datasets")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cfg = ProbeConfig(
        ckpt_path=args.ckpt_path,
        layer_idx=args.layer_idx,
        use_mlp=args.use_mlp,
        subset_name=args.subset_name,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    wan = WanModel.from_pretrained(cfg.ckpt_path).eval().to(cfg.device)

    for p in wan.parameters():
        p.requires_grad_(False)

    feature_model = WanFeatureModel(wan, layer_idx=cfg.layer_idx, pool=cfg.pool).to(
        cfg.device
    )

    # TODO: build loaders, VAE/text encoder setup, and run training loop.
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # NOTE: in_dim should be verified by running a real forward pass.
    in_dim = wan.dim
    probe = LinearProbe(in_dim, cfg.num_classes, use_mlp=cfg.use_mlp).to(cfg.device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=cfg.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=0.0
    )

    # TODO: plug real loaders
    # best_val = train_linear(
    #     feature_model,
    #     probe,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     scheduler,
    #     cfg.device,
    #     cfg.epochs,
    # )
    #
    # test_acc = evaluate(feature_model, probe, test_loader, cfg.device)
    # print(f"final test_top1={test_acc:.2f}")

    print("Stub created. Fill TODOs for data, VAE encode, text context, and loaders.")


if __name__ == "__main__":
    main()
