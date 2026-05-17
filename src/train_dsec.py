"""
train_dsec.py — DSEC 11-class semantic segmentation (LADS + Gating + ViT student)
================================================================================

Based on the structure of src/gra.py (init / start / train_step / validate /
checkpoint save), but specialised for the DSEC semantic segmentation task:

  - Data: precomputed pairs from pre_dse_lads.py
      {root}/{split}_images/{seq}/images/left/eventImage/{ts}.pt   (3,H,W) float32
      {root}/{split}_images/{seq}/images/left/warpped/{ts}.png     (H,W,3) uint8
      {root}/{split}_semantic_segmentation/{split}/{seq}/11classes/{ts}.png (H,W) uint8 in [0..11]
  - Model: SmartRouterFrontEnd (gating, online & learnable) → ViT-Small (DINOv2 init)
           → Linear decoder → upsample to label resolution
  - Loss: CrossEntropyLoss with ignore_index=255
  - Metric: mIoU (computed during validation, used to pick the best checkpoint)
  - Schedule: AdamW + warmup-cosine LR (same get_lr helper as gra.py)
  - AMP: torch.amp.autocast + GradScaler (same as gra.py)
  - TensorBoard: src/runs/{now}_dsec_seg/

New dependencies (vs gra.py):
  - src/pre_dse_lads.py            (NEW — to drive offline preprocessing)
  - src/smartrouterFrontend.py     (NEW — online learnable gating)
  - src/utils.IOU                  (NEW — used for mIoU)
"""

import argparse
import os
import sys
import math
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

# NEW: make the src/ modules importable (mirrors how gra.py is run from src/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if os.path.join(SRC_DIR, "dinov2") not in sys.path:
    sys.path.insert(0, os.path.join(SRC_DIR, "dinov2"))
if "dinov2" not in sys.path:
    sys.path.append("dinov2")

from dinov2.models.vision_transformer import vit_small, vit_base                # NEW
from smartrouterFrontend import SmartRouterFrontEnd                              # NEW
from utils import get_lr, IOU                                                    # NEW (IOU = mIoU accumulator)


# -----------------------------------------------------------------------------
# Determinism (matches gra.py)
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 0):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# =============================================================================
# Config
# =============================================================================
class DSECSegConfig:
    """All DSEC-segmentation-specific knobs in one place."""
    # ---- data ----
    root             = "D:/ANew_Stage/Achenteacher/generative_event_pretraining-master/data/DSEC"
    train_split      = "train"
    valid_split      = "test"          # DSEC's "test" split has labels available for our use
    label_dir_name   = "11classes"     # DSEC semantic segmentation: 11 classes
    num_classes      = 11              # NEW — segmentation head output classes
    ignore_index     = 255

    # ---- spatial (DIMENSION ADAPT) ----
    DSEC_H           = 224             # must be a multiple of patch_size (14)
    DSEC_W           = 224
    label_H          = 224             # resize labels to here for loss; nearest interpolation
    label_W          = 224

    # ---- LADS / event normalization (re-derive with compute_rgb_stats if desired) ----
    EV_MEAN          = [0.0, 0.0, 0.0]
    EV_STD           = [1.0, 1.0, 1.0]
    IMG_MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats (DINOv2 teacher convention)
    IMG_STD          = [0.229, 0.224, 0.225]

    # ---- model ----
    vit              = "small"                  # "small" | "base"
    vit_patch        = 14
    vit_init_ckpt    = "D:/ANew_Stage/Achenteacher/generative_event_pretraining-master/weights/dinov2_vits14_pretrain.pth"
    event_encoder_init = None                   # optional: GRA-trained event encoder
    use_gating       = True                     # NEW: enable SmartRouterFrontEnd
    gating_lr_mult   = 5.0                      # NEW: same ratio used in spikingjelly_dvs128_lads.py
    use_image_branch = False                    # NEW: if True, also feed warpped RGB through a (frozen) image encoder

    # ---- optimisation ----
    batch_size       = 4
    n_workers        = 4
    lr               = 1e-4
    min_lr           = 1e-6
    wd               = 0.05
    warmup_steps     = 500
    steps            = 40000
    valid_every      = 1000
    log_every        = 20
    device           = "cuda:0"

    # ---- checkpointing ----
    ckpt_root        = "src/runs"               # tensorboard + ckpt root
    restore_ckpt     = None                     # path to .pt to resume from

    # ---- preprocessing trigger ----
    run_preprocess   = False                    # NEW: when True, call pre_dse_lads.Processor.run(...) first


# =============================================================================
# Dataset: reads pre_dse_lads outputs
# =============================================================================
class DSECSegDataset(Dataset):
    """DSEC semantic segmentation dataset built on top of pre_dse_lads outputs.

    Each item returns:
      event  : (3, H, W)  float32 — LADS hybrid frame (gated offline OR not)
      image  : (3, H, W)  float32 — warpped RGB, normalized to ImageNet stats
      label  : (H, W)     int64   — 11-class id with 255 = ignore
      stem   : str                — timestamp filename stem (debug)
    """

    def __init__(self, cfg: DSECSegConfig, split: str):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.image_root = os.path.join(cfg.root, f"{split}_images")
        self.label_root = os.path.join(cfg.root, f"{split}_semantic_segmentation", split)

        # NEW: walk all sequences; collect (event_pt, warpped_png, label_png) triples
        self.samples = []
        if not os.path.isdir(self.image_root):
            raise FileNotFoundError(f"Missing image_root: {self.image_root}")

        for seq in sorted(os.listdir(self.image_root)):
            seq_root = os.path.join(self.image_root, seq, "images", "left")
            event_dir = os.path.join(seq_root, "eventImage")
            warp_dir  = os.path.join(seq_root, "warpped")
            label_dir = os.path.join(self.label_root, seq, cfg.label_dir_name)
            if not (os.path.isdir(event_dir) and os.path.isdir(warp_dir)):
                continue
            for fn in sorted(os.listdir(event_dir)):
                if not fn.endswith(".pt"):
                    continue
                stem = os.path.splitext(fn)[0]
                ev_p = os.path.join(event_dir, fn)
                wp_p = os.path.join(warp_dir, f"{stem}.png")
                lb_p = os.path.join(label_dir, f"{stem}.png")
                if not (os.path.exists(wp_p) and os.path.exists(lb_p)):
                    continue
                self.samples.append((ev_p, wp_p, lb_p, stem))

        if not self.samples:
            raise RuntimeError(
                f"No (eventImage/*.pt, warpped/*.png, label/*.png) triples found under "
                f"{self.image_root} / {self.label_root}. Did you run pre_dse_lads.run() first?"
            )

        # normalization tensors
        self._ev_mean = torch.tensor(cfg.EV_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._ev_std  = torch.tensor(cfg.EV_STD,  dtype=torch.float32).view(3, 1, 1)
        self._im_mean = torch.tensor(cfg.IMG_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._im_std  = torch.tensor(cfg.IMG_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def _load_event(self, path):
        ev = torch.load(path, map_location="cpu").to(torch.float32)  # (3,H,W)
        # DIMENSION ADAPT: snap to model input resolution
        if ev.shape[-2] != self.cfg.DSEC_H or ev.shape[-1] != self.cfg.DSEC_W:
            ev = F.interpolate(ev.unsqueeze(0), size=(self.cfg.DSEC_H, self.cfg.DSEC_W),
                               mode="bilinear", align_corners=False).squeeze(0)
        ev = (ev - self._ev_mean) / self._ev_std
        return ev

    def _load_image(self, path):
        im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
        im = torch.from_numpy(im).permute(2, 0, 1).contiguous()                      # (3,H,W)
        # DIMENSION ADAPT
        if im.shape[-2] != self.cfg.DSEC_H or im.shape[-1] != self.cfg.DSEC_W:
            im = F.interpolate(im.unsqueeze(0), size=(self.cfg.DSEC_H, self.cfg.DSEC_W),
                               mode="bilinear", align_corners=False).squeeze(0)
        im = (im - self._im_mean) / self._im_std
        return im

    def _load_label(self, path):
        lab = np.asarray(Image.open(path), dtype=np.int64)            # (H,W) raw
        lab_t = torch.from_numpy(lab).unsqueeze(0).unsqueeze(0).float()
        # DIMENSION ADAPT: nearest-neighbour resize (preserve class ids)
        if lab_t.shape[-2] != self.cfg.label_H or lab_t.shape[-1] != self.cfg.label_W:
            lab_t = F.interpolate(lab_t, size=(self.cfg.label_H, self.cfg.label_W), mode="nearest")
        lab_t = lab_t.squeeze(0).squeeze(0).long()                    # (H,W)
        # Push anything outside [0, num_classes-1] to ignore_index
        invalid = (lab_t < 0) | (lab_t >= self.cfg.num_classes)
        lab_t[invalid] = self.cfg.ignore_index
        return lab_t

    def __getitem__(self, idx):
        ev_p, wp_p, lb_p, stem = self.samples[idx]
        event = self._load_event(ev_p)
        image = self._load_image(wp_p)
        label = self._load_label(lb_p)
        return event, image, label, stem


# =============================================================================
# Model
# =============================================================================
class ViTSegHead(nn.Module):
    """Light Linear decoder on top of ViT patch tokens.

    Reshape patch tokens (B, N, D) → (B, D, Hp, Wp) → 1x1 Conv → bilinear upsample
    to the label resolution. Kept intentionally small so the LADS+gating signal
    can carry most of the segmentation capacity.
    """
    def __init__(self, embed_dim: int, num_classes: int, out_hw: tuple):
        super().__init__()
        self.cls_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.out_hw = out_hw

    def forward(self, tokens: torch.Tensor, patch_hw: tuple):
        # tokens: (B, N, D); N == Hp * Wp
        B, N, D = tokens.shape
        Hp, Wp = patch_hw
        assert N == Hp * Wp, f"token count {N} != Hp*Wp {Hp*Wp}"
        x = tokens.transpose(1, 2).reshape(B, D, Hp, Wp)         # (B,D,Hp,Wp)
        x = self.cls_head(x)                                     # (B,C,Hp,Wp)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return x


class DSECSegModel(nn.Module):
    """Front end (gating) + ViT encoder + linear seg head.

    NEW: this is the LADS+gating analogue of the GRA student. The image branch
    is optional — kept off by default since segmentation only needs the event
    pathway with labels.
    """
    def __init__(self, cfg: DSECSegConfig):
        super().__init__()
        self.cfg = cfg

        # NEW: online gating (learnable)
        if cfg.use_gating:
            self.front_end = SmartRouterFrontEnd(
                in_channels=3, hidden_channels=16, state_channels=3,
                H=cfg.DSEC_H, W=cfg.DSEC_W,
            )
        else:
            self.front_end = None

        # NEW: ViT encoder (DINOv2 init); identical hparams to gra.py
        if cfg.vit == "small":
            self.event_encoder = vit_small(
                patch_size=cfg.vit_patch, img_size=518, block_chunks=0,
                init_values=1e-6, num_register_tokens=4,
            )
            embed_dim = 384
        elif cfg.vit == "base":
            self.event_encoder = vit_base(
                patch_size=cfg.vit_patch, img_size=518, block_chunks=0,
                init_values=1e-6, num_register_tokens=4,
            )
            embed_dim = 768
        else:
            raise ValueError(f"Unsupported vit={cfg.vit!r}")

        # NEW: optional DINOv2 / GRA pretrained initialization
        if cfg.event_encoder_init and os.path.exists(cfg.event_encoder_init):
            sd = torch.load(cfg.event_encoder_init, map_location="cpu")
            if isinstance(sd, dict) and "event_encoder" in sd:
                sd = sd["event_encoder"]
            msg = self.event_encoder.load_state_dict(sd, strict=False)
            print(f"[init] event_encoder <- {cfg.event_encoder_init} "
                  f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
        elif cfg.vit_init_ckpt and os.path.exists(cfg.vit_init_ckpt):
            sd = torch.load(cfg.vit_init_ckpt, map_location="cpu", weights_only=True)
            msg = self.event_encoder.load_state_dict(sd, strict=False)
            print(f"[init] event_encoder <- {cfg.vit_init_ckpt} (DINOv2 pretrain) "
                  f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

        # NEW: segmentation head
        self.seg_head = ViTSegHead(
            embed_dim=embed_dim,
            num_classes=cfg.num_classes,
            out_hw=(cfg.label_H, cfg.label_W),
        )

        # Patch grid (used by ViTSegHead)
        self.patch_h = cfg.DSEC_H // cfg.vit_patch
        self.patch_w = cfg.DSEC_W // cfg.vit_patch

    def forward(self, event: torch.Tensor):
        """event: (B, 3, H, W) float32 already normalized."""
        x = event
        if self.front_end is not None:
            x = self.front_end(x)                                # (B, 3, H, W) — gated three-state map
        feats = self.event_encoder.forward_features(x)
        tokens = feats["x_norm_patchtokens"]                     # (B, N, D)
        logits = self.seg_head(tokens, (self.patch_h, self.patch_w))  # (B, C, label_H, label_W)
        return logits


# =============================================================================
# Trainer (mirrors gra.py's Gra.start/train_step/validate)
# =============================================================================
class DSECTrainer:
    def __init__(self, cfg: DSECSegConfig):
        self.cfg = cfg
        self.device = cfg.device

        # NEW: optionally drive preprocessing first (LADS + gating offline)
        if cfg.run_preprocess:
            self._run_preprocess()

        # data
        self.train_set = DSECSegDataset(cfg, split=cfg.train_split)
        self.valid_set = DSECSegDataset(cfg, split=cfg.valid_split)
        print(f"[data] train samples: {len(self.train_set)}, valid samples: {len(self.valid_set)}")

        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.n_workers, pin_memory=True, drop_last=True,
        )
        self.valid_loader = DataLoader(
            self.valid_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.n_workers, pin_memory=True, drop_last=False,
        )

        # model
        self.model = DSECSegModel(cfg).to(self.device)

        # loss + metric
        self.criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
        self.miou = IOU(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index, device=self.device)

        # optimizer — NEW: separate lr group for front_end (mirrors spikingjelly_dvs128_lads.py)
        front_params, base_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "front_end" in n:
                front_params.append(p)
            else:
                base_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": base_params,  "lr": cfg.lr, "lr_mult": 1.0,           "weight_decay": cfg.wd},
                {"params": front_params, "lr": cfg.lr * cfg.gating_lr_mult,      # NEW: gating LR boost
                                          "lr_mult": cfg.gating_lr_mult,         "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )

        self.amp = torch.amp.autocast(device_type="cuda")
        self.scaler = torch.amp.GradScaler(device="cuda")

        # logging / ckpt
        self.now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.run_dir = os.path.join(cfg.ckpt_root, f"{self.now}_dsec_seg")
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.best_miou = -1.0

        # resume
        self.start_step = 0
        if cfg.restore_ckpt and os.path.exists(cfg.restore_ckpt):
            self._restore(cfg.restore_ckpt)

    # -------- preprocessing hook --------
    def _run_preprocess(self):
        """NEW: drive pre_dse_lads.Processor.run() for both splits before training."""
        from pre_dse_lads import Processor as LADSPreproc
        for split in [self.cfg.train_split, self.cfg.valid_split]:
            class _A:  # tiny adapter for Processor's args.* lookups
                pass
            a = _A()
            a.root = self.cfg.root
            a.split = split
            a.device = self.device
            a.out_h = self.cfg.DSEC_H
            a.out_w = self.cfg.DSEC_W
            a.lads_decay_func = "er"
            a.lads_decay_param = 0.2
            a.lads_patch_size = 32
            a.lads_interpolate_patches = True
            a.lads_min_decay = 0.0
            a.lads_ts_to_seconds_factor = 1.0
            a.gate_ckpt = None
            a.gate_device = "cpu"
            a.save_png_preview = False
            proc = LADSPreproc(a)
            print(f"[preprocess] running pre_dse_lads on split={split}")
            proc.run(n_workers=1)

    # -------- ckpt --------
    def _restore(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_step = int(ckpt.get("step", 0)) + 1
        self.best_miou = float(ckpt.get("best_miou", -1.0))
        print(f"[resume] from {path} @ step {self.start_step}, best_miou={self.best_miou:.4f}")

    def _save(self, step, miou, tag="last"):
        path = os.path.join(self.run_dir, f"ckpt_{tag}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "best_miou": self.best_miou,
            "miou": miou,
        }, path)
        return path

    # -------- main loop --------
    def train_step(self, batch, step):
        self.model.train()
        event, image, label, _stems = batch
        event = event.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        # image branch isn't used by the seg head, but is loaded for symmetry / optional ext.
        # image = image.to(self.device, non_blocking=True)

        # warmup-cosine LR
        cur_lr = get_lr(step, self.cfg.warmup_steps, self.cfg.lr, self.cfg.steps, self.cfg.min_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = cur_lr * pg["lr_mult"]

        t0 = time.time()
        with self.amp:
            logits = self.model(event)                           # (B, C, label_H, label_W)
            loss = self.criterion(logits, label)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        t1 = time.time()

        if (step + 1) % self.cfg.log_every == 0:
            self.writer.add_scalar("loss/train", loss.item(), step + 1)
            self.writer.add_scalar("lr", cur_lr, step + 1)
            print(
                f"step {step+1}/{self.cfg.steps}, lr={cur_lr:.2e}, "
                f"loss={loss.item():.4f}, grad_norm={float(grad_norm):.3f}, "
                f"dt={t1 - t0:.2f}s, in={tuple(event.shape)}"
            )

    @torch.no_grad()
    def validate(self, step):
        self.model.eval()
        self.miou.reset()
        total_loss, n_batches = 0.0, 0
        for event, image, label, _stems in tqdm(self.valid_loader, desc=f"valid@{step+1}"):
            event = event.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            with self.amp:
                logits = self.model(event)
                loss = self.criterion(logits, label)
            pred = logits.argmax(dim=1)                          # (B, H, W)
            self.miou.update(pred, label)
            total_loss += float(loss.item())
            n_batches += 1

        ious = self.miou.compute()                               # (num_classes,)
        miou_value = float(torch.nanmean(ious).item())
        valid_loss = total_loss / max(1, n_batches)
        self.writer.add_scalar("loss/valid", valid_loss, step + 1)
        self.writer.add_scalar("miou/valid", miou_value, step + 1)
        for c, v in enumerate(ious.tolist()):
            if not math.isnan(v):
                self.writer.add_scalar(f"iou_per_class/{c}", v, step + 1)
        print(f"[valid] step={step+1} loss={valid_loss:.4f} mIoU={miou_value:.4f}")

        # best ckpt by mIoU
        last_path = self._save(step, miou_value, tag="last")
        if miou_value > self.best_miou:
            self.best_miou = miou_value
            best_path = self._save(step, miou_value, tag="best")
            print(f"[ckpt] new best mIoU={miou_value:.4f} → {best_path}")
        else:
            print(f"[ckpt] last → {last_path} (best so far {self.best_miou:.4f})")

    def start(self):
        train_iter = iter(self.train_loader)
        for step in range(self.start_step, self.cfg.steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            self.train_step(batch, step)
            if (step + 1) % self.cfg.valid_every == 0 or (step + 1) == self.cfg.steps:
                self.validate(step)
                self.model.train()


# =============================================================================
# CLI
# =============================================================================
def _build_arg_parser():
    p = argparse.ArgumentParser(description="Train ViT segmentation on DSEC (LADS+Gating preproc)")
    p.add_argument("--root", default=None, type=str)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--batch_size", default=None, type=int)
    p.add_argument("--n_workers", default=None, type=int)
    p.add_argument("--lr", default=None, type=float)
    p.add_argument("--steps", default=None, type=int)
    p.add_argument("--warmup_steps", default=None, type=int)
    p.add_argument("--valid_every", default=None, type=int)
    p.add_argument("--vit", default=None, type=str, choices=["small", "base"])
    p.add_argument("--use_gating", default=None, type=lambda s: s.lower() in ("1", "true", "yes"))
    p.add_argument("--restore_ckpt", default=None, type=str)
    p.add_argument("--event_encoder_init", default=None, type=str,
                   help="Optional path to a GRA-trained event_encoder ckpt to warm-start")
    p.add_argument("--run_preprocess", action="store_true",
                   help="Run pre_dse_lads on both splits before training")
    p.add_argument("--seed", default=0, type=int)
    return p


def _apply_overrides(cfg: DSECSegConfig, args):
    for k, v in vars(args).items():
        if v is None or k == "seed":
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = _apply_overrides(DSECSegConfig(), args)
    trainer = DSECTrainer(cfg)
    trainer.start()
    print("Training completed.")