#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PANNs CNN14 training script for noise segment detection (frame-wise).

Run:
  python v3.py

All defaults are embedded in Config below. You can still override via CLI flags.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    from sklearn.metrics import average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


@dataclass
class Config:
    # Paths (edit these if needed)
    manifest: str = "data/manifest.jsonl"
    out_dir: str = "runs/panns_cnn14"

    # Audio / feature
    sr: int = 48000
    n_fft: int = 1024
    hop: int = 320
    win: int = 1024
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 20000.0

    # Chunking
    chunk_s: float = 2.0
    val_stride_s: float = 2.0
    label_expand_ms: float = 20.0

    # Training
    epochs: int = 30
    batch_size: int = 32
    steps_per_epoch: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 1234

    # Imbalance handling
    pos_sample_prob: float = 0.6
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    # Regularization
    smooth_lambda: float = 0.02
    specaug_time_mask: int = 30
    specaug_freq_mask: int = 8
    specaug_time_masks: int = 2
    specaug_freq_masks: int = 2
    specaug_p: float = 0.6

    # Performance
    num_workers: int = 6
    prefetch_factor: int = 4
    amp: bool = True

    # Optional pretrained
    pretrained: str = ""  # path to PANNs CNN14 .pth (optional)


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def get_audio_info(path: str) -> Tuple[int, int, float]:
    info = sf.info(path)
    sr = int(info.samplerate)
    frames = int(info.frames)
    dur = frames / sr if sr > 0 else 0.0
    return sr, frames, dur


def load_audio_segment(path: str, target_sr: int, start_s: float, dur_s: float) -> torch.Tensor:
    orig_sr, total_frames, _ = get_audio_info(path)
    start_frame = int(round(start_s * orig_sr))
    num_frames = int(round(dur_s * orig_sr))

    start_frame = max(0, min(start_frame, total_frames))
    readable = max(0, min(num_frames, total_frames - start_frame))

    if readable > 0:
        with sf.SoundFile(path, "r") as f:
            f.seek(start_frame)
            audio = f.read(frames=readable, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio)
    else:
        wav = torch.zeros(0, dtype=torch.float32)

    if orig_sr != target_sr and wav.numel() > 0:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)

    target_len = int(round(dur_s * target_sr))
    if wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    else:
        wav = wav[:target_len]
    return wav


def ls_gain_align_batch(ref: torch.Tensor, rec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if ref.dim() == 1:
        ref = ref[None, :]
    if rec.dim() == 1:
        rec = rec[None, :]
    num = torch.sum(rec * ref, dim=1)
    den = torch.sum(ref * ref, dim=1) + eps
    g = num / den
    return ref * g[:, None]


def intervals_to_frame_labels_batch(
    noisy_batch: List[List[List[float]]],
    start_s: torch.Tensor,
    n_frames: int,
    hop: int,
    sr: int,
    expand_ms: float,
    device: torch.device,
) -> torch.Tensor:
    B = len(noisy_batch)
    expand_s = float(expand_ms) / 1000.0
    t = (torch.arange(n_frames, device=device, dtype=torch.float32) * (hop / sr))[None, :]
    times = start_s.to(device=device, dtype=torch.float32)[:, None] + t

    y = torch.zeros((B, n_frames), device=device, dtype=torch.float32)
    for b in range(B):
        intervals = noisy_batch[b]
        if not intervals:
            continue
        for s, e in intervals:
            s = float(s) - expand_s
            e = float(e) + expand_s
            y[b] = torch.where((times[b] >= s) & (times[b] <= e), torch.ones_like(y[b]), y[b])
    return y


def focal_bce_with_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    loss = alpha_t * torch.pow(1.0 - p_t, gamma) * bce
    return loss.mean()


def smoothness_loss(prob: torch.Tensor) -> torch.Tensor:
    if prob.size(1) < 2:
        return torch.tensor(0.0, device=prob.device)
    dp = torch.abs(prob[:, 1:] - prob[:, :-1])
    return dp.mean()


def pr_curve_by_threshold(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 101) -> Dict[str, np.ndarray]:
    thresholds = np.linspace(0.0, 1.0, int(bins), dtype=np.float64)
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)

    prec = np.zeros_like(thresholds, dtype=np.float64)
    rec = np.zeros_like(thresholds, dtype=np.float64)
    fpr = np.zeros_like(thresholds, dtype=np.float64)

    pos = (y_true == 1)
    neg = ~pos
    eps = 1e-12

    for i, thr in enumerate(thresholds):
        pred = (y_prob >= thr)
        tp = float(np.logical_and(pred, pos).sum())
        fp = float(np.logical_and(pred, neg).sum())
        fn = float(np.logical_and(~pred, pos).sum())
        tn = float(np.logical_and(~pred, neg).sum())
        prec[i] = tp / (tp + fp + eps)
        rec[i] = tp / (tp + fn + eps)
        fpr[i] = fp / (fp + tn + eps)

    return {"thr": thresholds, "precision": prec, "recall": rec, "fpr": fpr}


def save_pr_curve(out_dir: str, epoch: int, curves: Dict[str, np.ndarray]) -> None:
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"pr_threshold_epoch{epoch:03d}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("thr,precision,recall,fpr\n")
        for t, p, r, fp in zip(curves["thr"], curves["precision"], curves["recall"], curves["fpr"]):
            f.write(f"{t:.6f},{p:.6f},{r:.6f},{fp:.6f}\n")


# -----------------------------
# Dataset
# -----------------------------

class RandomChunkDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, Any]], cfg: Config):
        self.items = items
        self.cfg = cfg
        self.meta = []
        for it in items:
            _, _, dur0 = get_audio_info(it["ref"])
            self.meta.append({"dur_s": dur0})
        self.len_ = cfg.steps_per_epoch

    def __len__(self) -> int:
        return self.len_

    def _sample_start(self, it: Dict[str, Any], dur_s: float) -> float:
        chunk = self.cfg.chunk_s
        if dur_s <= chunk:
            return 0.0
        noisy = it.get("noisy", [])
        if noisy and np.random.rand() < self.cfg.pos_sample_prob:
            s, e = noisy[np.random.randint(0, len(noisy))]
            center = 0.5 * (float(s) + float(e))
            start = float(np.clip(center - 0.5 * chunk, 0.0, dur_s - chunk))
            start += float(np.random.uniform(-0.2, 0.2))
            return float(np.clip(start, 0.0, dur_s - chunk))
        return float(np.random.uniform(0.0, dur_s - chunk))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = np.random.randint(0, len(self.items))
        it = self.items[i]
        dur_s = self.meta[i]["dur_s"]
        start_s = self._sample_start(it, dur_s)

        ref = load_audio_segment(it["ref"], self.cfg.sr, start_s, self.cfg.chunk_s)
        rec = load_audio_segment(it["rec"], self.cfg.sr, start_s, self.cfg.chunk_s)

        return {
            "ref": ref,
            "rec": rec,
            "noisy": it.get("noisy", []),
            "start_s": float(start_s),
        }


class ValChunkDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, Any]], cfg: Config):
        self.items = items
        self.cfg = cfg
        self.segs: List[Tuple[int, float]] = []
        for i, it in enumerate(items):
            _, _, dur_s = get_audio_info(it["ref"])
            if dur_s <= 0:
                continue
            t = 0.0
            while t <= dur_s - 1e-6:
                self.segs.append((i, float(t)))
                t += cfg.val_stride_s
        if len(self.segs) == 0 and len(items) > 0:
            self.segs.append((0, 0.0))

    def __len__(self) -> int:
        return len(self.segs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i, start_s = self.segs[idx]
        it = self.items[i]
        ref = load_audio_segment(it["ref"], self.cfg.sr, start_s, self.cfg.chunk_s)
        rec = load_audio_segment(it["rec"], self.cfg.sr, start_s, self.cfg.chunk_s)
        return {
            "ref": ref,
            "rec": rec,
            "noisy": it.get("noisy", []),
            "start_s": float(start_s),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[float]]], torch.Tensor]:
    refs = torch.stack([b["ref"] for b in batch], dim=0)
    recs = torch.stack([b["rec"] for b in batch], dim=0)
    noisys = [b["noisy"] for b in batch]
    starts = torch.tensor([b["start_s"] for b in batch], dtype=torch.float32)
    return refs, recs, noisys, starts


# -----------------------------
# Feature extractor
# -----------------------------

class LogMel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            win_length=cfg.win,
            hop_length=cfg.hop,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
            center=True,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.mel(wav)
        x = self.db(x)
        return x


def spec_augment(feat: torch.Tensor, cfg: Config) -> torch.Tensor:
    if cfg.specaug_p <= 0.0:
        return feat
    if torch.rand(()) > cfg.specaug_p:
        return feat
    B, C, M, T = feat.shape
    for b in range(B):
        for _ in range(cfg.specaug_freq_masks):
            fw = int(torch.randint(0, cfg.specaug_freq_mask + 1, (1,)).item())
            if fw <= 0:
                continue
            f0_max = max(1, M - fw + 1)
            f0 = int(torch.randint(0, f0_max, (1,)).item())
            feat[b, :, f0:f0 + fw, :] = 0.0
        for _ in range(cfg.specaug_time_masks):
            tw = int(torch.randint(0, cfg.specaug_time_mask + 1, (1,)).item())
            if tw <= 0:
                continue
            t0_max = max(1, T - tw + 1)
            t0 = int(torch.randint(0, t0_max, (1,)).item())
            feat[b, :, :, t0:t0 + tw] = 0.0
    return feat


def norm_per_utt(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = feat.mean(dim=(2, 3), keepdim=True)
    std = feat.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (feat - mu) / std


# -----------------------------
# PANNs CNN14
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class Cnn14Frame(nn.Module):
    def __init__(self, n_mels: int = 64, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.block1 = ConvBlock(in_ch, base)
        self.block2 = ConvBlock(base, base)
        self.block3 = ConvBlock(base, base * 2)
        self.block4 = ConvBlock(base * 2, base * 2)
        self.block5 = ConvBlock(base * 2, base * 4)
        self.block6 = ConvBlock(base * 4, base * 4)

        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Conv1d(base * 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, M, T]
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        x = self.pool(x)
        x = self.block6(x)
        x = self.pool(x)

        # x: [B, C, M', T']
        x = torch.mean(x, dim=2)  # [B, C, T']
        x = self.dropout(x)
        x = self.fc(x).squeeze(1)  # [B, T']
        return x


# -----------------------------
# Train / Eval
# -----------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    mel: LogMel,
    dl,
    cfg: Config,
    device: torch.device,
    epoch: int,
    out_dir: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_frames = 0.0
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []

    for ref, rec, noisy, starts in dl:
        ref = ref.to(device)
        rec = rec.to(device)
        starts = starts.to(device)

        ref_aligned = ls_gain_align_batch(ref, rec)
        res = rec - ref_aligned

        feat = mel(res).unsqueeze(1)
        feat = norm_per_utt(feat)

        logits = model(feat)
        T = logits.size(1)
        y = intervals_to_frame_labels_batch(noisy, starts, T, cfg.hop * 64, cfg.sr, cfg.label_expand_ms, device)
        loss = focal_bce_with_logits(logits, y, cfg.focal_alpha, cfg.focal_gamma)
        prob = torch.sigmoid(logits)

        all_y.append(y.detach().cpu().numpy().reshape(-1))
        all_p.append(prob.detach().cpu().numpy().reshape(-1))

        total_loss += float(loss.item()) * float(T * ref.size(0))
        total_frames += float(T * ref.size(0))

    y_true = np.concatenate(all_y) if all_y else np.zeros((0,), dtype=np.float32)
    y_prob = np.concatenate(all_p) if all_p else np.zeros((0,), dtype=np.float32)
    ap = float("nan")
    if _HAS_SK and y_true.size > 0 and len(np.unique(y_true)) > 1:
        ap = float(average_precision_score(y_true, y_prob))

    curves = pr_curve_by_threshold(y_true, y_prob, bins=101) if y_true.size > 0 else {}
    fpr_thr = float("nan")
    if curves:
        valid = curves["fpr"] <= 0.005
        if np.any(valid):
            idx = int(np.argmax(curves["thr"][valid]))
            fpr_thr = float(curves["thr"][valid][idx])
        save_pr_curve(out_dir, epoch, curves)

    return {
        "loss": total_loss / max(1.0, total_frames),
        "ap": ap,
        "fpr_thr@0.005": fpr_thr,
    }


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ensure_dir(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    items = read_jsonl(cfg.manifest)
    if len(items) == 0:
        raise RuntimeError("manifest is empty")

    train_items = [it for it in items if str(it.get("split", "")).lower() != "val"]
    val_items = [it for it in items if str(it.get("split", "")).lower() == "val"]
    if len(val_items) == 0:
        val_items = items

    train_ds = RandomChunkDataset(train_items, cfg)
    val_ds = ValChunkDataset(val_items, cfg)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    mel = LogMel(cfg).to(device)
    model = Cnn14Frame(n_mels=cfg.n_mels).to(device)

    if cfg.pretrained:
        state = torch.load(cfg.pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val = float("inf")

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        run_frames = 0.0

        for ref, rec, noisy, starts in train_dl:
            ref = ref.to(device, non_blocking=True)
            rec = rec.to(device, non_blocking=True)
            starts = starts.to(device, non_blocking=True)

            ref_aligned = ls_gain_align_batch(ref, rec)
            res = rec - ref_aligned

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                feat = mel(res).unsqueeze(1)
                feat = norm_per_utt(feat)
                feat = spec_augment(feat, cfg)

                logits = model(feat)
                T = logits.size(1)
                y = intervals_to_frame_labels_batch(noisy, starts, T, cfg.hop * 64, cfg.sr, cfg.label_expand_ms, device)
                loss_det = focal_bce_with_logits(logits, y, cfg.focal_alpha, cfg.focal_gamma)
                prob = torch.sigmoid(logits)
                loss_sm = smoothness_loss(prob) * cfg.smooth_lambda
                loss = loss_det + loss_sm

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            run_loss += float(loss.item()) * float(T * ref.size(0))
            run_frames += float(T * ref.size(0))

        val_m = evaluate(model, mel, val_dl, cfg, device, ep, cfg.out_dir)
        train_loss = run_loss / max(1.0, run_frames)
        scheduler.step()

        dt = time.time() - t0
        print(
            f"[ep {ep:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_m['loss']:.4f} ap={val_m['ap']:.4f} "
            f"fpr_thr@0.005={val_m['fpr_thr@0.005']:.4f} "
            f"time={dt:.1f}s lr={opt.param_groups[0]['lr']:.6f}"
        )

        ckpt = {
            "epoch": ep,
            "cfg": asdict(cfg),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))
        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))

    print("Done. Output:", cfg.out_dir)


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=Config.manifest)
    ap.add_argument("--out_dir", type=str, default=Config.out_dir)
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch_size", type=int, default=Config.batch_size)
    ap.add_argument("--steps_per_epoch", type=int, default=Config.steps_per_epoch)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--pretrained", type=str, default=Config.pretrained)
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    cfg = Config(
        manifest=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        pretrained=args.pretrained,
    )
    train(cfg)


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("torchaudio:", getattr(torchaudio, "__version__", "?"))
    print("soundfile:", getattr(sf, "__version__", "?"))
    main()
