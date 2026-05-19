"""
EEGConformer + ERD-Align — Full Experiment
==========================================
Chạy trên máy local (Dell 7560 A2000 hoặc bất kỳ CUDA GPU nào).

Cách dùng:
    python EEGConformer_ERD_Align.py                    # cả 2 datasets
    python EEGConformer_ERD_Align.py --dataset 004      # chỉ BNCI2014_004
    python EEGConformer_ERD_Align.py --dataset 001      # chỉ BNCI2014_001
    python EEGConformer_ERD_Align.py --no-riemannian    # Euclidean align (nhanh hơn)
    python EEGConformer_ERD_Align.py --amp              # mixed precision (A2000 nhanh hơn)

Kết quả lưu vào: ./results/conformer/
Data MOABB lưu vào: ~/mne_data/ (lần đầu tự download)

Requirements:
    pip install moabb mne torch scipy scikit-learn pandas matplotlib seaborn tqdm
"""

import os
import copy
import json
import warnings
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import sqrtm, inv, logm, expm
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not found, progress bars disabled. pip install tqdm")

warnings.filterwarnings('ignore')

import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004
from moabb.paradigms import MotorImagery
moabb.set_log_level('warning')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────────
    'sfreq'         : 250,
    'tmin'          : -2.0,    # 2s baseline trước cue (cần cho ERD)
    'tmax'          :  4.0,    # 4s MI period
    'baseline_start':    0,    # index 0   = t = -2s
    'baseline_end'  :  500,    # index 500 = t =  0s
    'task_start'    :  625,    # index 625 = t =  0.5s (tránh onset artifact)
    'task_end'      : 1375,    # index 1375= t =  3.5s
    'model_start'   :  500,    # model thấy [0s, 4s]

    # ── EEGConformer ──────────────────────────────────────────────────────────
    'n_filters'   :  40,       # F: temporal filters, d_model = F*2 = 80
    'kernel_size' :  25,       # ~100ms @ 250Hz
    'pool_size'   :  75,
    'pool_stride' :  15,
    'n_heads'     :   8,       # 80 / 8 = 10 per head ✓
    'n_layers'    :   3,       # transformer depth (3 = good balance)
    'd_ff'        : 256,       # FFN hidden dim
    'n_proj'      : 128,       # classifier projection dim
    'dropout_cnn' : 0.50,
    'dropout_attn': 0.30,

    # ── Training ──────────────────────────────────────────────────────────────
    'lr'         : 5e-4,
    'epochs'     :  200,
    'batch_size' :   32,
    'patience'   :   30,       # early stopping patience
    'grad_clip'  :  1.0,       # gradient clipping (essential for Transformer)
    'weight_decay': 1e-4,

    # ── TTA ───────────────────────────────────────────────────────────────────
    'tta_lr'       : 1e-3,
    'tta_steps'    :     1,    # update steps per trial
    'alpha'        :   1.0,    # entropy loss weight
    'beta'         :   1.0,    # ERD loss weight
    'erd_threshold': -10.0,    # ERD% < threshold → subject có ERD

    # ── ERD bands ─────────────────────────────────────────────────────────────
    'mu_low'   :  8,
    'mu_high'  : 12,
    'beta_low' : 13,
    'beta_high': 30,

    # ── Output ────────────────────────────────────────────────────────────────
    'output_dir': './results/conformer',
    'seed'      :   42,
}

CHANNEL_NAMES = {
    'BNCI2014_001': [
        'Fz','FC3','FC1','FCz','FC2','FC4',
        'C5','C3','C1','Cz','C2','C4','C6',
        'CP3','CP1','CPz','CP2','CP4',
        'P1','Pz','P2','POz'
    ],
    'BNCI2014_004': ['C3','Cz','C4'],
}

METHODS = ['no_adapt','tent','ek_tta','erd_align','erd_align_ek']
LABELS  = ['No-adapt','TENT','EK-TTA','ERD-Align','ERD-Align+EK']
PALETTE = {
    'no_adapt'    : '#888780',
    'tent'        : '#E24B4A',
    'ek_tta'      : '#0F6E56',
    'erd_align'   : '#534AB7',
    'erd_align_ek': '#BA7517',
}


# ══════════════════════════════════════════════════════════════════════════════
# EEGConformer
# ══════════════════════════════════════════════════════════════════════════════
class EEGConformer(nn.Module):
    """
    EEG Conformer: Convolutional Transformer for EEG decoding.
    Song et al. (2022), IEEE TNSRE.

    Architecture:
        Block 1 — CNN Patch Embedding:
            Temporal conv (1×kernel) → local spectral features
            Spatial conv depthwise   → channel mixing per filter
            AvgPool                  → temporal downsampling → patches

        Block 2 — Transformer Encoder:
            Positional encoding (learnable)
            Multi-head self-attention across patches
            Feed-forward network per patch
            Pre-LayerNorm (training stability)

        Block 3 — Classification Head:
            Flatten → Linear(n_proj) → Linear(n_classes)

    TTA interface (same as EEGNet):
        get_bn_params() → BN + LayerNorm params
        set_bn_train()  → freeze all, enable only norm layers
    """
    def __init__(self, n_classes, n_channels, n_times,
                 n_filters=40, kernel_size=25,
                 pool_size=75, pool_stride=15,
                 n_heads=8, n_layers=3, d_ff=256, n_proj=128,
                 dropout_cnn=0.5, dropout_attn=0.3):
        super().__init__()
        d_model = n_filters * 2   # = 80

        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}")

        # ── Block 1: CNN ───────────────────────────────────────────────────
        self.conv_time = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, kernel_size),
                      padding=(0, kernel_size // 2), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(n_filters, d_model, (n_channels, 1),
                      groups=n_filters, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, pool_size), stride=(1, pool_stride)),
            nn.Dropout(dropout_cnn),
        )

        # ── Block 2: Transformer ───────────────────────────────────────────
        seq_len = self._compute_seq_len(n_channels, n_times)
        self.pos_emb = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout_attn,
            activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=False)

        # ── Block 3: Classifier ────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, n_proj),
            nn.ELU(),
            nn.Dropout(dropout_attn),
            nn.Linear(n_proj, n_classes),
        )
        self.d_model  = d_model
        self.seq_len  = seq_len
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    EEGConformer: seq_len={seq_len}, "
              f"d_model={d_model}, n_params={n_params:,}")

    def _compute_seq_len(self, n_ch, n_t):
        x = torch.zeros(1, 1, n_ch, n_t)
        with torch.no_grad():
            x = self.conv_time(x)
            x = self.conv_spatial(x)
        return x.shape[-1]

    def forward(self, x):
        x = self.conv_time(x)
        x = self.conv_spatial(x)              # (B, d, 1, T')
        B, D, _, T = x.shape
        x = x.squeeze(2).permute(0, 2, 1)    # (B, T', d)
        x = x + self.pos_emb[:, :T]
        x = self.transformer(x)
        return self.fc(x.flatten(1))

    def get_bn_params(self):
        """BatchNorm2d (CNN) + LayerNorm (Transformer) → TTA optimizer."""
        params = []
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if m.weight is not None: params.append(m.weight)
                if m.bias   is not None: params.append(m.bias)
        return params

    def set_bn_train(self):
        """Freeze all → enable only norm layers for TTA."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                m.train()
                if m.weight is not None: m.weight.requires_grad_(True)
                if m.bias   is not None: m.bias.requires_grad_(True)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# ERD Module + Loss Functions
# ══════════════════════════════════════════════════════════════════════════════
def compute_erd(epoch_np, c3_idx, c4_idx, cfg):
    """
    ERD% từ raw EEG (không cần nhãn).
    ERD% = (P_task - P_rest) / P_rest × 100
    Âm = ERD thực sự xảy ra.
    """
    sfreq   = cfg['sfreq']
    nperseg = min(128, (cfg['baseline_end'] - cfg['baseline_start']) // 2)
    erd = {}
    for name, idx in [('C3', c3_idx), ('C4', c4_idx)]:
        base = epoch_np[idx, cfg['baseline_start']:cfg['baseline_end']]
        task = epoch_np[idx, cfg['task_start']:cfg['task_end']]
        freqs, pb = welch(base, fs=sfreq, nperseg=nperseg)
        _,     pt = welch(task, fs=sfreq, nperseg=nperseg)
        for band, mask in [
            ('mu',   (freqs >= cfg['mu_low'])   & (freqs <= cfg['mu_high'])),
            ('beta', (freqs >= cfg['beta_low']) & (freqs <= cfg['beta_high']))]:
            pb_m = np.mean(pb[mask]) + 1e-10
            erd[f'{name}_{band}'] = (np.mean(pt[mask]) - pb_m) / pb_m * 100
        erd[f'{name}_comp'] = (erd[f'{name}_mu'] + erd[f'{name}_beta']) / 2
    return erd


def entropy_loss(logits):
    """Shannon entropy minimization (TENT)."""
    p = F.softmax(logits, dim=-1)
    return -(p * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def erd_consistency_loss(logits, erd, left_idx, right_idx, threshold):
    """
    ERD Consistency Loss — phần contribution chính.
    Penalize khi prediction không consistent với ERD signal.
    Predict left  → C4 (contralateral) phải có ERD âm.
    Predict right → C3 (contralateral) phải có ERD âm.
    """
    probs = F.softmax(logits, dim=-1)
    c4 = torch.tensor(erd['C4_comp'], dtype=torch.float32, device=logits.device)
    c3 = torch.tensor(erd['C3_comp'], dtype=torch.float32, device=logits.device)
    return (probs[0, left_idx]  * F.relu(c4 - threshold) +
            probs[0, right_idx] * F.relu(c3 - threshold))


# ══════════════════════════════════════════════════════════════════════════════
# TTA Methods
# ══════════════════════════════════════════════════════════════════════════════
class NoAdapt:
    def __init__(self, model, cfg):
        self.model = model
    def predict(self, x_raw, x_t):
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), None


class TENTMethod:
    def __init__(self, model, cfg):
        self.model = model.set_bn_train()
        self.opt   = torch.optim.Adam(model.get_bn_params(), lr=cfg['tta_lr'])
        self.cfg   = cfg
    def predict(self, x_raw, x_t):
        self.model.set_bn_train()
        for _ in range(self.cfg['tta_steps']):
            loss = entropy_loss(self.model(x_t))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), None


class EKTTAMethod:
    def __init__(self, model, cfg, c3, c4, li, ri):
        self.model               = model.set_bn_train()
        self.opt                 = torch.optim.Adam(model.get_bn_params(),
                                                     lr=cfg['tta_lr'])
        self.cfg, self.c3, self.c4 = cfg, c3, c4
        self.li, self.ri         = li, ri
    def predict(self, x_raw, x_t):
        erd = compute_erd(x_raw[0], self.c3, self.c4, self.cfg)
        self.model.set_bn_train()
        for _ in range(self.cfg['tta_steps']):
            logits = self.model(x_t)
            loss   = self.cfg['beta'] * erd_consistency_loss(
                logits, erd, self.li, self.ri, self.cfg['erd_threshold'])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), erd


# ══════════════════════════════════════════════════════════════════════════════
# ERD-Align Utilities
# ══════════════════════════════════════════════════════════════════════════════
def bandpass_batch(X, lo, hi, sfreq=250, order=4):
    """Bandpass filter. X: (n_trials, n_ch, n_times)."""
    nyq = sfreq / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    out  = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(X.shape[1]):
            out[i, j] = filtfilt(b, a, X[i, j])
    return out


def mean_cov(X_filt, reg=1e-5):
    """Mean covariance over trials. X_filt: (n_trials, n_ch, n_times)."""
    covs = [(x @ x.T) / x.shape[-1] + np.eye(x.shape[0]) * reg
            for x in X_filt]
    return np.mean(covs, axis=0)


def riemannian_mean(covs, n_iter=15, tol=1e-8):
    """
    Riemannian mean of SPD matrices (gradient descent on SPD manifold).
    M_{k+1} = M^(1/2) exp(1/n Σ log(M^(-1/2) C_i M^(-1/2))) M^(1/2)
    """
    n = len(covs)
    M = np.mean(covs, axis=0)
    for _ in range(n_iter):
        Ms  = np.real(sqrtm(M))
        Msi = np.real(inv(Ms))
        grad = np.zeros_like(M)
        for C in covs:
            grad += np.real(logm(Msi @ C @ Msi))
        grad /= n
        M_new = Ms @ np.real(expm(grad)) @ Ms
        if np.linalg.norm(M_new - M, 'fro') < tol:
            return M_new
        M = M_new
    return M


def whitening_transform(sigma_src, sigma_tgt):
    """W = sigma_tgt^(1/2) @ sigma_src^(-1/2). Maps src → tgt distribution."""
    return np.real(np.real(sqrtm(sigma_tgt)) @ np.real(inv(np.real(sqrtm(sigma_src)))))


class ERDAlignMethod:
    """
    ERD-Selective Riemannian Alignment.

    Workflow:
        fit_source(X_src)          — offline: reference covariance from source
        screen_and_fit(Xf, Xm)    — per-subject: ERD screen + alignment W
        predict(x_raw, x_model)   — per-trial: align → frozen model → predict

    Key design:
        - Covariance computed from mu+beta band (8-30Hz) — ERD-relevant
        - ERD screening: ERS subjects → no alignment (no-adapt)
        - No model update: transform input instead
    """
    def __init__(self, model, cfg, c3, c4,
                 erd_threshold=-10.0, use_riemannian=True):
        self.model         = model
        self.cfg           = cfg
        self.c3, self.c4   = c3, c4
        self.threshold     = erd_threshold
        self.use_riemannian= use_riemannian
        self.sigma_ref     = None
        self.W             = None
        self.has_erd       = None
        self.erd_val       = None

    def fit_source(self, X_src_model, sfreq=250):
        lo, hi = self.cfg['mu_low'], self.cfg['beta_high']
        Xf     = bandpass_batch(X_src_model, lo, hi, sfreq)
        n_ch   = Xf.shape[1]
        covs   = [(x @ x.T) / x.shape[-1] + np.eye(n_ch) * 1e-5
                  for x in Xf]
        if self.use_riemannian:
            # Subsample để tránh quá chậm
            if len(covs) > 150:
                idx  = np.random.choice(len(covs), 150, replace=False)
                covs = [covs[i] for i in idx]
            self.sigma_ref = riemannian_mean(covs)
        else:
            self.sigma_ref = np.mean(covs, axis=0)

    def screen_and_fit(self, X_full, X_model, sfreq=250):
        assert self.sigma_ref is not None, "Chạy fit_source() trước."
        erds = [compute_erd(X_full[i], self.c3, self.c4, self.cfg)
                for i in range(len(X_full))]
        c3m  = np.mean([e['C3_comp'] for e in erds])
        c4m  = np.mean([e['C4_comp'] for e in erds])
        self.erd_val = (c3m + c4m) / 2
        self.has_erd = min(c3m, c4m) < self.threshold

        if not self.has_erd:
            self.W = None; return

        lo, hi = self.cfg['mu_low'], self.cfg['beta_high']
        Xf     = bandpass_batch(X_model, lo, hi, sfreq)
        sig_t  = mean_cov(Xf)
        self.W = whitening_transform(sig_t, self.sigma_ref)

    def predict(self, x_raw, x_model_np, device):
        if self.W is not None:
            xa  = self.W @ x_model_np[0]
            x_t = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
        else:
            x_t = torch.FloatTensor(x_model_np[:, np.newaxis]).to(device)
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), self.erd_val


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset(ds_name, cfg):
    print(f"\n{'='*55}")
    print(f"  Loading {ds_name} via MOABB...")
    print(f"  (Lần đầu sẽ download vào ~/mne_data/)")

    ds       = BNCI2014_001() if ds_name == 'BNCI2014_001' else BNCI2014_004()
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand'], n_classes=2,
        fmin=0.5, fmax=45.0,
        tmin=cfg['tmin'], tmax=cfg['tmax'],
        resample=cfg['sfreq'])
    X, y, meta = paradigm.get_data(ds)
    print(f"  X shape  : {X.shape}")
    print(f"  Duration : {X.shape[2]/cfg['sfreq']:.1f}s per trial")

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    cls    = list(le.classes_)
    li, ri = cls.index('left_hand'), cls.index('right_hand')

    ch     = CHANNEL_NAMES[ds_name]
    c3, c4 = ch.index('C3'), ch.index('C4')
    print(f"  C3={c3}, C4={c4} | left_hand={li}, right_hand={ri}")

    ms, sdata = cfg['model_start'], {}
    for s in sorted(meta['subject'].unique()):
        mask = meta['subject'] == s
        Xs   = X[mask]
        sdata[s] = dict(
            X_full=Xs, X_model=Xs[:, :, ms:],
            y=y_enc[mask], c3_idx=c3, c4_idx=c4,
            left_idx=li, right_idx=ri)
    return sdata


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def train_backbone(X_tr, y_tr, n_ch, n_t, cfg, device, use_amp=False):
    """Train EEGConformer on source subjects with optional mixed precision."""
    model = EEGConformer(
        n_classes=2, n_channels=n_ch, n_times=n_t,
        n_filters   = cfg['n_filters'],
        kernel_size = cfg['kernel_size'],
        pool_size   = cfg['pool_size'],
        pool_stride = cfg['pool_stride'],
        n_heads     = cfg['n_heads'],
        n_layers    = cfg['n_layers'],
        d_ff        = cfg['d_ff'],
        n_proj      = cfg['n_proj'],
        dropout_cnn = cfg['dropout_cnn'],
        dropout_attn= cfg['dropout_attn'],
    ).to(device)

    X_t    = torch.FloatTensor(X_tr[:, np.newaxis]).to(device)
    y_t    = torch.LongTensor(y_tr).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=cfg['batch_size'], shuffle=True,
                        pin_memory=False)

    opt    = torch.optim.Adam(model.parameters(),
                               lr=cfg['lr'],
                               weight_decay=cfg['weight_decay'])
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    crit   = nn.CrossEntropyLoss()
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    best_loss, best_state, pat = float('inf'), None, 0
    ep_iter = range(cfg['epochs'])
    if HAS_TQDM:
        ep_iter = tqdm(ep_iter, desc="    epoch", leave=False, ncols=70)

    for ep in ep_iter:
        model.train(); total = 0
        for xb, yb in loader:
            opt.zero_grad()
            if scaler:
                with autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['grad_clip'])
                scaler.step(opt); scaler.update()
            else:
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg['grad_clip'])
                opt.step()
            total += loss.item()

        sch.step()
        avg = total / len(loader)
        if HAS_TQDM:
            ep_iter.set_postfix(loss=f"{avg:.4f}")

        if avg < best_loss:
            best_loss, pat = avg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= cfg['patience']:
                break

    model.load_state_dict(best_state)
    return model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# LOSO Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def run_loso(subject_data, cfg, device, ds_name,
             use_riemannian=True, use_amp=False):
    """
    LOSO evaluation: 5 methods × 9 subjects.
    Returns DataFrame with accuracy, kappa, ERD info per fold.
    """
    subjects = sorted(subject_data.keys())
    rows     = []
    t0_total = time.time()

    for i, ts in enumerate(subjects):
        t0 = time.time()
        print(f"\n  Fold {i+1}/{len(subjects)} — Test subject: {ts}")
        info = subject_data[ts]
        c3, c4, li, ri = (info['c3_idx'], info['c4_idx'],
                          info['left_idx'], info['right_idx'])

        # Source data
        src_X = np.concatenate(
            [subject_data[s]['X_model'] for s in subjects if s != ts])
        src_y = np.concatenate(
            [subject_data[s]['y']       for s in subjects if s != ts])
        n_ch, n_t = src_X.shape[1], src_X.shape[2]

        print(f"    Train: {src_X.shape[0]} trials, "
              f"n_ch={n_ch}, n_t={n_t}")

        # ── Train backbone ───────────────────────────────────────────────
        model = train_backbone(src_X, src_y, n_ch, n_t, cfg, device, use_amp)
        orig  = copy.deepcopy(model.state_dict())

        Xf, Xm, yt = info['X_full'], info['X_model'], info['y']

        # ── 1. No-adapt ──────────────────────────────────────────────────
        model.load_state_dict(orig); model.eval()
        with torch.no_grad():
            xt   = torch.FloatTensor(Xm[:, np.newaxis]).to(device)
            p_na = model(xt).argmax(-1).cpu().numpy()

        # ── 2. TENT ──────────────────────────────────────────────────────
        model.load_state_dict(orig)
        tm     = TENTMethod(model, cfg)
        p_tent = []
        trial_iter = range(len(Xm))
        if HAS_TQDM:
            trial_iter = tqdm(trial_iter, desc="    TENT", leave=False, ncols=60)
        for j in trial_iter:
            xt = torch.FloatTensor(Xm[j:j+1, np.newaxis]).to(device)
            p, _ = tm.predict(None, xt); p_tent.append(p)
        p_tent = np.array(p_tent)

        # ── 3. EK-TTA ────────────────────────────────────────────────────
        model.load_state_dict(orig)
        em   = EKTTAMethod(model, cfg, c3, c4, li, ri)
        p_ek, erds = [], []
        trial_iter = range(len(Xm))
        if HAS_TQDM:
            trial_iter = tqdm(trial_iter, desc="    EK-TTA", leave=False, ncols=60)
        for j in trial_iter:
            xt = torch.FloatTensor(Xm[j:j+1, np.newaxis]).to(device)
            p, e = em.predict(Xf[j:j+1], xt); p_ek.append(p); erds.append(e)
        p_ek = np.array(p_ek)

        # ── 4. ERD-Align ─────────────────────────────────────────────────
        model.load_state_dict(orig)
        am = ERDAlignMethod(model, cfg, c3, c4,
                            cfg['erd_threshold'], use_riemannian)
        print(f"    Fitting ERD-Align source covariance...", end=' ', flush=True)
        t_fit = time.time()
        am.fit_source(src_X, cfg['sfreq'])
        am.screen_and_fit(Xf, Xm, cfg['sfreq'])
        grp = "ERD" if am.has_erd else "ERS"
        print(f"done ({time.time()-t_fit:.1f}s) | "
              f"group={grp}, ERD={am.erd_val:.1f}%")

        p_align = []
        trial_iter = range(len(Xm))
        if HAS_TQDM:
            trial_iter = tqdm(trial_iter, desc="    Align", leave=False, ncols=60)
        for j in trial_iter:
            p, _ = am.predict(Xf[j:j+1], Xm[j:j+1], device)
            p_align.append(p)
        p_align = np.array(p_align)

        # ── 5. ERD-Align + EK-TTA ────────────────────────────────────────
        model.load_state_dict(orig)
        am2 = ERDAlignMethod(model, cfg, c3, c4,
                             cfg['erd_threshold'], use_riemannian)
        am2.fit_source(src_X, cfg['sfreq'])
        am2.screen_and_fit(Xf, Xm, cfg['sfreq'])
        ek2    = EKTTAMethod(model, cfg, c3, c4, li, ri)
        p_aek  = []
        trial_iter = range(len(Xm))
        if HAS_TQDM:
            trial_iter = tqdm(trial_iter, desc="    Align+EK", leave=False, ncols=60)
        for j in trial_iter:
            xm_j = (am2.W @ Xm[j])[np.newaxis] if am2.W is not None else Xm[j:j+1]
            xt   = torch.FloatTensor(xm_j[:, np.newaxis]).to(device)
            p, _ = ek2.predict(Xf[j:j+1], xt); p_aek.append(p)
        p_aek = np.array(p_aek)

        # ── Log results ──────────────────────────────────────────────────
        erd_means = {}
        if erds and erds[0]:
            erd_means = {f'erd_{k}': np.mean([e[k] for e in erds])
                         for k in erds[0].keys()}

        print(f"    {'Method':<16} {'Acc':>8} {'Kappa':>8}")
        print(f"    {'-'*34}")
        for method, preds in [
            ('no_adapt',     p_na),
            ('tent',         p_tent),
            ('ek_tta',       p_ek),
            ('erd_align',    p_align),
            ('erd_align_ek', p_aek),
        ]:
            acc   = accuracy_score(yt, preds)
            kappa = cohen_kappa_score(yt, preds)
            row   = dict(dataset=ds_name, subject=ts, method=method,
                         accuracy=acc, kappa=kappa,
                         erd_group=grp, erd_value=am.erd_val)
            row.update(erd_means); rows.append(row)
            print(f"    {method:<16} {acc:>8.4f} {kappa:>8.4f}")

        elapsed = time.time() - t0
        remaining = elapsed * (len(subjects) - i - 1)
        print(f"    Fold time: {elapsed:.0f}s | "
              f"Est. remaining: {remaining/60:.1f} min")

    total_t = time.time() - t0_total
    print(f"\n  Total time: {total_t/60:.1f} min")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis + Visualization
# ══════════════════════════════════════════════════════════════════════════════
def print_summary_table(df_all):
    print("\n" + "="*68)
    print("TABLE — EEGConformer LOSO (mean ± std, n=9 subjects)")
    print("="*68)
    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df  = df_all[df_all['dataset'] == ds]
        na  = df[df['method'] == 'no_adapt']['accuracy'].mean()
        print(f"\n  {ds}")
        print(f"  {'Method':<20} {'Acc':>8} {'±std':>7} {'κ':>8} {'vs NA':>8}")
        print(f"  {'-'*55}")
        for m, lb in zip(METHODS, LABELS):
            sub = df[df['method'] == m]['accuracy']
            d   = sub.mean() - na
            mk  = ' ↑' if d > 0.003 else (' ↓' if d < -0.003 else '  ')
            print(f"  {lb:<20} {sub.mean():>8.4f} {sub.std():>7.4f}"
                  f" {df[df['method']==m]['kappa'].mean():>8.4f}"
                  f" {d:>+8.4f}{mk}")

    print("\n" + "-"*68)
    print("WILCOXON — methods vs No-adapt")
    print("-"*68)
    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df   = df_all[df_all['dataset'] == ds]
        subs = sorted(df['subject'].unique())
        na   = [df[(df['subject']==s)&(df['method']=='no_adapt')]['accuracy'].values[0]
                for s in subs]
        print(f"\n  {ds}:")
        for m, lb in zip(METHODS[1:], LABELS[1:]):
            vals  = [df[(df['subject']==s)&(df['method']==m)]['accuracy'].values[0]
                     for s in subs]
            delta = np.array(vals) - np.array(na)
            if np.all(delta == 0):
                print(f"    {lb:<18}: identical"); continue
            try:
                stat, p = wilcoxon(vals, na)
                d   = delta.mean() / (delta.std() + 1e-10)
                sig = '✓ p<0.05' if p < 0.05 else '✗ ns'
                print(f"    {lb:<18}: W={stat:.0f}, p={p:.4f}, "
                      f"d={d:.2f}, Δ={delta.mean():+.4f}  {sig}")
            except Exception as e:
                print(f"    {lb:<18}: {e}")


def print_group_analysis(df_all):
    print("\n" + "="*60)
    print("GROUP ANALYSIS — ERD vs ERS subjects")
    print("="*60)
    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df = df_all[df_all['dataset'] == ds]
        for grp in ['ERD', 'ERS']:
            dg = df[df['erd_group'] == grp]
            if len(dg) == 0: continue
            ns   = dg['subject'].nunique()
            na_g = dg[dg['method']=='no_adapt']['accuracy'].mean()
            print(f"\n  {ds} | {grp} ({ns} subjects):")
            print(f"  {'Method':<20} {'Acc':>8} {'vs NA':>8}")
            for m, lb in zip(METHODS, LABELS):
                sub = dg[dg['method']==m]['accuracy']
                if len(sub) == 0: continue
                d  = sub.mean() - na_g
                mk = ' ↑' if d > 0.005 else (' ↓' if d < -0.005 else '  ')
                print(f"  {lb:<20} {sub.mean():>8.4f} {d:>+8.4f}{mk}")

    print("\n" + "-"*60)
    print("KEY: ERD-Align > No-adapt on ERD subjects?")
    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df     = df_all[df_all['dataset'] == ds]
        dg     = df[df['erd_group'] == 'ERD']
        if len(dg) == 0: continue
        subs   = sorted(dg['subject'].unique())
        align  = [dg[(dg['subject']==s)&(dg['method']=='erd_align')]['accuracy'].values[0]
                  for s in subs]
        na     = [dg[(dg['subject']==s)&(dg['method']=='no_adapt')]['accuracy'].values[0]
                  for s in subs]
        win    = sum(1 for a, n in zip(align, na) if a > n)
        delta  = np.array(align) - np.array(na)
        print(f"  {ds}: {win}/{len(subs)} ERD subjects "
              f"Align>NA (Δmean={delta.mean():+.4f})")


def plot_results(df_all, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, ds in enumerate(['BNCI2014_004', 'BNCI2014_001']):
        df   = df_all[df_all['dataset'] == ds]
        subs = sorted(df['subject'].unique())

        # Bar chart
        ax   = axes[row, 0]
        ms   = [df[df['method']==m]['accuracy'].mean() for m in METHODS]
        ss   = [df[df['method']==m]['accuracy'].std()  for m in METHODS]
        cs   = [PALETTE[m] for m in METHODS]
        bars = ax.bar(LABELS, ms, yerr=ss, color=cs, alpha=0.87,
                      capsize=5, width=0.6, error_kw={'lw': 1.5})
        ax.set_ylim(max(0, min(ms)-0.10), min(1, max(ms)+0.10))
        ax.set_title(f'{ds} — EEGConformer', fontsize=11, fontweight='500')
        ax.set_ylabel('Accuracy')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(LABELS, rotation=20, ha='right', fontsize=8)
        for bar, m in zip(bars, ms):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f'{m:.3f}', ha='center', fontsize=8)

        # Scatter: ERD-Align vs No-adapt
        ax   = axes[row, 1]
        a_v  = [df[(df['subject']==s)&(df['method']=='erd_align')]['accuracy'].values[0]
                for s in subs]
        n_v  = [df[(df['subject']==s)&(df['method']=='no_adapt')]['accuracy'].values[0]
                for s in subs]
        grps = [df[df['subject']==s]['erd_group'].values[0] for s in subs]
        gc   = {'ERD': '#534AB7', 'ERS': '#E24B4A'}

        for s, av, nv, g in zip(subs, a_v, n_v, grps):
            ax.scatter(nv, av, c=gc.get(g, '#888780'), s=80, zorder=5)
            ax.annotate(f'S{s}', (nv, av), xytext=(4, 3),
                        textcoords='offset points', fontsize=8)

        lo = min(n_v+a_v)-0.02; hi = max(n_v+a_v)+0.02
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4, label='y=x')
        ax.set_xlabel('No-adapt accuracy', fontsize=10)
        ax.set_ylabel('ERD-Align accuracy', fontsize=10)
        ax.set_title(f'{ds} — ERD-Align vs No-adapt\n(above line = improvement)',
                     fontsize=10)
        ax.legend(handles=[mpatches.Patch(color='#534AB7', label='ERD'),
                            mpatches.Patch(color='#E24B4A', label='ERS')],
                  fontsize=8)
        ax.grid(alpha=0.25)

    plt.suptitle('EEGConformer + ERD-Align — LOSO Results', fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, 'conformer_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='EEGConformer + ERD-Align experiment')
    p.add_argument('--dataset', choices=['001','004','both'], default='both',
                   help='Dataset to run (default: both)')
    p.add_argument('--no-riemannian', action='store_true',
                   help='Use Euclidean alignment (faster, less accurate)')
    p.add_argument('--amp', action='store_true',
                   help='Mixed precision training (A2000 supports this)')
    p.add_argument('--output', default=CONFIG['output_dir'],
                   help=f"Output directory (default: {CONFIG['output_dir']})")
    p.add_argument('--compare', default=None,
                   help='Path to previous EEGNet results CSV for comparison')
    return p.parse_args()


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    CONFIG['output_dir'] = args.output
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    use_riemannian = not args.no_riemannian
    use_amp        = args.amp
    print(f"Riemannian alignment: {use_riemannian}")
    print(f"Mixed precision (AMP): {use_amp}")

    # Dataset selection
    ds_map = {
        '004': ['BNCI2014_004'],
        '001': ['BNCI2014_001'],
        'both': ['BNCI2014_004', 'BNCI2014_001'],
    }
    datasets = ds_map[args.dataset]

    # Run
    all_dfs = []
    for ds_name in datasets:
        print(f"\n{'='*60}\n  DATASET: {ds_name}\n{'='*60}")
        sdata = load_dataset(ds_name, CONFIG)
        df    = run_loso(sdata, CONFIG, device, ds_name,
                         use_riemannian=use_riemannian, use_amp=use_amp)
        path  = os.path.join(CONFIG['output_dir'], f'{ds_name}_conformer.csv')
        df.to_csv(path, index=False)
        print(f"\n  Saved → {path}")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    all_path = os.path.join(CONFIG['output_dir'], 'all_conformer.csv')
    df_all.to_csv(all_path, index=False)

    # Analysis
    print_summary_table(df_all)
    print_group_analysis(df_all)

    # Compare with previous EEGNet results
    compare_path = args.compare or os.path.join(
        os.path.dirname(CONFIG['output_dir']), 'all_results_v2.csv')
    if os.path.exists(compare_path):
        df_prev = pd.read_csv(compare_path)
        print("\n" + "─"*60)
        print("BACKBONE COMPARISON — EEGConformer vs EEGNet (No-adapt)")
        print("─"*60)
        for ds in datasets:
            c_acc = df_all[(df_all['dataset']==ds)&
                           (df_all['method']=='no_adapt')]['accuracy'].mean()
            if ds in df_prev['dataset'].values:
                e_acc = df_prev[(df_prev['dataset']==ds)&
                                (df_prev['method']=='no_adapt')]['accuracy'].mean()
                print(f"  {ds}: Conformer={c_acc:.4f}  "
                      f"EEGNet={e_acc:.4f}  Δ={c_acc-e_acc:+.4f}")

    # Plot
    if len(all_dfs) == 2:
        plot_results(df_all, CONFIG['output_dir'])

    print(f"\n✓ Done. Results in: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
