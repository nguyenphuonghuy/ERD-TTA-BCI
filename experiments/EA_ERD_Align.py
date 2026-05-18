"""
EA + ERD-Align Pipeline — Complete Experiment
=============================================
Euclidean Alignment (EA) tại training + ERD-guided Alignment tại test.

Hypothesis: EA tại training giảm domain gap giữa source subjects →
            model học representation tốt hơn →
            ERD-Align tại test hiệu quả hơn vì align vào cùng 1 không gian.

Experiment matrix:
  Rows (training):   Raw | EA
  Cols (test):       No-adapt | TENT | Std-EA | ERD-Align | ERD-Align+EK

Cách chạy:
    python EA_ERD_Align.py                    # cả 2 datasets
    python EA_ERD_Align.py --dataset 004      # chỉ BNCI2014_004
    python EA_ERD_Align.py --amp              # mixed precision

Kết quả: ./results/ea_erd/
"""

import os, copy, warnings, argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import sqrtm, inv
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore')
import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_004
from moabb.paradigms import MotorImagery
moabb.set_log_level('warning')


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    'sfreq'         : 250,
    'tmin'          : -2.0,
    'tmax'          :  4.0,
    'baseline_start':    0,
    'baseline_end'  :  500,
    'task_start'    :  625,
    'task_end'      : 1375,
    'model_start'   :  500,

    # EEGNet (proven tốt hơn Conformer cho small-N LOSO)
    'F1'      :  8,
    'D'       :  2,
    'dropout' : 0.5,

    'lr'         : 1e-3,
    'epochs'     :  150,
    'batch_size' :   32,
    'patience'   :   25,
    'grad_clip'  :  5.0,
    'weight_decay': 1e-4,

    # TTA
    'tta_lr'       : 1e-3,
    'tta_steps'    :    1,
    'alpha'        :  1.0,
    'beta'         :  1.0,
    'erd_threshold': -10.0,

    # ERD bands
    'mu_low'   :  8,
    'mu_high'  : 12,
    'beta_low' : 13,
    'beta_high': 30,

    # EA regularization
    'ea_reg'   : 1e-5,

    'output_dir': './results/ea_erd',
    'seed'      :   42,
}

CH = {
    'BNCI2014_001': ['Fz','FC3','FC1','FCz','FC2','FC4',
                     'C5','C3','C1','Cz','C2','C4','C6',
                     'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'],
    'BNCI2014_004': ['C3','Cz','C4'],
}

# Methods: (train_mode, test_mode)
EXPERIMENTS = [
    ('raw',  'no_adapt',    'No-adapt (baseline)'),
    ('raw',  'tent',        'TENT'),
    ('raw',  'erd_align',   'ERD-Align (raw train)'),
    ('ea',   'no_adapt',    'EA-train + No-adapt'),
    ('ea',   'std_ea',      'EA-train + Std-EA'),
    ('ea',   'erd_align',   'EA-train + ERD-Align ★'),
    ('ea',   'erd_align_ek','EA-train + ERD-Align+EK ★'),
    ('ea',   'tent',        'EA-train + TENT'),
]

PALETTE = {
    'No-adapt (baseline)'      : '#888780',
    'TENT'                     : '#E24B4A',
    'ERD-Align (raw train)'    : '#5DCAA5',
    'EA-train + No-adapt'      : '#B0AEA8',
    'EA-train + Std-EA'        : '#BA7517',
    'EA-train + ERD-Align ★'   : '#534AB7',
    'EA-train + ERD-Align+EK ★': '#0F6E56',
    'EA-train + TENT'          : '#F4857F',
}


# ══════════════════════════════════════════════════════════════════════
# EEGNet
# ══════════════════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    def __init__(self, n_classes, n_channels, n_times, F1=8, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        flat = self._flat(n_channels, n_times)
        self.fc = nn.Linear(flat, n_classes)

    def _flat(self, nc, nt):
        with torch.no_grad():
            x = torch.zeros(1, 1, nc, nt)
            return self.block3(self.block2(self.block1(x))).numel()

    def forward(self, x):
        return self.fc(self.block3(self.block2(self.block1(x))).flatten(1))

    def get_norm_params(self):
        p = []
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                p += [m.weight, m.bias]
        return p

    def set_norm_train(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
        return self


# ══════════════════════════════════════════════════════════════════════
# Euclidean Alignment (EA)
# ══════════════════════════════════════════════════════════════════════
def compute_ref_cov(X, reg=1e-5):
    """
    Tính reference covariance matrix từ batch trials.
    X: (n_trials, n_ch, n_times)
    Returns: R (n_ch, n_ch) — regularized mean covariance
    """
    covs = [x @ x.T / x.shape[-1] for x in X]
    R = np.mean(covs, axis=0) + np.eye(X.shape[1]) * reg
    return R


def ea_whiten(X, R):
    """
    Whiten: X_aligned = R^(-1/2) @ X
    Maps subject's distribution → near-identity covariance.
    """
    R_inv_sqrt = np.real(inv(np.real(sqrtm(R))))
    return np.array([R_inv_sqrt @ x for x in X]), R_inv_sqrt


def ea_align_subjects(subject_data, cfg):
    """
    Áp dụng EA cho từng source subject riêng lẻ.
    Trả về dict {subj: {X_model_ea, R_inv_sqrt, ...}}
    và mean_ref_cov (reference của training distribution).

    Lý do align từng subject riêng:
    - Standard EA: R_s = mean cov của subject s
    - X_s_aligned = R_s^(-1/2) @ X_s
    - Sau alignment, mọi subject có cov ~ identity
    - Model train trên không gian "universal"
    """
    aligned = {}
    all_R_inv = []

    for s, info in subject_data.items():
        X = info['X_model']  # (n_trials, n_ch, n_times)
        R = compute_ref_cov(X, cfg['ea_reg'])
        X_ea, R_inv_sqrt = ea_whiten(X, R)
        aligned[s] = {**info,
                      'X_model'   : X_ea,
                      'R_inv_sqrt': R_inv_sqrt,
                      'R'         : R}
        all_R_inv.append(R_inv_sqrt)

    # Mean của R_inv_sqrt → dùng làm reference khi align test subject
    mean_R_inv = np.mean(all_R_inv, axis=0)
    return aligned, mean_R_inv


# ══════════════════════════════════════════════════════════════════════
# ERD Utilities
# ══════════════════════════════════════════════════════════════════════
def compute_erd(epoch_np, c3, c4, cfg):
    sfreq   = cfg['sfreq']
    nperseg = min(128, (cfg['baseline_end'] - cfg['baseline_start']) // 2)
    erd = {}
    for name, idx in [('C3', c3), ('C4', c4)]:
        base = epoch_np[idx, cfg['baseline_start']:cfg['baseline_end']]
        task = epoch_np[idx, cfg['task_start']:cfg['task_end']]
        f, pb = welch(base, fs=sfreq, nperseg=nperseg)
        _, pt = welch(task, fs=sfreq, nperseg=nperseg)
        for band, mask in [
            ('mu',   (f >= cfg['mu_low'])   & (f <= cfg['mu_high'])),
            ('beta', (f >= cfg['beta_low']) & (f <= cfg['beta_high']))]:
            pb_m = np.mean(pb[mask]) + 1e-10
            erd[f'{name}_{band}'] = (np.mean(pt[mask]) - pb_m) / pb_m * 100
        erd[f'{name}_comp'] = (erd[f'{name}_mu'] + erd[f'{name}_beta']) / 2
    return erd


def bandpass_batch(X, lo, hi, sfreq=250, order=4):
    nyq = sfreq / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    out  = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(X.shape[1]):
            out[i, j] = filtfilt(b, a, X[i, j])
    return out


def mean_cov(X_filt, reg=1e-5):
    return np.mean(
        [(x @ x.T) / x.shape[-1] + np.eye(x.shape[0]) * reg for x in X_filt],
        axis=0)


# ══════════════════════════════════════════════════════════════════════
# Loss Functions
# ══════════════════════════════════════════════════════════════════════
def entropy_loss(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def erd_loss(logits, erd, li, ri, threshold):
    p  = F.softmax(logits, dim=-1)
    c4 = torch.tensor(erd['C4_comp'], dtype=torch.float32, device=logits.device)
    c3 = torch.tensor(erd['C3_comp'], dtype=torch.float32, device=logits.device)
    return (p[0, li] * F.relu(c4 - threshold) +
            p[0, ri] * F.relu(c3 - threshold))


# ══════════════════════════════════════════════════════════════════════
# Test-Time Adaptation Methods
# ══════════════════════════════════════════════════════════════════════
class TestAdapter:
    """Base class cho tất cả test-time methods."""
    def __init__(self, model, cfg, c3, c4, li, ri,
                 src_X_model=None, src_mean_R_inv=None,
                 train_mode='raw'):
        self.model        = model
        self.cfg          = cfg
        self.c3, self.c4  = c3, c4
        self.li, self.ri  = li, ri
        self.src_X_model  = src_X_model
        self.mean_R_inv   = src_mean_R_inv
        self.train_mode   = train_mode

        # ERD-Align state
        self.W_erd        = None
        self.has_erd      = None
        self.erd_val      = None

        # Std EA state
        self.R_inv_test   = None

    def fit_test(self, X_full, X_model):
        """Pre-compute alignment transforms từ tất cả test trials."""
        # 1. ERD screening
        erds      = [compute_erd(X_full[i], self.c3, self.c4, self.cfg)
                     for i in range(len(X_full))]
        c3m       = np.mean([e['C3_comp'] for e in erds])
        c4m       = np.mean([e['C4_comp'] for e in erds])
        self.erd_val = (c3m + c4m) / 2
        self.has_erd = min(c3m, c4m) < self.cfg['erd_threshold']

        # 2. Standard EA transform (cho Std-EA test)
        R_test        = compute_ref_cov(X_model, self.cfg['ea_reg'])
        _, self.R_inv_test = ea_whiten(X_model[:1], R_test)

        # 3. ERD-selective alignment
        if self.has_erd and self.src_X_model is not None:
            lo, hi   = self.cfg['mu_low'], self.cfg['beta_high']
            # Source reference: ERD-band covariance của training data
            Xf_src   = bandpass_batch(self.src_X_model, lo, hi)
            sig_src  = mean_cov(Xf_src)
            # Test: ERD-band covariance
            Xf_test  = bandpass_batch(X_model, lo, hi)
            sig_test = mean_cov(Xf_test)
            # W: map test ERD-band distribution → source ERD-band distribution
            sqrt_src     = np.real(sqrtm(sig_src))
            sqrt_inv_tst = np.real(inv(np.real(sqrtm(sig_test))))
            self.W_erd   = np.real(sqrt_src @ sqrt_inv_tst)

    def _apply_std_ea(self, x_np):
        """Apply standard EA to single trial."""
        if self.R_inv_test is not None:
            return self.R_inv_test @ x_np
        return x_np

    def _apply_erd_align(self, x_np):
        """Apply ERD-selective alignment to single trial."""
        if self.W_erd is not None and self.has_erd:
            return self.W_erd @ x_np
        return x_np

    def _to_tensor(self, x_np, device):
        return torch.FloatTensor(x_np[np.newaxis, np.newaxis]).to(device)


class NoAdaptAdapter(TestAdapter):
    def predict(self, x_full, x_model, device):
        self.model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(x_model, device)
            return self.model(x_t).argmax(-1).item(), self.erd_val


class TENTAdapter(TestAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.set_norm_train()
        self.opt = torch.optim.Adam(
            self.model.get_norm_params(), lr=self.cfg['tta_lr'])

    def predict(self, x_full, x_model, device):
        self.model.set_norm_train()
        x_t = self._to_tensor(x_model, device)
        for _ in range(self.cfg['tta_steps']):
            loss = entropy_loss(self.model(x_t))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), self.erd_val


class StdEAAdapter(TestAdapter):
    """Standard EA at test time: whiten test data with test R_inv."""
    def predict(self, x_full, x_model, device):
        x_aligned = self._apply_std_ea(x_model)
        self.model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(x_aligned, device)
            return self.model(x_t).argmax(-1).item(), self.erd_val


class ERDAlignAdapter(TestAdapter):
    """
    ERD-Selective Alignment at test time.

    Difference vs Std-EA:
    - Alignment guided by mu+beta band covariance (8-30Hz)
    - Only applied if subject shows ERD (not ERS)
    - For ERS subjects: fall back to no-adapt
    
    When combined with EA training:
    - Training space is already whitened
    - Test alignment maps to SAME whitened space
    - ERD band ensures motor-relevant alignment
    """
    def predict(self, x_full, x_model, device):
        x_aligned = self._apply_erd_align(x_model)
        self.model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(x_aligned, device)
            return self.model(x_t).argmax(-1).item(), self.erd_val


class ERDAlignEKAdapter(TestAdapter):
    """ERD-Align + EK-TTA: align input THEN update norm layers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.set_norm_train()
        self.opt = torch.optim.Adam(
            self.model.get_norm_params(), lr=self.cfg['tta_lr'])

    def predict(self, x_full, x_model, device):
        erd       = compute_erd(x_full, self.c3, self.c4, self.cfg)
        x_aligned = self._apply_erd_align(x_model)
        self.model.set_norm_train()
        x_t = self._to_tensor(x_aligned, device)
        for _ in range(self.cfg['tta_steps']):
            logits = self.model(x_t)
            loss   = (self.cfg['alpha'] * entropy_loss(logits) +
                      self.cfg['beta']  * erd_loss(
                          logits, erd, self.li, self.ri,
                          self.cfg['erd_threshold']))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), self.erd_val


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════
def load_dataset(ds_name, cfg):
    print(f"\n{'='*52}")
    print(f"  Loading {ds_name}...")
    ds       = BNCI2014_001() if ds_name == 'BNCI2014_001' else BNCI2014_004()
    paradigm = MotorImagery(
        events=['left_hand','right_hand'], n_classes=2,
        fmin=0.5, fmax=45.0,
        tmin=cfg['tmin'], tmax=cfg['tmax'],
        resample=cfg['sfreq'])
    X, y, meta = paradigm.get_data(ds)
    print(f"  X shape: {X.shape}")

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    cls    = list(le.classes_)
    li, ri = cls.index('left_hand'), cls.index('right_hand')

    ch     = CH[ds_name]
    c3, c4 = ch.index('C3'), ch.index('C4')
    print(f"  C3={c3}, C4={c4} | left={li}, right={ri}")

    ms, sdata = cfg['model_start'], {}
    for s in sorted(meta['subject'].unique()):
        mask = meta['subject'] == s
        Xs   = X[mask]
        sdata[s] = dict(
            X_full=Xs, X_model=Xs[:, :, ms:], y=y_enc[mask],
            c3_idx=c3, c4_idx=c4, left_idx=li, right_idx=ri)
    return sdata


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════
def train_eegnet(X_tr, y_tr, n_ch, n_t, cfg, device, use_amp=False):
    model  = EEGNet(2, n_ch, n_t, cfg['F1'], cfg['D'], cfg['dropout']).to(device)
    X_t    = torch.FloatTensor(X_tr[:, np.newaxis]).to(device)
    y_t    = torch.LongTensor(y_tr).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=cfg['batch_size'], shuffle=True)
    opt    = torch.optim.Adam(model.parameters(),
                               lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    crit   = nn.CrossEntropyLoss()
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    best_loss, best_state, pat = float('inf'), None, 0
    ep_iter = tqdm(range(cfg['epochs']), desc="    train", leave=False, ncols=65) \
              if HAS_TQDM else range(cfg['epochs'])

    for ep in ep_iter:
        model.train(); total = 0
        for xb, yb in loader:
            opt.zero_grad()
            if scaler:
                with autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(opt); scaler.update()
            else:
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                opt.step()
            total += loss.item()
        sch.step()
        avg = total / len(loader)
        if HAS_TQDM: ep_iter.set_postfix(loss=f"{avg:.4f}")
        if avg < best_loss:
            best_loss, pat = avg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= cfg['patience']: break

    model.load_state_dict(best_state)
    return model.eval()


# ══════════════════════════════════════════════════════════════════════
# LOSO — Full Experiment Matrix
# ══════════════════════════════════════════════════════════════════════
def run_loso_full(subject_data, cfg, device, ds_name, use_amp=False):
    """
    Chạy đầy đủ experiment matrix:
    Training mode (raw / ea) × Test mode (5 methods)
    """
    subjects = sorted(subject_data.keys())
    rows     = []

    for i, ts in enumerate(subjects):
        t0 = time.time()
        print(f"\n  Fold {i+1}/{len(subjects)} — Test: {ts}")
        info = subject_data[ts]
        c3, c4, li, ri = (info['c3_idx'], info['c4_idx'],
                           info['left_idx'], info['right_idx'])

        # ── Source data ──────────────────────────────────────────────
        src_subjs = {s: subject_data[s] for s in subjects if s != ts}

        # Raw source
        src_X_raw = np.concatenate([v['X_model'] for v in src_subjs.values()])
        src_y     = np.concatenate([v['y']       for v in src_subjs.values()])

        # EA-aligned source
        src_subjs_ea, mean_R_inv = ea_align_subjects(src_subjs, cfg)
        src_X_ea = np.concatenate([v['X_model'] for v in src_subjs_ea.values()])

        n_ch, n_t = src_X_raw.shape[1], src_X_raw.shape[2]

        # ── Train 2 backbones ────────────────────────────────────────
        print(f"    [1/2] Train EEGNet (raw)...", end=' ', flush=True)
        model_raw = train_eegnet(src_X_raw, src_y, n_ch, n_t, cfg, device, use_amp)
        print("done")

        print(f"    [2/2] Train EEGNet (EA)...", end=' ', flush=True)
        model_ea  = train_eegnet(src_X_ea, src_y, n_ch, n_t, cfg, device, use_amp)
        print("done")

        orig_raw = copy.deepcopy(model_raw.state_dict())
        orig_ea  = copy.deepcopy(model_ea.state_dict())

        Xf   = info['X_full']    # (n_trials, n_ch, 1500)
        Xm   = info['X_model']   # (n_trials, n_ch, 1000)
        yt   = info['y']

        # ── Pre-compute test alignment ───────────────────────────────
        # (dùng chung cho tất cả methods)
        dummy = ERDAlignAdapter(model_raw, cfg, c3, c4, li, ri,
                                src_X_raw, mean_R_inv, 'raw')
        dummy.fit_test(Xf, Xm)
        has_erd  = dummy.has_erd
        erd_val  = dummy.erd_val
        W_erd    = dummy.W_erd
        R_inv_t  = dummy.R_inv_test
        grp      = "ERD" if has_erd else "ERS"
        print(f"    ERD screening: {grp} (ERD%={erd_val:.1f}%)")

        # ── EA-aligned test data ─────────────────────────────────────
        R_test    = compute_ref_cov(Xm, cfg['ea_reg'])
        _, R_inv_test = ea_whiten(Xm[:1], R_test)
        Xm_ea     = np.array([R_inv_test @ x for x in Xm])

        def run_adapter(AdapterClass, model, orig, Xm_input,
                        train_mode, test_mode, label,
                        extra_kwargs=None):
            """Helper: clone model, create adapter, run prediction."""
            model.load_state_dict(orig)
            kwargs = dict(
                model=model, cfg=cfg, c3=c3, c4=c4, li=li, ri=ri,
                src_X_model=src_X_raw, src_mean_R_inv=mean_R_inv,
                train_mode=train_mode)
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            adapter = AdapterClass(**kwargs)
            adapter.has_erd  = has_erd
            adapter.erd_val  = erd_val
            adapter.W_erd    = W_erd
            adapter.R_inv_test = R_inv_test

            preds  = []
            erds_t = []
            it = (tqdm(range(len(Xm_input)),
                       desc=f"    {label[:20]}", leave=False, ncols=60)
                  if HAS_TQDM else range(len(Xm_input)))
            for j in it:
                p, ev = adapter.predict(Xf[j], Xm_input[j], device)
                preds.append(p); erds_t.append(ev)
            return np.array(preds)

        # ── Run all experiments ──────────────────────────────────────
        print(f"    {'Label':<35} {'Acc':>8} {'κ':>8}")
        print(f"    {'-'*53}")

        for train_mode, test_mode, label in EXPERIMENTS:
            # Select model and input
            if train_mode == 'raw':
                model_use = model_raw; orig_use = orig_raw
                Xm_use    = Xm
            else:  # 'ea'
                model_use = model_ea; orig_use = orig_ea
                # EA-train: test input should also be EA-aligned
                # (either Std-EA or ERD-Align, depending on test_mode)
                Xm_use    = Xm  # raw — adapter handles alignment

            # Select adapter class
            cls_map = {
                'no_adapt'    : NoAdaptAdapter,
                'tent'        : TENTAdapter,
                'std_ea'      : StdEAAdapter,
                'erd_align'   : ERDAlignAdapter,
                'erd_align_ek': ERDAlignEKAdapter,
            }
            AdaptCls = cls_map[test_mode]
            preds    = run_adapter(AdaptCls, model_use, orig_use,
                                   Xm_use, train_mode, test_mode, label)

            acc   = accuracy_score(yt, preds)
            kappa = cohen_kappa_score(yt, preds)
            rows.append(dict(
                dataset=ds_name, subject=ts,
                train_mode=train_mode, test_mode=test_mode,
                label=label, accuracy=acc, kappa=kappa,
                erd_group=grp, erd_value=erd_val))
            print(f"    {label:<35} {acc:>8.4f} {kappa:>8.4f}")

        elapsed = time.time() - t0
        print(f"    Fold time: {elapsed:.0f}s")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════
def print_summary(df_all):
    print("\n" + "="*72)
    print("SUMMARY TABLE — EA + ERD-Align (mean ± std, n=9 subjects)")
    print("="*72)

    # Baseline = raw no-adapt
    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df  = df_all[df_all['dataset'] == ds]
        na  = df[df['label']=='No-adapt (baseline)']['accuracy'].mean()
        print(f"\n  {ds}  (baseline No-adapt = {na:.4f})")
        print(f"  {'Method':<36} {'Acc':>8} {'±std':>7} {'vs NA':>8} {'p-val':>8}")
        print(f"  {'-'*70}")

        subs = sorted(df['subject'].unique())
        na_v = [df[(df['subject']==s)&
                   (df['label']=='No-adapt (baseline)')]
                ['accuracy'].values[0] for s in subs]

        for _, _, label in EXPERIMENTS:
            sub  = df[df['label']==label]['accuracy']
            if len(sub) == 0: continue
            dlt  = sub.mean() - na
            mk   = ' ↑' if dlt > 0.003 else (' ↓' if dlt < -0.003 else '  ')
            vals = [df[(df['subject']==s)&(df['label']==label)]['accuracy'].values[0]
                    for s in subs]
            try:
                _, p = wilcoxon(vals, na_v)
                p_str = f"{p:.3f}"
            except:
                p_str = "  —  "
            star = "★" if label.endswith('★') else " "
            print(f"  {star}{label:<35} {sub.mean():>8.4f}"
                  f" {sub.std():>7.4f} {dlt:>+8.4f} {p_str:>8}{mk}")


def print_key_comparison(df_all):
    """Bảng so sánh quan trọng nhất: 4 methods trên ERD subjects."""
    print("\n" + "="*72)
    print("KEY COMPARISON — EA-train methods vs Raw-train No-adapt")
    print("(Đây là bảng chính cho paper)")
    print("="*72)

    key_labels = [
        'No-adapt (baseline)',
        'TENT',
        'EA-train + No-adapt',
        'EA-train + Std-EA',
        'EA-train + ERD-Align ★',
    ]

    for ds in ['BNCI2014_004', 'BNCI2014_001']:
        df   = df_all[df_all['dataset'] == ds]
        subs = sorted(df['subject'].unique())
        na_v = [df[(df['subject']==s)&
                   (df['label']=='No-adapt (baseline)')]
                ['accuracy'].values[0] for s in subs]
        na   = np.mean(na_v)

        print(f"\n  {ds}")
        print(f"  {'Method':<30} {'All':>8} {'ERD':>8} {'ERS':>8} {'p(vs NA)':>10}")
        print(f"  {'-'*62}")

        for label in key_labels:
            sub = df[df['label']==label]
            if len(sub) == 0: continue
            acc_all = sub['accuracy'].mean()

            erd_sub = sub[sub['erd_group']=='ERD']['accuracy']
            ers_sub = sub[sub['erd_group']=='ERS']['accuracy']
            acc_erd = erd_sub.mean() if len(erd_sub) > 0 else float('nan')
            acc_ers = ers_sub.mean() if len(ers_sub) > 0 else float('nan')

            vals = [sub[sub['subject']==s]['accuracy'].values[0] for s in subs]
            try:
                _, p = wilcoxon(vals, na_v)
                p_str = f"{p:.3f}" + ('*' if p < 0.05 else ' ')
            except:
                p_str = "  —   "

            diff = acc_all - na
            mk   = '↑' if diff > 0.003 else ('↓' if diff < -0.003 else '~')
            print(f"  {label:<30} {acc_all:>8.4f}"
                  f" {acc_erd:>8.4f} {acc_ers:>8.4f} {p_str:>10} {mk}")


def plot_results(df_all, output_dir):
    key_labels = [
        'No-adapt (baseline)',
        'TENT',
        'EA-train + No-adapt',
        'EA-train + Std-EA',
        'EA-train + ERD-Align ★',
        'EA-train + ERD-Align+EK ★',
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, ds in zip(axes, ['BNCI2014_004', 'BNCI2014_001']):
        df   = df_all[df_all['dataset'] == ds]
        ms   = [df[df['label']==lb]['accuracy'].mean() for lb in key_labels]
        ss   = [df[df['label']==lb]['accuracy'].std()  for lb in key_labels]
        cs   = [PALETTE[lb] for lb in key_labels]

        short = ['No-adapt', 'TENT', 'EA+NoAdapt', 'EA+StdEA',
                 'EA+ERDAlign★', 'EA+ERDAlign+EK★']

        bars = ax.bar(short, ms, yerr=ss, color=cs, alpha=0.87,
                      capsize=5, width=0.6, error_kw={'lw': 1.5})
        ax.axhline(ms[0], color='#888780', lw=1, ls='--', alpha=0.5,
                   label=f'No-adapt={ms[0]:.3f}')
        ax.set_ylim(max(0, min(ms)-0.10), min(1, max(ms)+0.12))
        ax.set_title(f'{ds}', fontsize=12, fontweight='500')
        ax.set_ylabel('Accuracy')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(short, rotation=25, ha='right', fontsize=9)
        ax.legend(fontsize=8)
        for bar, m in zip(bars, ms):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.004,
                    f'{m:.3f}', ha='center', fontsize=8)

    plt.suptitle('EA + ERD-Align: Full Experiment Results', fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, 'ea_erd_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['001','004','both'], default='both')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output', default=CONFIG['output_dir'])
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    CONFIG['output_dir'] = args.output
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    ds_map = {
        '004' : ['BNCI2014_004'],
        '001' : ['BNCI2014_001'],
        'both': ['BNCI2014_004', 'BNCI2014_001'],
    }

    all_dfs = []
    for ds_name in ds_map[args.dataset]:
        print(f"\n{'='*60}\n  DATASET: {ds_name}\n{'='*60}")
        sdata = load_dataset(ds_name, CONFIG)
        df    = run_loso_full(sdata, CONFIG, device, ds_name, args.amp)
        path  = os.path.join(CONFIG['output_dir'], f'{ds_name}_ea_erd.csv')
        df.to_csv(path, index=False)
        print(f"\n  Saved → {path}")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(CONFIG['output_dir'], 'all_ea_erd.csv'), index=False)

    print_summary(df_all)
    print_key_comparison(df_all)

    if len(all_dfs) == 2:
        plot_results(df_all, CONFIG['output_dir'])

    print(f"\n✓ Done. Results → {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
