"""
PhysioNetMI — EA + ERD-Align Experiment
========================================
Dataset thứ 3 để validate TENT failure finding.

PhysioNetMI specs:
  - 109 subjects, 64 kênh (10-20 system), 160Hz → resample 250Hz
  - Classes: left_hand, right_hand (lọc từ 4 classes gốc)
  - Download: ~3GB vào ~/mne_data/MNE-eegbci-data/

Cách chạy:
    python PhysioNetMI_Experiment.py                  # 20 subjects đầu
    python PhysioNetMI_Experiment.py --n_subjects 30  # 30 subjects
    python PhysioNetMI_Experiment.py --amp            # mixed precision
    python PhysioNetMI_Experiment.py --analyze_only   # chỉ plot (đã có CSV)

So sánh với kết quả cũ:
    python PhysioNetMI_Experiment.py --combine_path ./results/ea_erd/all_ea_erd.csv
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
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
moabb.set_log_level('warning')


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    'sfreq'         : 250,
    'tmin'          : -2.0,
    'tmax'          :  4.0,
    'baseline_start':    0,   # t = -2s
    'baseline_end'  :  500,   # t =  0s
    'task_start'    :  625,   # t =  0.5s
    'task_end'      : 1375,   # t =  3.5s
    'model_start'   :  500,

    # EEGNet
    'F1'      :  8,
    'D'       :  2,
    'dropout' : 0.5,

    'lr'          : 1e-3,
    'epochs'      :  150,
    'batch_size'  :   32,
    'patience'    :   25,
    'grad_clip'   :  5.0,
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

    # EA
    'ea_reg'   : 1e-5,

    'output_dir'  : './results/physionet',
    'seed'        :   42,
    'n_subjects'  :   20,   # subjects dùng cho LOSO (từ 109)
}

# PhysioNetMI 64-channel layout (chuẩn 10-20)
# C3 = index 7, C4 = index 11 trong layout chuẩn MOABB
# Sẽ được detect tự động khi load
PHYSIONET_C3_NAME = 'C3'
PHYSIONET_C4_NAME = 'C4'

METHODS_ORDER = [
    'No-adapt (baseline)',
    'TENT',
    'ERD-Align (raw train)',
    'EA-train + No-adapt',
    'EA-train + Std-EA',
    'EA-train + ERD-Align ★',
    'EA-train + ERD-Align+EK ★',
    'EA-train + TENT',
]

PALETTE = {
    'No-adapt (baseline)'        : '#888780',
    'TENT'                       : '#E24B4A',
    'ERD-Align (raw train)'      : '#5DCAA5',
    'EA-train + No-adapt'        : '#B0AEA8',
    'EA-train + Std-EA'          : '#BA7517',
    'EA-train + ERD-Align ★'     : '#534AB7',
    'EA-train + ERD-Align+EK ★'  : '#0F6E56',
    'EA-train + TENT'            : '#F4857F',
}


# ══════════════════════════════════════════════════════════════════════
# EEGNet
# ══════════════════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    def __init__(self, n_classes, n_channels, n_times,
                 F1=8, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.b3 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8),
                      groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.fc = nn.Linear(self._flat(n_channels, n_times), n_classes)

    def _flat(self, nc, nt):
        with torch.no_grad():
            return self.b3(self.b2(self.b1(
                torch.zeros(1, 1, nc, nt)))).numel()

    def forward(self, x):
        return self.fc(self.b3(self.b2(self.b1(x))).flatten(1))

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
# Signal processing utilities
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


def mean_cov(Xf, reg=1e-5):
    return np.mean(
        [(x @ x.T) / x.shape[-1] + np.eye(x.shape[0]) * reg
         for x in Xf], axis=0)


def compute_ref_cov(X, reg=1e-5):
    return np.mean(
        [x @ x.T / x.shape[-1] for x in X], axis=0) + np.eye(X.shape[1]) * reg


def ea_whiten(X, R):
    R_inv_sqrt = np.real(inv(np.real(sqrtm(R))))
    return np.array([R_inv_sqrt @ x for x in X]), R_inv_sqrt


def ea_align_subjects(subject_data, cfg):
    aligned, all_R_inv = {}, []
    for s, info in subject_data.items():
        X  = info['X_model']
        R  = compute_ref_cov(X, cfg['ea_reg'])
        Xe, Ri = ea_whiten(X, R)
        aligned[s] = {**info, 'X_model': Xe, 'R_inv_sqrt': Ri, 'R': R}
        all_R_inv.append(Ri)
    return aligned, np.mean(all_R_inv, axis=0)


# ══════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════
def entropy_loss(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def erd_loss(logits, erd, li, ri, threshold):
    p  = F.softmax(logits, dim=-1)
    c4 = torch.tensor(erd['C4_comp'], dtype=torch.float32,
                       device=logits.device)
    c3 = torch.tensor(erd['C3_comp'], dtype=torch.float32,
                       device=logits.device)
    return (p[0, li] * F.relu(c4 - threshold) +
            p[0, ri] * F.relu(c3 - threshold))


# ══════════════════════════════════════════════════════════════════════
# Test adapters (same as EA_ERD_Align.py)
# ══════════════════════════════════════════════════════════════════════
class BaseAdapter:
    def __init__(self, model, cfg, c3, c4, li, ri,
                 src_X_model=None, src_mean_R_inv=None):
        self.model, self.cfg    = model, cfg
        self.c3, self.c4        = c3, c4
        self.li, self.ri        = li, ri
        self.src_X_model        = src_X_model
        self.mean_R_inv         = src_mean_R_inv
        self.W_erd = self.R_inv_test = None
        self.has_erd = self.erd_val = None

    def fit_test(self, X_full, X_model):
        erds = [compute_erd(X_full[i], self.c3, self.c4, self.cfg)
                for i in range(len(X_full))]
        c3m = np.mean([e['C3_comp'] for e in erds])
        c4m = np.mean([e['C4_comp'] for e in erds])
        self.erd_val = (c3m + c4m) / 2
        self.has_erd = min(c3m, c4m) < self.cfg['erd_threshold']

        R_test = compute_ref_cov(X_model, self.cfg['ea_reg'])
        _, self.R_inv_test = ea_whiten(X_model[:1], R_test)

        if self.has_erd and self.src_X_model is not None:
            lo, hi  = self.cfg['mu_low'], self.cfg['beta_high']
            Xf_src  = bandpass_batch(self.src_X_model, lo, hi)
            Xf_test = bandpass_batch(X_model, lo, hi)
            sig_src = mean_cov(Xf_src)
            sig_tst = mean_cov(Xf_test)
            self.W_erd = np.real(
                np.real(sqrtm(sig_src)) @
                np.real(inv(np.real(sqrtm(sig_tst)))))

    def _t(self, x_np, device):
        return torch.FloatTensor(x_np[np.newaxis, np.newaxis]).to(device)


class NoAdaptAdapter(BaseAdapter):
    def predict(self, xf, xm, device):
        self.model.eval()
        with torch.no_grad():
            return self.model(self._t(xm, device)).argmax(-1).item(), self.erd_val


class TENTAdapter(BaseAdapter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model.set_norm_train()
        self.opt = torch.optim.Adam(
            self.model.get_norm_params(), lr=self.cfg['tta_lr'])

    def predict(self, xf, xm, device):
        self.model.set_norm_train()
        x_t = self._t(xm, device)
        for _ in range(self.cfg['tta_steps']):
            loss = entropy_loss(self.model(x_t))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(x_t).argmax(-1).item(), self.erd_val


class StdEAAdapter(BaseAdapter):
    def predict(self, xf, xm, device):
        xa = self.R_inv_test @ xm if self.R_inv_test is not None else xm
        self.model.eval()
        with torch.no_grad():
            return self.model(self._t(xa, device)).argmax(-1).item(), self.erd_val


class ERDAlignAdapter(BaseAdapter):
    def predict(self, xf, xm, device):
        xa = (self.W_erd @ xm
              if (self.W_erd is not None and self.has_erd)
              else xm)
        self.model.eval()
        with torch.no_grad():
            return self.model(self._t(xa, device)).argmax(-1).item(), self.erd_val


class ERDAlignEKAdapter(BaseAdapter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model.set_norm_train()
        self.opt = torch.optim.Adam(
            self.model.get_norm_params(), lr=self.cfg['tta_lr'])

    def predict(self, xf, xm, device):
        erd = compute_erd(xf, self.c3, self.c4, self.cfg)
        xa  = (self.W_erd @ xm
               if (self.W_erd is not None and self.has_erd)
               else xm)
        self.model.set_norm_train()
        x_t = self._t(xa, device)
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
# Data loading — PhysioNetMI
# ══════════════════════════════════════════════════════════════════════
def load_physionet(cfg, n_subjects=20):
    """
    Load PhysioNetMI qua MOABB.

    Lưu ý quan trọng:
    - 109 subjects, 64 kênh, 160Hz → resample 250Hz
    - Chỉ dùng n_subjects đầu tiên để LOSO feasible
    - C3, C4 ở vị trí khác so với BNCI (64-ch layout)
    - Download ~3GB vào ~/mne_data/MNE-eegbci-data/
    """
    print(f"\n{'='*55}")
    print(f"  Loading PhysioNetMI (n_subjects={n_subjects})")
    print(f"  Download ~3GB vào ~/mne_data/ — lần đầu chậm")
    print(f"{'='*55}")

    ds       = PhysionetMI()
    paradigm = MotorImagery(
        events   =['left_hand', 'right_hand'],
        n_classes=2,
        fmin=0.5, fmax=45.0,
        tmin=cfg['tmin'], tmax=cfg['tmax'],
        resample=cfg['sfreq'])

    # Load chỉ n_subjects subjects để giới hạn thời gian
    X, y, meta = paradigm.get_data(
        ds, subjects=list(range(1, n_subjects + 1)))

    print(f"  X shape : {X.shape}")
    print(f"  Duration: {X.shape[2]/cfg['sfreq']:.1f}s per trial")
    print(f"  Subjects: {sorted(meta['subject'].unique())[:5]}...")

    # Encode labels
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    cls    = list(le.classes_)
    li, ri = cls.index('left_hand'), cls.index('right_hand')

    # Tìm C3, C4 channel indices
    # MOABB trả về channel names qua info — dùng cách an toàn hơn
    try:
        ch_names = paradigm.get_data(
            ds, subjects=[1])[2].index.tolist()
    except:
        pass

    # PhysioNetMI 64-ch standard: tìm C3, C4 trong list
    # Dùng MNE để lấy channel names
    import mne
    raw_sample = mne.io.read_raw_edf(
        ds.data_path(subject=1)[0], preload=False, verbose=False)
    ch_names_all = [ch.upper().replace('EEG ', '').replace('.', '')
                    for ch in raw_sample.ch_names]

    # PhysioNetMI dùng notation EEG C3. → C3.
    # Tìm C3 và C4
    c3_idx, c4_idx = None, None
    for i, ch in enumerate(ch_names_all):
        ch_clean = ch.strip('.')
        if ch_clean in ['C3', 'EEG C3']:
            c3_idx = i
        if ch_clean in ['C4', 'EEG C4']:
            c4_idx = i

    # Fallback: MOABB đã filter channels, dùng paradigm channel list
    if c3_idx is None or c4_idx is None:
        # Sau khi MOABB process, X shape = (trials, n_ch, times)
        # Số kênh có thể đã bị reduce. Dùng index từ paradigm info
        from moabb.datasets import PhysionetMI as PMI
        pmi = PMI()
        raw = mne.io.read_raw_edf(
            pmi.data_path(subject=1)[0],
            preload=False, verbose=False)
        ch_names_mne = raw.ch_names
        # Standard PhysioNetMI channel order (64ch)
        # C3 = channel index sau filter
        # MOABB thường giữ nguyên thứ tự, lấy theo tên
        ch_lower = [c.lower().replace(' ', '').replace('.', '')
                    for c in ch_names_mne]
        try:
            c3_idx = ch_lower.index('eegc3')
        except ValueError:
            try: c3_idx = ch_lower.index('c3')
            except ValueError: c3_idx = 7  # fallback default

        try:
            c4_idx = ch_lower.index('eegc4')
        except ValueError:
            try: c4_idx = ch_lower.index('c4')
            except ValueError: c4_idx = 11  # fallback default

    # Sau MOABB preprocessing, X có thể có số kênh khác
    # Đảm bảo indices không vượt quá
    n_ch_actual = X.shape[1]
    c3_idx = min(c3_idx, n_ch_actual - 1)
    c4_idx = min(c4_idx, n_ch_actual - 1)
    print(f"  C3 idx={c3_idx}, C4 idx={c4_idx}")
    print(f"  left_hand={li}, right_hand={ri}")
    print(f"  n_channels={n_ch_actual}")

    # Build subject_data dict
    ms     = cfg['model_start']
    sdata  = {}
    for s in sorted(meta['subject'].unique())[:n_subjects]:
        mask = meta['subject'] == s
        Xs   = X[mask]
        sdata[s] = dict(
            X_full=Xs, X_model=Xs[:, :, ms:], y=y_enc[mask],
            c3_idx=c3_idx, c4_idx=c4_idx,
            left_idx=li, right_idx=ri)
        print(f"  Subject {s}: {Xs.shape[0]} trials")

    return sdata


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════
def train_eegnet(X_tr, y_tr, n_ch, n_t, cfg, device, use_amp=False):
    model  = EEGNet(2, n_ch, n_t,
                    cfg['F1'], cfg['D'], cfg['dropout']).to(device)
    X_t    = torch.FloatTensor(X_tr[:, np.newaxis]).to(device)
    y_t    = torch.LongTensor(y_tr).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=cfg['batch_size'], shuffle=True)
    opt    = torch.optim.Adam(model.parameters(),
                               lr=cfg['lr'],
                               weight_decay=cfg['weight_decay'])
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    crit   = nn.CrossEntropyLoss()
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    best_loss, best_state, pat = float('inf'), None, 0
    pbar = (tqdm(range(cfg['epochs']), desc="    train",
                 leave=False, ncols=65)
            if HAS_TQDM else range(cfg['epochs']))

    for _ in pbar:
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
            pbar.set_postfix(loss=f"{avg:.4f}")
        if avg < best_loss:
            best_loss, pat = avg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            pat += 1
            if pat >= cfg['patience']:
                break

    model.load_state_dict(best_state)
    return model.eval()


# ══════════════════════════════════════════════════════════════════════
# LOSO Evaluation
# ══════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    ('raw', 'no_adapt',     'No-adapt (baseline)'),
    ('raw', 'tent',         'TENT'),
    ('raw', 'erd_align',    'ERD-Align (raw train)'),
    ('ea',  'no_adapt',     'EA-train + No-adapt'),
    ('ea',  'std_ea',       'EA-train + Std-EA'),
    ('ea',  'erd_align',    'EA-train + ERD-Align ★'),
    ('ea',  'erd_align_ek', 'EA-train + ERD-Align+EK ★'),
    ('ea',  'tent',         'EA-train + TENT'),
]

ADAPTER_MAP = {
    'no_adapt'    : NoAdaptAdapter,
    'tent'        : TENTAdapter,
    'std_ea'      : StdEAAdapter,
    'erd_align'   : ERDAlignAdapter,
    'erd_align_ek': ERDAlignEKAdapter,
}


def run_loso(subject_data, cfg, device, use_amp=False):
    subjects = sorted(subject_data.keys())
    rows     = []
    t0_total = time.time()

    for i, ts in enumerate(subjects):
        t0 = time.time()
        print(f"\n  Fold {i+1}/{len(subjects)} — Test subject: {ts}")

        info   = subject_data[ts]
        c3, c4 = info['c3_idx'], info['c4_idx']
        li, ri = info['left_idx'], info['right_idx']

        src_subjs = {s: subject_data[s] for s in subjects if s != ts}

        src_X_raw = np.concatenate(
            [v['X_model'] for v in src_subjs.values()])
        src_y = np.concatenate(
            [v['y'] for v in src_subjs.values()])

        src_subjs_ea, mean_R_inv = ea_align_subjects(src_subjs, cfg)
        src_X_ea = np.concatenate(
            [v['X_model'] for v in src_subjs_ea.values()])

        n_ch, n_t = src_X_raw.shape[1], src_X_raw.shape[2]
        print(f"    n_ch={n_ch}, n_t={n_t}, "
              f"train_trials={src_X_raw.shape[0]}")

        print(f"    [1/2] Train EEGNet (raw)...", end=' ', flush=True)
        model_raw = train_eegnet(src_X_raw, src_y,
                                  n_ch, n_t, cfg, device, use_amp)
        print("done")

        print(f"    [2/2] Train EEGNet (EA)...", end=' ', flush=True)
        model_ea  = train_eegnet(src_X_ea, src_y,
                                  n_ch, n_t, cfg, device, use_amp)
        print("done")

        orig_raw = copy.deepcopy(model_raw.state_dict())
        orig_ea  = copy.deepcopy(model_ea.state_dict())

        Xf = info['X_full']
        Xm = info['X_model']
        yt = info['y']

        # Pre-compute alignment
        dummy = ERDAlignAdapter(model_raw, cfg, c3, c4, li, ri,
                                src_X_raw, mean_R_inv)
        dummy.fit_test(Xf, Xm)
        has_erd   = dummy.has_erd
        erd_val   = dummy.erd_val
        W_erd     = dummy.W_erd
        R_inv_t   = dummy.R_inv_test
        grp       = "ERD" if has_erd else "ERS"
        print(f"    ERD: {grp} ({erd_val:.1f}%)")

        print(f"    {'Label':<36} {'Acc':>8} {'κ':>8}")
        print(f"    {'-'*55}")

        for train_mode, test_mode, label in EXPERIMENTS:
            model_use = model_raw if train_mode == 'raw' else model_ea
            orig_use  = orig_raw  if train_mode == 'raw' else orig_ea

            model_use.load_state_dict(orig_use)
            adapter = ADAPTER_MAP[test_mode](
                model_use, cfg, c3, c4, li, ri,
                src_X_raw, mean_R_inv)
            adapter.has_erd    = has_erd
            adapter.erd_val    = erd_val
            adapter.W_erd      = W_erd
            adapter.R_inv_test = R_inv_t

            preds = []
            it = (tqdm(range(len(Xm)),
                       desc=f"    {label[:22]}",
                       leave=False, ncols=60)
                  if HAS_TQDM else range(len(Xm)))
            for j in it:
                p, _ = adapter.predict(Xf[j], Xm[j], device)
                preds.append(p)
            preds = np.array(preds)

            acc   = accuracy_score(yt, preds)
            kappa = cohen_kappa_score(yt, preds)
            rows.append(dict(
                dataset='PhysioNetMI', subject=ts,
                train_mode=train_mode, test_mode=test_mode,
                label=label, accuracy=acc, kappa=kappa,
                erd_group=grp, erd_value=erd_val))
            print(f"    {label:<36} {acc:>8.4f} {kappa:>8.4f}")

        elapsed = time.time() - t0
        rem     = elapsed * (len(subjects) - i - 1)
        print(f"    Fold: {elapsed:.0f}s | Remaining: {rem/60:.0f}min")

    total = time.time() - t0_total
    print(f"\n  Total: {total/60:.1f} min")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════
def print_summary(df, n_subjects):
    print(f"\n{'='*70}")
    print(f"SUMMARY — PhysioNetMI (n={n_subjects} subjects, LOSO)")
    print(f"{'='*70}")

    subs = sorted(df['subject'].unique())
    na   = df[df['label']=='No-adapt (baseline)']['accuracy'].mean()
    na_v = [df[(df['subject']==s)&
               (df['label']=='No-adapt (baseline)')]['accuracy'].values[0]
            for s in subs]

    print(f"\n  {'Method':<36} {'Acc':>8} {'±std':>7} {'κ':>7}"
          f" {'vs NA':>8} {'p':>7} {'d':>6}")
    print(f"  {'-'*72}")

    for label in METHODS_ORDER:
        sub  = df[df['label']==label]['accuracy']
        kap  = df[df['label']==label]['kappa']
        if len(sub) == 0: continue
        dlt  = sub.mean() - na
        vals = [df[(df['subject']==s)&(df['label']==label)]['accuracy'].values[0]
                for s in subs]
        try:
            delta = np.array(vals) - np.array(na_v)
            _, p  = wilcoxon(vals, na_v)
            cd    = delta.mean() / (delta.std() + 1e-10)
            p_str = f"{p:.3f}*" if p < 0.05 else f"{p:.3f} "
        except:
            p_str, cd = "  —   ", 0
        mk   = ' ↑' if dlt > 0.003 else (' ↓' if dlt < -0.003 else '  ')
        star = '★' if '★' in label else ' '
        print(f"  {star}{label:<35} {sub.mean():>8.4f} {sub.std():>7.4f}"
              f" {kap.mean():>7.4f} {dlt:>+8.4f} {p_str:>7} {cd:>+6.2f}{mk}")


def print_tent_finding(df):
    """Tóm tắt finding quan trọng nhất: TENT failure."""
    print(f"\n{'='*70}")
    print("KEY FINDING — TENT failure trên PhysioNetMI")
    print(f"{'='*70}")
    subs  = sorted(df['subject'].unique())
    n     = len(subs)
    na_v  = [df[(df['subject']==s)&
                (df['label']=='No-adapt (baseline)')]['accuracy'].values[0]
             for s in subs]
    tent_v= [df[(df['subject']==s)&
                (df['label']=='TENT')]['accuracy'].values[0]
             for s in subs]
    delta = np.array(tent_v) - np.array(na_v)
    n_hurt= sum(1 for d in delta if d < -0.01)
    try:
        _, p = wilcoxon(tent_v, na_v)
        cd   = delta.mean() / (delta.std() + 1e-10)
        print(f"\n  TENT vs No-adapt:")
        print(f"    Mean Δ = {delta.mean():+.4f}")
        print(f"    Subjects hurt (Δ < -1%): {n_hurt}/{n}")
        print(f"    Wilcoxon: p = {p:.4f}, Cohen's d = {cd:.2f}")
        print(f"    Significant: {'YES ✓' if p < 0.05 else 'NO ✗'}")
    except Exception as e:
        print(f"  Statistical test: {e}")

    print(f"\n  TENT Δ per subject:")
    for s, d in zip(subs, delta):
        grp = df[df['subject']==s]['erd_group'].values[0]
        tag = " ← HURT" if d < -0.01 else ""
        print(f"    S{s:02d} [{grp}]: {d:+.4f}{tag}")


def combined_analysis(df_physio, combine_path):
    """So sánh TENT failure qua cả 3 datasets."""
    print(f"\n{'='*70}")
    print("COMBINED — TENT failure trên 3 datasets")
    print(f"{'='*70}")

    try:
        df_prev = pd.read_csv(combine_path)
        all_ds  = [('BNCI2014_004', df_prev),
                   ('BNCI2014_001', df_prev),
                   ('PhysioNetMI',  df_physio)]

        print(f"\n  {'Dataset':<16} {'NA':>8} {'TENT':>8} {'Δ':>8}"
              f" {'n_hurt':>8} {'p':>8} {'d':>7}")
        print(f"  {'-'*65}")

        for ds_name, df_src in all_ds:
            df_ds = df_src[df_src['dataset']==ds_name] \
                    if 'dataset' in df_src.columns else df_src
            subs  = sorted(df_ds['subject'].unique())
            if len(subs) == 0:
                continue

            na_v = [df_ds[(df_ds['subject']==s)&
                          (df_ds['label']=='No-adapt (baseline)')]['accuracy'].values[0]
                    for s in subs]
            tent_v = [df_ds[(df_ds['subject']==s)&
                            (df_ds['label']=='TENT')]['accuracy'].values[0]
                      for s in subs]
            delta = np.array(tent_v) - np.array(na_v)
            n_hurt = sum(1 for d in delta if d < -0.01)

            try:
                _, p = wilcoxon(tent_v, na_v)
                cd   = delta.mean() / (delta.std() + 1e-10)
                sig  = '*' if p < 0.05 else ' '
                print(f"  {ds_name:<16}"
                      f" {np.mean(na_v):>8.4f}"
                      f" {np.mean(tent_v):>8.4f}"
                      f" {delta.mean():>+8.4f}"
                      f" {n_hurt:>4}/{len(subs):<3}"
                      f" {p:>8.4f}{sig}"
                      f" {cd:>7.2f}")
            except Exception as e:
                print(f"  {ds_name}: {e}")

    except FileNotFoundError:
        print(f"  Không tìm thấy file combine: {combine_path}")
        print(f"  Chạy PhysioNetMI standalone.")


def plot_summary(df, output_dir, n_subjects):
    KEY = [
        'No-adapt (baseline)', 'TENT',
        'ERD-Align (raw train)',
        'EA-train + No-adapt',
        'EA-train + Std-EA',
        'EA-train + ERD-Align ★',
    ]
    ms  = [df[df['label']==lb]['accuracy'].mean() for lb in KEY]
    ss  = [df[df['label']==lb]['accuracy'].std()  for lb in KEY]
    cs  = [PALETTE[lb] for lb in KEY]
    lbs = ['No-adapt','TENT','ERD-Align','EA+NA','EA+StdEA','EA+ERD★']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = ax1.bar(lbs, ms, yerr=ss, color=cs, alpha=0.87,
                   capsize=5, width=0.6, error_kw={'lw': 1.5})
    ax1.axhline(ms[0], color='#888780', lw=1, ls='--', alpha=0.5)
    ax1.set_ylim(max(0, min(ms) - 0.10), min(1, max(ms) + 0.12))
    ax1.set_title(f'PhysioNetMI — n={n_subjects} subjects', fontsize=12)
    ax1.set_ylabel('Accuracy')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(lbs, rotation=20, ha='right', fontsize=9)
    for bar, m in zip(bars, ms):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.004,
                 f'{m:.3f}', ha='center', fontsize=8)

    # TENT delta per subject
    subs   = sorted(df['subject'].unique())
    na_v   = [df[(df['subject']==s)&
                 (df['label']=='No-adapt (baseline)')]['accuracy'].values[0]
              for s in subs]
    tent_v = [df[(df['subject']==s)&
                 (df['label']=='TENT')]['accuracy'].values[0]
              for s in subs]
    delta  = np.array(tent_v) - np.array(na_v)
    grps   = [df[df['subject']==s]['erd_group'].values[0] for s in subs]
    colors = ['#E24B4A' if d < -0.01 else '#5DCAA5' for d in delta]

    ax2.bar([f'S{s}' for s in subs], delta,
            color=colors, alpha=0.85, width=0.6)
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.axhline(-0.10, color='#E24B4A', lw=0.8, ls='--',
                alpha=0.5, label='−10% threshold')
    ax2.set_title('TENT Δ accuracy per subject\n'
                  '(red = hurt by TENT)', fontsize=11)
    ax2.set_ylabel('TENT − No-adapt')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels([f'S{s}\n[{g}]' for s, g in zip(subs, grps)],
                        rotation=30, ha='right', fontsize=8)
    ax2.legend(fontsize=8)

    plt.suptitle('PhysioNetMI — EA + ERD-Align Results', fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, 'physionet_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_subjects', type=int, default=CONFIG['n_subjects'],
                   help='Number of subjects for LOSO (default 20)')
    p.add_argument('--amp', action='store_true',
                   help='Mixed precision training')
    p.add_argument('--output', default=CONFIG['output_dir'])
    p.add_argument('--analyze_only', action='store_true',
                   help='Only plot from existing CSV')
    p.add_argument('--combine_path', default='./results/ea_erd/all_ea_erd.csv',
                   help='Path to previous EA_ERD results for combined analysis')
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

    csv_path = os.path.join(CONFIG['output_dir'], 'physionet_results.csv')

    if args.analyze_only:
        print("Analyze only mode — loading existing CSV")
        df = pd.read_csv(csv_path)
    else:
        sdata = load_physionet(CONFIG, args.n_subjects)
        df    = run_loso(sdata, CONFIG, device, args.amp)
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved → {csv_path}")

    # Analysis
    print_summary(df, args.n_subjects)
    print_tent_finding(df)
    combined_analysis(df, args.combine_path)
    plot_summary(df, CONFIG['output_dir'], args.n_subjects)

    print(f"\n✓ Done → {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
