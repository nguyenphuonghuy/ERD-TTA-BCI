"""
Inference Time Benchmark — EA + ERD-Align Methods
==================================================
Đo inference time per trial cho từng method trên CPU.
Chạy trên Dell 7560 A2000 (CPU mode để simulate edge).

Cách chạy:
    python benchmark_timing.py                    # 3 channel config (BNCI2014_004)
    python benchmark_timing.py --n_ch 22          # 22 channel (BNCI2014_001)
    python benchmark_timing.py --n_ch 64          # 64 channel (PhysioNetMI)
    python benchmark_timing.py --all              # cả 3 configs
    python benchmark_timing.py --device cuda      # GPU timing (for reference)

Output:
    ./results/timing/timing_report.csv
    ./results/timing/timing_report.png
"""

import os
import time
import copy
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch, butter, filtfilt
from scipy.linalg import sqrtm, inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
N_TRIALS    = 500    # số trials để average (nhiều hơn = ổn định hơn)
N_REPS      = 7      # số lần lặp để lấy median (loại bỏ outlier)
N_WARMUP    = 20     # warm-up trials trước khi đo
SFREQ       = 250
N_TIMES     = 1000   # task period: [0s, 4s] @ 250Hz
N_TIMES_FULL= 1500   # full epoch: [-2s, 4s] @ 250Hz

CONFIGS = [
    {'n_ch': 3,  'label': 'BNCI2014_004 (3ch)',  'c3': 0, 'c4': 2},
    {'n_ch': 22, 'label': 'BNCI2014_001 (22ch)', 'c3': 7, 'c4': 11},
    {'n_ch': 64, 'label': 'PhysioNetMI (64ch)',  'c3': 7, 'c4': 11},
]

ERD_CFG = {
    'sfreq'         : 250,
    'baseline_start':   0,
    'baseline_end'  : 500,
    'task_start'    : 625,
    'task_end'      : 1375,
    'mu_low'        :   8,
    'mu_high'       :  12,
    'beta_low'      :  13,
    'beta_high'     :  30,
    'erd_threshold' : -10.0,
    'ea_reg'        : 1e-5,
}

METHODS = [
    'No-adapt',
    'EA + No-adapt',
    'TENT (1 step)',
    'ERD Screening only',
    'ERD-Align',
    'EA + Std-EA',
    'EA + ERD-Align',
    'EA + ERD-Align + EK (1 step)',
]

PALETTE = {
    'No-adapt'                    : '#888780',
    'EA + No-adapt'               : '#B0AEA8',
    'TENT (1 step)'               : '#E24B4A',
    'ERD Screening only'          : '#5DCAA5',
    'ERD-Align'                   : '#5DCAA5',
    'EA + Std-EA'                 : '#BA7517',
    'EA + ERD-Align'              : '#534AB7',
    'EA + ERD-Align + EK (1 step)': '#0F6E56',
}


# ══════════════════════════════════════════════════════════════════════
# EEGNet
# ══════════════════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    def __init__(self, n_classes=2, n_channels=3, n_times=1000,
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
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            flat = self.b3(self.b2(self.b1(
                torch.zeros(1, 1, n_channels, n_times)))).numel()
        self.fc = nn.Linear(flat, n_classes)
        self._flat = flat

    def forward(self, x):
        return self.fc(self.b3(self.b2(self.b1(x))).flatten(1))

    def get_bn_params(self):
        p = []
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                p += [m.weight, m.bias]
        return p

    def set_bn_train(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
        return self


# ══════════════════════════════════════════════════════════════════════
# Signal Processing (same as experiment files)
# ══════════════════════════════════════════════════════════════════════
def compute_erd_single(epoch_np, c3_idx, c4_idx, cfg):
    """ERD computation for 1 trial — core operation được đo."""
    nperseg = min(128, (cfg['baseline_end'] - cfg['baseline_start']) // 2)
    erd = {}
    for name, idx in [('C3', c3_idx), ('C4', c4_idx)]:
        base = epoch_np[idx, cfg['baseline_start']:cfg['baseline_end']]
        task = epoch_np[idx, cfg['task_start']:cfg['task_end']]
        f, pb = welch(base, fs=cfg['sfreq'], nperseg=nperseg)
        _, pt = welch(task, fs=cfg['sfreq'], nperseg=nperseg)
        mu_m   = (f >= cfg['mu_low'])   & (f <= cfg['mu_high'])
        beta_m = (f >= cfg['beta_low']) & (f <= cfg['beta_high'])
        for band, mask in [('mu', mu_m), ('beta', beta_m)]:
            pb_m = np.mean(pb[mask]) + 1e-10
            erd[f'{name}_{band}'] = (np.mean(pt[mask]) - pb_m) / pb_m * 100
        erd[f'{name}_comp'] = (erd[f'{name}_mu'] + erd[f'{name}_beta']) / 2
    return erd


def make_whitening_matrix(n_ch, seed=42):
    """Tạo ma trận whitening giả lập — đại diện R_inv_sqrt từ EA."""
    rng = np.random.default_rng(seed)
    A   = rng.standard_normal((n_ch, n_ch))
    R   = A @ A.T + np.eye(n_ch) * ERD_CFG['ea_reg']
    return np.real(inv(np.real(sqrtm(R)))).astype(np.float32)


def bandpass_single(x, lo=8, hi=30, sfreq=250, order=4):
    """Bandpass filter for 1 trial (n_ch, n_times)."""
    nyq = sfreq / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return np.array([filtfilt(b, a, x[j]) for j in range(x.shape[0])])


def make_erd_align_matrix(X_src_model, n_ch, seed=42):
    """Tạo W_erd giả lập — đại diện ERD-band alignment transform."""
    rng  = np.random.default_rng(seed)
    # Fake source covariance
    Xf   = np.array([bandpass_single(x) for x in X_src_model[:50]])
    covs = [(x @ x.T) / x.shape[-1] + np.eye(n_ch)*1e-5 for x in Xf]
    sig_src = np.mean(covs, axis=0)
    # Fake test covariance
    sig_tst = sig_src + rng.standard_normal((n_ch, n_ch)) * 0.01
    sig_tst = sig_tst @ sig_tst.T + np.eye(n_ch) * 1e-5
    W = np.real(np.real(sqrtm(sig_src)) @
                np.real(inv(np.real(sqrtm(sig_tst)))))
    return W.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Timing measurement
# ══════════════════════════════════════════════════════════════════════
def measure_one_method(method_name, model, orig_state,
                       X_model, X_full, R_inv, W_erd,
                       c3_idx, c4_idx, device,
                       n_trials, n_reps, n_warmup):
    """
    Đo inference time cho 1 method.
    Returns: dict với mean, median, std, p5, p95 (tất cả tính bằng ms/trial)
    """
    X_t_all = torch.FloatTensor(X_model[:, np.newaxis]).to(device)
    all_times = []

    for rep in range(n_reps + 1):  # rep 0 là warmup
        model.load_state_dict(orig_state)

        if method_name == 'No-adapt':
            model.eval()
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                with torch.no_grad():
                    _ = model(X_t_all[j:j+1]).argmax(-1)

        elif method_name == 'EA + No-adapt':
            model.eval()
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                xa  = R_inv @ X_model[j]
                xt  = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
                with torch.no_grad():
                    _ = model(xt).argmax(-1)

        elif method_name == 'TENT (1 step)':
            bn_params = model.get_bn_params()
            opt = torch.optim.Adam(bn_params, lr=1e-3)
            t0  = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                model.set_bn_train()
                xt = X_t_all[j:j+1]
                logits = model(xt)
                p    = F.softmax(logits, -1)
                loss = -(p * F.log_softmax(logits, -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    _ = model(xt).argmax(-1)

        elif method_name == 'ERD Screening only':
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                _ = compute_erd_single(X_full[j], c3_idx, c4_idx, ERD_CFG)

        elif method_name == 'ERD-Align':
            model.eval()
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                # ERD screen + align + infer
                _ = compute_erd_single(X_full[j], c3_idx, c4_idx, ERD_CFG)
                xa  = W_erd @ X_model[j]
                xt  = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
                with torch.no_grad():
                    _ = model(xt).argmax(-1)

        elif method_name == 'EA + Std-EA':
            model.eval()
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                # R_inv is per-subject (pre-computed offline for test subject)
                xa  = R_inv @ X_model[j]
                xt  = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
                with torch.no_grad():
                    _ = model(xt).argmax(-1)

        elif method_name == 'EA + ERD-Align':
            model.eval()
            t0 = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                _ = compute_erd_single(X_full[j], c3_idx, c4_idx, ERD_CFG)
                xa  = W_erd @ X_model[j]
                xt  = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
                with torch.no_grad():
                    _ = model(xt).argmax(-1)

        elif method_name == 'EA + ERD-Align + EK (1 step)':
            bn_params = model.get_bn_params()
            opt = torch.optim.Adam(bn_params, lr=1e-3)
            t0  = time.perf_counter()
            for j in range(n_trials if rep > 0 else n_warmup):
                erd = compute_erd_single(X_full[j], c3_idx, c4_idx, ERD_CFG)
                xa  = W_erd @ X_model[j]
                xt  = torch.FloatTensor(xa[np.newaxis, np.newaxis]).to(device)
                # EK update
                model.set_bn_train()
                logits = model(xt)
                p    = F.softmax(logits, -1)
                c4t  = torch.tensor(erd['C4_comp'], dtype=torch.float32)
                c3t  = torch.tensor(erd['C3_comp'], dtype=torch.float32)
                loss = (p[0, 0] * F.relu(c4t - ERD_CFG['erd_threshold']) +
                        p[0, 1] * F.relu(c3t - ERD_CFG['erd_threshold']))
                opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    _ = model(xt).argmax(-1)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        n_this     = n_trials if rep > 0 else n_warmup
        ms_per_trial = elapsed_ms / n_this

        if rep > 0:  # bỏ warmup rep
            all_times.append(ms_per_trial)

    all_times = np.array(all_times)
    return {
        'mean'  : float(np.mean(all_times)),
        'median': float(np.median(all_times)),
        'std'   : float(np.std(all_times)),
        'p5'    : float(np.percentile(all_times, 5)),
        'p95'   : float(np.percentile(all_times, 95)),
        'min'   : float(np.min(all_times)),
        'max'   : float(np.max(all_times)),
    }


def run_benchmark(n_ch, c3_idx, c4_idx, label, device, n_trials, n_reps, n_warmup):
    """Chạy benchmark đầy đủ cho 1 channel configuration."""
    print(f"\n{'='*62}")
    print(f"  Config: {label}")
    print(f"  n_ch={n_ch}, n_times={N_TIMES}, device={device}")
    print(f"  n_trials={n_trials}, n_reps={n_reps}, warmup={n_warmup}")
    print(f"{'='*62}")

    # Synthetic data
    rng      = np.random.default_rng(42)
    X_model  = rng.standard_normal((n_trials, n_ch, N_TIMES)).astype(np.float32)
    X_full   = rng.standard_normal((n_trials, n_ch, N_TIMES_FULL)).astype(np.float32)
    # Simulate some ERD in C3/C4 channels
    t_task   = np.arange(N_TIMES) / SFREQ
    mu_osc   = 0.3 * np.sin(2 * np.pi * 10 * t_task)  # 10Hz mu oscillation
    X_model[:, c3_idx] += mu_osc[np.newaxis, :]

    # Pre-computed alignment matrices (offline cost — not measured)
    R_inv = make_whitening_matrix(n_ch)
    W_erd = make_erd_align_matrix(X_model, n_ch)

    # Model
    model      = EEGNet(2, n_ch, N_TIMES).eval().to(device)
    orig_state = copy.deepcopy(model.state_dict())

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  EEGNet: {n_params:,} params, {n_bytes/1024:.1f} KB")

    # Benchmark each method
    rows = []
    print(f"\n  {'Method':<36} {'Median':>8} {'Mean':>8} {'Std':>7} {'p5':>7} {'p95':>7}")
    print(f"  {'-'*70}")

    na_time = None
    for method in METHODS:
        t = measure_one_method(
            method, model, orig_state,
            X_model, X_full, R_inv, W_erd,
            c3_idx, c4_idx, device,
            n_trials, n_reps, n_warmup)

        if method == 'No-adapt':
            na_time = t['median']

        overhead = t['median'] / na_time if na_time else 1.0
        print(f"  {method:<36} "
              f"{t['median']:>7.3f}ms "
              f"{t['mean']:>7.3f}ms "
              f"{t['std']:>6.3f}ms "
              f"{t['p5']:>6.3f}ms "
              f"{t['p95']:>6.3f}ms  "
              f"({overhead:.1f}x)")

        rows.append({
            'config'  : label,
            'n_ch'    : n_ch,
            'method'  : method,
            'n_params': n_params,
            **t,
            'overhead': overhead,
        })

    return pd.DataFrame(rows), n_params, n_bytes


# ══════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════
def plot_timing(df_all, output_dir, device_str):
    configs  = df_all['config'].unique()
    n_config = len(configs)

    fig, axes = plt.subplots(1, n_config, figsize=(6*n_config, 6), sharey=False)
    if n_config == 1:
        axes = [axes]

    for ax, config in zip(axes, configs):
        df  = df_all[df_all['config'] == config]
        na  = df[df['method']=='No-adapt']['median'].values[0]

        methods  = df['method'].tolist()
        medians  = df['median'].tolist()
        p5s      = df['p5'].tolist()
        p95s     = df['p95'].tolist()
        colors   = [PALETTE.get(m, '#888780') for m in methods]
        errs_lo  = [m - p for m, p in zip(medians, p5s)]
        errs_hi  = [p - m for m, p in zip(p95s, medians)]

        bars = ax.barh(methods, medians, color=colors, alpha=0.87, height=0.6)
        ax.errorbar(medians, range(len(methods)),
                    xerr=[errs_lo, errs_hi],
                    fmt='none', color='gray', capsize=4, lw=1.2)

        # No-adapt reference line
        ax.axvline(na, color='#888780', lw=1.2, ls='--', alpha=0.6,
                   label=f'No-adapt = {na:.2f}ms')

        ax.set_xlabel('Inference time (ms/trial)', fontsize=10)
        ax.set_title(f'{config}\n(CPU, n={df["n_params"].iloc[0]:,} params)',
                     fontsize=11, fontweight='500')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(fontsize=8)

        # Value labels
        for bar, m in zip(bars, medians):
            ax.text(m + 0.05,
                    bar.get_y() + bar.get_height()/2,
                    f'{m:.2f}ms', va='center', fontsize=8)

    plt.suptitle(f'Inference Time Benchmark — {device_str}',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, f'timing_benchmark_{device_str.replace(" ","_")}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot → {out}")


def print_paper_table(df_all):
    """In bảng theo format paper — dễ copy vào LaTeX."""
    print(f"\n{'='*70}")
    print("PAPER TABLE — Inference time (ms/trial, CPU, median ± std)")
    print(f"{'='*70}")

    key_methods = [
        'No-adapt',
        'TENT (1 step)',
        'ERD Screening only',
        'EA + Std-EA',
        'EA + ERD-Align',
    ]

    configs = df_all['config'].unique()
    header  = f"  {'Method':<30}" + "".join(f" {c[:12]:>14}" for c in configs)
    print(header)
    print("  " + "-"*(28 + 14*len(configs)))

    for m in key_methods:
        row = f"  {m:<30}"
        for c in configs:
            sub = df_all[(df_all['config']==c)&(df_all['method']==m)]
            if len(sub) == 0:
                row += f"{'—':>14}"
            else:
                med = sub['median'].values[0]
                std = sub['std'].values[0]
                row += f" {med:.2f}±{std:.2f}ms".rjust(14)
        print(row)

    print(f"\n  Overhead vs No-adapt:")
    na_times = {c: df_all[(df_all['config']==c)&(df_all['method']=='No-adapt')]['median'].values[0]
                for c in configs}
    for m in ['TENT (1 step)', 'EA + Std-EA', 'EA + ERD-Align']:
        row = f"  {m:<30}"
        for c in configs:
            sub = df_all[(df_all['config']==c)&(df_all['method']==m)]
            if len(sub) == 0:
                row += f"{'—':>14}"
            else:
                oh = sub['overhead'].values[0]
                row += f"{'×'+f'{oh:.1f}':>14}"
        print(row)

    # System info
    print(f"\n  Key claim for paper:")
    for c in configs:
        na  = na_times[c]
        ea  = df_all[(df_all['config']==c)&(df_all['method']=='EA + Std-EA')]['median'].values
        tnt = df_all[(df_all['config']==c)&(df_all['method']=='TENT (1 step)')]['median'].values
        if len(ea) > 0 and len(tnt) > 0:
            print(f"  {c}: EA+StdEA={ea[0]:.2f}ms (+{ea[0]-na:.2f}ms vs NA), "
                  f"TENT={tnt[0]:.2f}ms (×{tnt[0]/na:.1f} vs NA)")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='Inference time benchmark')
    p.add_argument('--n_ch', type=int, default=3,
                   help='Number of channels (default: 3)')
    p.add_argument('--all', action='store_true',
                   help='Run all 3 channel configs')
    p.add_argument('--device', default='cpu',
                   choices=['cpu', 'cuda'],
                   help='Device (default: cpu — simulates edge)')
    p.add_argument('--n_trials', type=int, default=N_TRIALS,
                   help=f'Number of trials to average (default: {N_TRIALS})')
    p.add_argument('--n_reps', type=int, default=N_REPS,
                   help=f'Number of repetitions for median (default: {N_REPS})')
    p.add_argument('--output', default='./results/timing')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Device info
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("NOTE: GPU timing for reference only — edge devices use CPU")
    else:
        import platform
        print(f"CPU: {platform.processor()}")

    print(f"PyTorch: {torch.__version__}")
    print(f"n_trials={args.n_trials}, n_reps={args.n_reps}")

    # Select configs to run
    if args.all:
        run_configs = CONFIGS
    else:
        match = [c for c in CONFIGS if c['n_ch'] == args.n_ch]
        if not match:
            print(f"No config for n_ch={args.n_ch}. Available: {[c['n_ch'] for c in CONFIGS]}")
            return
        run_configs = match

    # Run benchmarks
    all_dfs = []
    for cfg in run_configs:
        df, n_params, n_bytes = run_benchmark(
            n_ch=cfg['n_ch'], c3_idx=cfg['c3'], c4_idx=cfg['c4'],
            label=cfg['label'], device=device,
            n_trials=args.n_trials, n_reps=args.n_reps,
            n_warmup=N_WARMUP)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    csv_path = os.path.join(args.output, 'timing_report.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"\n  CSV → {csv_path}")

    # Paper table
    print_paper_table(df_all)

    # Model info summary
    print(f"\n{'='*60}")
    print("MODEL SIZE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'Params':>10} {'Size (KB)':>12}")
    print(f"  {'-'*50}")
    for cfg in run_configs:
        model = EEGNet(2, cfg['n_ch'], N_TIMES)
        np_   = sum(p.numel() for p in model.parameters())
        nb_   = sum(p.numel()*p.element_size() for p in model.parameters())
        print(f"  {cfg['label']:<25} {np_:>10,} {nb_/1024:>10.1f} KB")

    # Plot
    device_str = f"{'GPU' if device.type == 'cuda' else 'CPU'}"
    plot_timing(df_all, args.output, device_str)

    # Edge deployment claim
    print(f"\n{'='*60}")
    print("EDGE DEPLOYMENT CLAIM — for paper Discussion")
    print(f"{'='*60}")
    for cfg in run_configs:
        df = df_all[df_all['config']==cfg['label']]
        na = df[df['method']=='No-adapt']['median'].values[0]
        ea = df[df['method']=='EA + Std-EA']['median']
        ea = ea.values[0] if len(ea)>0 else None
        print(f"\n  {cfg['label']}:")
        print(f"    No-adapt baseline : {na:.2f} ms/trial")
        if ea:
            print(f"    EA + Std-EA       : {ea:.2f} ms/trial (+{ea-na:.2f}ms overhead)")
            print(f"    Real-time @ 250Hz : trial every 4000ms")
            print(f"    → Adaptation overhead: {(ea-na)/4000*100:.2f}% of trial duration")
            print(f"    → Feasible for real-time BCI: {'YES ✓' if ea < 50 else 'MARGINAL'}")

    print(f"\n✓ Done → {args.output}/")


if __name__ == '__main__':
    main()
