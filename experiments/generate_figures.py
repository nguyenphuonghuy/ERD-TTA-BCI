"""
Generate Figure 3 and Figure 4 for paper
==========================================
Fig. 3: TENT Δ accuracy per subject across 3 datasets (bar chart)
Fig. 4: ERD% vs Δ accuracy scatter plot (EA+Std-EA vs No-adapt)

Input files:
    ./results/ea_erd/all_ea_erd.csv        (BNCI2014_004 + BNCI2014_001)
    ./results/physionet/physionet_results.csv  (PhysioNetMI)

Output:
    ./results/figures/fig3_tent_delta.png
    ./results/figures/fig4_erd_scatter.png

Cách chạy:
    python generate_figures.py
    python generate_figures.py --dpi 300     # high-res cho paper
    python generate_figures.py --format pdf  # vector format
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Style chuẩn IEEE/BSPC ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'Times New Roman',
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.labelsize'   : 10,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── Colors ────────────────────────────────────────────────────────────────────
C_HURT    = '#E24B4A'   # đỏ — TENT làm hại
C_HELP    = '#0F6E56'   # xanh lá — TENT có ích
C_NEUTRAL = '#B0AEA8'   # xám — gần 0
C_ERD     = '#534AB7'   # tím — ERD subject
C_ERS     = '#E24B4A'   # đỏ — ERS subject
C_REF     = '#888780'   # xám — no-adapt reference

HURT_THRESHOLD  = -0.01   # Δ < -1%: TENT làm hại
HELP_THRESHOLD  =  0.01   # Δ > +1%: TENT có ích


def load_data(ea_erd_path, physionet_path):
    """Load và chuẩn bị data từ hai file CSV."""
    dfs = []

    if os.path.exists(ea_erd_path):
        df1 = pd.read_csv(ea_erd_path)
        dfs.append(df1)
        print(f"Loaded: {ea_erd_path} ({len(df1)} rows)")
    else:
        print(f"WARNING: không tìm thấy {ea_erd_path}")

    if os.path.exists(physionet_path):
        df2 = pd.read_csv(physionet_path)
        dfs.append(df2)
        print(f"Loaded: {physionet_path} ({len(df2)} rows)")
    else:
        print(f"WARNING: không tìm thấy {physionet_path}")

    if not dfs:
        raise FileNotFoundError("Không tìm thấy file CSV nào!")

    return pd.concat(dfs, ignore_index=True)


def get_delta(df, dataset, method_test, baseline='No-adapt (baseline)'):
    """
    Tính Δ accuracy = method - baseline cho từng subject.
    Returns: dict {subject: delta}
    """
    d = df[df['dataset'] == dataset]
    subs = sorted(d['subject'].unique())
    result = {}
    for s in subs:
        m_acc = d[(d['subject']==s) & (d['label']==method_test)]['accuracy']
        b_acc = d[(d['subject']==s) & (d['label']==baseline)]['accuracy']
        if len(m_acc) > 0 and len(b_acc) > 0:
            result[s] = float(m_acc.values[0]) - float(b_acc.values[0])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — TENT Δ per subject
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig3(df, output_path, fmt='png', dpi=300):
    """
    Bar chart: TENT Δ accuracy per subject trên 3 datasets.
    Bars đỏ = TENT làm hại (Δ < −1%)
    Bars xanh = TENT có ích (Δ > +1%)
    Bars xám = không đáng kể
    """
    datasets = ['BNCI2014_004', 'BNCI2014_001', 'PhysioNetMI']
    ds_labels = ['BNCI2014-004\n(3 channels, n=9)',
                 'BNCI2014-001\n(22 channels, n=9)',
                 'PhysioNetMI\n(64 channels, n=20)']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              gridspec_kw={'width_ratios': [9, 9, 20]})

    for ax, ds, ds_label in zip(axes, datasets, ds_labels):
        delta_dict = get_delta(df, ds, 'TENT')
        if not delta_dict:
            ax.set_title(f'{ds_label}\n(no data)')
            ax.axis('off')
            continue

        subs   = sorted(delta_dict.keys())
        deltas = [delta_dict[s] for s in subs]
        labels = [f'S{s}' for s in subs]

        # Màu sắc theo giá trị
        colors = []
        for d in deltas:
            if d < HURT_THRESHOLD:
                colors.append(C_HURT)
            elif d > HELP_THRESHOLD:
                colors.append(C_HELP)
            else:
                colors.append(C_NEUTRAL)

        x = np.arange(len(subs))
        bars = ax.bar(x, deltas, color=colors, alpha=0.88, width=0.65,
                      edgecolor='white', linewidth=0.5)

        # Zero line
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

        # Annotation: số liệu trên/dưới bar
        for bar, val in zip(bars, deltas):
            va    = 'bottom' if val >= 0 else 'top'
            ypos  = val + (0.004 if val >= 0 else -0.004)
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:+.3f}', ha='center', va=va,
                    fontsize=7, color='#333333')

        # Hurt/help counts
        n_hurt = sum(1 for d in deltas if d < HURT_THRESHOLD)
        n_help = sum(1 for d in deltas if d > HELP_THRESHOLD)
        ax.text(0.98, 0.98,
                f'Hurt: {n_hurt}/{len(subs)}\nHelp: {n_help}/{len(subs)}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='#444444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ds_label, fontsize=10, fontweight='500', pad=6)
        ax.set_ylabel('Δ Accuracy (TENT − No-adapt)' if ax == axes[0] else '',
                      fontsize=9)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

        # Y-axis range: symmetric around 0, với chút padding
        max_abs = max(abs(d) for d in deltas) * 1.3
        ax.set_ylim(-max_abs, max_abs)

    # Legend chung
    legend_elements = [
        mpatches.Patch(facecolor=C_HURT,    label='Δ < −1% (TENT hurts)'),
        mpatches.Patch(facecolor=C_NEUTRAL, label='|Δ| ≤ 1% (negligible)'),
        mpatches.Patch(facecolor=C_HELP,    label='Δ > +1% (TENT helps)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Fig. 3. TENT Δ accuracy per subject across three datasets',
                 fontsize=11, fontweight='normal', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                format=fmt.replace('.', ''))
    plt.close()
    print(f"Fig. 3 saved → {output_path}")

    # Print summary stats
    print("\n  Summary TENT Δ:")
    for ds in datasets:
        delta_dict = get_delta(df, ds, 'TENT')
        if delta_dict:
            vals = list(delta_dict.values())
            print(f"  {ds}: mean={np.mean(vals):+.4f}, "
                  f"hurt={sum(1 for v in vals if v<HURT_THRESHOLD)}/{len(vals)}, "
                  f"help={sum(1 for v in vals if v>HELP_THRESHOLD)}/{len(vals)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Scatter: ERD% vs Δaccuracy
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig4(df, output_path, fmt='png', dpi=300):
    """
    Scatter plot: ERD% composite vs Δ accuracy (EA+Std-EA − No-adapt).
    Điểm màu tím = ERD subject, màu đỏ cam = ERS subject.
    Hai panel: BNCI2014_004 và BNCI2014_001.
    """
    # PhysioNetMI bỏ qua vì EA+Std-EA có vấn đề ở 64ch (×49 overhead)
    datasets   = ['BNCI2014_004', 'BNCI2014_001']
    ds_labels  = ['BNCI2014-004 (3 channels, n=9)',
                  'BNCI2014-001 (22 channels, n=9)']

    ERD_THRESH = -10.0   # ngưỡng ERD screening (%)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, ds, ds_label in zip(axes, datasets, ds_labels):
        d    = df[df['dataset'] == ds]
        subs = sorted(d['subject'].unique())

        # Lấy ERD value (từ cột erd_value hoặc erd_C3_comp + erd_C4_comp)
        erd_vals, delta_vals, groups = [], [], []

        for s in subs:
            sub_df = d[d['subject'] == s]

            # Lấy Δ accuracy
            ea_std = sub_df[sub_df['label'] == 'EA-train + Std-EA']['accuracy']
            na_acc = sub_df[sub_df['label'] == 'No-adapt (baseline)']['accuracy']

            if len(ea_std) == 0 or len(na_acc) == 0:
                continue

            delta = float(ea_std.values[0]) - float(na_acc.values[0])

            # Lấy ERD value
            erd_row = sub_df[sub_df['label'] == 'EA-train + Std-EA']
            if 'erd_value' in erd_row.columns and len(erd_row) > 0:
                erd = float(erd_row['erd_value'].values[0])
            elif ('erd_C3_comp' in erd_row.columns and
                  'erd_C4_comp' in erd_row.columns):
                erd = (float(erd_row['erd_C3_comp'].values[0]) +
                       float(erd_row['erd_C4_comp'].values[0])) / 2
            else:
                continue

            grp = 'ERD' if erd < ERD_THRESH else 'ERS'
            erd_vals.append(erd)
            delta_vals.append(delta)
            groups.append(grp)
            sub_list = subs

        if not erd_vals:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        erd_arr   = np.array(erd_vals)
        delta_arr = np.array(delta_vals)

        # Plot points
        for erd, delta, grp, s in zip(erd_vals, delta_vals, groups, subs):
            color = C_ERD if grp == 'ERD' else C_ERS
            ax.scatter(erd, delta, c=color, s=80, zorder=5,
                       edgecolors='white', linewidths=0.5)
            ax.annotate(f'S{s}', (erd, delta),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=8, color='#444444')

        # Trend line nếu đủ điểm
        if len(erd_vals) >= 3:
            try:
                z     = np.polyfit(erd_arr, delta_arr, 1)
                x_fit = np.linspace(erd_arr.min()-5, erd_arr.max()+5, 100)
                ax.plot(x_fit, np.poly1d(z)(x_fit),
                        '--', color='#888780', linewidth=1, alpha=0.6,
                        label='Linear trend')
            except Exception:
                pass

        # Reference lines
        ax.axhline(0,         color='black', linewidth=0.8, alpha=0.5)
        ax.axvline(ERD_THRESH, color='#534AB7', linewidth=1,
                   linestyle='--', alpha=0.6,
                   label=f'θ_ERD = {ERD_THRESH}%')

        # Pearson correlation
        if len(erd_vals) >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(erd_arr, delta_arr)
            p_str = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
            ax.text(0.04, 0.97, f'r = {r:.2f}\n{p_str}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#cccccc', alpha=0.8))

        ax.set_xlabel('ERD composite (%) — (C3 + C4) / 2', fontsize=9)
        ax.set_ylabel('Δ Accuracy (EA + Std-EA − No-adapt)',
                      fontsize=9 if ax == axes[0] else 0.1)
        ax.set_title(ds_label, fontsize=10, fontweight='500', pad=6)
        ax.grid(alpha=0.25, linewidth=0.5)

        # Shade regions
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.axvspan(xlim[0], ERD_THRESH, alpha=0.04, color=C_ERD)
        ax.axvspan(ERD_THRESH, xlim[1], alpha=0.04, color=C_ERS)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=C_ERD, label='ERD subject (ERD% < −10%)'),
        mpatches.Patch(facecolor=C_ERS, label='ERS subject (ERD% ≥ −10%)'),
        Line2D([0],[0], color='#534AB7', linestyle='--',
               label='ERD threshold θ = −10%'),
        Line2D([0],[0], color='#888780', linestyle='--',
               label='Linear trend'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.1))

    fig.suptitle('Fig. 4. ERD composite (%) vs Δ accuracy of EA + Std-EA per subject',
                 fontsize=11, fontweight='normal', y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                format=fmt.replace('.', ''))
    plt.close()
    print(f"Fig. 4 saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='Generate Fig. 3 and Fig. 4')
    p.add_argument('--ea_erd',    default='./results/ea_erd/all_ea_erd.csv')
    p.add_argument('--physionet', default='./results/physionet/physionet_results.csv')
    p.add_argument('--outdir',    default='./results/figures')
    p.add_argument('--dpi',       type=int, default=300)
    p.add_argument('--format',    default='png', choices=['png','pdf','svg'])
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.ea_erd, args.physionet)
    print(f"Total rows: {len(df)}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Labels: {df['label'].unique()[:5]}...")

    ext = f".{args.format}"

    print("\n── Generating Fig. 3 ──")
    plot_fig3(df,
              os.path.join(args.outdir, f'fig3_tent_delta{ext}'),
              fmt=args.format, dpi=args.dpi)

    print("\n── Generating Fig. 4 ──")
    plot_fig4(df,
              os.path.join(args.outdir, f'fig4_erd_scatter{ext}'),
              fmt=args.format, dpi=args.dpi)

    print(f"\n✓ Done. Figures saved in: {args.outdir}/")


if __name__ == '__main__':
    main()
