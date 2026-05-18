ERD-TTA-BCI
ERD-Guided Test-Time Adaptation for Cross-Subject Motor Imagery EEG
Paper: ERD-Guided Test-Time Adaptation for Cross-Subject Motor Imagery EEG: Reliability Analysis and Edge Deployment Guidelines
Journal: 
---
Overview
This repository provides code and data for a systematic evaluation of Test-Time Adaptation (TTA) methods for Motor Imagery (MI) EEG decoding, along with an ERD-guided adaptation framework suitable for edge deployment.
Key findings:
TENT — the most widely used TTA method — exhibits unpredictable behavior in cross-subject MI-EEG: catastrophic failure on BNCI2014-004 (−15.3%, p=0.004) but beneficial on PhysioNetMI (+2.3%, p=0.025)
EA + Std-EA is the only method that consistently improves accuracy across all three datasets (+1.6%, +0.9%, +3.0%) with minimal latency overhead (×1.1 vs no-adapt)
ERD Screening adds less than 0.5 ms overhead regardless of electrode count — O(1) w.r.t. channels
---
Datasets
All datasets are downloaded automatically via MOABB:
Dataset	Subjects	Channels	Sampling Rate	Classes
BNCI2014-004 (BCI-IV 2b)	9	3 (C3/Cz/C4)	250 Hz	LH / RH
BNCI2014-001 (BCI-IV 2a)	9	22	250 Hz	LH / RH
PhysioNetMI	20 (of 109)	64	160→250 Hz	LH / RH
Data is stored locally in `~/mne_data/` on first run (~3 GB total). No manual download required.
---
Repository Structure
```
ERD-TTA-BCI/
├── experiments/
│   ├── run_main.py           # Main experiment: BNCI2014-004 + BNCI2014-001
│   ├── run_physionet.py      # PhysioNetMI validation
│   ├── benchmark_timing.py   # Inference time benchmark (CPU)
│   └── generate_figures.py   # Reproduce Fig. 3 and Fig. 4
│
├── notebooks/
│   ├── EK_TTA_Experiment.ipynb       # Google Colab — baseline experiment
│   ├── ERD_Align_Addon.ipynb         # Google Colab — ERD-Align add-on
│   └── EEGConformer_ERD_Align.ipynb  # Google Colab — EEGConformer experiment
│
├── results/
│   ├── all_ea_erd.csv        # Main results (BNCI2014-004 + BNCI2014-001)
│   ├── physionet_results.csv # PhysioNetMI results
│   └── timing_report.csv     # Inference time benchmark
│
├── figures/
│   ├── fig3_tent_delta.png   # Fig. 3: TENT Δ per subject
│   └── fig4_erd_scatter.png  # Fig. 4: ERD% vs Δ accuracy scatter
│
├── requirements.txt
└── README.md
```
---
Quick Start
Installation
```bash
git clone https://github.com/[your-username]/ERD-TTA-BCI.git
cd ERD-TTA-BCI
pip install -r requirements.txt
```
Reproduce main results (Table 4)
```bash
# Run on BNCI2014-004 and BNCI2014-001 (~2h on GPU)
python experiments/run_main.py --amp

# Run on BNCI2014-004 only (~45 min)
python experiments/run_main.py --dataset 004 --amp

# Run on PhysioNetMI (20 subjects, ~2h)
python experiments/run_physionet.py --n_subjects 20 --amp
```
Results are saved to `./results/`.
Reproduce timing benchmark (Table 6)
```bash
# All 3 channel configurations
python experiments/benchmark_timing.py --all --n_trials 500

# Output: ./results/timing/timing_report.csv
```
Reproduce figures
```bash
python experiments/generate_figures.py \
    --ea_erd    ./results/all_ea_erd.csv \
    --physionet ./results/physionet_results.csv \
    --outdir    ./figures \
    --dpi 300

# Output: figures/fig3_tent_delta.png, figures/fig4_erd_scatter.png
```
Google Colab (no local GPU required)
Open `notebooks/EK_TTA_Experiment.ipynb` in Google Colab
Select Runtime → T4 GPU
Run all cells (data downloads automatically)
---
Methods Compared
Method	Training	Test-time	Description
No-adapt	Raw	None	Baseline
TENT	Raw	Entropy min.	Wang et al. (2021)
ERD-Align	Raw	ERD-guided align	Proposed
EA + No-adapt	EA	None	Ablation
EA + Std-EA	EA	Std-EA	Proposed (best)
EA + ERD-Align	EA	ERD-guided align	Proposed
EA + TENT	EA	Entropy min.	Ablation
---
System Requirements
Python ≥ 3.9
PyTorch ≥ 2.0 (GPU recommended, CPU supported)
~3 GB disk space for datasets
~4 GB RAM minimum
Tested on:
Ubuntu 22.04, Python 3.10, PyTorch 2.1, CUDA 11.8
Google Colab (T4 GPU)
---
Pre-computed Results
To skip the experiments and directly reproduce figures and analysis:
```bash
# Figures are already generated from pre-computed CSVs in results/
python experiments/generate_figures.py
```
---
Acknowledgements
Datasets accessed via MOABB [Jayaram & Barachant, 2018]
EEGNet implementation adapted from [Lawhern et al., 2018]
TENT implementation adapted from [Wang et al., 2021]
---
Citation
If you use this code, please cite:
```bibtex
@article{[author]2025erd,
  title   = {ERD-Guided Test-Time Adaptation for Cross-Subject Motor Imagery EEG:
             Reliability Analysis and Edge Deployment Guidelines},
  author  = {[Author names]},
  journal = {Biomedical Signal Processing and Control},
  year    = {2025},
  note    = {Under review}
}
```
---
License
This project is licensed under the MIT License. See LICENSE for details.
