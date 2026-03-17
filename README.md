# AI-Driven Predictive Maintenance with Real-Time Contextual Data Fusion for Connected Vehicles

[![arXiv](https://img.shields.io/badge/arXiv-2603.13343-b31b1b.svg)](https://arxiv.org/abs/2603.13343)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](vehiclepm/LICENSE)

Official code, data, and results for:

> **Khemani, K. & Qureshi, A.N. (2026).** *AI-Driven Predictive Maintenance with Real-Time Contextual Data Fusion for Connected Vehicles: A Multi-Dataset Evaluation.* arXiv:2603.13343

---

## Overview

This repository provides:
- **`vehiclepm/`** — Python library for V2X-augmented vehicle predictive maintenance with live OBD-II support
- **`experiments/`** — Reproducible scripts for all 7 experiments in the paper
- **`results/`** — Pre-computed results, plots, and CSVs matching the paper
- **`data/`** — AI4I 2020 benchmark dataset and OBD-II driving datasets

### Key Results

| Experiment | Key Finding |
|---|---|
| Exp 1 — Ablation | V2X features contribute −0.026 F1 when removed; internal-only = −0.049 |
| Exp 2 — Classification | LightGBM F1=0.837, AUC=0.949 on synthetic contextual dataset |
| Exp 3 — AI4I 2020 | LightGBM F1=0.814, AUC=0.973 on real industrial failure data |
| Exp 4 — Noise | F1 > 0.88 for σ≤0.5; degrades to 0.74 at σ=2.0 |
| Exp 5 — SHAP | 4 contextual/interaction features in top 9 predictors |
| Exp 6 — Calibration | Platt scaling reduces Brier score: 0.082 → 0.080 |
| Exp 7 — Regression | LightGBM MAE=2.33 days, R²=0.9949 (simulation only) |

---

## Installation

```bash
git clone https://github.com/Kushalk0677/AI-Driven-Predictive-Maintenance-with-Real-Time-Contextual-Data-Fusion-for-Connected-Vehicles
cd AI-Driven-Predictive-Maintenance-with-Real-Time-Contextual-Data-Fusion-for-Connected-Vehicles

# Install the library
pip install -e vehiclepm/

# Install all dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
from vehiclepm.features import build_feature_matrix

# Generate synthetic data
df = generate_synthetic_dataset(n_samples=2000)
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]

# Train and evaluate
clf = VehiclePMClassifier(model_type="lightgbm")
results = clf.cross_validate(X, y)
print(f"F1: {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
print(f"AUC: {results['auc_mean']:.3f} ± {results['auc_std']:.3f}")
```

---

## 🔌 Live OBD-II Prediction

Plug any ELM327 Bluetooth dongle into your car and get real-time predictions:

```python
from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, LivePredictor, VehicleContext
from vehiclepm.features import build_feature_matrix

# Train model
df = generate_synthetic_dataset()
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
clf = VehiclePMClassifier()
clf.fit(X, df["maintenance_needed"])

# Set your vehicle details
ctx = VehicleContext(
    mileage=95000, vehicle_age=11,
    brake_thickness=5.5, tire_tread=4.2,
    oil_degradation=0.35, road_type="Urban",
    weather_condition="Clear",
)

# Connect dongle and run
predictor = LivePredictor(model=clf, context=ctx, port="COM6")  # Windows
predictor.run()
```

Output:
```
✅ [OK]       Risk: 18% | Engine: 89.2°C | Battery: 87% | DTCs: 0
⚠️  [WARNING]  Risk: 61% | Engine: 108.7°C | Battery: 79% | DTCs: 1
🚨 [CRITICAL] Risk: 84% | Engine: 115.2°C | Battery: 71% | DTCs: 2
```

> **Note:** Currently trained on synthetic data. For accurate real-world predictions, collect labelled data from your vehicle and retrain using `clf.fit()`.

---

## Reproducing Paper Results

```bash
# Run all 7 experiments
python run_all_experiments.py

# Or run individually
python experiments/exp1_ablation_study.py
python experiments/exp2_classification_benchmark.py
python experiments/exp3_ai4i_benchmark.py      # requires data/raw/ai4i2020.csv
python experiments/exp4_noise_sensitivity.py
python experiments/exp5_shap_importance.py
python experiments/exp6_calibration.py
python experiments/exp7_regression.py
```

Pre-computed results are in `results/` and match the paper exactly.

---

## Repository Structure

```
├── vehiclepm/                  # Python library (pip install -e vehiclepm/)
│   └── vehiclepm/
│       ├── features/           # Feature engineering (Groups A–D)
│       ├── models/             # VehiclePMClassifier
│       ├── evaluation/         # Ablation study, noise sensitivity
│       ├── interpretability/   # SHAP analysis
│       ├── calibration/        # Probability calibration
│       ├── data/               # Synthetic dataset generator
│       └── obd/                # Live OBD-II reader & predictor
│
├── experiments/                # Reproducible experiment scripts
│   ├── exp1_ablation_study.py
│   ├── exp2_classification_benchmark.py
│   ├── exp3_ai4i_benchmark.py
│   ├── exp4_noise_sensitivity.py
│   ├── exp5_shap_importance.py
│   ├── exp6_calibration.py
│   └── exp7_regression.py
│
├── results/                    # Pre-computed plots and CSVs
├── data/
│   └── raw/
│       ├── ai4i2020.csv                    # AI4I 2020 benchmark
│       ├── exp2_19drivers_1car_1route.csv  # Real OBD-II driving data
│       └── exp3_4drivers_1car_1route.csv   # Real OBD-II driving data
│
├── run_all_experiments.py
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Description | Used in |
|---|---|---|
| Physics-informed synthetic | 2000 vehicle-month observations, probabilistic labels | Exp 1, 2, 4, 5, 6, 7 |
| [AI4I 2020](https://archive.ics.uci.edu/dataset/601/) | 10,000 industrial milling machine failures, 5 failure modes | Exp 3 |
| OBD-II driving (19 drivers) | Real OBD-II signals from instrumented vehicle | Reference |
| OBD-II driving (4 drivers) | Real OBD-II signals from instrumented vehicle | Reference |

---

## Citation

```bibtex
@article{khemani2026vehiclepm,
  title   = {AI-Driven Predictive Maintenance with Real-Time Contextual Data
             Fusion for Connected Vehicles: A Multi-Dataset Evaluation},
  author  = {Khemani, Kushal and Qureshi, Anjum Nazir},
  journal = {arXiv preprint arXiv:2603.13343},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.13343}
}
```

---

## License

MIT License — see [vehiclepm/LICENSE](vehiclepm/LICENSE)

## Author

**Kushal Khemani** — [kushal.khemani@gmail.com](mailto:kushal.khemani@gmail.com)

**Dr. Anjum Nazir Qureshi** — Rajiv Gandhi College of Engineering Research and Technology, Chandrapur, India
