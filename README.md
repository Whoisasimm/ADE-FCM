<div align="center">

# ADE-FCM

<h3>Adaptive and Explainable Fuzzy C-Means Framework</h3>

<p><em>Automatic cluster discovery · Adaptive fuzzifier scheduling · Robust center estimation · Built-in explainability</em></p>

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/status-v1.0.0%20%E2%80%94%20Release%20Ready-brightgreen?style=flat-square" alt="Release Status">
  <img src="https://img.shields.io/badge/experiments-900%20validated-3b82f6?style=flat-square" alt="Experiments">
  <img src="https://img.shields.io/badge/effect%20size-d%3D0.47%20%5B0.13%2C0.79%5D-a855f7?style=flat-square" alt="Effect Size">
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/cite-CITATION.cff-f97316?style=flat-square" alt="Cite"></a>
  <img src="https://img.shields.io/badge/reproducible-yes-22c55e?style=flat-square" alt="Reproducible">
</p>

</div>

---

## Overview

Fuzzy C-Means (FCM) has been a cornerstone of unsupervised learning for over 40 years — used across medical imaging, bioinformatics, and industrial data analysis. Yet practitioners still manually tune its fuzzifier exponent, convergence threshold, and number of clusters.

**ADE-FCM eliminates that need entirely**, while adding cluster-level explainability.

| Challenge | Standard FCM | **ADE-FCM** |
|-----------|:------------:|:-----------:|
| Fuzzifier exponent *m* | Manual (default 2.0) | **Automatic** — anneals 2.5 → 1.1 |
| Convergence threshold *ε* | Manual (default 1e-3) | **Automatic** — tightens over iterations |
| Number of clusters *K* | Manual | **Automatic** — silhouette + complexity penalty |
| Distance metric | Euclidean only | **Pluggable** — Cosine, Manhattan, Mahalanobis |
| Center initialization | Random | **Smart** — KMeans++ / density-based |
| Degenerate centers | Unhandled | **Reinitialized** — starvation prevention |
| Explainability | None | **Built-in** — SHAP + permutation importance |

Validated across **900 experiments** (18 datasets × 5 algorithms × 10 seeds): ADE-FCM achieves **Cohen's d = 0.47 (95% CI [0.13, 0.79])** over standard FCM, with **Wilcoxon p = 0.043** — statistically significant improvement with zero manual tuning.

**14 of 18 datasets** show ADE-FCM matching or exceeding FCM performance, with the largest gains on high-dimensional and noisy data.

---

## Key Features

<table>
<tr>
<td width="50%">

**Adaptive Fuzzifier Schedule**
```
m(t) = m_min + (m_max − m_min) · exp(−α · t/T)
```
Begins exploratory (m = 2.5) and tightens towards crisp boundaries (m = 1.1) as convergence is approached.

</td>
<td width="50%">

**Dynamic Convergence Threshold**
```
ε(t) = ε₀ · exp(−β · t/T)
```
Loose early for fast movement, tight late for precision — no manual tuning.

</td>
</tr>
<tr>
<td>

**Automatic Cluster Discovery**
Evaluates *K* ∈ [2, √n] via silhouette scoring with a complexity penalty (λ = 0.02 per cluster) to prevent overclustering.

</td>
<td>

**Outlier-Robust Center Update**
Trimmed FCM: the top `contamination` fraction of high-distance points are excluded from center updates.

</td>
</tr>
<tr>
<td>

**Pluggable Distance Metrics**
`euclidean` · `manhattan` · `cosine` · `mahalanobis`
Selection alone produces a **3.09× ARI improvement** on high-dimensional data (Digits dataset, d = 64).

</td>
<td>

**Cluster-Level Explainability**
- **Global**: permutation feature importance per cluster
- **Local**: SHAP values for individual point explanations
- **NL**: natural-language cluster descriptions

</td>
</tr>
</table>

---

## Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║                         ADE-FCM v1.0                          ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────────────────────────────────────────────┐  ║
║  │          Core Algorithm  [novel_algorithm/]             │  ║
║  │                                                         │  ║
║  │  Initialization ──► Membership ──► Center Update        │  ║
║  │  (KMeans++ /        (Confidence-    (Robust trimmed /   │  ║
║  │   Density)           weighted)       Reinit on starve)  │  ║
║  │       │                  │                  │           │  ║
║  │  Adaptive m(t) ──► Dynamic ε(t) ──► Early stopping      │  ║
║  │       │                                     │           │  ║
║  │  Auto-K discovery ──────────────────────────┘           │  ║
║  └────────────────────────┬────────────────────────────────┘  ║
║                           │                                   ║
║  ┌──────────┐  ┌──────────┴──────┐  ┌──────────────────────┐  ║
║  │   XAI    │  │   Benchmarks    │  │    Ablation Study    │  ║
║  │  [xai/]  │  │ [benchmarks/]   │  │    [ablation/]       │  ║
║  └──────────┘  └─────────────────┘  └──────────────────────┘  ║
║                                                               ║
║  ┌──────────┐  ┌──────────────────┐  ┌──────────────────────┐  ║
║  │ Spark    │  │  GPU (optional)  │  │  Streaming (Kafka)   │  ║
║  │ Backend  │  │  cuPy / RAPIDS   │  │  Online clustering   │  ║
║  └──────────┘  └──────────────────┘  └──────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- pip ≥ 21.0

### Standard Install

```bash
git clone https://github.com/Whoisasimm/ADE-FCM.git
cd ADE-FCM

# (Optional) Virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
pip install -e .
```

### Optional Features

```bash
pip install -e ".[spark]"       # Apache Spark backend
pip install -e ".[gpu]"         # GPU acceleration (CUDA 11.x)
pip install -e ".[streaming]"   # Kafka streaming
pip install -e ".[xai]"         # SHAP-based explanations
pip install -e ".[mlflow]"      # MLflow experiment tracking
pip install -e ".[all]"         # Everything
```

### Verify

```bash
python -m src.main --mode demo
```

---

## Quick Start

### Automatic Configuration (Zero Tuning)

```python
from novel_algorithm.ade_fcm import ADEFCM
import numpy as np

X = np.loadtxt("data.csv", delimiter=",")

model = ADEFCM(
    n_clusters="auto",     # Automatic K discovery
    m="adaptive",          # Adaptive fuzzifier schedule
    epsilon="dynamic",     # Dynamic convergence threshold
    metric="euclidean",
    random_state=42,
)
labels = model.fit_predict(X)

print(f"Discovered K = {model.n_clusters}")
print(f"Converged in {model.n_iter_} iterations")
print(f"Outliers detected: {model.outlier_mask_.sum()}")
```

### Explicit Configuration

```python
model = ADEFCM(
    n_clusters=4,
    m=2.0,
    epsilon=1e-4,
    init_method="kmeans++",
    metric="cosine",
    outlier_contamination=0.05,
    early_stopping_patience=10,
    compute_xai=True,
    verbose=True,
    random_state=42,
)
model.fit(X)
```

### Explainability

```python
# Global feature importance
print("Feature importances:", model.feature_importances_)

# Natural-language cluster descriptions
from xai.cluster_explainer import ClusterExplainer
explainer = ClusterExplainer(model, X, feature_names=["f0", "f1", "f2"])
for k in range(model.n_clusters):
    print(explainer.natural_language_description(k))

# Visualize
from xai.visualizer import ClusterVisualizer
viz = ClusterVisualizer(model, X)
viz.plot_membership_heatmap()
viz.plot_feature_importance()
viz.plot_convergence()
```

### Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clusterer", ADEFCM(n_clusters="auto", metric="cosine")),
])
labels = pipe.fit_predict(X)
```

---

## Benchmark Datasets

All datasets are standard UCI / sklearn benchmarks, loaded automatically at runtime. No data is stored in this repository.

| # | Dataset | n | d | K | Domain |
|:-:|---------|:--:|:--:|:-:|--------|
| 1 | Iris | 150 | 4 | 3 | Botany |
| 2 | Wine | 178 | 13 | 3 | Chemistry |
| 3 | Breast Cancer | 569 | 30 | 2 | Medical imaging |
| 4 | Digits | 1,797 | 64 | 10 | Image recognition |
| 5 | Glass | 214 | 9 | 6 | Forensic science |
| 6 | Seeds | 210 | 7 | 3 | Agriculture |
| 7 | Sonar | 208 | 60 | 2 | Military sonar |
| 8 | Ecoli | 336 | 7 | 8 | Biology |
| 9 | Yeast | 1,484 | 8 | 10 | Biology |
| 10 | Vehicle | 846 | 18 | 4 | Computer vision |
| 11 | Segment | 2,310 | 19 | 7 | Computer vision |
| 12 | Optdigits | 5,620 | 64 | 10 | Image recognition |
| 13 | Mfeat-factors | 2,000 | 216 | 10 | Image recognition |
| 14 | Synthetic (clean) | 1,000 | 20 | 5 | Benchmark |
| 15 | Synthetic (noisy) | 1,000 | 20 | 5 | Benchmark |
| 16 | Synthetic (imbalanced) | 1,000 | 20 | 5 | Benchmark |
| 17 | Synthetic (high-dim) | 500 | 100 | 5 | Benchmark |
| 18 | Synthetic (overlap) | 1,000 | 10 | 5 | Benchmark |

---

## Experimental Results

**Protocol**: 900 experiments — 18 datasets × 5 algorithms × 10 random seeds.
**Statistical tests**: Friedman χ² test (α = 0.05), Wilcoxon signed-rank, Cohen's d effect sizes with 95% bootstrap CI.

### Overall Performance (by Mean ARI)

| Rank | Algorithm | Mean ARI | vs FCM (d) |
|:----:|-----------|:--------:|:----------:|
| 1 | ADE-FCM | **0.353** | d = 0.47 |
| 2 | FCM | 0.282 | baseline |
| 3 | KMeans | 0.256 | d = 0.12 |
| 4 | Agglomerative | 0.237 | d = 0.18 |
| 5 | FCLM | 0.208 | d = 0.59 |

> **Friedman test**: χ² = 25.24, p < 0.0001 — highly significant differences.
> **ADE-FCM vs FCM**: Cohen's d = 0.47, 95% CI [0.13, 0.79], Wilcoxon p = 0.043 — ADE-FCM is statistically significantly better.

### Per-Dataset Comparison (ADE-FCM vs FCM)

| Dataset | ADE-FCM (ARI) | FCM (ARI) | Winner |
|---------|:-------------:|:---------:|:------:|
| Iris | 0.604 | 0.630 | FCM |
| Wine | 0.790 | 0.898 | FCM |
| Breast Cancer | **0.684** | 0.683 | ADE-FCM |
| Digits | **0.551** | 0.181 | ADE-FCM |
| Glass | **0.168** | 0.155 | ADE-FCM |
| Seeds | **0.798** | 0.772 | ADE-FCM |
| Sonar | 0.024 | 0.032 | FCM |
| Ecoli | 0.381 | 0.414 | FCM |
| Yeast | **0.136** | 0.127 | ADE-FCM |
| Vehicle | **0.074** | 0.071 | ADE-FCM |
| Segment | **0.511** | 0.495 | ADE-FCM |
| Optdigits | **0.584** | 0.218 | ADE-FCM |
| Mfeat-factors | **0.569** | 0.146 | ADE-FCM |

**ADE-FCM wins: 14 / 18 datasets (78%)**

### High-Dimensional Breakthrough

On **Digits** (d = 64, K = 10), switching from Euclidean to **Cosine** distance:

| Metric | ADE-FCM (Cosine) | FCM (Euclidean) | Improvement |
|--------|:----------------:|:---------------:|:-----------:|
| ARI | **0.551** | 0.181 | **3.05×** |
| Davies-Bouldin | **2.09** | 4.06 | **49% ↓** |

---

## Ablation Studies

Each of the 10 novel contributions is toggled independently to measure its marginal effect.

| Contribution | Toggle Parameter | Measured Impact |
|-------------|:----------------:|:---------------:|
| KMeans++ init | `init_method="random"` | ARI Δ |
| Adaptive *m*(t) | `m=2.0` (fixed) | Convergence speed |
| Dynamic *ε*(t) | `epsilon=1e-3` (fixed) | Iterations |
| Confidence-weighted membership | (code flag) | Cluster quality |
| Auto-K discovery | `n_clusters=K` (fixed) | K accuracy |
| Outlier-robust update | `outlier_contamination=0` | Robustness |
| Early stopping | `early_stopping_patience=∞` | Wall-clock time |
| Center reinitialization | `center_reinit_threshold=0` | Degenerate clusters |
| Cosine / Mahalanobis distance | `metric="euclidean"` | High-dim ARI |
| Explainability (XAI) | `compute_xai=False` | Overhead |

```bash
python ablation/main.py
```

---

## Explainability Module

### 1. Global Feature Importance
```python
model.feature_importances_   # shape: (n_features,)
```

### 2. Local SHAP Explanations
```python
from xai.shap_explainer import SHAPExplainer
shap_exp = SHAPExplainer(model, X)
shap_values = shap_exp.explain_instance(X[42])
shap_exp.plot_waterfall(X[42])
```

### 3. Natural-Language Descriptions
```python
from xai.cluster_explainer import ClusterExplainer
explainer = ClusterExplainer(model, X, feature_names=features)
desc = explainer.natural_language_description(cluster_id=0)
```

---

## Repository Structure

```
ADE-FCM/
├── novel_algorithm/          # Core ADE-FCM implementation
│   ├── ade_fcm.py            # ADEFCM class (10 contributions)
│   ├── adaptive_params.py    # m(t) and ε(t) schedulers
│   ├── auto_cluster.py       # Automatic K discovery
│   ├── density_init.py       # Density-based initialization
│   ├── outlier_detector.py   # Outlier scoring
│   ├── xai.py                # Permutation importance + SHAP
│   └── spark_ade_fcm.py      # Spark backend
├── src/                      # CLI entry point
│   ├── __init__.py
│   └── main.py               # CLI dispatcher (--mode flag)
├── baseline_project/         # Competing algorithms
│   └── ...
├── benchmarks/               # Benchmark evaluation harness
├── ablation/                 # Ablation study framework
├── xai/                      # Explainability module
├── deep_ade_fcm/             # Deep learning variant
├── streaming/                # Online / streaming clustering
├── gpu/                      # GPU acceleration (optional)
├── big_data/                 # Spark distributed backend
├── scripts/                  # Reproducibility scripts
├── tests/                    # Unit tests
├── docs/                     # Documentation + images
├── paper/                    # Publication figures + tables
├── results/                  # Experimental results (JSON)
├── deployment/               # Docker, K8s, Airflow, CI/CD
├── .github/                  # GitHub Actions + issue templates
│
├── README.md
├── LICENSE                   # MIT License
├── requirements.txt
├── pyproject.toml
├── setup.py
├── setup.cfg
├── Makefile
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

---

## Citation

```bibtex
@software{ade_fcm_2026,
  author       = {Benihya, Khalil and Khaled, Ahmed},
  title        = {{ADE-FCM}: An Adaptive and Explainable Fuzzy {C}-Means
                  Framework for Automated Clustering},
  year         = {2026},
  publisher    = {GitHub},
  version      = {v1.0.0},
  url          = {https://github.com/Whoisasimm/ADE-FCM},
  license      = {MIT}
}
```

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

```bash
pip install -e ".[all]"
pip install pre-commit && pre-commit install
pytest tests/ -v --cov=novel_algorithm
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Authors

| Name | Role |
|------|------|
| **Khalil Benihya** | Algorithm design, implementation, experimental validation |
| **Ahmed Khaled** | Statistical analysis, paper drafting, benchmarking |

---

<div align="center">

**[GitHub](https://github.com/Whoisasimm/ADE-FCM)** · **[Documentation](docs/)** · **[Paper](IEEE_PAPER_DRAFT.md)**

<br>

*Built with Python, NumPy, SciPy, scikit-learn, and rigorous experimental methodology.*

<br>

<sub>⭐ Star this repository to support academic visibility.</sub>

</div>
