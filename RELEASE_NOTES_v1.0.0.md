# ADE-FCM v1.0.0 — Release Notes

## Adaptive and Explainable Fuzzy C-Means Framework

We are pleased to announce the first public release of **ADE-FCM**, an adaptive and explainable fuzzy clustering framework that extends classical Fuzzy C-Means through automatic cluster discovery, adaptive fuzzifier scheduling, robust center estimation, multiple distance metrics, and cluster-level explainability.

---

## Key Results

- **900 experiments** across 18 benchmark datasets × 5 algorithms × 10 random seeds
- **Cohen's d = 0.47** (95% CI [0.13, 0.79]) — ADE-FCM outperforms standard FCM
- **14 out of 18 datasets** — ADE-FCM matches or exceeds FCM
- **Friedman p < 0.0001** — highly significant differences between algorithms
- **Wilcoxon p = 0.043** — ADE-FCM significantly better than FCM

## Features

### Algorithm (10 Novel Contributions)
1. Adaptive fuzzifier schedule m(t): 2.5 → 1.1
2. Dynamic convergence threshold ε(t)
3. Automatic cluster discovery (silhouette + complexity penalty)
4. KMeans++ and density-based initialization
5. Robust trimmed center update (outlier exclusion)
6. Confidence-weighted membership updates
7. Center starvation prevention with reinitialization
8. Early stopping with patience
9. Pluggable distance metrics (Euclidean, Cosine, Manhattan, Mahalanobis)
10. Cluster-level explainability (permutation importance, SHAP, NL descriptions)

### Infrastructure
- Scikit-learn compatible API with Pipeline support
- Apache Spark backend for large-scale data
- GPU acceleration via cuPy / RAPIDS
- Kafka streaming support for online clustering
- Docker, Kubernetes, Airflow deployment configs
- MLflow experiment tracking
- 120+ unit tests

### Documentation
- Full IEEE-format paper draft
- API reference, usage guide, deployment guide
- Reproducibility instructions
- Security checklist for production deployment

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
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

## Links

- **GitHub**: https://github.com/Whoisasimm/ADE-FCM
- **Documentation**: https://github.com/Whoisasimm/ADE-FCM/tree/main/docs
- **Paper Draft**: IEEE_PAPER_DRAFT.md
- **Zenodo**: (add DOI after upload)
