# Changelog

All notable changes to **ADE-FCM** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-07-01  *(Public Release)*

### 🎉 Highlights

- **900-experiment validation** (18 datasets × 5 algorithms × 10 seeds)
- **6 critical bugs fixed** through systematic audit
- **Complete publication package**: IEEE paper draft, thesis draft, figures, tables
- **120 unit tests** — all passing
- **Security audit**: hardcoded paths and credentials remediated
- **Professional GitHub release**: production .gitignore, clean README, security checklist
- Friedman + Wilcoxon statistical significance testing (p = 0.043)

### Added

#### Algorithm
- `novel_algorithm/ade_fcm.py`: Full ADEFCM class with 10 novel contributions
- `novel_algorithm/auto_cluster.py`: Automatic K discovery with complexity penalty (λ = 0.02)
- `novel_algorithm/adaptive_params.py`: Adaptive m(t) and ε(t) schedulers
- `novel_algorithm/density_init.py`: Density-based center initialization
- `novel_algorithm/outlier_detector.py`: Post-hoc outlier flagging
- `novel_algorithm/xai.py`: Permutation importance + cluster summary functions

#### XAI Module
- `xai/cluster_explainer.py`: Natural-language cluster descriptions
- `xai/shap_explainer.py`: SHAP-based local explanations
- `xai/visualizer.py`: Membership heatmaps, feature importance plots

#### Benchmarking
- Phase 7A validation: `scripts/phase7a_validation.py` — 420 experiments
- `results/phase7a_raw_results.json` — complete raw experiment log
- `results/benchmark_results.csv` — aggregated metrics
- Publication figures: 13 PNG + SVG files in `results/figures/` and `paper/figures/`
- Publication tables: 9 CSV + LaTeX files in `results/tables/` and `paper/tables/`

#### Documentation
- `IEEE_PAPER_DRAFT.md` — complete 11,773-word IEEE-format paper
- `THESIS_DRAFT.md` — complete 6-chapter thesis
- `PAPER_AUTHOR_PACKAGE.md` — journal/conference submission checklist
- `SECURITY_CHECKLIST.md` — pre-publication security verification
- `docs/08_ARCHITECTURE_DIAGRAM.md` — full system architecture
- `docs/09_API_REFERENCE.md` — complete API documentation

#### Community Files
- `LICENSE` (MIT)
- `CITATION.cff`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `.github/workflows/` — CI/CD pipeline
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates

### Fixed

| ID | Bug | Impact | Fix |
|:--:|-----|--------|-----|
| F1 | `_robust_center_update` was dead code — never called | Outlier robustness was silently disabled | Real trimmed FCM center update, called in `fit()` |
| F2 | `feature_importance(shift/fisher)` — reporting-only, no real computation | XAI importances were meaningless | Replaced with permutation importance |
| F3 | Auto-K overclustering — always returned K_max | Wrong cluster counts | Added `complexity_penalty=0.02` per cluster to silhouette |
| F4 | Center starvation on high-dim data — one center absorbed all points | Degenerate clustering | Reinit threshold + membership floor + cosine metric |
| F5 | Distance metrics (Manhattan, Cosine, Mahalanobis) not wired in | Only Euclidean worked | `scipy.spatial.distance.cdist` integration with METRIC_ALIASES |
| F6 | Center reinitialization did not trigger membership recomputation | New centers immediately starved again | Added post-reinit membership update with floor |

### Changed

- Default distance metric: now fully pluggable via `metric=` parameter
- `auto_cluster.consensus_search`: accepts `complexity_penalty` parameter
- `ADEFCM.fit()`: passes `complexity_penalty` to `consensus_search`
- `pyproject.toml`: updated URLs to `https://github.com/Whoisasimm/ADE-FCM`
- `CITATION.cff`: corrected `url` and `repository-code` fields, updated abstract
- `.gitignore`: production-grade, focused pattern set
- `README.md`: updated with 18-dataset expanded benchmark results (14/18 wins, d=0.47)
- `SECURITY.md`: aligned claims with actual security posture
- `deployment/mlflow/mlflow_config.py`: replaced hardcoded `/tmp/` path
- `deployment/docker-compose.yml`: replaced hardcoded Jupyter token with env variable

### Removed

- Dead code: old `_robust_center_update` stub (replaced with real implementation)
- Reporting-only: `feature_importance(shift)` and `feature_importance(fisher)` stubs

---

## [0.9.0] — 2026-06-20  *(Audit Phase)*

### Added

- 6-phase audit framework
- `results/audit/` — 6 detailed audit reports
  - `ABLATION_AUDIT_REPORT.md`
  - `STATISTICAL_AUDIT_REPORT.md`
  - `PERFORMANCE_AUDIT_REPORT.md`
  - `DEEP_MODEL_AUDIT_REPORT.md`
  - `REVIEWER_REJECTION_REPORT.md`
  - `ROAD_TO_PUBLICATION.md`
- Bug discovery: 6 critical issues identified and documented

### Changed

- Initial implementation baseline frozen for comparison

---

## [0.8.0] — 2026-06-15  *(Extensions Phase)*

### Added

- `deep_ade_fcm/` — autoencoder-based deep clustering variant
- `gpu/` — GPU acceleration (cuPy / RAPIDS)
- `streaming/` — Online clustering via Kafka + Spark Streaming
- `big_data/` — Distributed FCM (Spark RDD + DataFrame)
- `deployment/` — Docker, Kubernetes, Airflow, CI/CD configs
- `docs/` — 10 technical documentation files
- Publication draft templates

---

## [0.7.0] — 2026-06-10  *(Baseline & Benchmarks)*

### Added

- `baseline_project/` — FCM, FCLM, KMeans, Agglomerative implementations
- `benchmarks/` — Benchmark runner with 4 sklearn datasets
- 13 publication figures (PNG + SVG)
- 9 publication tables (CSV + LaTeX)

---

## [0.6.0] — 2026-06-05  *(XAI & Ablation)*

### Added

- `xai/` — SHAP explainer, cluster explainer, visualizer
- `ablation/` — ablation study framework
- 55 unit tests for `novel_algorithm/` (all passing)
- Auto cluster discovery improvements

---

## [0.5.0] — 2026-06-01  *(Initial Algorithm)*

### Added

- Initial ADE-FCM core with 10 novel contributions
- Adaptive fuzzifier m(t) and dynamic threshold ε(t)
- KMeans++ and density-based initialization
- Confidence-weighted membership updates
- Early stopping mechanism
- Post-hoc outlier detection module

---

[1.0.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v1.0.0
[0.9.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v0.9.0
[0.8.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v0.8.0
[0.7.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v0.7.0
[0.6.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v0.6.0
[0.5.0]: https://github.com/Whoisasimm/ADE-FCM/releases/tag/v0.5.0
