# GitHub Repository Structure

## Repository Layout

```
ADE-FCM/
│
├── novel_algorithm/           # Core ADE-FCM implementation (10 contributions)
│   ├── __init__.py            # Public API exports
│   ├── ade_fcm.py             # ADEFCM class (contributions 1-9)
│   ├── adaptive_params.py     # AdaptiveFuzzifier, DynamicThreshold, EarlyStopping
│   ├── density_init.py        # DensityInitializer, KMeansPlusPlusInitializer
│   ├── auto_cluster.py        # AutomaticClusterDiscovery, ClusterEvaluator
│   ├── outlier_detector.py    # OutlierDetector (3 methods)
│   ├── xai.py                 # XAI: explain_clusters, feature_importance, shap
│   ├── spark_ade_fcm.py       # SparkADEFCM (contribution 10)
│   └── main.py                # Pipeline: ade_fcm_pipeline, ADEFCMPipeline, demo
│
├── baseline_project/          # FCM/FCLM baselines replicating the reference paper
│   ├── main.py                # CLI entry, run_fcm(), run_fclm()
│   ├── data_loader.py         # DataLoader for benchmark/synthetic/weblog
│   ├── preprocessing.py       # Preprocessor (weblog cleaning, TOH1 sessions)
│   ├── distance_metrics.py    # 7 distance functions + pairwise + point-to-centers
│   ├── membership_update.py   # MembershipUpdater (FCM/FCLM)
│   ├── cluster_update.py      # ClusterUpdater (FCM/FCLM centers)
│   ├── objective_function.py  # ObjectiveFunction (J, PC, PEC, SSE)
│   ├── convergence.py         # ConvergenceChecker (Frobenius, early stopping)
│   ├── evaluation.py          # Evaluator (silhouette, DB, CH, ARI, NMI, RI)
│   ├── spark_engine.py        # SparkFCMEngine, SparkFCLMEngine
│   ├── visualization.py       # Visualizer (clusters, convergence, confusion)
│   ├── utils.py               # setup_logging, set_random_seed, save_results, etc.
│   └── requirements.txt       # Baseline dependencies
│
├── benchmarks/                # Comprehensive benchmark suite
│   ├── __init__.py
│   ├── main.py                # run_benchmarks() entry point
│   ├── benchmark_runner.py    # BenchmarkRunner (11 algorithms, 8 datasets)
│   ├── metrics_collector.py   # MetricsCollector (time, memory, metrics)
│   ├── results_analyzer.py    # ResultsAnalyzer (ranking, significance, LaTeX)
│   └── plots.py               # BenchmarkPlotter (bar, radar, scalability)
│
├── streaming/                 # Real-time streaming pipeline
│   ├── __init__.py
│   ├── main.py                # StreamingPipeline (CLI entry point)
│   ├── online_clustering.py   # OnlineFCM (incremental partial_fit)
│   ├── kafka_producer.py      # DataProducer (batch + stream modes)
│   └── spark_streaming_consumer.py  # ADEFCMStreaming (Structured Streaming)
│
├── big_data/                  # Large-scale distributed FCM on Spark
│   ├── __init__.py
│   ├── main.py                # Benchmark pipeline (sequential/RDD/DF/SQL)
│   ├── large_scale_fcm.py     # LargeScaleFCM, ChunkedMembershipUpdate
│   ├── spark_rdd_optimizer.py # SparkRDDOptimizer, RDDMembershipComputer
│   └── spark_dataframe_optimizer.py  # SparkDataFrameOptimizer
│
├── gpu/                       # GPU-accelerated FCM
│   ├── __init__.py
│   ├── main.py                # GPU benchmark pipeline (CLI)
│   ├── gpu_fcm.py             # GPUFCMManager (CuPy CPU/GPU)
│   ├── cuda_kernels.py        # Custom CUDA kernels (ElementwiseKernel, RawKernel)
│   ├── rapids_fcm.py          # RAPIDSFCM (cuML KMeans + fuzzy membership)
│   └── spark_gpu_hybrid.py    # SparkGPUHybridEngine (Spark partition + GPU)
│
├── xai/                       # Explainable AI module
│   ├── __init__.py
│   ├── main.py                # XAI report generator (CLI)
│   ├── cluster_explainer.py   # ClusterExplainer (importance, summaries)
│   ├── shap_explainer.py      # ShapExplainer (SHAP proxy)
│   └── visualizer.py          # XAIVisualizer (importance, radar, parallel coordinates)
│
├── ablation/                  # Ablation study
│   ├── __init__.py
│   ├── main.py                # Ablation study runner (CLI + HTML report)
│   └── ablation_study.py      # AblationStudy class
│
├── deployment/                # Production deployment
│   ├── Dockerfile             # Multi-stage Docker build
│   ├── docker-compose.yml     # Full stack: ZK, Kafka, Spark, Jupyter, MLflow
│   ├── kubernetes/
│   │   ├── configmap.yaml     # ConfigMap for env vars
│   │   ├── deployment.yaml    # Deployments + PVCs + HPA
│   │   └── service.yaml       # Services (ClusterIP, NodePort, LoadBalancer)
│   ├── mlflow/
│   │   └── mlflow_config.py   # ADEFCMMLflowConfig (tracking, logging, registry)
│   ├── monitoring/
│   │   ├── prometheus.yml     # Prometheus scrape config
│   │   └── grafana_dashboard.json  # Pre-built Grafana dashboard
│   ├── ci_cd/.github/workflows/
│   │   ├── ci.yml             # Test → Build → Deploy pipeline
│   │   └── research_pipeline.yml  # Research workflow
│   └── airflow/dags/
│       ├── research_pipeline_dag.py  # Weekly research pipeline DAG
│       └── benchmark_dag.py          # Benchmark DAG
│
├── docs/                      # Documentation
│   ├── 01_PAPER_UNDERSTANDING.md
│   ├── 02_MATHEMATICAL_DERIVATIONS.md
│   ├── 03_INSTALLATION.md
│   ├── 04_USAGE_GUIDE.md
│   ├── 05_REPRODUCIBILITY.md
│   ├── 06_DEPLOYMENT_GUIDE.md
│   ├── 07_GITHUB_REPO_STRUCTURE.md
│   ├── 08_ARCHITECTURE_DIAGRAM.md
│   └── 09_API_REFERENCE.md
│
├── data/                      # Dataset directory (gitignored, user-provided)
├── notebooks/                 # Jupyter notebooks (gitignored, user-created)
├── results/                   # Benchmark results output (gitignored)
├── publication/               # Paper PDFs and supplementary material
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── conftest.py            # Shared fixtures
│
├── src/                       # Package entry point
│   ├── __init__.py
│   └── main.py                # CLI entry for the installed package
│
├── README.md                  # Project overview
├── LICENSE                    # MIT License
├── AGENTS.md                  # (optional) AI-assisted development notes
├── requirements.txt           # Full project dependencies
└── setup.py / pyproject.toml  # Package metadata
```

---

## Branch Strategy

```
main                    # Production-ready, protected
├── develop             # Integration branch
│   ├── feature/*       # New features (e.g., feature/gpu-kernel-opt)
│   ├── fix/*           # Bug fixes (e.g., fix/memory-leak-streaming)
│   ├── bench/*         # Benchmark additions (e.g., bench/new-dataset)
│   └── docs/*          # Documentation (e.g., docs/api-refresh)
└── release/*           # Release candidates (e.g., release/v1.0.0)
```

### Workflow

1. Create feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/your-feature
   ```

2. Make changes, commit with conventional commits:
   ```
   feat: add GPU kernel for membership update
   fix: resolve OutOfMemory in large_scale_fcm
   docs: update API reference for ADEFCM
   test: add unit tests for DensityInitializer
   bench: add Fashion-MNIST dataset
   ```

3. Open PR to `develop` with description template (see below).

4. After review and CI passes, squash-merge to `develop`.

5. For releases: `develop` → `release/vX.Y.Z` → tag + merge to `main`.

---

## Contribution Guidelines

### Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ADE-FCM.git
   cd ADE-FCM
   ```
3. Set up development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   pip install flake8 black mypy pytest pytest-cov
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/my-contribution
   ```

### Code Standards

- **Style**: Follow PEP 8. Use `black --line-length=100` for formatting.
- **Linting**: `flake8 . --max-complexity=10 --max-line-length=100`
- **Types**: Use type hints for all public functions. Verify with `mypy src/ --strict --ignore-missing-imports`.
- **Docstrings**: NumPy/Google style docstrings for all public classes and methods.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Testing**: Unit tests go in `tests/`. Run with `pytest tests/ --cov=src/ --cov-fail-under=80`.
- **Imports**: Standard lib → third-party → local. Absolute imports preferred.

### Commit Messages

Use conventional commits:
```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `bench`, `perf`, `ci`, `chore`

### PR Process

1. Ensure all tests pass and coverage >= 80%.
2. Run linting and type checking.
3. Update documentation if API changes.
4. Open PR against `develop` using the template.
5. At least one maintainer review required.
6. Squash-merge with a descriptive commit message.

---

## Issue Templates

### Bug Report

```markdown
**Describe the bug**
A clear and concise description.

**To Reproduce**
Minimal code example:
```python
from novel_algorithm import ADEFCM
model = ADEFCM(n_clusters=5)
model.fit(X)  # crashes here
```

**Expected behavior**
What should happen.

**Environment:**
- Python version:
- ADE-FCM version:
- Spark version (if applicable):
- CUDA version (if applicable):
- OS:

**Additional context**
Logs, screenshots, etc.
```

### Feature Request

```markdown
**Is your feature request related to a problem?**
Clear description.

**Describe the solution**
What you want to happen.

**Describe alternatives**
Other approaches considered.

**Additional context**
API design ideas, references, etc.
```

### Benchmark / Result Report

```markdown
**Algorithm & Dataset**

**Hardware**
- CPU:
- GPU:
- RAM:
- Spark config:

**Results**
| Metric | Value |
|--------|-------|
| Silhouette | |
| DB Index | |
| Time (s) | |

**Comparison with baseline**
```

---

## Pull Request Template

```markdown
## Description

Closes #(issue)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Benchmark addition
- [ ] Performance improvement
- [ ] Code refactor

## Changes Made
- `file1.py`: description
- `file2.py`: description

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual verification

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have added tests that prove my fix/feature works
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] All CI checks pass (lint, type, test)

## Performance Impact
- [ ] No significant change
- [ ] Improvement: [description]
- [ ] Regression: [mitigation]

## Additional Notes
Any relevant information for reviewers.
```
