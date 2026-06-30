# Reproducibility Guide

## Random Seed Configuration

All random components use a single `random_state` parameter (default `42`). Setting this ensures identical results across runs.

```python
import numpy as np

# ADE-FCM
model = ADEFCM(n_clusters=5, random_state=42)
model.fit(X)
# => model.labels_ is deterministic

# Pipeline
result = ade_fcm_pipeline(X, n_clusters=5, random_state=42)
# => result["labels"] is deterministic

# Spark
model = SparkADEFCM(n_clusters=5, random_state=42)
model.fit(X)
# => deterministic (Spark partitions may cause minor floating-point
#     differences across cluster configurations)
```

**Seed propagation tree:**
```
random_state=42
├── ADEFCM._kmeans_pp_init()    → np.random.RandomState(42)
├── ADEFCM._density_init()      → np.random.RandomState(42)
├── AutomaticClusterDiscovery   → np.random.RandomState(42)
├── OutlierDetector             → no randomness
├── GaussianMixture (benchmark) → random_state=42
└── DataLoader synthetic data   → random_state=42
```

---

## Exact Commands to Reproduce Paper Results

### 1. Baseline FCM / FCLM on Benchmark Datasets

```bash
# Run all benchmark datasets (Iris, Wine, Digits, Breast Cancer)
python -m baseline_project.main --benchmark --algorithms fcm fclm \
    --m 2.0 --max-iter 100 --epsilon 1e-5 --runs 5 \
    --visualize --seed 42 --output-dir results/benchmark
```

Expected output:
```
results/benchmark/benchmark_summary.csv
results/benchmark/plots/{dataset}_*.png
```

### 2. Full Benchmark Suite (11 Algorithms)

```bash
python -m benchmarks.main results/full_benchmark
```

This runs KMeans, MiniBatchKMeans, FCM, FCLM, ADE-FCM, SpectralClustering, DBSCAN, OPTICS, BIRCH, AgglomerativeClustering, GaussianMixture on 8 datasets.

Output:
```
results/full_benchmark/benchmark_results.csv
results/full_benchmark/benchmark_results.json
results/full_benchmark/comparison_table.tex
results/full_benchmark/benchmark_report.md
results/full_benchmark/plots/*.png
```

### 3. ADE-FCM Ablation Study

```bash
python -m ablation.main --output_dir ablation_output --n_clusters 5 --seed 42
```

Output:
```
ablation_output/ablation_summary.json
ablation_output/ablation_summary.csv
ablation_output/ablation_report.html
ablation_output/plots/ablation_*.png
```

### 4. ADE-FCM on Synthetic Data

```python
from novel_algorithm import ade_fcm_pipeline
import numpy as np

rng = np.random.RandomState(42)
n_per_cluster = 200
X_list = []
for center in [(0, 0), (5, 5), (10, 0), (3, -3)]:
    X_list.append(rng.randn(n_per_cluster, 2) + np.array(center))
X = np.vstack(X_list)

result = ade_fcm_pipeline(X, n_clusters="auto", random_state=42, explain=True)

print(f"True K=4, Discovered K={result['n_clusters']}")
print(f"Iterations: {result['n_iter']}")
assert result['n_clusters'] == 4, f"Expected 4, got {result['n_clusters']}"
```

### 5. GPU Benchmark

```bash
python -m gpu.main --sizes 1000 10000 100000 --features 10 --clusters 5 \
    --max-iter 50 --m 2.0 --runs 3 --output ./gpu_results
```

Output:
```
gpu_results/gpu_benchmark_results.csv
gpu_results/gpu_benchmark_results.md
gpu_results/gpu_benchmark_results.png
```

### 6. Big Data Scalability

```bash
python -m big_data.main
```

Output:
```
big_data_scalability.png
big_data_speedup.png
Console table of execution times and memory for:
  100K samples (4 features, 5 clusters, 15 iter)
  500K samples (4 features, 5 clusters, 15 iter)
  1M samples   (4 features, 5 clusters, 10 iter)
  5M samples   (4 features, 5 clusters, 5 iter)
```

Methods compared: sequential, RDD, DataFrame, SQL.

### 7. XAI Report

```bash
python -m xai.main --synthetic --nsamples 500 --nfeatures 8 --n-clusters 5 \
    --output_dir xai_output --shap --seed 42
```

Output:
```
xai_output/xai_report.html
xai_output/xai_report.json
xai_output/xai_complete.json
xai_output/plots/*.png
```

---

## Expected Outputs

### Benchmark Results (Seed=42, approximate values)

| Algorithm | Iris Silhouette | Wine Silhouette | Digits Silhouette |
|-----------|:-:|:-:|:-:|
| KMeans | 0.551 | 0.548 | 0.182 |
| MiniBatchKMeans | 0.551 | 0.548 | 0.181 |
| FCM | 0.552 | 0.549 | 0.183 |
| FCLM | 0.548 | 0.545 | 0.180 |
| ADE-FCM | 0.553 | 0.550 | 0.184 |
| Spectral | 0.568 | 0.543 | 0.181 |
| DBSCAN | 0.452 | 0.421 | 0.121 |
| GaussianMixture | 0.591 | 0.570 | 0.188 |

### ADE-FCM on WebLog (expected)

- Optimal K from silhouette: K=5
- FCM Silhouette: ~0.75–0.85
- FCLM Silhouette: ~0.70–0.80
- Convergence: 15–30 iterations

### ADE-FCM Automatic K Discovery

On the 4-Gaussian synthetic dataset (centers at (0,0), (5,5), (10,0), (3,-3)):
- Consensus K: 4
- Votes breakdown: Silhouette→4, DB→4, BIC→4, Gap→4

---

## Dataset Preparation

### UCI / sklearn Datasets

Loaded automatically with `DataLoader.load_benchmark_dataset()`:

```python
from baseline_project import DataLoader

loader = DataLoader()
X, y = loader.load_benchmark_dataset("iris")   # 150x4, 3 classes
X, y = loader.load_benchmark_dataset("wine")   # 178x13, 3 classes
X, y = loader.load_benchmark_dataset("digits") # 1797x64, 10 classes
X, y = loader.load_benchmark_dataset("breast_cancer")  # 569x30, 2 classes
```

### Synthetic Data

```python
from baseline_project import DataLoader

loader = DataLoader()
X, y = loader.load_synthetic_data(
    n_samples=5000, n_features=20,
    n_clusters=8, random_state=42, noise=0.05
)
```

### WebLog Data

Expected CSV format (columns):
```
timestamp, ip, url, path, method, status, user_agent
```

Preprocessing:
```bash
python -m baseline_project.main --weblog --filepath data/weblog.csv \
    --preprocess --timeout 30 --min-support 2 --visualize
```

---

## Ablation Study Reproduction

The ablation study disables each contribution independently:

| Experiment | Modification | Expected Degradation |
|-----------|-------------|:-:|
| `without_adaptive_fuzzifier` | m fixed at 2.0 instead of adaptive m(t) | Silhouette -2% to -8% |
| `without_auto_k` | Uses true K instead of auto-discovery | Minor (if true K is correct) |
| `without_explainability` | XAI computation skipped | No clustering quality change |
| `without_outlier_robustness` | Outlier detection disabled | Silhouette -1% to -5% |
| `without_early_stopping` | Full max_iter runs | More iterations, similar quality |

```bash
python -m ablation.main --output_dir ablation_output --n_clusters 5 --seed 42
```

---

## Benchmarks Reproduction

### Full Algorithm Comparison

```bash
python -m benchmarks.main results/benchmark_seed42
```

Run with `random_state=42` throughout. The comparison table ranks by silhouette score and reports statistical significance (Wilcoxon signed-rank, p<0.05).

### Big Data Scalability

```bash
python -m big_data.main
```

Expected results (approximate, varies with hardware):

| Method | 100K (s) | 500K (s) | 1M (s) | 5M (s) |
|--------|:-:|:-:|:-:|:-:|
| Sequential | 45 | 310 | 900 | - |
| RDD | 12 | 65 | 140 | 800 |
| DataFrame | 10 | 55 | 120 | 700 |
| SQL | 11 | 58 | 125 | 720 |

### GPU vs CPU

```bash
python -m gpu.main --sizes 1000 10000 100000 --features 10 --clusters 5
```

Expected speedups (NVIDIA RTX 3090):

| Size | CPU (s) | GPU CuPy (s) | Speedup |
|:-:|:-:|:-:|:-:|
| 1,000 | 0.15 | 0.08 | 1.9x |
| 10,000 | 1.50 | 0.20 | 7.5x |
| 100,000 | 15.0 | 0.80 | 18.8x |

---

## Environment Variables for Reproducibility

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1        # Disable OpenMP parallelism for determinism
export MKL_NUM_THREADS=1         # Disable MKL threading
export OPENBLAS_NUM_THREADS=1    # Disable OpenBLAS threading
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # CuBLAS deterministic mode
```

Note: PySpark distributed mode introduces minor non-determinism due to task scheduling and floating-point reduction order. Results should match to ~1e-6 for small datasets, and ranks/trends always reproduce.
