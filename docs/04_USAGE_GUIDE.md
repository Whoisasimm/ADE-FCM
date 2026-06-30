# Usage Guide

## Command-Line Interface

### Baseline Project (FCM / FCLM experiments)

```bash
# Single dataset
python -m baseline_project.main --dataset iris --algorithms fcm fclm --visualize

# Synthetic data
python -m baseline_project.main --synthetic --n-samples 5000 --n-features 20 --n-clusters 8

# Full benchmark across all datasets
python -m baseline_project.main --benchmark --algorithms fcm fclm --visualize

# Weblog preprocessing pipeline
python -m baseline_project.main --weblog --filepath data/weblog.csv --preprocess

# Spark execution
python -m baseline_project.main --dataset iris --spark --spark-master local[4]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `iris` | Dataset: iris, wine, digits, breast_cancer, synthetic, weblog |
| `--filepath` | None | Path to weblog CSV file |
| `--synthetic` | False | Use synthetic data |
| `--n-samples` | 1000 | Number of synthetic samples |
| `--n-features` | 10 | Number of features |
| `--n-clusters` | 5 | Ground truth clusters |
| `--noise` | 0.05 | Noise level |
| `--algorithms` | fcm fclm | Algorithms: fcm, fclm, both |
| `--m` | 2.0 | Fuzzifier exponent |
| `--max-iter` | 100 | Maximum iterations |
| `--epsilon` | 1e-5 | Convergence threshold |
| `--runs` | 1 | Number of runs for statistics |
| `--spark` | False | Enable Spark execution |
| `--spark-master` | local[*] | Spark master URL |
| `--visualize` | False | Generate plots |
| `--output-dir` | results | Output directory |
| `--seed` | 42 | Random seed |

### Benchmarks

```bash
# Run full benchmark suite
python -m benchmarks.main results/

# Run with custom results directory
python -m benchmarks.main /path/to/results
```

This runs 11 algorithms (KMeans, MiniBatchKMeans, FCM, FCLM, ADE-FCM, SpectralClustering, DBSCAN, OPTICS, BIRCH, AgglomerativeClustering, GaussianMixture) across 8 datasets.

### Streaming Pipeline

```bash
# Online mode (direct NumPy, no Kafka)
python -m streaming.main --mode online --data-path data.csv --n-clusters 5 --batch-size 100

# Kafka producer mode
python -m streaming.main --mode produce --data-path data.csv --bootstrap-servers localhost:9092

# Spark Structured Streaming mode
python -m streaming.main --mode spark --bootstrap-servers localhost:9092 --n-clusters 5 --use-spark
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `online` | Mode: produce, online, spark, stream |
| `--data-path` | None | Path to input CSV |
| `--bootstrap-servers` | localhost:9092 | Kafka brokers |
| `--topic` | clustering-data | Kafka topic |
| `--n-clusters` | 5 | Number of clusters |
| `--fuzzifier` | 2.0 | Fuzzy exponent m |
| `--learning-rate` | 0.3 | Online learning rate |
| `--batch-size` | 100 | Mini-batch size |
| `--interval` | 0.1 | Sleep between batches (s) |
| `--use-spark` | False | Enable Spark backend |

### GPU Benchmark

```bash
# Run GPU benchmarks
python -m gpu.main --sizes 1000 10000 100000 --features 10 --clusters 5 --max-iter 50

# All options
python -m gpu.main --sizes 1000 5000 10000 50000 --features 20 --clusters 10 \
    --max-iter 100 --m 2.0 --runs 3 --output ./gpu_results --no-plot
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--sizes` | 1000 10000 100000 | Dataset sizes |
| `--features` | 10 | Number of features |
| `--clusters` | 5 | Number of clusters |
| `--max-iter` | 50 | Maximum iterations |
| `--m` | 2.0 | Fuzziness |
| `--runs` | 2 | Runs per config |
| `--output` | `.` | Output directory |
| `--no-plot` | False | Skip plots |

### XAI Report Generator

```bash
# Using synthetic data
python -m xai.main --synthetic --nsamples 500 --nfeatures 8 --n-clusters 5 --output_dir xai_output

# Using custom CSV data
python -m xai.main --data data.csv --n-clusters 5 --output_dir xai_output --shap

# All options
python -m xai.main --synthetic --nsamples 1000 --nfeatures 10 --n-clusters 5 \
    --output_dir ./xai_report --shap --seed 42
```

### Ablation Study

```bash
python -m ablation.main --output_dir ablation_output --n_clusters 5 --seed 42
```

---

## Python API Examples

### ADEFCM Class

```python
from novel_algorithm import ADEFCM
import numpy as np

# Configuration options
model = ADEFCM(
    n_clusters="auto",           # int or "auto" for automatic discovery
    max_iter=300,                # Maximum iterations
    m="adaptive",                # float or "adaptive" for m(t) schedule
    epsilon="dynamic",           # float or "dynamic" for epsilon(t) schedule
    init_method="kmeans++",      # "kmeans++", "density", or "random"
    early_stopping_patience=10,  # Consecutive converged iterations
    outlier_threshold=2.0,       # Std multiplier for outlier detection
    random_state=42,             # Reproducibility seed
    verbose=True,                # Log iteration details
)

# Fit
model.fit(X)

# Attributes
print(model.n_clusters_)         # Number of clusters found
print(model.centers_)            # Final cluster centers (n_clusters x n_features)
print(model.U_)                  # Membership matrix (n_samples x n_clusters)
print(model.labels_)             # Hard labels (n_samples,)
print(model.n_iter_)             # Iterations taken
print(model.J_history_)          # Objective values per iteration
print(model.outlier_mask_)       # Boolean outlier mask (n_samples,)
print(model.outlier_scores_)     # Outlier scores (n_samples,)
print(model.feature_importances_)# Per-cluster feature importance
print(model.cluster_summaries_)  # Cluster summary dicts
print(model.convergence_history_)# Membership change per iteration

# Predict new points
labels = model.predict(X_new)

# Fit + predict
labels = model.fit_predict(X)

# XAI explanation
explanation = model.explain(X, feature_names=["age", "income"])
print(explanation["descriptions"])
print(explanation["top_features_global"])
```

### Full Pipeline

```python
from novel_algorithm import ade_fcm_pipeline, ADEFCMPipeline

# Functional API
result = ade_fcm_pipeline(
    X,
    n_clusters="auto",
    max_iter=300,
    m="adaptive",
    epsilon="dynamic",
    init_method="kmeans++",
    early_stopping_patience=10,
    outlier_threshold=2.0,
    random_state=42,
    verbose=True,
    use_spark=False,
    spark_master="local[*]",
    explain=True,
    feature_names=["f1", "f2", "f3"],
)

# Result structure
print(result["labels"])          # Hard labels
print(result["centers"])         # Cluster centers
print(result["U"])               # Membership matrix
print(result["n_iter"])          # Iterations
print(result["J_history"])       # Objective history
print(result["outlier_mask"])    # Outlier flags
print(result["summaries"])       # Cluster summaries
print(result["descriptions"])    # NL descriptions
print(result["feature_importances"])  # Importance scores
print(result["explanation"])     # Full XAI result
print(result["outlier_indices"]) # Indices of outliers
print(result["outlier_details"]) # Outlier count/ratio

# Pipeline class
pipeline = ADEFCMPipeline(n_clusters="auto", use_spark=False)
pipeline.fit(X, feature_names=["f1", "f2", "f3"])
print(pipeline.summary())

# Predict using fitted pipeline
new_labels = pipeline.predict(X_new)
```

### Spark Distributed

```python
from novel_algorithm import SparkADEFCM

model = SparkADEFCM(
    n_clusters="auto",
    max_iter=300,
    m="adaptive",
    epsilon="dynamic",
    init_method="kmeans++",
    early_stopping_patience=10,
    outlier_threshold=2.0,
    random_state=42,
    verbose=True,
    spark_master="local[*]",
    checkpoint_dir="/tmp/spark-checkpoints",
    num_partitions=None,  # auto-inferred
)

model.fit(X)

# Results
print(model.labels_)
print(model.centers_)

# Predict
new_labels = model.predict(X_new)

# Clean up
model.stop()
```

### Adaptive Parameters (standalone)

```python
from novel_algorithm import AdaptiveFuzzifier, DynamicThreshold, EarlyStopping

# Adaptive fuzzifier
af = AdaptiveFuzzifier(m_min=1.1, m_max=2.5, alpha=3.0, max_iter=300)
for t in range(300):
    m_t = af(t)
print(af.summary())

# Dynamic threshold
dt = DynamicThreshold(eps_0=1e-3, beta=5.0, max_iter=300, min_eps=1e-8)
for t in range(300):
    eps_t = dt(t)
print(dt.summary())

# Early stopping
es = EarlyStopping(patience=10, verbose=True)
for iteration in range(300):
    change = compute_membership_change(...)
    if es.check(change, threshold=1e-4, iteration=iteration):
        print(f"Stopped at iteration {iteration}")
        break
```

### Density Initialization

```python
from novel_algorithm import DensityInitializer, KMeansPlusPlusInitializer

# Density init
di = DensityInitializer(n_clusters=5, density_percentile=90.0,
                        subsample_size=1000, random_state=42)
centers = di.initialize(X)

# KMeans++ init
kpp = KMeansPlusPlusInitializer(n_clusters=5, random_state=42)
centers = kpp.initialize(X)
```

### Automatic Cluster Discovery

```python
from novel_algorithm import AutomaticClusterDiscovery, ClusterEvaluator

# Full consensus search
discover = AutomaticClusterDiscovery(random_state=42)
k_range = range(2, 11)
results = discover.search(X, k_range)
best_k = discover.consensus_search(X, k_range)

# Elbow curve
wss = discover.elbow_curve(X, k_range)

# Evaluate a single K
metrics = discover.evaluate_k(X, 5)
print(metrics)

# ClusterEvaluator (for given labels)
from novel_algorithm import ClusterEvaluator
evaluator = ClusterEvaluator(X, labels, centers)
print(evaluator.evaluate_all())
# {'silhouette': ..., 'davies_bouldin': ..., 'bic': ..., 'gap': ...}
```

### Outlier Detection

```python
from novel_algorithm import OutlierDetector

# Weighted distance method
detector = OutlierDetector(method="weighted_distance", threshold_multiplier=2.0)
detector.fit(X, model.U_, model.centers_, m=2.0)
print(detector.outlier_summary())

# Entropy method
detector = OutlierDetector(method="entropy", contamination=0.05)
detector.fit(X, model.U_, model.centers_)

# LOF approximation
detector = OutlierDetector(method="lof", threshold_multiplier=3.0)
detector.fit(X, model.U_, model.centers_)

# Predict on new scores
mask, scores = detector.predict(scores=new_scores)
```

### XAI Functions (standalone)

```python
from novel_algorithm import (
    feature_importance, cluster_summary,
    describe_cluster_natural, shap_explain, explain_clusters
)

# Feature importance (methods: "shift", "fisher", "permutation")
fi = feature_importance(X, labels, centers, method="shift")

# Cluster summaries
summaries = cluster_summary(X, labels, centers, feature_names=["f1", "f2"])
for s in summaries:
    print(s["size"], s["top_features"])

# Natural language description
for s in summaries:
    print(describe_cluster_natural(s))

# SHAP-style explanation for a single point
shap_values, text = shap_explain(X, labels, centers, sample_idx=0)
print(text)

# Full explanation
exp = explain_clusters(X, labels, centers, feature_names=["f1", "f2"],
                       outlier_mask=outlier_mask)
print(exp["n_clusters"], exp["n_outliers"])
print(exp["top_features_global"])
```

### Online Streaming

```python
from streaming import OnlineFCM, DataProducer, StreamingPipeline

# OnlineFCM (incremental)
model = OnlineFCM(n_clusters=5, m=2.0, learning_rate=0.3, random_state=42)
model.partial_fit(batch1)
model.partial_fit(batch2)
labels = model.predict(new_data)
U = model.get_membership(new_data)
model.reset()

# StreamingPipeline
pipeline = StreamingPipeline(
    n_clusters=5, fuzzifier=2.0, learning_rate=0.3,
    bootstrap_servers="localhost:9092", topic="clustering-data",
    checkpoint_dir="/tmp/ade-fcm-checkpoint",
    use_spark=False,
)
pipeline.setup_model()
pipeline.run_online("data.csv", batch_size=100)

# Kafka producer
producer = DataProducer(bootstrap_servers="localhost:9092", topic="clustering-data")
producer.produce_batch(X, batch_size=100, sleep=0.1)
producer.produce_stream(X, interval=0.5)
producer.close()
```

### GPU Acceleration

```python
from gpu import GPUFCMManager, RAPIDSFCM, SparkGPUHybridEngine

# GPUFCMManager (CuPy)
model = GPUFCMManager(use_gpu=True, n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42)
centers, U, J_history = model.fit(X)
labels = model.predict(X_new)
print(f"Fit time: {model.fit_time:.3f}s")
print(model.get_memory_report())

# Benchmark CPU vs GPU
bench = model.benchmark_cpu_vs_gpu(X, n_runs=3)
print(f"CPU: {bench['cpu']['mean']:.3f}s, GPU: {bench['gpu']['mean']:.3f}s, Speedup: {bench['speedup']:.1f}x")

# RAPIDSFCM
model = RAPIDSFCM(n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42)
centers, U, J = model.fit_fuzzy(X)
centers_km = model.fit_rapids_kmeans(X)  # cuML KMeans first, then fuzzy

# Spark GPU hybrid
engine = SparkGPUHybridEngine(n_clusters=5, spark_mode="spark_gpu")
centers, U = engine.fit_spark_gpu(X)
comparison = engine.compare_modes(X, include_cpu=True, include_spark_gpu=True)
print(comparison)
engine.stop_spark()
```

### Big Data FCM

```python
from big_data import LargeScaleFCM
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ADE-FCM").getOrCreate()
fcm = LargeScaleFCM(
    spark_session=spark, n_clusters=5, max_iter=20,
    m=2.0, epsilon=1e-4, chunk_size=50000,
)

# Sequential mode
centers, U, J = fcm.fit_sequential(X)

# RDD mode
centers, U, J = fcm.fit_rdd(X)

# DataFrame mode
df = generate_dataframe(spark, X)
centers, U, J = fcm.fit_dataframe(df)

# SQL mode
centers, U, J = fcm.fit_sql(df)

print(fcm.get_metadata())
```

---

## Configuration Options Summary

| Module | Key Parameters | Default |
|--------|---------------|---------|
| ADEFCM | `n_clusters, max_iter, m, epsilon, init_method, early_stopping_patience, outlier_threshold, random_state` | `auto, 300, adaptive, dynamic, kmeans++, 10, 2.0, 42` |
| SparkADEFCM | Same as ADEFCM + `spark_master, checkpoint_dir, num_partitions` | `local[*]` |
| AdaptiveFuzzifier | `m_min, m_max, alpha, max_iter` | `1.1, 2.5, 3.0, 300` |
| DynamicThreshold | `eps_0, beta, max_iter, min_eps` | `1e-3, 5.0, 300, 1e-8` |
| EarlyStopping | `patience, min_delta, verbose` | `10, 0.0, True` |
| DensityInitializer | `n_clusters, density_percentile, subsample_size, random_state` | `90.0, 1000, 42` |
| AutomaticClusterDiscovery | `random_state` | `42` |
| OutlierDetector | `method, threshold_multiplier, contamination` | `weighted_distance, 2.0, None` |
| OnlineFCM | `n_clusters, m, learning_rate, random_state` | `5, 2.0, 0.3, 42` |
| GPUFCMManager | `use_gpu, n_clusters, max_iter, m, tol, seed` | `True, 5, 100, 2.0, 1e-4, 42` |
| LargeScaleFCM | `n_clusters, max_iter, m, epsilon, chunk_size` | `5, 100, 2.0, 1e-5, 50000` |

---

## Output Interpretation

### Cluster Output

- **centers_** (`ndarray` `n_clusters x n_features`): Final cluster centroid coordinates.
- **U_** (`ndarray` `n_samples x n_clusters`): Fuzzy membership matrix. Row i sums to 1; `U[i,j]` is the membership of point i to cluster j.
- **labels_** (`ndarray` `n_samples`): Hard cluster assignment via `argmax(U, axis=1)`.
- **J_history_** (`list` of float): Objective function `J` at each iteration. Monotonically decreasing indicates proper convergence.
- **outlier_mask_** (`ndarray` of bool): `True` for statistically flagged outliers. Points where weighted distance > `mean + threshold * std`.

### XAI Output

- **feature_importances** (`ndarray` `n_clusters x n_features`): Higher = more discriminative for that cluster.
- **cluster_summaries** (list of dict): Each dict has `size`, `proportion`, `center`, `radius`, `max_distance`, `top_features`.
- **descriptions** (list of str): Human-readable cluster descriptions.
- **shap_values**: Per-feature contribution to a point's cluster assignment.

### Streaming Output

- **centers_history** (list of `ndarray`): Evolution of centers over batches.
- **batches_processed / total_points**: Pipeline throughput stats.
- **Throughput (pts/s)**: Processing rate.

### GPU Benchmark Output

- **cpu_time / gpu_time / rapids_time / spark_cpu_time / spark_gpu_time**: Execution times for each mode.
- **gpu_speedup**: `cpu_time / gpu_time` ratio (>1 means GPU faster).

### Ablation Output

- **full_ade_fcm**: Baseline metrics (silhouette, DB, time, iterations).
- **ablation_results**: Metrics per ablated component.
- **degradation**: Percentage change relative to full model.
