# API Reference

## `novel_algorithm` — Core ADE-FCM

### Classes

#### `ADEFCM`

Adaptive Distributed Explainable Fuzzy C-Means clustering algorithm integrating all 10 novel contributions.

**Constructor:**
```python
ADEFCM(
    n_clusters: int | str = "auto",
    max_iter: int = 300,
    m: float | str = "adaptive",
    epsilon: float | str = "dynamic",
    init_method: str = "kmeans++",
    early_stopping_patience: int = 10,
    outlier_threshold: float = 2.0,
    random_state: int = 42,
    verbose: bool = True,
)
```

**Parameters:**
- `n_clusters` — Number of clusters. If `"auto"`, uses `AutomaticClusterDiscovery.consensus_search()` (Contribution 5).
- `max_iter` — Maximum number of iterations.
- `m` — Fuzzifier exponent. If `"adaptive"`, uses `AdaptiveFuzzifier` schedule (Contribution 3).
- `epsilon` — Convergence threshold. If `"dynamic"`, uses `DynamicThreshold` schedule (Contribution 8).
- `init_method` — Center initialization: `"kmeans++"` (C1), `"density"` (C2), or `"random"`.
- `early_stopping_patience` — Consecutive converged iterations before early stop (C7).
- `outlier_threshold` — Standard deviation multiplier for outlier flagging (C6).
- `random_state` — Seed for reproducibility.
- `verbose` — Log iteration details.

**Attributes:**
- `centers_` — `ndarray (n_clusters, n_features)`. Final cluster centers.
- `U_` — `ndarray (n_samples, n_clusters)`. Fuzzy membership matrix.
- `labels_` — `ndarray (n_samples,)`. Hard cluster assignments (`argmax(U_, axis=1)`).
- `J_history_` — `list[float]`. Objective value per iteration.
- `n_iter_` — `int`. Number of iterations executed.
- `n_clusters` — `int`. Number of clusters (after auto-discovery if applicable).
- `outlier_mask_` — `ndarray[bool] (n_samples,)`. `True` for flagged outliers.
- `outlier_scores_` — `ndarray (n_samples,)`. Outlier scores per point.
- `feature_importances_` — `ndarray (n_clusters, n_features)`. Shift-based feature importance.
- `cluster_summaries_` — `list[dict]`. Cluster summary dicts from `cluster_summary()`.
- `convergence_history_` — `list[float]`. Membership Frobenius norm change per iteration.

**Methods:**

```python
fit(X: ndarray, y=None) -> ADEFCM
```
Fit ADE-FCM to data. Automatically discovers K if `n_clusters="auto"`. Returns self.

```python
predict(X: ndarray) -> ndarray
```
Predict cluster labels for new data using learned centers.

```python
fit_predict(X: ndarray, y=None) -> ndarray
```
Fit and return labels.

```python
explain(X: ndarray, feature_names: list[str] | None = None) -> dict
```
Return a full XAI explanation: summaries, descriptions, feature importances, outlier info.

**Internal methods (contributions):**

```python
_adaptive_fuzzifier(t: int) -> float       # C3: m(t)
_dynamic_threshold(t: int) -> float         # C8: epsilon(t)
_kmeans_pp_init(X: ndarray) -> ndarray     # C1: KMeans++ init
_density_init(X: ndarray) -> ndarray        # C2: Density init
_confidence_weighted_membership(U) -> U     # C4: Confidence weighting
_outlier_detection(X, centers, U, m) -> mask, scores  # C6: Outlier detection
_update_membership(X, centers, m) -> U      # Standard FCM membership
_update_centers(X, U, m) -> centers         # Standard FCM center update
_compute_objective(X, U, centers, m) -> float  # Objective J
_initialize_centers(X) -> centers           # Dispatch init method
```

---

#### `AdaptiveFuzzifier`

Time-varying fuzzifier exponent (Contribution 3).

```python
AdaptiveFuzzifier(
    m_min: float = 1.1,
    m_max: float = 2.5,
    alpha: float = 3.0,
    max_iter: int = 300,
)
```

**Methods:**
```python
__call__(t: int) -> float          # m(t) for iteration t
schedule() -> ndarray              # Full m(t) schedule of length max_iter
summary() -> dict                  # Dict with m_start, m_end, m_mean, etc.
```

---

#### `DynamicThreshold`

Time-varying convergence threshold (Contribution 8).

```python
DynamicThreshold(
    eps_0: float = 1e-3,
    beta: float = 5.0,
    max_iter: int = 300,
    min_eps: float = 1e-8,
)
```

**Methods:**
```python
__call__(t: int) -> float          # epsilon(t) for iteration t
schedule() -> ndarray              # Full schedule
summary() -> dict                  # Dict with eps_start, eps_end, etc.
```

---

#### `EarlyStopping`

Patience-based early stopping (Contribution 7).

```python
EarlyStopping(
    patience: int = 10,
    min_delta: float = 0.0,
    verbose: bool = True,
)
```

**Methods:**
```python
reset()                            # Reset state
check(change: float, threshold: float, iteration: int) -> bool  # True if stop
summary() -> dict                  # State summary
stopped -> bool                    # Property: whether stopped
```

---

#### `DensityInitializer`

Density-based center initialization (Contribution 2).

```python
DensityInitializer(
    n_clusters: int,
    density_percentile: float = 90.0,
    subsample_size: int = 1000,
    random_state: int = 42,
)
```

**Methods:**
```python
initialize(X: ndarray) -> ndarray  # Returns (n_clusters, n_features) centers
```

---

#### `KMeansPlusPlusInitializer`

KMeans++ initialization (Contribution 1).

```python
KMeansPlusPlusInitializer(
    n_clusters: int,
    random_state: int = 42,
)
```

**Methods:**
```python
initialize(X: ndarray) -> ndarray  # Returns (n_clusters, n_features) centers
```

---

#### `RandomInitializer`

Random center selection.

```python
RandomInitializer(
    n_clusters: int,
    random_state: int = 42,
)
```

**Methods:**
```python
initialize(X: ndarray) -> ndarray
```

---

#### `AutomaticClusterDiscovery`

Automatic determination of optimal K via multi-index consensus (Contribution 5).

```python
AutomaticClusterDiscovery(random_state: int = 42)
```

**Methods:**
```python
search(X: ndarray, k_range: iterable, base_estimator: ADEFCM | None = None) -> dict
```
Evaluate all K in `k_range`. Returns `{k: metrics_dict}`.

```python
consensus_search(X: ndarray, k_range: iterable, base_estimator=None) -> int
```
Vote across Silhouette, DB, BIC, Gap. Returns best K (mode).

```python
evaluate_k(X: ndarray, k: int, base_estimator=None) -> dict
```
Evaluate a single K.

```python
elbow_curve(X: ndarray, k_range: iterable, base_estimator=None) -> dict
```
Return within-cluster SSE for each K.

**Attributes:**
- `results_` — `dict`. Stored results from last `search()` call.

---

#### `ClusterEvaluator`

Compute internal validation indices for a given partition.

```python
ClusterEvaluator(X: ndarray, labels: ndarray, centers: ndarray)
```

**Methods:**
```python
silhouette_score() -> float            # [-1, 1], higher better
davies_bouldin_score() -> float        # [0, inf), lower better
gap_statistic(n_reference=50) -> float # Higher gap → better K
bic_score() -> float                   # More negative → better
evaluate_all() -> dict                 # All indices at once
```

---

#### `OutlierDetector`

Outlier detection for fuzzy clustering results (Contribution 6).

```python
OutlierDetector(
    method: str = "weighted_distance",
    threshold_multiplier: float = 2.0,
    contamination: float | None = None,
)
```

**Methods:**
```python
fit(X, U, centers, m=2.0) -> OutlierDetector
```
Compute outlier scores and flag outliers.

```python
predict(X=None, U=None, centers=None, scores=None) -> tuple[ndarray, ndarray]
```
Return `(mask, scores)` for given data or precomputed scores.

```python
outlier_summary() -> dict
```
Return count, ratio, threshold, score statistics.

**Attributes:**
- `outlier_mask_` — `ndarray[bool]`
- `outlier_scores_` — `ndarray`
- `threshold_` — `float`

---

#### `SparkADEFCM`

Distributed ADE-FCM using PySpark (Contribution 10).

```python
SparkADEFCM(
    n_clusters: int | str = "auto",
    max_iter: int = 300,
    m: float | str = "adaptive",
    epsilon: float | str = "dynamic",
    init_method: str = "kmeans++",
    early_stopping_patience: int = 10,
    outlier_threshold: float = 2.0,
    random_state: int = 42,
    verbose: bool = True,
    spark_master: str = "local[*]",
    checkpoint_dir: str | None = None,
    num_partitions: int | None = None,
)
```

**Methods:**
```python
fit(X: ndarray, y=None) -> SparkADEFCM
```
Distributed fitting. Initializes Spark session automatically.

```python
predict(X: ndarray) -> ndarray
```
Distributed prediction.

```python
stop()
```
Stop the Spark session.

**Attributes:** Same as `ADEFCM`.

---

### Functions

```python
ade_fcm_pipeline(X, n_clusters="auto", max_iter=300, m="adaptive",
                 epsilon="dynamic", init_method="kmeans++",
                 early_stopping_patience=10, outlier_threshold=2.0,
                 random_state=42, verbose=True, use_spark=False,
                 spark_master="local[*]", spark_checkpoint_dir=None,
                 explain=True, feature_names=None) -> dict
```
Full pipeline returning `{"labels", "centers", "U", "n_iter", "n_clusters",
"J_history", "outlier_mask", "outlier_scores", "feature_importances",
"summaries", "descriptions", "explanation", "outlier_indices", "outlier_details"}`.

```python
explain_clusters(X, labels, centers, feature_names=None, outlier_mask=None) -> dict
```
Full XAI explanation: summaries, descriptions, global feature importance.

```python
feature_importance(X, labels, centers, method="shift") -> ndarray
```
Per-cluster feature importance. Methods: `"shift"`, `"fisher"`, `"permutation"`.

```python
cluster_summary(X, labels, centers, feature_names=None) -> list[dict]
```
Detailed per-cluster statistics.

```python
describe_cluster_natural(summary, feature_names=None) -> str
```
Human-readable cluster description.

```python
shap_explain(X, labels, centers, sample_idx=0) -> tuple[ndarray, str]
```
Approximate SHAP explanation for a single point.

---

## `baseline_project` — FCM / FCLM Baselines

### Classes

#### `DataLoader`

```python
DataLoader()
load_weblog_data(filepath: str) -> DataFrame
load_synthetic_data(n_samples=1000, n_features=10, n_clusters=5,
                    random_state=42, noise=0.05) -> tuple[ndarray, ndarray]
load_benchmark_dataset(name="iris") -> tuple[ndarray, ndarray]
```

#### `Preprocessor`

```python
Preprocessor()
clean_weblog_data(df) -> DataFrame          # Remove images, robots, errors
identify_users(df) -> DataFrame             # Assign user_id from IP
identify_sessions(df, timeout_minutes=30) -> DataFrame  # TOH1 heuristic
reduce_dimensions(df, min_support=1) -> DataFrame
assign_session_weights(df) -> DataFrame
build_session_matrix(df) -> tuple[csr_matrix, list]  # Session-page matrix
normalize(matrix) -> csr_matrix              # L_inf per column
standardize(X) -> ndarray                    # Z-score
min_max_scale(X) -> ndarray                  # [0, 1] scaling
```

#### `MembershipUpdater`

```python
MembershipUpdater()
update_fcm(X, centers, m=2.0) -> ndarray     # FCM membership matrix
update_fclm(X, centers, m=2.0) -> ndarray    # Same as FCM
initialize_random(n, c, seed=42) -> ndarray
initialize_uniform(n, c) -> ndarray
```

#### `ClusterUpdater`

```python
ClusterUpdater()
update_centers_fcm(X, U, m=2.0) -> ndarray   # Weighted mean centers
update_centers_fclm(X, U, m=2.0) -> ndarray  # Median-based centers
initialize_kmeans_plus_plus(X, n, seed=42) -> ndarray
initialize_random(X, n, seed=42) -> ndarray
```

#### `ObjectiveFunction`

```python
ObjectiveFunction()
compute_fcm(X, U, centers, m=2.0) -> float   # J objective
compute_fclm(X, U, centers, m=2.0) -> float  # Same as FCM
compute_partition_coefficient(U) -> float     # PC
compute_partition_entropy(U) -> float         # PEC
compute_sse(X, labels, centers) -> float      # Sum of squared errors
```

#### `ConvergenceChecker`

```python
ConvergenceChecker()
check(U_new, U_old, epsilon=1e-5) -> bool    # Frobenius norm check
compute_change(U_new, U_old) -> float         # Frobenius norm
check_objective(J_new, J_old, epsilon=1e-6) -> bool
early_stopping(J_history, patience=5, min_delta=1e-6) -> bool
```

#### `Evaluator`

```python
Evaluator()
silhouette_score(X, labels) -> float
davies_bouldin_index(X, labels) -> float
calinski_harabasz_score(X, labels) -> float
adjusted_rand_index(y_true, y_pred) -> float
normalized_mutual_info(y_true, y_pred) -> float
rand_index(y_true, y_pred) -> float
compute_all(X, labels, y_true=None) -> dict
execution_time(func, *args, **kwargs) -> tuple[result, elapsed]
```

#### `SparkFCMEngine` / `SparkFCLMEngine`

```python
SparkFCMEngine(spark=None, n_clusters=5, max_iter=100, m=2.0, epsilon=1e-5)
fit(df, feature_columns) -> tuple[engine, centers, U, J_history]
predict(df, feature_columns) -> ndarray

SparkFCLMEngine(spark=None, n_clusters=5, max_iter=100, m=2.0, epsilon=1e-5)
# Same API, median-based center updates
```

#### `Visualizer`

```python
Visualizer(output_dir="plots", dpi=150, style="seaborn")
plot_clusters_2d(X, labels, centers, title, save_name, method="pca")
plot_convergence(J_history, title, save_name)
plot_metrics_comparison(fcm_metrics, fclm_metrics, title, save_name)
plot_confusion_matrix(y_true, y_pred, title, save_name)
plot_combined_results(X, labels_fcm, labels_fclm, centers_fcm, centers_fclm,
                      J_fcm, J_fclm, metrics_fcm, metrics_fclm, y_true, ...)
plot_membership_heatmap(U, title, save_name)
plot_cluster_sizes(labels, title, save_name)
plot_elbow_curve(inertias, max_k, title, save_name)
plot_silhouette_analysis(X, labels, title, save_name)
```

---

## `benchmarks` — Benchmark Suite

### Classes

#### `BenchmarkRunner`

```python
BenchmarkRunner(random_state=42, results_dir=None)
run_single(algorithm_name: str, dataset_name: str) -> dict
run_all() -> DataFrame
compare_algorithms(datasets=None, metrics=None) -> DataFrame
```

Registered algorithms: `"KMeans"`, `"MiniBatchKMeans"`, `"FCM"`, `"FCLM"`,
`"ADE-FCM"`, `"SpectralClustering"`, `"DBSCAN"`, `"OPTICS"`, `"BIRCH"`,
`"AgglomerativeClustering"`, `"GaussianMixture"`.

Datasets: `"blobs"`, `"moons"`, `"circles"`, `"varied"`, `"iris"`, `"wine"`,
`"digits"`, `"breast_cancer"`.

#### `MetricsCollector`

```python
MetricsCollector()
collect_all(X, labels, y_true=None, centers=None) -> dict
collect_clustering_metrics(X, labels) -> dict
collect_classification_metrics(y_true, y_pred) -> dict
measure_time(func) -> tuple[result, elapsed]
measure_memory(func) -> tuple[result, memory_mb]
```

#### `ResultsAnalyzer`

```python
ResultsAnalyzer(results_dir=None)
load_results(path=None) -> DataFrame
rank_algorithms(df, metric, ascending=False) -> DataFrame
statistical_significance(df, metric, method="wilcoxon") -> DataFrame
generate_latex_table(comparison_df) -> str
generate_report(df, output_path) -> str
```

#### `BenchmarkPlotter`

```python
BenchmarkPlotter(output_dir="results/plots")
plot_comparison_bar(df, metric, title=None) -> Path
plot_scalability(sizes, times, algorithms, title=None) -> Path
plot_radar_chart(df, metrics, title=None) -> Path
generate_all_plots(df) -> dict[str, Path]
```

---

## `streaming` — Real-Time Streaming

### Classes

#### `OnlineFCM`

```python
OnlineFCM(n_clusters=5, m=2.0, learning_rate=0.3, random_state=42)
partial_fit(X, y=None) -> OnlineFCM     # Incremental update
fit(X, y=None) -> OnlineFCM             # Fit via mini-batches
predict(X) -> ndarray                    # Cluster assignment
get_membership(X) -> ndarray            # Membership matrix
reset()                                 # Clear state
```

#### `DataProducer`

```python
DataProducer(bootstrap_servers="localhost:9092", topic="clustering-data")
produce_batch(X, batch_size=100, sleep=0.1, key_func=None)
produce_stream(X, interval=0.5)
close()
```

#### `ADEFCMStreaming`

```python
ADEFCMStreaming(spark=None, n_clusters=5, model=None)
create_kafka_stream(bootstrap_servers, topic) -> DataFrame
online_update(micro_batch_df, batch_id)
start_streaming(bootstrap_servers, topic, output_mode, trigger_interval) -> StreamingQuery
start_console_stream(bootstrap_servers, topic, ...) -> StreamingQuery
```

#### `StreamingPipeline`

```python
StreamingPipeline(n_clusters=5, fuzzifier=2.0, learning_rate=0.3,
                  bootstrap_servers="localhost:9092", topic="clustering-data",
                  checkpoint_dir="/tmp/ade-fcm-checkpoint",
                  spark_master="local[*]", use_spark=False)
setup_model()
setup_producer()
setup_spark_consumer()
run_producer(data_path, batch_size=100, interval=0.1, stream_mode="batch")
run_online(data_path, batch_size=100)
run_spark_foreground(data_path=None)
```

---

## `big_data` — Large-Scale Distributed FCM

### Classes

#### `LargeScaleFCM`

```python
LargeScaleFCM(spark_session, n_clusters, max_iter=100, m=2.0, epsilon=1e-5,
              chunk_size=50000, cache_storage_level="MEMORY_AND_DISK",
              use_rdd=True, use_dataframe=False, checkpoint_dir=None, verbose=True)
fit_sequential(X) -> tuple[centers, U, J_history]
fit_rdd(X) -> tuple[centers, U, J_history]
fit_dataframe(df, feature_columns=None) -> tuple[centers, U, J_history]
fit_sql(df, temp_view="fcm_data") -> tuple[centers, U, J_history]
get_metadata() -> dict
```

#### `SparkRDDOptimizer`

```python
SparkRDDOptimizer(spark_session)
optimize_partitioning(rdd, target_partitions) -> RDD
cache_strategy(rdd, storage_level) -> RDD
checkpoint(rdd, checkpoint_dir) -> RDD
broadcast_centers(sc, centers) -> Broadcast
tree_aggregate(rdd, seq_op, comb_op, depth=2) -> any
map_partitions_with_centers(rdd, centers_bc, func) -> RDD
fault_tolerant_compute(rdd, n_retries=3) -> list
```

#### `SparkDataFrameOptimizer`

```python
SparkDataFrameOptimizer(spark_session)
optimize_with_sql(df, temp_view_name, sql_query) -> DataFrame
adaptive_query_execution(enabled=True) -> SparkSession
broadcast_hash_join(df1, df2, join_key) -> DataFrame
partition_by_column(df, column, num_partitions) -> DataFrame
cache_dataframe(df) -> DataFrame
optimize_shuffle(conf, shuffle_partitions=200)
dynamic_allocation(enabled=True, min_executors=1, max_executors=10) -> SparkSession
```

#### `ChunkedMembershipUpdate`

```python
ChunkedMembershipUpdate(chunk_size=50000, m=2.0)
compute(X_chunk, centers) -> ndarray
```

#### `ChunkedCenterUpdate`

```python
ChunkedCenterUpdate(m=2.0)
compute(X_chunk, U_chunk) -> tuple[numerator, denominator]
aggregate(partials) -> ndarray  # Weighted mean centers
```

---

## `gpu` — GPU Acceleration

### Classes

#### `GPUFCMManager`

```python
GPUFCMManager(use_gpu=True, n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42)
fit(X) -> tuple[centers, U, J_history]
predict(X) -> ndarray
benchmark_cpu_vs_gpu(X, n_runs=3) -> dict  # {"cpu": {}, "gpu": {}, "speedup": float}
get_memory_report() -> dict
to_gpu(X) -> cupy.ndarray
to_cpu(X) -> ndarray
```

**Properties:** `centers`, `U`, `J_history`, `fit_time`.

#### `RAPIDSFCM`

```python
RAPIDSFCM(n_clusters=5, max_iter=100, m=2.0, tol=1e-4, seed=42)
fit_rapids_kmeans(X) -> tuple[centers, U, J_history]  # cuML KMeans + fuzzy
fit_fuzzy(X) -> tuple[centers, U, J_history]           # Full GPU FCM
predict(X, use_rapids=False) -> ndarray
silhouette_score(X, labels, use_rapids=True) -> float
pairwise_distances_gpu(X, Y=None, metric="euclidean") -> ndarray
to_gpu(X) -> cupy.ndarray
to_cpu(X) -> ndarray
```

**Properties:** `centers`, `U`, `J_history`.

#### `SparkGPUHybridEngine`

```python
SparkGPUHybridEngine(spark_session=None, n_clusters=5, max_iter=100, m=2.0,
                     tol=1e-4, seed=42, spark_mode="spark_gpu")
fit_spark_gpu(df) -> tuple[centers, U]
fit_spark_cpu(df) -> tuple[centers, U]
fit_gpu(df) -> tuple[centers, U]
fit_cpu(df) -> tuple[centers, U]
compare_modes(df, include_cpu=True, include_spark_cpu=True,
              include_gpu=True, include_spark_gpu=True) -> DataFrame
stop_spark()
```

**Properties:** `centers`, `U`, `results` (dict of mode→time).

#### `FCMKernels` (CUDA kernel factory)

```python
FCMKernels()
get_membership_update_kernel() -> ElementwiseKernel
get_center_update_kernel() -> RawKernel        # Grid: n_clusters, Block: n_features
get_distance_kernel() -> ElementwiseKernel
get_objective_kernel() -> ReductionKernel
```

#### Kernel functions:

```python
check_cuda()
compute_membership_gpu(distances, m) -> cupy.ndarray
compute_centers_gpu(data, U, fuzziness) -> cupy.ndarray
compute_distances_gpu(data, centers) -> cupy.ndarray
compute_objective_gpu(U, distances, m) -> float
```

---

## `xai` — Explainable AI

### Classes

#### `ClusterExplainer`

```python
ClusterExplainer(model=None, X=None, feature_names=None)
feature_importance(cluster_id) -> dict       # Per-cluster feature importances
cluster_summary(cluster_id) -> dict          # Stats: size, mean, std, min, max
global_explanation() -> list[dict]           # All clusters
natural_language_description(cluster_id) -> str  # Human-readable
generate_report(output_path=None) -> dict    # Full JSON report
```

#### `XAIVisualizer`

```python
XAIVisualizer(model=None, X=None, feature_names=None, output_dir="xai_plots")
plot_feature_importance(cluster_id, top_n=10, save=True) -> Figure
plot_all_feature_importance(top_n=10) -> list[Path]
plot_radar(cluster_id, save=True) -> Figure
plot_all_radars() -> list[Path]
plot_parallel_coordinates(save=True) -> Path
plot_scatter(save=True) -> Path
plot_outlier_analysis(save=True) -> Path
plot_membership_heatmap(save=True) -> Path
generate_all_plots() -> dict[str, Path | None]
```

#### `ShapExplainer`

```python
ShapExplainer(model, X, feature_names=None, nsamples=200)
fit()
global_shap_summary() -> list[dict]
plot_summary(save_path=None) -> Figure
```

---

## `ablation` — Ablation Study

### Classes

#### `AblationStudy`

```python
AblationStudy(X, y_true=None, n_clusters=5, random_state=42)
run_full_ade_fcm() -> tuple[ADEFCM, dict]       # Full model + metrics
run_without_adaptive_fuzzifier() -> dict         # Fixed m=2.0
run_without_auto_k() -> dict                     # Fixed K
run_without_explainability() -> dict             # Skip XAI
run_without_outlier_robustness() -> dict         # No outlier detection
run_without_early_stopping() -> dict             # Full max_iter
run_all() -> dict                                # All experiments
generate_report(output_path="ablation_report.json") -> dict
```

Report dict structure:
```python
{
    "full_ade_fcm": {"silhouette": ..., "davies_bouldin": ..., "time": ..., "iterations": ..., "objective": ...},
    "ablation_results": {
        "without_adaptive_fuzzifier": {...},
        "without_auto_k": {...},
        ...
    },
    "degradation": {
        "without_adaptive_fuzzifier": {"silhouette": -5.2, "davies_bouldin": 8.1, "time": -12.3},
        ...
    }
}
```
