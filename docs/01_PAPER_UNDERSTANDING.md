# Paper Understanding: Parallel Fuzzy C-Median Clustering Using Spark for Big Data

## Executive Summary

The paper by Mallik et al. (2024, IEEE Access) addresses the challenge of clustering big data by proposing a **Parallel Fuzzy C-Median (FCLM) Clustering Algorithm** on Apache Spark. The algorithm replaces the traditional mean-based centroid computation with a median-based approach to eliminate mean squared error and reduce outlier impact. Implemented on Databricks cloud platform using PySpark, the method achieves near-ideal silhouette scores (~1.0) with optimal cluster count k=5 on weblog datasets, outperforming MiniBatchKMeans, AffinityPropagation, SpectralClustering, Ward, OPTICS, and BIRCH in computational time.

## Mathematical Breakdown

### 1. Fuzzy C-Means (FCM) Foundation

**Objective Function:**
```
J = Σᵢ₌₁ᴺ Σⱼ₌₁ᶜ Uᵢⱼᵐ · d(xᵢ, cⱼ)
```

**Constraints:**
- Σⱼ₌₁ᶜ Uᵢⱼ = 1, ∀i (membership sums to 1)
- 0 ≤ Uᵢⱼ ≤ 1 (fuzzy membership)
- m > 1 (fuzzifier parameter)

### 2. Membership Update (Original FCM)
```
Uᵢⱼ = 1 / Σₖ₌₁ᶜ (dᵢⱼ / dᵢₖ)^(2/(m-1))
```

### 3. Cluster Center Update (FCM - Mean based)
```
Cⱼ = (Σᵢ₌₁ᴺ Uᵢⱼᵐ · xᵢ) / (Σᵢ₌₁ᴺ Uᵢⱼᵐ)
```

### 4. Fuzzy C-Median (FCLM) - Paper's Novelty

**Membership Update (same as FCM):**
```
Uᵢⱼ = 1 / Σₖ₌₁ᶜ (dᵢⱼ / dᵢₖ)^(2/(m-1))
```

**Distance to New Cluster Centers:**
```
Dᵢ = Median{(Dᵢⱼ(Sₖ - Sᵢ) · Uᵢⱼ)}  ∀i ≠ k; k = 1...n
```

**New Center Selection:**
```
p = Argmin{(Dᵢ : n); ∀i = 1...n}
```

**Cluster Center Update:**
```
Vⱼ = (Σᵢ₌₁ᴺ Uᵢⱼᵐ · xᵢ) / (Σᵢ₌₁ᴺ Uᵢⱼᵐ)
```

### 5. Distance Metrics Used
- **Manhattan (L1):** d(x, y) = Σ|x - y|
- Used for median-based clustering (robust to outliers)

### 6. Convergence Condition
```
||U⁽ᵏ⁺¹⁾ - U⁽ᵏ⁾|| < ε
where ε ∈ [0, 1] is the termination criterion
```

### 7. Evaluation Metrics
**Partition Coefficient (PC):**
```
PC = (1/n) · Σⱼ₌₁ᵏ Σᵢ₌₁ᴺ Uⱼᵢᵐ
```

**Partition Entropy Coefficient (PEC):**
```
PEC = -(1/n) · Σⱼ₌₁ᵏ Σᵢ₌₁ᴺ Uⱼᵢ · log(Uⱼᵢ)
```

**Silhouette Coefficient:**
```
S(i) = (b(i) - a(i)) / max{a(i), b(i)}
```
where a(i) = intra-cluster distance, b(i) = inter-cluster distance

**Cost Function:**
```
J = Σᵢ₌₁ᴺ Σⱼ₌₁ᶜ Uᵢⱼᵐ · d(xᵢ, cⱼ)
```

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: Raw Weblog Data                  │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: DATA PREPROCESSING                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │Data Cleaning│→│User ID Setup │→│Session ID (TOH)│  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐ │
│  │Dim Reduction  │→│Session Weight │→│Session Matrix  │ │
│  └──────────────┘  └───────────────┘  └───────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 2: SAMPLING                            │
│         Random sample subset of dataset                    │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 3: PARTITIONING                        │
│      Divide data into partitions across Spark nodes        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 4: PARALLEL FCLM                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Initialize membership matrix U randomly          │   │
│  │  Broadcast cluster centers V to all nodes         │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │  WHILE not converged:                             │   │
│  │    Map: Compute Uij using Equation (3)            │   │
│  │    MapPartitions: Compute local Vi                │   │
│  │    Reduce: Aggregate Vi to global V               │   │
│  │    Check convergence: ||U(k+1) - U(k)|| < ε      │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              PHASE 5: EVALUATION                          │
│  ┌──────────┐ ┌─────────┐ ┌───────┐ ┌──────┐ ┌───────┐ │
│  │Silhouette│ │PC / PEC │ │Rand   │ │F-Meas│ │SSE    │ │
│  │Score     │ │         │ │Index  │ │ure   │ │       │ │
│  └──────────┘ └─────────┘ └───────┘ └──────┘ └───────┘ │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              OUTPUT: Cluster Assignments + Metrics        │
└─────────────────────────────────────────────────────────┘
```

## Apache Spark Architecture (as implemented in paper)

```
┌───────────────────────────────────────────────────────────┐
│                    DRIVER NODE                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  SparkContext / SparkSession                         │  │
│  │  - Main program entry point                         │  │
│  │  - Creates RDDs, manages DAG                        │  │
│  │  - Broadcasts cluster centers                       │  │
│  │  - Aggregates results via Reduce                    │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────┬──────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     ↓               ↓               ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Worker 1 │   │ Worker 2 │   │ Worker N │
│ ┌──────┐ │   │ ┌──────┐ │   │ ┌──────┐ │
│ │RDD   │ │   │ │RDD   │ │   │ │RDD   │ │
│ │Part 1│ │   │ │Part 2│ │   │ │Part N│ │
│ └──────┘ │   │ └──────┘ │   │ └──────┘ │
│ ┌──────┐ │   │ ┌──────┐ │   │ ┌──────┐ │
│ │Exec  │ │   │ │Exec  │ │   │ │Exec  │ │
│ │tor   │ │   │ │tor   │ │   │ │tor   │ │
│ └──────┘ │   │ └──────┘ │   │ └──────┘ │
│ Map: Uij │   │ Map: Uij │   │ Map: Uij │
│ Part: Vi │   │ Part: Vi │   │ Part: Vi │
└──────────┘   └──────────┘   └──────────┘
     │               │               │
     └───────────────┼───────────────┘
                     ↓
            ┌─────────────────┐
            │  Reduce: V_new  │
            └─────────────────┘
```

## Complexity Analysis

### Time Complexity

| Component | Sequential | Parallel (Spark) |
|-----------|-----------|------------------|
| Membership Update | O(N·C·D·T) | O((N/P)·C·D·T) |
| Center Update | O(N·C·D·T) | O((N/P)·C·D·T) |
| Distance Computation | O(N·C·D·T) | O((N/P)·C·D·T) |
| Convergence Check | O(N·C·T) | O((N/P)·C·T) |
| **Overall** | **O(N·C·D·T)** | **O((N/P)·C·D·T)** |

Where:
- N = number of data points
- C = number of clusters
- D = number of dimensions
- T = number of iterations
- P = number of partitions/workers

### Space Complexity

| Component | Complexity |
|-----------|-----------|
| Data Storage (RDD) | O(N/P · D) per node |
| Membership Matrix | O(N/P · C) per node |
| Cluster Centers | O(C · D) broadcast |
| **Total per Node** | **O((N/P)·(D + C))** |

### Communication Complexity
- Broadcast: O(C·D) per iteration
- Reduce: O(C·D) per iteration
- Shuffle: O(N·C) per iteration

## Datasets Used
- **Source:** filewatcher.net → `pa.sanitized-access.20070109.gz`
- **Type:** Web server log data (FTP access logs)
- **Content:** IP addresses, timestamps, URLs, status codes, bytes transferred
- **Preprocessing:** Cleaning (remove graphics, robots, query strings), user identification, session identification (TOH1 with 30-min threshold), dimensionality reduction, session weighting, session matrix construction

## Limitations Identified (from paper + analysis)
1. Manual selection of cluster count K
2. Fixed fuzzifier m = 2
3. No automatic convergence threshold adaptation
4. No streaming data support
5. Single distance metric (Manhattan)
6. No GPU acceleration
7. No explainability module
8. No automatic parameter tuning
9. Limited to batch processing
10. No fault tolerance mechanisms beyond Spark defaults
