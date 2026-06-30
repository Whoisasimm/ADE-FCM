# Architecture Diagrams

## 1. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ADE-FCM System Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      PRESENTATION LAYER                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Python   │  │  CLI     │  │ Jupyter  │  │ HTML Reports     │   │   │
│  │  │ API      │  │  Commands│  │ Notebooks│  │ (XAI, Ablation)  │   │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │   │
│  └───────┼─────────────┼─────────────┼──────────────────┼─────────────┘   │
│          │             │             │                  │                 │
│  ┌───────┴─────────────┴─────────────┴──────────────────┴─────────────┐   │
│  │                      APPLICATION LAYER                              │   │
│  │                                                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │                   ADE-FCM Core Algorithm                     │   │   │
│  │  │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │   │   │
│  │  │  │ KMeans++   │  │ Density-     │  │ Adaptive Fuzzifier │   │   │   │
│  │  │  │ Init (C1)  │  │ Based Init   │  │ m(t) (C3)          │   │   │   │
│  │  │  └────────────┘  │ (C2)         │  └────────────────────┘   │   │   │
│  │  │                  └──────────────┘                            │   │   │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │   │   │
│  │  │  │ Confidence-    │  │ Auto Cluster   │  │ Outlier-Robust │ │   │   │
│  │  │  │ Weighted Mem   │  │ Discovery (C5) │  │ Membership (C6)│ │   │   │
│  │  │  │ (C4)           │  │                │  │                │ │   │   │
│  │  │  └────────────────┘  └────────────────┘  └────────────────┘ │   │   │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │   │   │
│  │  │  │ Early Stopping │  │ Dynamic Conv   │  │ XAI            │ │   │   │
│  │  │  │ (C7)           │  │ Threshold (C8) │  │ (C9)           │ │   │   │
│  │  │  └────────────────┘  └────────────────┘  └────────────────┘ │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ Pipeline │  │ Baseline │  │ Ablation │  │ Benchmarks       │   │   │
│  │  │ Module   │  │ Project  │  │ Study    │  │ Suite            │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     COMPUTATION LAYER                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │   │
│  │  │ Single-Node  │  │ Apache Spark │  │ GPU (CuPy)   │  │ RAPIDS │ │   │
│  │  │ NumPy/CPU    │  │ Distributed  │  │ CUDA Kernels │  │ cuML   │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     DATA LAYER                                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ sklearn  │  │ CSV/TSV  │  │ Parquet  │  │ Kafka Stream     │   │   │
│  │  │ Datasets │  │ Weblog   │  │ HDFS     │  │ Topics           │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Pipeline

```
                           DATA FLOW PIPELINE

  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  Input   │────▶│  Pre-    │────▶│  Cluster │────▶│  Post-   │
  │  Data    │     │  process │     │  (ADE-   │     │  process │
  └──────────┘     └──────────┘     │  FCM)    │     └──────────┘
        │               │           └──────────┘           │
        │               │                │                 │
        ▼               ▼                ▼                 ▼
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ sklearn  │     │ Standard │     │ Fit ADE- │     │ XAI      │
  │ CSV      │     │ -ize     │     │ FCM      │     │ Reports  │
  │ Parquet  │     │ PCA      │     │ Discover │     │ Visual-  │
  │ Kafka    │     │ Session  │     │ K        │     │ izations │
  └──────────┘     │ Matrix   │     │ Outlier  │     │ Metrics  │
                   └──────────┘     │ Detect   │     │ Export   │
                                    └──────────┘     └──────────┘


  WITHIN CLUSTERING ITERATION:

   ┌──────────────┐
   │ 1. Init      │  KMeans++ / Density / Random
   │  Centers     │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ 2. Adaptive  │  m(t) = m_min + (m_max-m_min)*exp(-alpha*t/T)
   │  Fuzzifier   │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ 3. Update    │  U_ij = 1 / sum_k (d_ij/d_ik)^(2/(m(t)-1))
   │  Membership  │  + Confidence weighting
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ 4. Update    │  v_j = sum_i u_ij^m(t) * x_i / sum_i u_ij^m(t)
   │  Centers     │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ 5. Check     │  ||U_new - U_old||_F < epsilon(t)
   │  Converged   │  + patience counter (early stopping)
   └──────┬───────┘
          │
     ┌────┴────┐
     │         │
    YES        NO
     │         │
     ▼         └──────→ back to step 2
   ┌──────────────┐
   │ 6. Outlier   │  O_i = sum_j u_ij^m * d(x_i, v_j)
   │  Detection   │  Flag if O_i > mean + threshold*std
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ 7. XAI       │  Feature importance, cluster summaries,
   │  Explain     │  SHAP values, NL descriptions
   └──────────────┘
```

---

## 3. Spark Cluster Architecture

```
                    ┌─────────────────────────┐
                    │    Driver Node           │
                    │  ┌─────────────────────┐ │
                    │  │  SparkContext        │ │
                    │  │  ADE-FCM Driver      │ │
                    │  │  ┌───────────────┐   │ │
                    │  │  │ SparkADEFCM   │   │ │
                    │  │  │ .fit()        │   │ │
                    │  │  └───────┬───────┘   │ │
                    │  └──────────┼──────────┘ │
                    └─────────────┼────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
  ┌───────────┐            ┌───────────┐            ┌───────────┐
  │  Worker 1 │            │  Worker 2 │            │  Worker N │
  │           │            │           │            │           │
  │ ┌───────┐ │            │ ┌───────┐ │            │ ┌───────┐ │
  │ │Executor│ │            │ │Executor│ │            │ │Executor│ │
  │ │ 4 cores│ │            │ │ 4 cores│ │            │ │ 4 cores│ │
  │ │ 8 GB   │ │            │ │ 8 GB   │ │            │ │ 8 GB   │ │
  │ └───────┘ │            │ └───────┘ │            │ └───────┘ │
  │     │     │            │     │     │            │     │     │
  │ ┌───┴───┐ │            │ ┌───┴───┐ │            │ ┌───┴───┐ │
  │ │ RDD   │ │            │ │ RDD   │ │            │ │ RDD   │ │
  │ │Partition│ │            │ │Partition│ │            │ │Partition│ │
  │ └───────┘ │            │ └───────┘ │            │ └───────┘ │
  └───────────┘            └───────────┘            └───────────┘

  SPARK OPTIMIZATIONS FOR ADE-FCM:

  ┌────────────────────────────────────────────────────────────┐
  │ broadcast(centers)     Map each partition to all workers   │
  │                                                           │
  │ mapPartitions(update)  Compute U locally per partition    │
  │                                                           │
  │ treeAggregate(center)  Combine partial numerators/denoms  │
  │   depth=2              Efficient distributed reduction     │
  │                                                           │
  │ checkpoint(dir)        Fault tolerance for long jobs       │
  └────────────────────────────────────────────────────────────┘

  DISTRIBUTED CENTER UPDATE PROTOCOL:

  1. Broadcast current centers to all workers
  2. Each partition: compute distances → membership U → partial stats
  3. treeAggregate: combine (sum U^m * x, sum U^m) across partitions
  4. Driver: new_centers = numerator_total / denominator_total
  5. Repeat until convergence
```

---

## 4. Streaming Pipeline

```
  KAFKA STREAMING PIPELINE

  ┌──────────────┐
  │ Data Source  │
  │ (CSV file    │
  │  or sensor)  │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────────┐
  │  Kafka       │────▶│  Kafka Topic     │
  │  Producer    │     │  "clustering-    │
  │  (batch or   │     │  data"           │
  │   stream)    │     └────────┬─────────┘
  └──────────────┘              │
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │  Direct Online   │  │  Spark Structured│  │  Console /       │
  │  (NumPy)         │  │  Streaming       │  │  Debug           │
  │                  │  │                  │  │                  │
  │  OnlineFCM       │  │  ADEFCMStreaming │  │  writeStream     │
  │  .partial_fit()  │  │  .foreachBatch() │  │  .format(console)│
  │                  │  │                  │  │                  │
  │  Per-batch:      │  │  Per micro-batch:│  │  For monitoring  │
  │  ┌──────────┐   │  │  ┌────────────┐  │  └──────────────────┘
  │  │Compute U │   │  │  │Collect pts │  │
  │  │Update    │   │  │  │Compute U   │  │
  │  │centers   │   │  │  │Update      │  │
  │  └──────────┘   │  │  │centers     │  │
  └──────────────────┘  │  └────────────┘  │
                         └──────────────────┘


  ONLINE FCM UPDATE RULE:
  (per mini-batch)

  centers_j_new = (1 - lr) * centers_j_old + lr * batch_center_j

  where:
    batch_center_j = sum_i u_ij^m * x_i / sum_i u_ij^m
    lr = learning_rate (default 0.3)
```

---

## 5. GPU Acceleration Pipeline

```
  GPU ACCELERATION ARCHITECTURE

  ┌─────────────────────────────────────────────────────────────┐
  │                      HOST (CPU)                             │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │  GPUFCMManager                                       │   │
  │  │                                                      │   │
  │  │  ┌──────────┐    ┌──────────┐    ┌────────────────┐ │   │
  │  │  │to_gpu()  │───▶│_fit_gpu()│───▶│to_cpu()        │ │   │
  │  │  │X→cupy    │    │          │    │result→numpy    │ │   │
  │  │  └──────────┘    └────┬─────┘    └────────────────┘ │   │
  │  └───────────────────────┼──────────────────────────────┘   │
  └──────────────────────────┼──────────────────────────────────┘
                             │ PCIe
  ┌──────────────────────────┼──────────────────────────────────┐
  │                      GPU (Device)                           │
  │  ┌───────────────────────┴──────────────────────────────┐   │
  │  │                  CuPy / CUDA Kernels                 │   │
  │  │                                                      │   │
  │  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │   │
  │  │  │distances │  │membership│  │center_update        │ │   │
  │  │  │ GPU      │──▶│ GPU      │──▶│ GPU (RawKernel)   │ │   │
  │  │  │ cdist    │   │ElemKernel│  │ blocks=n_clusters  │ │   │
  │  │  └──────────┘  └──────────┘  │ threads=n_features  │ │   │
  │  │                              └────────────────────┘ │   │
  │  │                                                      │   │
  │  │  ┌────────────────────────────────────────────────┐ │   │
  │  │  │  Loop until convergence (max_iter iterations)   │ │   │
  │  │  │  All operations stay on GPU, no CPU copies      │ │   │
  │  │  └────────────────────────────────────────────────┘ │   │
  │  └──────────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────┘


  CUDA KERNEL STRUCTURE:

  ┌─────────────────────────────────────────────────────────────┐
  │  // ElementwiseKernel (per element)                         │
  │  membership_update:                                          │
  │    membership = pow(dist, inv_exponent)                      │
  │                                                              │
  │  // RawKernel (block = feature, grid = cluster)             │
  │  center_update:                                              │
  │    for each sample:                                          │
  │        num += u * data[f]                                    │
  │        den += u                                              │
  │    new_center[f] = num / den                                 │
  │                                                              │
  │  // ReductionKernel                                          │
  │  objective:                                                  │
  │    sum_i val_i  where val = (U^m) * distances                │
  └─────────────────────────────────────────────────────────────┘


  SPARK + GPU HYBRID MODE:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Partition│    │ Partition│    │ Partition│    │ Partition│
  │ 1        │    │ 2        │    │ 3        │    │ 4        │
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │               │
  ┌────▼────┐     ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
  │ GPU     │     │ GPU     │     │ GPU     │     │ GPU     │
  │ FCM     │     │ FCM     │     │ FCM     │     │ FCM     │
  │ on-part │     │ on-part │     │ on-part │     │ on-part │
  │ centers │     │ centers │     │ centers │     │ centers │
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │               │               │               │
       └───────────────┼───────────────┼───────────────┘
                       │               │
                  ┌────▼───────────────▼────┐
                  │  Collect partition       │
                  │  centers → aggregate     │
                  │  → final centers (CPU)   │
                  └─────────────────────────┘
```
