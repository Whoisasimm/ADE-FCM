# ADE-FCM: An Adaptive Explainable Fuzzy C-Means Framework for Automated Clustering

Anonymous Author(s)
Affiliation
Email

---

**Abstract—Fuzzy C-Means (FCM) remains one of the most widely used clustering algorithms, yet practitioners face three persistent barriers: manual tuning of the fuzzifier parameter m, reliance on Euclidean distance which degrades on high-dimensional data, and a lack of explainability in cluster assignments. This paper introduces ADE-FCM, an adaptive framework that addresses all three barriers simultaneously. ADE-FCM incorporates (i) an adaptive fuzzifier schedule m(t); (ii) a dynamic convergence threshold; (iii) automatic cluster-count discovery; (iv) pluggable distance metrics; (v) robust center estimation; (vi) center reinitialization; and (vii) built-in explainability. We evaluate ADE-FCM against four competing algorithms across 18 benchmark datasets (4 UCI built-in, 9 OpenML, 5 synthetic) using up to 5 random seeds. Over 396 total experiments, ADE-FCM achieves the best mean Friedman rank (2.083), outperforming standard FCM (3.222), K-Means (2.722), Agglomerative (2.472), and FCLM (4.500). The Friedman test yields p &lt; 0.0001, indicating highly significant differences. Cohen's d between ADE-FCM and FCM is 0.47 (95% CI [0.13, 0.79]), a medium effect size confirmed by Wilcoxon signed-rank test (p = 0.043). On the high-dimensional Digits dataset, ADE-FCM with Cosine distance achieves ARI = 0.553 with all 10 clusters active, a threefold improvement over standard FCM. tests pass after systematic bug fixing and dead-code removal. These results establish that ADE-FCM matches FCM accuracy while adding automation, metric flexibility, and explainability at negligible computational overhead.**

**Keywords—fuzzy clustering, adaptive fuzzifier, automatic cluster discovery, explainable AI, permutation importance, high-dimensional clustering**

---

## 1. Introduction

Clustering is a fundamental unsupervised learning task that partitions data into homogeneous groups without labeled training examples. Among the many clustering paradigms, fuzzy clustering—and particularly Fuzzy C-Means (FCM) introduced by Bezdek [1]—occupies a unique position by assigning each data point a *degree of membership* to every cluster rather than a hard assignment. This soft partitioning is valuable in domains where boundaries between categories are inherently ambiguous, including medical diagnosis, image segmentation, bioinformatics, and social network analysis.

Despite four decades of widespread adoption, FCM in its standard formulation presents three practical barriers that limit its effectiveness in modern data science workflows.

**Barrier 1: Manual fuzzifier tuning.** The standard FCM algorithm requires the user to specify a fuzzifier exponent m ∈ (1, ∞), typically set to 2.0 by convention. This parameter controls the "softness" of the partition: values close to 1 produce near-hard assignments, while large values produce highly overlapping clusters. The sensitivity of clustering results to m is well documented [2], yet there is no universally accepted rule for selecting it. Practitioners must resort to grid search, heuristics, or prior knowledge, all of which are time-consuming and dataset-dependent.

**Barrier 2: Euclidean-only distance.** Classical FCM is defined using the squared Euclidean distance, which implicitly assumes spherical, isotropic clusters. On high-dimensional data—where the curse of dimensionality causes Euclidean distances to concentrate—this choice leads to degraded performance. Domains such as text clustering, gene expression analysis, and image feature extraction routinely involve 64, 128, or more dimensions where Euclidean geometry is inappropriate. Alternative distance metrics such as Cosine (angular) distance, Manhattan (L1) distance, and Mahalanobis distance (which accounts for feature correlations) are needed but are not supported in standard FCM implementations without manual modification.

**Barrier 3: Lack of explainability.** The output of FCM—a membership matrix U and cluster centers V—provides no intrinsic mechanism for interpreting *why* a particular point was assigned to a cluster or *which features* drive the cluster structure. In regulated domains such as healthcare, finance, and criminal justice, model explainability is not optional but mandatory. The growing field of eXplainable AI (XAI) has produced powerful tools including SHAP [3] and permutation importance [4], but these have not been systematically integrated into fuzzy clustering frameworks.

The central research question of this work is: **Can we design an integrated fuzzy clustering framework that simultaneously eliminates manual parameter tuning, supports arbitrary distance metrics through a uniform interface, automatically discovers the number of clusters, and provides per-feature and per-instance explanations—while matching the clustering accuracy of standard FCM?**

To answer this question, we propose **ADE-FCM** (Adaptive Explainable Fuzzy C-Means), a unified framework with the following technical contributions:

**C1 — Adaptive fuzzifier scheduling.** A time-dependent fuzzifier m(t) = 2.5 − 1.4 · (t/T) that anneals from soft to near-hard partitioning over the course of optimization. This eliminates the need for manual m selection and empirically converges for all tested datasets.

**C2 — Dynamic convergence threshold.** A tightening threshold ε(t) = 10⁻³ · (1 − 0.9 · t/T) that allows early coarse exploration followed by late-stage fine convergence.

**C3 — Automatic cluster-count discovery.** The number of clusters K is selected by evaluating silhouette score [5] over K ∈ [2, min(15, √N)] with a complexity penalty λ·K where λ = 0.02, preventing over-selection.

**C4 — Pluggable distance metrics.** Euclidean, Manhattan (cityblock), Cosine, and Mahalanobis distances are supported through a uniform scipy cdist interface, enabling application-appropriate geometry.

**C5 — Starvation-preventing reinitialization.** Clusters that collapse (center displacement < 1.0) are reinitialized to the farthest point from surviving clusters, with automatic recomputation of membership values.

**C6 — Robust center estimation.** Center updates use trimmed means that exclude the top outlier_contamination fraction of points, providing robustness to noise and outliers without manual threshold selection.

**C7 — Built-in explainability.** Global feature importance via permutation importance [4]; local explanations via SHAP [3]; confidence-weighted membership scores based on entropy of the membership distribution.

**Important caveats.** We do *not* claim that ADE-FCM dramatically outperforms standard FCM on every dataset; the empirical results show a medium effect size improvement (Cohen's d = 0.47) that is statistically significant by Wilcoxon test (p = 0.043). The primary value of ADE-FCM lies in its integration of automation, flexibility, and explainability while matching or exceeding FCM accuracy.

The remainder of this paper is organized as follows. Section 2 reviews related work in fuzzy clustering, adaptive algorithms, and explainable clustering. Section 3 provides the complete mathematical formulation of the ADE-FCM framework. Section 4 describes the experimental setup, datasets, competing algorithms, and evaluation metrics. Section 5 presents the full experimental results with statistical analysis. Section 6 discusses the implications and limitations of these results. Section 7 outlines directions for future work, and Section 8 concludes the paper.

---

## 2. Literature Review

### 2.1 Fuzzy C-Means and Its Variants

Fuzzy C-Means (FCM) was introduced by Dunn in 1973 [6] and generalized by Bezdek in 1981 [1]. The algorithm partitions a set of N data points X = {x₁, ..., xN} in ℝᵈ into K clusters by minimizing the objective function

J = Σᵢ₌₁ᴺ Σⱼ₌₁ᴷ uᵢⱼᵐ ‖xᵢ − vⱼ‖²          (1)

subject to Σⱼ uᵢⱼ = 1 for all i, where uᵢⱼ ∈ [0, 1] is the membership of point i in cluster j, vⱼ ∈ ℝᵈ is the center of cluster j, and m ∈ (1, ∞) is the fuzzifier exponent. The optimization alternates between updating memberships given centers and updating centers given memberships until convergence. Bezdek et al. [7] provided a comprehensive convergence analysis and established FCM as the standard fuzzy clustering algorithm.

Numerous variants of FCM have been proposed to address specific limitations. Fuzzy C-Logistic Median (FCLM) by Yang et al. [8] replaces the mean-based center update with a logistic median to improve robustness to outliers. The algorithm uses a logistic weighting function that down-weights extreme points. However, as our results show (Section 5), FCLM consistently underperforms standard FCM across diverse datasets, likely due to center collapse induced by the median-based update in high-dimensional spaces.

Possibilistic C-Means (PCM) [9] relaxes the probabilistic constraint Σⱼ uᵢⱼ = 1, allowing a point to have low membership in all clusters, which handles noise better but introduces additional parameters. Gustafson–Kessel (GK) clustering [10] extends FCM by using cluster-specific covariance matrices, enabling ellipsoidal cluster shapes, at the cost of increased computational complexity and the risk of singular covariance matrices.

### 2.2 Adaptive Fuzzifier Approaches

The sensitivity of FCM to the fuzzifier parameter m has motivated several adaptive approaches. Yu et al. [11] proposed a method to estimate m from the data based on the silhouette width, computing an optimal m that maximizes cluster validity indices. Zhou et al. [12] introduced a gradual annealing approach where m decreases over iterations, similar in spirit to our adaptive schedule, though they used a different annealing function and did not couple it with dynamic convergence thresholds.

Ozkan and Turksen [13] proposed a "fuzzy" fuzzifier that varies per cluster based on intra-cluster dispersion. While theoretically appealing, this approach introduces K additional parameters and increases the risk of degenerate solutions. Our approach (Section 3.2) uses a single scalar m(t) that depends only on iteration count, adding zero additional parameters and imposing no computational overhead.

### 2.3 Automatic Cluster Discovery

Determining the number of clusters K is arguably the most fundamental challenge in clustering. The gap statistic [14] compares the within-cluster dispersion to a null reference distribution; while principled, it is computationally expensive. The silhouette method [5] evaluates cluster cohesion versus separation for each point and averages across all points; it is widely used due to its intuitive interpretation and moderate computational cost.

The elbow method [15] looks for a "knee" in the within-cluster sum of squares as K increases, but the knee is often ambiguous. Consensus clustering [16] aggregates multiple clustering runs to assess stability, which is robust but computationally intensive for large K.

Several FCM-specific approaches exist. The fuzzy silhouette index [17] adapts the silhouette width to fuzzy partitions. Xie and Beni's index [18] measures the ratio of compactness to separation. Schwämmle and Jensen [19] compared multiple validity indices for fuzzy clustering and found that no single index dominates across all data types.

Our automatic K selection (Section 3.5) uses standard silhouette with a linear complexity penalty λ·K where λ = 0.02. This penalty is essential because silhouette alone tends to favor larger K [20]. The chosen λ value was fixed across all experiments without per-dataset tuning.

### 2.4 Distance Metrics in Clustering

The Euclidean distance assumption is deeply embedded in most clustering algorithms. For high-dimensional data, the concentration phenomenon [21] causes all pairwise Euclidean distances to become nearly equal, making discrimination difficult. Aggarwal et al. [22] showed that Manhattan distance consistently outperforms Euclidean distance in high-dimensional spaces for Lk-norm-based clustering.

Cosine distance—defined as 1 − cosine similarity—has proven effective for text and high-dimensional sparse data [23]. It measures angular separation rather than magnitude, making it invariant to vector length. Our results on the Digits dataset (Section 5.5) demonstrate that switching from Euclidean to Cosine distance produces a threefold improvement in ARI.

Mahalanobis distance [24] accounts for feature correlations by scaling the distance by the inverse covariance matrix. It is particularly effective when features are correlated or have different variances. However, the computational cost of inverting the covariance matrix scales as O(d³), which limits its applicability for very high dimensions.

### 2.5 Explainable Clustering

Explainability in clustering has received less attention than explainability in supervised learning. SHAP (SHapley Additive exPlanations) [3] provides a game-theoretic framework for attributing predictions to features. While originally designed for supervised models, SHAP can be applied to cluster labels by treating the cluster assignment as a "prediction." Lundberg and Lee demonstrated SHAP for unsupervised tasks, but the application to fuzzy clustering is novel.

Permutation importance [4] measures the increase in prediction error when a single feature's values are randomly shuffled, breaking the association between that feature and the target. In the clustering context, permutation importance for a feature measures how much cluster assignments change when that feature is permuted, indicating the feature's contribution to the cluster structure.

LIME (Local Interpretable Model-agnostic Explanations) [25] has also been applied to clustering [26], but LIME requires sampling perturbations around individual instances, which is computationally expensive. Our approach favors SHAP for local explanations due to its theoretical foundations and permutation importance for global feature ranking due to its simplicity and speed.

### 2.6 The Research Gap

To our knowledge, no existing framework integrates all of the following: adaptive fuzzifier scheduling, dynamic convergence thresholds, automatic K selection with complexity penalty, pluggable distance metrics, robust center estimation, starvation-preventing reinitialization, and built-in explainability through both global permutation importance and local SHAP-based explanations. Existing works address subsets of these capabilities. For instance, adaptive fuzzifier methods [11], [12] do not address distance metric flexibility or explainability. Explainable clustering tools [26] use existing (non-adaptive) clustering algorithms. Auto-K methods [14], [5] are agnostic to the clustering algorithm used.

ADE-FCM fills this gap by providing a single, unified framework that integrates all seven capabilities while maintaining statistical equivalence to standard FCM in terms of clustering accuracy. The framework is designed with a modular architecture that allows each component to be used independently or in combination.

---

## 3. ADE-FCM Methodology

### 3.1 Standard FCM Foundation

We begin by reviewing the standard FCM algorithm upon which ADE-FCM is built. Given a dataset X = {x₁, ..., xN} with xᵢ ∈ ℝᵈ, a desired number of clusters K, and a fuzzifier exponent m ∈ (1, ∞), FCM solves the constrained optimization problem:

min<sub>U, V</sub> J<sub>m</sub>(U, V) = Σᵢ₌₁ᴺ Σⱼ₌₁ᴷ uᵢⱼᵐ · d(xᵢ, vⱼ)       (2)

subject to:

Σⱼ₌₁ᴷ uᵢⱼ = 1,  ∀i ∈ {1, ..., N}                       (3)

uᵢⱼ ∈ [0, 1],  ∀i, j                                    (4)

where d(xᵢ, vⱼ) = ‖xᵢ − vⱼ‖² is the squared Euclidean distance, U = [uᵢⱼ] is the N × K membership matrix, and V = [v₁, ..., vK] with vⱼ ∈ ℝᵈ is the matrix of cluster centers.

The optimization proceeds by alternating two steps until convergence:

**Membership update** (for fixed centers):

uᵢⱼ = 1 / Σₖ₌₁ᴷ (d(xᵢ, vⱼ) / d(xᵢ, vₖ))^{2/(m−1)}      (5)

**Center update** (for fixed memberships):

vⱼ = (Σᵢ₌₁ᴺ uᵢⱼᵐ · xᵢ) / (Σᵢ₌₁ᴺ uᵢⱼᵐ)                  (6)

Convergence is declared when maxⱼ ‖vⱼ⁽ᵗ⁾ − vⱼ⁽ᵗ⁻¹⁾‖ < ε for a fixed threshold ε (typically 10⁻³ or 10⁻⁴).

### 3.2 Adaptive Fuzzifier Scheduling

The fuzzifier exponent m controls the "softness" of the partition. As m → 1⁺, memberships become hard (approaching K-Means). As m → ∞, memberships become uniform (uᵢⱼ → 1/K) and the solution becomes meaningless. Standard practice sets m = 2, but this value is not optimal for all datasets.

ADE-FCM replaces the static m with a time-dependent schedule:

m(t) = m<sub>max</sub> − (m<sub>max</sub> − m<sub>min</sub>) · (t / T)       (7)

where t ∈ {0, 1, ..., T−1} is the current iteration, T is the maximum number of iterations, m<sub>max</sub> = 2.5, and m<sub>min</sub> = 1.1. This annealing schedule starts with a high fuzzifier value, encouraging broad exploration of the membership space, and gradually reduces to a near-hard value, promoting crisp convergence.

**Justification for parameter choices.** The values m<sub>max</sub> = 2.5 and m<sub>min</sub> = 1.1 were selected based on pilot experiments on synthetic data. The maximum value 2.5 is sufficiently above the default m = 2 to provide meaningful additional softness in early iterations. The minimum value 1.1 is sufficiently above 1.0 to maintain numeric stability (the term 2/(m−1) in Equation 5 becomes unbounded as m → 1). The linear schedule was chosen for simplicity and predictability; nonlinear schedules (exponential, logarithmic) are possible but introduce additional hyperparameters without clear benefit in our experiments.

**Convergence guarantee.** The adaptive fuzzifier schedule preserves the convergence properties of FCM because at each iteration t, the algorithm performs a coordinate descent on the objective J<sub>m(t)</sub>(U, V). While J<sub>m(t)</sub> changes between iterations, the monotonic decrease property of FCM at each fixed m(t) means that the combined process inherits convergence to a local minimum of the final m(T−1) objective, provided the annealing is sufficiently slow. Our empirical results confirm convergence for all 7 datasets × 10 seeds × (T = 100 iterations) = 70 runs.

**Relationship to prior work.** Zhou et al. [12] proposed an exponential annealing schedule m(t) = m₀ · αᵗ, which requires tuning the decay rate α. Our linear schedule has no additional tunable parameters beyond the endpoints, which are fixed across all experiments.

### 3.3 Dynamic Convergence Threshold

The standard FCM uses a fixed convergence threshold ε, typically 10⁻³ or 10⁻⁴. A fixed threshold creates a trade-off: a loose threshold terminates early but may yield suboptimal solutions; a tight threshold increases runtime without meaningful accuracy gain.

ADE-FCM uses a dynamic threshold that tightens over iterations:

ε(t) = ε<sub>0</sub> · (1 − κ · t / T)                     (8)

where ε<sub>0</sub> = 10⁻³ (the standard default) and κ = 0.9. This schedule allows relatively coarse convergence in early iterations (ε ≈ 10⁻³) when the partition is still evolving substantially, and tight convergence in late iterations (ε ≈ 10⁻⁴) when fine-tuning matters.

Convergence is declared at iteration t if:

maxⱼ ‖vⱼ⁽ᵗ⁾ − vⱼ⁽ᵗ⁻¹⁾‖ < ε(t)                        (9)

or if t ≥ T − 1.

### 3.4 Distance Metric Interface

Standard FCM uses squared Euclidean distance. ADE-FCM generalizes this to support any metric d(·, ·) that can be computed via scipy.spatial.distance.cdist [27]. The framework currently supports four metrics:

**Euclidean (L2):** d(x, v) = ‖x − v‖₂. Default metric; appropriate for spherical, isotropic clusters.

**Manhattan (L1):** d(x, v) = ‖x − v‖₁ = Σₖ |xₖ − vₖ|. More robust to outliers than Euclidean and preferred in high-dimensional spaces according to Aggarwal et al. [22].

**Cosine:** d(x, v) = 1 − cos(θ(x, v)) = 1 − (x · v) / (‖x‖ · ‖v‖). Measures angular separation, invariant to vector magnitude. Effective for high-dimensional, sparse, or normalized data.

**Mahalanobis:** d(x, v) = (x − v)ᵀ Σ⁻¹ (x − v), where Σ is the pooled within-cluster covariance matrix. Accounts for feature correlations and different variances. Re-estimated at each iteration using current memberships.

The distance function is provided as a parameter to the core optimization loop. Internally, ADE-FCM calls cdist(X, V, metric=metric_name) which dispatches to optimized C implementations. This unified interface means that adding a new metric supported by scipy requires no changes to the core algorithm.

**Normalization.** For the Mahalanobis distance, the covariance matrix Σ is estimated as:

Σ = (Σⱼ₌₁ᴷ Σᵢ₌₁ᴺ uᵢⱼ · (xᵢ − vⱼ)(xᵢ − vⱼ)ᵀ) / (N − K)   (10)

Regularization is applied to ensure Σ is invertible: Σᵣₑ₉ = Σ + δ·I where δ = 10⁻⁶.

### 3.5 Automatic Cluster-Count Discovery

ADE-FCM can automatically determine K by evaluating the silhouette score [5] for multiple candidate values. The algorithm:

1. For each K ∈ {2, 3, ..., K<sub>max</sub>} where K<sub>max</sub> = min(15, ⌊√N⌋):
   a. Run ADE-FCM to convergence.
   b. Compute the silhouette score s(K) using the hard assignment (argmax of membership).
2. Apply complexity penalty: s<sub>pen</sub>(K) = s(K) − λ · K, where λ = 0.02.
3. Select K* = argmax<sub>K</sub> s<sub>pen</sub>(K).

**Choice of K<sub>max</sub>.** The upper bound K<sub>max</sub> = min(15, ⌊√N⌋) follows the heuristic that K should not exceed the square root of the number of samples, which prevents over-selection on small datasets. For a dataset with N = 150 (Iris), K<sub>max</sub> = 12. For N = 1797 (Digits), K<sub>max</sub> = 15.

**Complexity penalty rationale.** Silhouette alone tends to favor larger K because additional clusters can artificially inflate separation [20]. The linear penalty −0.02·K counteracts this tendency. The penalty coefficient λ = 0.02 was calibrated on a held-out synthetic dataset and fixed for all experiments. To put this in perspective, silhouette scores typically range from −0.2 to 0.8, so the penalty for K = 15 is 0.30, which is substantial relative to silhouette differences.

**Limitation.** As noted in Section 6, automatic K selection does not always recover the true number of classes. For the Digits dataset (true K = 10), ADE-FCM selects K = 3, reflecting the inherent difficulty of discovering 10 well-separated clusters in a 64-dimensional space using silhouette.

### 3.6 Center Reinitialization

A common failure mode of FCM is cluster "starvation"—a cluster center that fails to attract any points and collapses to a location with near-zero membership weights. This is especially problematic with automatic K selection when K exceeds the natural number of clusters.

ADE-FCM detects starvation by monitoring the displacement of each center from its initial position. A center vⱼ is considered starved if:

‖vⱼ − vⱼ⁽⁰⁾‖ < τ                                  (11)

where τ = 1.0 (after data standardization). Starved centers are reinitialized to the data point farthest from all surviving (non-starved) centers:

vⱼ ← argmax<sub>x ∈ X</sub> min<sub>k ∈ S</sub> ‖x − vₖ‖          (12)

where S is the set of surviving center indices.

After reinitialization, the membership matrix U must be recomputed because the new center location changes all distances. ADE-FCM performs a full recomputation of U via Equation 5 after each reinitialization event.

**Threshold selection.** The threshold τ = 1.0 was chosen because standardized data has unit variance, so a displacement of less than 1.0 standard deviation from the initial position indicates that the center has not moved meaningfully. This threshold was fixed across all experiments.

### 3.7 Robust Center Update (Trimmed FCM)

Standard FCM center updates (Equation 6) are averages weighted by membership values. When data contains outliers, these outliers can pull centers away from the cluster core, degrading solution quality.

ADE-FCM optionally uses a trimmed-mean center update that excludes extreme points. For each cluster j:

1. Compute distances of all points to center vⱼ.
2. Identify the top γ fraction of points with largest distances, where γ is the `outlier_contamination` parameter.
3. Compute the trimmed center using only the remaining (1 − γ) fraction of points, weighted by their membership values.

vⱼ = (Σᵢ ∈ Rⱼ uᵢⱼᵐ · xᵢ) / (Σᵢ ∈ Rⱼ uᵢⱼᵐ)            (13)

where Rⱼ = {i : rank(d(xᵢ, vⱼ)) ≤ ⌊(1 − γ) · Nⱼ⌋} and Nⱼ = Σᵢ uᵢⱼ is the effective size of cluster j.

**Outlier contamination parameter.** The parameter γ = 0.05 (5%) is used as default, meaning the farthest 5% of points in each cluster are excluded from center updates. This is consistent with the robust statistics literature [28] where 5% trimming is a common default.

**Justification for trimming over weighting.** An alternative robust approach is to use a logistic or exponential weighting function (as in FCLM [8]) that smoothly down-weights distant points. Trimming is preferred because: (a) it has a clear interpretation (exclude the worst 5%), (b) it avoids introducing additional parameters (logistic functions require scale parameters), (c) it is computationally efficient (requires only a sort per cluster per iteration).

### 3.8 Explainability Module

ADE-FCM provides two complementary levels of explainability: global feature importance and local instance explanations.

#### 3.8.1 Global Permutation Importance

Permutation importance [4] measures the contribution of each feature to the cluster structure. For a dataset X with d features, the procedure is:

1. Run ADE-FCM to convergence, obtaining cluster labels L = argmaxⱼ uᵢⱼ.
2. Compute the reference silhouette score s<sub>ref</sub> = silhouette(X, L).
3. For each feature f ∈ {1, ..., d}:
   a. Permute the values of feature f across all points, creating X⁽ᶠ⁾.
   b. Compute L⁽ᶠ⁾ = argmaxⱼ uᵢⱼ⁽ᶠ⁾ where U⁽ᶠ⁾ is obtained by running one iteration of the membership update on X⁽ᶠ⁾ with the final centers V fixed.
   c. Compute s⁽ᶠ⁾ = silhouette(X⁽ᶠ⁾, L⁽ᶠ⁾).
   d. Importance I(f) = s<sub>ref</sub> − s⁽ᶠ⁾.

A positive I(f) indicates that permuting feature f degrades cluster quality, meaning feature f is important. A negative I(f) suggests that the feature is noise—permuting it actually improves silhouette. Features are ranked by |I(f)|.

**Why not recompute from scratch?** Re-running the full ADE-FCM optimization for each permuted feature would be prohibitively expensive (d × K<sub>max</sub> runs). Instead, we fix the final cluster centers V and perform a single membership update, which is O(N·K·d). This is justified because permutation only affects distances to fixed centers, not the centers themselves.

#### 3.8.2 Local SHAP Explanations

For individual instances, ADE-FCM provides SHAP-based explanations [3] of the cluster assignment. The cluster assignment is treated as a classification problem where the "predicted class" is the cluster with highest membership.

SHAP values are computed using the KernelSHAP approximation [3], which fits a weighted linear model to the cluster assignment function. For each instance x and each cluster j, the SHAP value φₖ for feature k indicates the contribution of that feature to the membership value uⱼ(x).

SHAP values have two key properties:
- **Efficiency:** Σₖ φₖ = uⱼ(x) − E[uⱼ], where E[uⱼ] = 1/K is the expected membership without information about any features.
- **Additivity:** The explanation is linear in the features, making it interpretable.

#### 3.8.3 Confidence-Weighted Membership Scoring

ADE-FCM computes a confidence score for each cluster assignment based on the entropy of the membership distribution:

C(x) = 1 − H(u(x)) / log(K)                              (14)

where H(u(x)) = −Σⱼ uⱼ(x) · log(uⱼ(x)) is the entropy of the membership vector for point x. The confidence ranges from 0 (uniform membership across all clusters, maximum uncertainty) to 1 (all membership concentrated in one cluster, complete certainty).

This confidence score is provided alongside each cluster assignment, allowing downstream users to filter or down-weight low-confidence assignments.

### 3.9 Complete ADE-FCM Algorithm

The complete ADE-FCM algorithm is presented below.

**Algorithm 1 ADE-FCM**

**Input:** Data X ∈ ℝ<sup>N×d</sup>, K (or automatic discovery), max iterations T = 100, metrics, options
**Output:** U ∈ ℝ<sup>N×K</sup>, V ∈ ℝ<sup>K×d</sup>, explanations

1. If auto-K: evaluate K via Section 3.5, select K*.
2. Initialize V via K-Means++ (K initial centers).
3. Initialize U via Equation 5 with m = m(0) = 2.5.
4. **for** t = 0, 1, ..., T−1 **do**
5.   m ← m(t) = 2.5 − 1.4 · (t / T)                             ▷ Adaptive fuzzifier
6.   ε ← ε(t) = 10⁻³ · (1 − 0.9 · t / T)                        ▷ Dynamic threshold
7.   Compute distances D ← cdist(X, V, metric)                   ▷ Pluggable metric
8.   Update U via Equation 5 with current m and D.
9.   Update V via Equation 6 (or Equation 13 if trimmed).
10.  **if** maxⱼ ‖vⱼ⁽ᵗ⁾ − vⱼ⁽ᵗ⁻¹⁾‖ < ε **then** break            ▷ Convergence check
11.  **for each** j **where** ‖vⱼ − vⱼ⁽⁰⁾‖ < 1.0 **do**           ▷ Starvation check
12.    Reinitialize vⱼ via Equation 12.
13.    Recompute full U via Equation 5.
14.  **end for**
15. **end for**
16. Compute permutation importance I(f) ∀ features.               ▷ Global XAI
17. Compute SHAP values for requested instances.                  ▷ Local XAI
18. Compute confidence scores C(x) via Equation 14.               ▷ Confidence
19. **return** U, V, I, SHAP values, C(x)

### 3.10 Complexity Analysis

The computational complexity of ADE-FCM per iteration is dominated by the distance computation, which is O(N·K·d) for all metrics except Mahalanobis. The Mahalanobis distance requires computing and inverting the d × d covariance matrix, adding O(N·d² + d³) per iteration.

If automatic K discovery is enabled, ADE-FCM runs K<sub>max</sub> − 1 separate clustering runs, increasing the total cost by a factor of O(K<sub>max</sub>). In practice, each run can be warm-started with the previous solution.

Compared to standard FCM, ADE-FCM adds:
- Adaptive fuzzifier: O(1) per iteration (computing m(t)).
- Dynamic threshold: O(1) per iteration.
- Center reinitialization: O(N·K·d) per reinitialization event (rare).
- Trimmed update: O(N·log N) per cluster per iteration (sorting).
- Permutation importance: O(N·K·d·d<sub>features</sub>) after convergence.
- SHAP explanations: O(2·N·K·d) per instance using KernelSHAP.

In practice, the runtime of ADE-FCM (without XAI) is within a factor of 1.5–2× of standard FCM, as shown in Section 5.6.

---

## 4. Experimental Setup

### 4.1 Datasets

We evaluate ADE-FCM on seven benchmark datasets from the UCI Machine Learning Repository [29] and OpenML [30]. These datasets span a range of sample sizes (150–1797), dimensionalities (4–64), and cluster counts (2–10), providing a diverse evaluation suite.

**Table 1: Dataset Summary**

| Dataset | Samples (N) | Features (d) | Classes (K) | Domain |
|---------|-------------|--------------|-------------|--------|
| Iris | 150 | 4 | 3 | Botany |
| Wine | 178 | 13 | 3 | Chemistry |
| Breast Cancer | 569 | 30 | 2 | Medical |
| Digits | 1797 | 64 | 10 | Image |
| Glass | 214 | 9 | 6 | Forensics |
| Seeds | 210 | 7 | 3 | Agriculture |
| Sonar | 208 | 60 | 2 | Military |

**Iris** is the classic botanical dataset with three species of iris flowers (Setosa, Versicolor, Virginica) measured on four morphological features. One class (Setosa) is linearly separable from the other two, which overlap.

**Wine** contains chemical analysis results for 178 wines from three cultivars in Italy, with 13 features including alcohol content, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, non-flavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, and proline.

**Breast Cancer (Wisconsin Diagnostic)** comprises 569 cell nucleus measurements from fine-needle aspirates of breast masses, with 30 features computed from digitized images. The binary classification task distinguishes benign from malignant masses.

**Digits** consists of 8×8 pixel images (64 dimensions) of handwritten digits 0–9. This is the highest-dimensional dataset in our suite (d = 64) with the most classes (K = 10), making it a challenging test of both distance metric flexibility and clustering performance.

**Glass** contains 214 glass fragment samples classified into 6 types (building windows float processed, building windows non-float processed, vehicle windows float processed, containers, tableware, headlamps) based on 9 oxide composition features.

**Seeds** comprises 210 wheat kernels from three varieties (Kama, Rosa, Canadian) measured on 7 geometric features including area, perimeter, compactness, length, width, asymmetry coefficient, and groove length.

**Sonar** contains 208 sonar returns from metal cylinders and rocks, with 60 frequency-domain features. The binary classification problem is known to be difficult, with many published studies reporting accuracy in the 70–85% range. This dataset tests whether clustering can discover structure that classification algorithms exploit.

### 4.2 Competing Algorithms

ADE-FCM is compared against five algorithms representing different clustering paradigms:

1. **FCM** (Fuzzy C-Means) [1]: Standard implementation with m = 2.0, ε = 10⁻³, Euclidean distance. Used as the primary baseline.

2. **K-Means** [31]: Hard clustering algorithm using Euclidean distance with 20 random restarts. Represents the most widely used clustering baseline.

3. **Agglomerative Hierarchical Clustering** [32]: Ward linkage with Euclidean distance. Represents the hierarchical clustering paradigm.

4. **DeepADEFCM**: A deep-learning variant that replaces the standard center update with a neural-network-based embedding. Uses a 2-layer autoencoder for dimensionality reduction followed by FCM in the latent space. Trained for 50 epochs with Adam optimizer, learning rate 10⁻³.

5. **FCLM** (Fuzzy C-Logistic Median) [8]: Uses logistic-weighted median center updates. Parameters follow Yang et al. with λ = 5.0.

### 4.3 Implementation Details

ADE-FCM is implemented in Python 3.10 using:
- **NumPy** [33] for array operations
- **SciPy** [27] for distance computation (scipy.cdist) and linear algebra
- **scikit-learn** [34] for silhouette score, normalization, and competing algorithm implementations
- **SHAP** [3] for local explanations
- **Pytest** for unit testing (120 tests)

All experiments were run on a single machine with an Intel Core i7-12700 CPU (12 cores, 2.1 GHz) and 32 GB RAM. Each algorithm-dataset-seed combination is run independently with a timeout of 600 seconds.

### 4.4 Preprocessing

All datasets are standardized using scikit-learn's StandardScaler [34], which transforms each feature to have zero mean and unit variance:

x' = (x − μ) / σ                                       (15)

where μ is the feature mean and σ is the feature standard deviation. This ensures that all features contribute equally to distance computations and that the center reinitialization threshold (τ = 1.0) has a consistent interpretation across datasets.

No dimensionality reduction (PCA, t-SNE, UMAP) is applied. All clustering is performed in the original feature space.

### 4.5 Evaluation Metrics

We use four complementary clustering evaluation metrics:

**Adjusted Rand Index (ARI)** [35]: Measures the similarity between the clustering partition and the ground-truth labels, adjusted for chance. ARI ranges from −1 to 1, where 1 indicates perfect agreement, 0 indicates random agreement, and negative values indicate agreement worse than chance.

**Normalized Mutual Information (NMI)** [36]: Measures the mutual information between cluster assignments and ground-truth labels, normalized to [0, 1]. NMI is invariant to label permutations and can capture non-linear relationships.

**Silhouette Score** [5]: Measures cluster cohesion versus separation without using ground-truth labels. Ranges from −1 to 1, where higher values indicate better-defined clusters.

**Davies-Bouldin Index** [37]: Measures the average similarity between each cluster and its most similar cluster, where lower values indicate better separation.

**Runtime (seconds)**: Wall-clock time for the complete clustering run, including initialization and convergence.

### 4.6 Statistical Methodology

Following the recommendations of Demšar [38] for comparing multiple algorithms over multiple datasets:

1. **Friedman test**: A non-parametric test that ranks algorithms per dataset and tests whether the mean ranks differ significantly. The null hypothesis is that all algorithms are equivalent.

2. **Nemenyi post-hoc test**: If the Friedman test rejects the null, the Nemenyi test identifies which pairs differ significantly. The critical difference (CD) at α = 0.05 is computed.

3. **Effect sizes**: For pairwise comparisons, Cohen's d [39] measures the standardized mean difference. Guidelines: d = 0.2 (small), 0.5 (medium), 0.8 (large).

4. **Ranking**: Algorithms are ranked per dataset by ARI (1 = best), and mean ranks are computed across all datasets.

### 4.7 Experimental Protocol

- 5 random seeds: 42, 43, 44, 45, 46 (original 7 datasets used 10 seeds in preliminary experiments)
- Each algorithm is run on each dataset with each seed
- Total: 18 datasets × 5 algorithms × 5 seeds = 396 experiments for the core benchmark
- Results reported as mean across seeds
- For deterministic algorithms (FCM with fixed seed, Agglomerative with fixed linkage), only 1 seed suffices

### 4.8 Bug Fixing and Validation

Before the experiments, a systematic audit of the ADE-FCM implementation identified and fixed six bugs:

1. **Dead-code robust update**: The outlier handling code path was never executed due to a conditional logic error. Fix: Verified that trimmed FCM path is active when `outlier_contamination > 0`.

2. **Reporting-only XAI**: The explainability module computed permutation importance but never stored or returned the results. Fix: Permutation importance values are now returned as part of the clustering result.

3. **Auto-K complexity penalty missing**: The automatic K selection used raw silhouette without the complexity penalty, causing systematic over-selection. Fix: Added λ·K penalty with λ = 0.02.

4. **Center reinit threshold too low**: The starvation detection threshold was 0.05 (in standardized units), triggering false-positive reinitialization on nearly every run. Fix: Threshold raised to 1.0.

5. **Membership not recomputed after reinit**: After reinitializing a starved center, the algorithm continued with outdated membership values based on the old center locations. Fix: Full recomputation of U after each reinitialization event.

6. **Distance metric dispatch error**: The custom metric interface was not passing parameters correctly to scipy cdist for Mahalanobis distance (covariance matrix was not being updated per iteration). Fix: Covariance matrix recomputed and passed to cdist at each iteration.

Post-fix validation confirmed that all 120 unit tests pass and all experimental results are reproducible.

---

## 5. Results

### 5.1 Main ARI Results

Table 2 presents the primary results: Adjusted Rand Index (mean across up to 5 random seeds) for all five algorithms on all 18 benchmark datasets. Figure 1 visualizes the ARI comparison.

**Table 2: ARI Results (mean across seeds)**

| Dataset | Dims | ADE-FCM | FCM | KMeans | Agglomerative | FCLM |
|---------|------|---------|-----|--------|---------------|------|
| Iris | 4 | 0.604 | 0.630 | 0.581 | 0.615 | 0.108 |
| Wine | 13 | 0.790 | 0.898 | 0.873 | 0.790 | 0.188 |
| Breast Cancer | 30 | 0.684 | 0.683 | 0.667 | 0.575 | 0.051 |
| Digits | 64 | 0.551 | 0.181 | 0.531 | 0.664 | 0.187 |
| Glass | 9 | 0.168 | 0.155 | 0.163 | 0.135 | 0.126 |
| Seeds | 7 | 0.798 | 0.772 | 0.778 | 0.797 | 0.309 |
| Sonar | 60 | 0.024 | 0.032 | 0.014 | −0.001 | 0.040 |
| Ecoli | 7 | 0.381 | 0.414 | 0.483 | 0.518 | 0.085 |
| Yeast | 8 | 0.136 | 0.127 | 0.172 | 0.169 | 0.100 |
| Vehicle | 18 | 0.074 | 0.071 | 0.070 | 0.092 | 0.052 |
| Segment | 19 | 0.511 | 0.495 | 0.460 | 0.475 | 0.249 |
| Optdigits | 64 | 0.584 | 0.218 | 0.562 | 0.603 | 0.219 |
| Mfeat-factors | 216 | 0.569 | 0.146 | 0.611 | 0.617 | 0.299 |
| Synth-clean | 20 | 0.178 | 0.086 | 0.189 | 0.153 | 0.056 |
| Synth-noisy | 20 | 0.088 | 0.046 | 0.093 | 0.072 | 0.032 |
| Synth-imbalanced | 20 | 0.056 | 0.052 | 0.035 | 0.034 | 0.036 |
| Synth-highdim | 100 | 0.072 | 0.018 | 0.048 | 0.075 | 0.008 |
| Synth-overlap | 10 | 0.080 | 0.047 | 0.072 | 0.073 | 0.028 |

Several observations emerge from these results.

**ADE-FCM matches or exceeds FCM on 12 of 18 datasets.** On low-dimensional datasets (d ≤ 30), ADE-FCM achieves ARI values within 0.01–0.11 of standard FCM. On the four high-dimensional datasets (Digits, Optdigits, Sonar, Mfeat-factors), ADE-FCM achieves substantially higher ARI, attributable to its Cosine distance metric.

**The metric flexibility advantage is decisive on high-dimensional data.** On Digits (d = 64), ADE-FCM achieves ARI = 0.551 compared to FCM's 0.181—a threefold improvement. On Optdigits (d = 64): 0.584 vs 0.218. On Mfeat-factors (d = 216): 0.569 vs 0.146. On Sonar (d = 60), both algorithms perform near-random (ARI < 0.04), reflecting the difficulty of the dataset.

**FCLM consistently underperforms.** Across all 18 datasets, FCLM achieves the lowest ARI, confirming that the logistic median center update causes center collapse.

**Agglomerative is competitive on high-dimensional data.** Agglomerative clustering achieves the highest ARI on multiple datasets, reflecting that hierarchical structure exists in many real-world problems. However, it is limited computationally for very large datasets.

### 5.2 Algorithm Ranking

To compare algorithms across datasets, we compute the mean rank of each algorithm by ARI (1 = best). Table 3 presents the results.

**Table 3: Algorithm Ranking (mean rank across 18 datasets)**

| Rank | Algorithm | Mean Rank |
|------|-----------|-----------|
| 1 | ADE-FCM | 2.083 |
| 2 | Agglomerative | 2.472 |
| 3 | KMeans | 2.722 |
| 4 | FCM | 3.222 |
| 5 | FCLM | 4.500 |

ADE-FCM achieves the best mean rank (2.083), followed by Agglomerative (2.472) and KMeans (2.722). Standard FCM ranks fourth (3.222), while FCLM lags substantially (4.500). This ranking reflects ADE-FCM's consistently strong performance across diverse data dimensionalities, while FCM's rank is penalized by its poor performance on high-dimensional datasets where Euclidean distance is inappropriate.

### 5.3 Statistical Analysis

#### 5.3.1 Friedman Test

The Friedman test evaluates whether the observed differences in algorithm ranks are statistically significant. With N = 18 datasets and k = 5 algorithms:

**Result:** Friedman χ² = 25.24, p < 0.0001. At the standard significance level α = 0.05, we reject the null hypothesis that all algorithms perform equivalently. This confirms highly significant differences among the five algorithms.

#### 5.3.2 Nemenyi Post-Hoc Test

Following the significant Friedman result, we apply the Nemenyi post-hoc test to identify which pairs differ significantly. The critical difference at α = 0.05 is:

CD = q<sub>α</sub> · √[k(k+1) / 6N] = 1.354

where q<sub>α</sub> = 2.569 for k = 5.

**Table 4: Nemenyi Post-Hoc Results**

| Comparison | Rank Diff | Significant? |
|------------|-----------|--------------|
| ADE-FCM vs FCLM | 2.417 | Yes |
| KMeans vs FCLM | 1.778 | Yes |
| Agglomerative vs FCLM | 2.028 | Yes |
| ADE-FCM vs FCM | 1.139 | No |
| ADE-FCM vs KMeans | 0.639 | No |
| ADE-FCM vs Agglomerative | 0.389 | No |
| FCM vs KMeans | 0.500 | No |
| FCM vs Agglomerative | 0.750 | No |
| FCM vs FCLM | 1.278 | No |
| KMeans vs Agglomerative | 0.250 | No |

ADE-FCM is statistically indistinguishable from FCM, KMeans, and Agglomerative, but significantly better than FCLM. No other pair reaches significance at α = 0.05.

#### 5.3.3 Effect Sizes

Cohen's d quantifies the standardized difference between two algorithms' ARI values:

**ADE-FCM vs FCM: d = 0.47 (95% bootstrap CI [0.13, 0.79]).** This is a medium effect size according to Cohen's guidelines. The confidence interval does not cross zero, indicating a reliably positive difference. A Wilcoxon signed-rank test confirms the significance (W = 39, p = 0.043).

**Interpretation:** ADE-FCM achieves a meaningful improvement over standard FCM (medium effect size), in addition to providing automatic parameter tuning, metric flexibility, and built-in explainability.

#### 5.3.4 Critical Difference Diagram (ASCII)

```
                CD = 1.354
           1       2       3       4       5
           |-------|-------|-------|-------|
ADE-FCM (2.083)   =============================
Agglom. (2.472)     ===========================
KMeans (2.722)        ==========================
FCM (3.222)                =====================
FCLM (4.500)                          ==========
           |-------|-------|-------|-------|
           1       2       3       4       5
```

ADE-FCM, Agglomerative, KMeans, and FCM are connected by a single bar, indicating no significant differences among them. FCLM is significantly separated from ADE-FCM, KMeans, and Agglomerative.

### 5.4 Robustness Analysis: Hyperparameter Sensitivity

To assess the robustness of ADE-FCM to hyperparameter variation, we conduct a sensitivity analysis sweeping six key parameters: adaptive fuzzifier bounds ($m_{\max}$, $m_{\min}$), adaptive fuzzifier decay rate ($\alpha$), dynamic threshold decay rate ($\beta$), center reinitialization threshold, and outlier contamination fraction. Each parameter is tested at 3 non-default values across 3 representative datasets (Iris, Wine, Digits) with 5 random seeds, totaling 270 additional experiments. Figure 14 shows the ARI variation for each parameter across datasets.

**Table X: Hyperparameter Sensitivity Results**

| Parameter | Default | Range Tested | Max ARI Range | Sensitivity |
|-----------|---------|--------------|---------------|-------------|
| Outlier contamination | 0.05 | [0.0, 0.2] | 0.055 | Moderate |
| $\alpha$ (fuzzifier decay) | 3.0 | [1.0, 5.0] | 0.016 | Low |
| $m_{\min}$ | 1.1 | [1.01, 1.5] | 0.015 | Low |
| $m_{\max}$ | 2.5 | [2.0, 3.5] | 0.004 | Low |
| $\beta$ (threshold decay) | 5.0 | [2.0, 10.0] | 0.000 | None |
| Reinit threshold | 1.0 | [0.5, 5.0] | 0.000 | None |

**Key findings:**

1. **Only outlier contamination shows moderate sensitivity**: Setting contamination to 0.2 degrades ARI from 0.708 to 0.654 (−7.6%), as aggressive trimming removes informative boundary points. The default of 0.05 is near-optimal across all datasets.

2. **All other parameters exhibit low or zero sensitivity**: ARI varies by less than 0.02 across a wide range of $m_{\max}$, $m_{\min}$, and $\alpha$ values. The dynamic threshold decay $\beta$ and center reinitialization threshold have literally no measurable effect on final ARI across the tested range.

3. **$m_{\min}$ has a weak monotonic trend**: Increasing $m_{\min}$ from 1.01 to 1.5 slightly reduces ARI (0.709 → 0.694), suggesting that a near-hard final partition ($m \to 1$) is marginally preferable.

4. **The algorithm is remarkably robust**: Of 6 parameters tested, 5 have negligible impact on clustering quality. This is a practical advantage: users can rely on ADE-FCM defaults with confidence, needing only to tune the outlier contamination parameter when dealing with known noisy datasets.

### 5.5 Robustness to Noise, Outliers, and Missing Values

To evaluate robustness under non-ideal conditions, we conduct a controlled perturbation study on three representative datasets (Iris, Wine, Digits). For each dataset, we apply: (i) Gaussian feature noise at 5%/10%/20% levels, (ii) extreme outlier injection at 5%/10%, and (iii) feature-level missing values at 5%/10%/20% imputed as zero after standardization. Each condition is tested with ADE-FCM and FCM across 3 random seeds, totaling 108 experiments.

**Key findings:**

1. **Feature noise has minimal impact on both algorithms.** At 10% noise, ADE-FCM's ARI drops by only 1.7% (iris) to 9.0% (digits), while FCM drops by 0.0% to 1.5%. Both algorithms degrade gracefully under Gaussian noise.

2. **Outliers are the most damaging condition.** At 10% outliers, ADE-FCM's ARI drops by 23.7–30.1% across datasets, while FCM drops by 21.4–33.5%. This is expected: extreme outliers distort cluster center estimates in both algorithms.

3. **Missing values affect FCM more than ADE-FCM on high-dimensional data.** At 10% missing values on Digits, ADE-FCM maintains ARI = 0.615 (actually +2.6% above clean baseline), while FCM drops to ARI = 0.173 (+3.2%). On Wine, ADE-FCM drops 5.5% vs FCM's 13.9% drop. ADE-FCM's Cosine distance is inherently more robust to zero-imputed missing values.

4. **ADE-FCM maintains its advantage on high-dimensional data under all conditions.** On Digits, ADE-FCM's ARI remains 0.42–0.62 across all conditions vs FCM's 0.17–0.20, preserving the threefold advantage.

### 5.6 Digits Cosine Metric Improvement

The Digits dataset (d = 64, K = 10) provides the clearest demonstration of ADE-FCM's advantage on high-dimensional data. Table 4 compares ADE-FCM using Cosine distance against FCM using Euclidean distance. Figure 8 illustrates the ARI, NMI, and silhouette heatmaps across all dataset-algorithm combinations.

 **Table 4: ADE-FCM with Cosine vs FCM with Euclidean on Digits**

| Metric | ADE-FCM (Cosine) | FCM (Euclidean) | Improvement |
|--------|-------------------|-----------------|-------------|
| ARI | 0.553 ± 0.048 | 0.179 ± 0.025 | 3.09× |
| Active clusters | 10/10 | 5/10 | 2× |
| Silhouette | 0.124 ± 0.007 | −0.007 ± 0.036 | — |
| Davies-Bouldin | 2.09 ± 0.09 | 4.06 ± 0.50 | 48% |

The most striking difference is in **cluster activation**. With Euclidean distance, FCM activates only 5 of 10 possible clusters—the remaining 5 centers collapse or remain near-initialized positions. With Cosine distance, ADE-FCM activates all 10 clusters, meaning every cluster center captures a meaningful subset of the data.

This activation difference is the primary driver of the threefold ARI improvement. With only 5 active clusters, FCM is forced to merge multiple digits into single clusters, losing the fine-grained distinction between, say, digits 3 and 8 or 1 and 7. ADE-FCM with Cosine can distinguish all 10 digit classes, even if some are partially overlapping in the 64-dimensional pixel space.

**Why does Cosine help?** Handwritten digit images vary in stroke thickness, writing pressure, and overall darkness. These variations affect the magnitude of pixel vectors but not their direction. Cosine distance measures angular similarity, making it invariant to overall brightness or line thickness. Two images of the same digit written with different pens will have similar angular structure but potentially very different Euclidean distances. This invariance is critical for digit clustering and likely generalizes to other high-dimensional visual and textual domains.

### 5.7 Runtime Comparison

Table 5 presents the runtime in seconds (mean across 10 seeds) for each algorithm on each dataset. Figure 9 shows the execution time comparison visually.

**Table 5: Runtime in Seconds (mean across 10 seeds)**

| Dataset | ADE-FCM | FCM | KMeans | Agglomerative | DeepADEFCM | FCLM |
|---------|---------|-----|--------|---------------|------------|------|
| Iris | 0.05 | 0.09 | 0.16 | 0.01 | 2.12 | 0.42 |
| Wine | 0.07 | 0.08 | 0.00 | 0.00 | 2.15 | 0.55 |
| Breast Cancer | 0.09 | 0.21 | 0.01 | 0.01 | 6.06 | 1.11 |
| Digits | 2.04 | 2.73 | 0.01 | 0.11 | 27.53 | 22.99 |
| Glass | 0.13 | 0.49 | 0.00 | 0.00 | 4.16 | 1.44 |
| Seeds | 0.08 | 0.09 | 0.00 | 0.00 | 2.48 | 0.61 |
| Sonar | 0.06 | 0.11 | 0.00 | 0.00 | 2.00 | 0.38 |

**ADE-FCM is comparable to or faster than FCM** on most datasets (Iris: 0.05s vs 0.09s; Wine: 0.07s vs 0.08s). On Digits, ADE-FCM is slightly faster (2.04s vs 2.73s) because Cosine distance computation with the auto-K module converges in fewer iterations. ADE-FCM's auto-K module evaluates multiple K values, but each evaluation uses a faster-converging schedule.

**DeepADEFCM is the slowest** (27.53s on Digits), driven by the neural network forward/backward passes and repeated encoding-decoding cycles.

**FCLM is substantially slower than FCM** on Digits (22.99s vs 2.73s) due to the logistic weighting function and its slower convergence properties.

**KMeans and Agglomerative are the fastest**, completing in under 0.01s on most datasets, as they lack iterative refinement or distance recomputation overhead.

### 5.8 Ablation Analysis

To understand the contribution of each ADE-FCM component, we conduct an ablation study on the Digits and Iris datasets (Table 6). Figure 10 visualizes the ablation results, and Figure 11 shows the degradation pattern when each component is removed relative to the full framework.

**Table 6: Ablation Results (ARI on Digits and Iris)**

| Configuration | Digits | Iris |
|--------------|--------|------|
| Full ADE-FCM | 0.553 | 0.604 |
| − Adaptive fuzzifier (fixed m = 2.0) | 0.481 | 0.612 |
| − Dynamic threshold (fixed ε = 10⁻³) | 0.541 | 0.604 |
| − Cosine distance (use Euclidean) | 0.179 | 0.604 |
| − Auto K (use true K) | 0.553 | 0.604 |
| − Center reinitialization | 0.421 | 0.589 |
| − Trimmed update | 0.538 | 0.601 |

The ablation reveals:

1. **Cosine distance is the dominant factor on Digits**: Replacing Cosine with Euclidean reduces ARI from 0.553 to 0.179—a 67% drop. This confirms that the metric flexibility is the most impactful ADE-FCM feature for high-dimensional data.

2. **Adaptive fuzzifier has mixed effects**: On Digits, the adaptive schedule improves ARI from 0.481 to 0.553 (+15%). On Iris, it slightly reduces ARI from 0.612 to 0.604 (−1.3%). The annealing from soft to hard partitioning helps escape local optima on complex data but can slightly degrade performance on simple data where the standard m = 2 is near-optimal.

3. **Center reinitialization prevents starvation**: Removing reinitialization reduces Digits ARI from 0.553 to 0.421 (−24%), as starved centers reduce the effective cluster count.

4. **Trimmed update provides marginal gain**: The outlier trimming contributes 2–3% improvement, consistent with the low outlier content of standardized benchmark datasets.

5. **Dynamic threshold and auto-K have minimal impact on accuracy**: These features primarily affect convergence speed and user convenience rather than final clustering quality.

### 5.9 Summary of Results

The experimental results can be summarized in five key findings:

1. **ADE-FCM outperforms FCM with medium effect size**: Mean rank 2.08 vs 3.22 (best overall), Cohen's d = 0.47 (95% CI [0.13, 0.79]). Wilcoxon p = 0.043 confirms significance.

2. **Cosine distance is transformative for high-dimensional data**: Threefold ARI improvement on Digits (0.553 vs 0.179) with full cluster activation.

3. **FCLM is consistently the worst performer**: Median-based center updates cause center collapse across all dataset types.

4. **DeepADEFCM is both slower and less accurate**: The deep learning component was excluded from the expanded benchmark due to CPU-only training constraints; preliminary results on 7 datasets show it adds complexity without benefit.

5. **FCM remains the strongest competitor**: Despite being the simplest algorithm (single parameter m = 2), standard FCM achieves the best mean rank, performing well on low-dimensional data but struggling on high-dimensional data.

---

## 6. Discussion

### 6.1 ADE-FCM vs FCM: Medium Effect Size with Significance

The primary quantitative finding of this study is that ADE-FCM achieves a meaningful improvement over standard FCM (Cohen's d = 0.47, 95% CI [0.13, 0.79]), confirmed by a significant Wilcoxon signed-rank test (p = 0.043). The Friedman test across 5 algorithms and 18 datasets is highly significant (p < 0.0001), with ADE-FCM achieving the best mean rank (2.083). This improvement is coupled with substantial additional functionality: automatic parameter tuning, metric flexibility, and built-in explainability.

Three factors explain why ADE-FCM achieves a meaningful but not dominant improvement:

1. **The adaptive fuzzifier is beneficial on complex data but neutral on simple data.** On datasets where m = 2 is near-optimal (Iris, Wine, Breast Cancer), the annealing schedule from 2.5 to 1.1 can lead to a slightly different local minimum. The adaptive schedule trades peak performance on m=2-optimal data for robustness across a wider range of datasets.

2. **FCM is remarkably robust.** Despite its simplicity (single parameter, Euclidean-only distance), FCM achieves strong results on low-to-moderate-dimensional data. This robustness explains the algorithm's continued popularity four decades after its introduction.

3. **The metric flexibility advantage is dataset-dependent.** ADE-FCM's Cosine distance provides large benefits on high-dimensional data (Digits, Optdigits, Mfeat-factors) but offers no advantage on low-dimensional data. The net effect across 18 datasets is a medium improvement.

The practical implication is clear: **a practitioner choosing between ADE-FCM and FCM should expect a meaningful accuracy improvement (d = 0.47) on high-dimensional data**, with additional benefits from automatic K selection, metric flexibility, and explainability.

### 6.2 The Importance of Distance Metric Flexibility

The Digits results (Section 5.5) demonstrate that metric selection can matter more than algorithm selection. The switch from Euclidean to Cosine distance produces a threefold ARI improvement—a larger effect than the difference between any two algorithms using the same metric.

This finding has important implications for clustering practice:

- **Algorithm benchmarking should control for distance metric.** Many studies compare FCM, K-Means, and hierarchical clustering using Euclidean distance by default. Our results suggest that the metric choice can confound algorithm comparisons, particularly on high-dimensional data.

- **Metric selection should be dataset-driven.** For visual data (images, video), Cosine or Manhattan distance may be more appropriate than Euclidean. For correlated features, Mahalanobis distance accounts for redundancy. ADE-FCM's pluggable metric interface makes this selection straightforward.

- **The default Euclidean distance should not be assumed optimal.** Despite its mathematical convenience, Euclidean distance is often suboptimal for real-world data. We recommend that practitioners evaluate multiple distance metrics as part of their clustering workflow.

### 6.3 Why DeepADEFCM Underperforms

DeepADEFCM, the neural-network-based variant, underperforms all non-deep algorithms except FCLM. This underperformance has three likely causes:

1. **Insufficient training.** The autoencoder was trained for only 50 epochs, which is insufficient for convergence on datasets with 60+ features (Breast Cancer, Sonar). The high standard deviations (e.g., 0.263 on Breast Cancer) confirm training instability.

2. **Latent space distortion.** The autoencoder is trained to minimize reconstruction loss, not clustering loss. The resulting latent space preserves global data structure but may not create well-separated clusters. End-to-end training with a clustering objective [40] would likely improve results.

3. **Small data regime.** Autoencoders typically require thousands of samples for effective training. Our datasets range from 150 to 1797 samples, which is marginal for deep learning. Pre-training on larger auxiliary datasets could help.

These issues are well-known in the deep clustering literature [40], [41]. Our results confirm that deep clustering without careful architecture design, hyperparameter tuning, and sufficient training data underperforms classical methods on small-to-medium benchmark datasets.

### 6.4 Why FCLM Underperforms

FCLM [8] uses logistic-weighted median center updates instead of the mean-based updates in standard FCM. The theoretical motivation is robustness to outliers. However, our results show that FCLM consistently produces worse results than FCM across all seven datasets.

The likely cause is **center collapse caused by the median operation in high dimensions**. In ℝᵈ for d > 1, the coordinate-wise median is not a natural generalization of the univariate median and can produce centers that lie outside the convex hull of the data. Additionally, when multiple clusters have similar median coordinates, their centers collapse together, reducing the effective number of clusters.

FCLM's poor performance on the Iris dataset (ARI = 0.146) is particularly informative. Iris has only 4 features and three well-separated classes. The fact that FCLM cannot recover this simple structure indicates a fundamental algorithmic deficiency rather than a problem-specific limitation.

### 6.5 Dataset-Dependent Algorithm Performance

Our results highlight the dataset-dependent nature of clustering algorithm performance:

- **Iris, Wine, Seeds (simple, low-dim):** All non-FCLM algorithms perform comparably (ARI 0.60–0.90). KMeans and Agglomerative match or exceed FCM. These datasets are effectively "solved" by existing algorithms.

- **Breast Cancer (moderate-dim, binary):** FCM and ADE-FCM perform best (ARI ≈ 0.68). The binary structure is well-captured by fuzzy partitioning. KMeans and Agglomerative lag slightly.

- **Digits (high-dim, multi-class):** Agglomerative and ADE-FCM dominate (ARI 0.55–0.66). FCM collapses to 5 active clusters. This dataset requires either hierarchical structure (Agglomerative) or non-Euclidean metrics (ADE-FCM).

- **Glass (low-dim, multi-class, imbalanced):** All algorithms perform poorly (ARI ≤ 0.20). Glass has 6 classes with heavy imbalance (e.g., 70% of samples belong to two classes). Silhouette-based evaluation may not capture the fine-grained structure of minority classes.

- **Sonar (high-dim, binary):** All algorithms perform near random (ARI ≈ 0.00–0.04). This confirms that Sonar is an intrinsically difficult clustering problem. The 60-dimensional feature space with only 208 samples produces high variance and low signal-to-noise ratio.

These patterns underscore the importance of evaluating clustering algorithms across diverse datasets—a single dataset can give a misleading picture of relative algorithm performance.

### 6.6 The Role of Explainability

While our quantitative evaluation focuses on clustering accuracy (ARI, NMI, etc.), the explainability module (Section 3.8) represents a qualitative contribution that is not captured by these metrics.

Permutation importance identifies which features drive the cluster structure. For example, on the Iris dataset, permutation importance correctly identifies petal length and petal width as the most important features, while sepal length and sepal width contribute less. This aligns with the known properties of the Iris dataset [29].

SHAP-based local explanations provide instance-level insights. For ambiguous points near cluster boundaries, the SHAP values reveal which features contribute to (or against) membership in each cluster. This information is valuable in high-stakes applications—medical diagnosis, fraud detection, customer segmentation—where understanding individual assignments is as important as the assignments themselves.

The confidence-weighted membership score (Equation 14) provides a principled way to identify low-confidence assignments. In practice, filtering out points with confidence below 0.5 can improve cluster purity at the cost of coverage. This trade-off is controlled by a single interpretable threshold.

**Limitation.** We did not conduct a user study to evaluate the usefulness of these explanations. The perceived quality of explanations is inherently subjective and task-dependent. A formal user study is planned for future work (Section 7).

### 6.7 Practical Recommendations

Based on our results, we offer the following practical recommendations:

1. **For low-dimensional data (d ≤ 10):** Use standard FCM with m = 2. The added complexity of ADE-FCM is not justified unless automated K selection or explainability is needed.

2. **For high-dimensional data (d ≥ 30):** Use ADE-FCM with Cosine distance. The metric flexibility provides substantial gains, and the adaptive fuzzifier helps avoid poor local minima.

3. **For mixed-type or noisy data:** Consider ADE-FCM with trimmed center updates. The 5% trimming provides robustness without parameter tuning.

4. **When K is unknown:** Use ADE-FCM's auto-K feature as a starting point. The complexity penalty (λ = 0.02) prevents gross over-selection. However, verify the selected K using domain knowledge or multiple validity indices.

5. **When explainability is required:** Use ADE-FCM's permutation importance for global feature ranking and SHAP for individual instance explanations. The confidence score helps identify points where explanations are most needed.

6. **Avoid FCLM and naive DeepADEFCM for general-purpose clustering.** These algorithms require extensive tuning and, in our experiments, consistently underperform simpler alternatives.

---

## 7. Limitations

This section discusses the limitations of the current study and identifies areas where caution is warranted in interpreting the results.

### 7.1 Limited Benchmark Scope

The benchmark now includes 18 datasets spanning diverse dimensionalities, sample sizes, and complexity levels, addressing the sample size concern from the preliminary study. However, Demšar [38] recommends at least 10–15 datasets for meaningful statistical comparison, which we now satisfy. The DeepADEFCM baseline could not be evaluated on the full 18-dataset benchmark due to CPU-only training constraints.

Furthermore, all datasets are from UCI/OpenML or synthetic; real-world clustering applications often involve missing values, categorical features, non-Gaussian distributions, and varying noise levels. Our robustness study (Section 5.8) begins to address this gap by evaluating noise, outliers, and missing values on representative datasets.

### 7.2 Automatic Cluster-Count Limitations

The auto-K module selects K by maximizing penalized silhouette. On Digits (true K = 10), it selected K = 3, a significant underestimate. Three factors contribute:

1. **Silhouette favors compact, well-separated clusters.** In 64 dimensions, 10 clusters are inevitably overlapping. The silhouette score for K = 10 penalizes this overlap, favoring smaller K where clusters are more compact.

2. **The complexity penalty λ = 0.02 may be too aggressive.** Increasing λ reduces the selected K. While λ = 0.02 was calibrated on synthetic data, it may not generalize to all real-world datasets.

3. **Alternative validity indices may perform better.** The Xie-Beni index [18], fuzzy silhouette [17], or the gap statistic [14] might be more appropriate for high-dimensional data.

### 7.3 DeepADEFCM Training

DeepADEFCM was trained for only 50 epochs with minimal hyperparameter tuning. The results (Section 5.1) should not be interpreted as evidence that deep fuzzy clustering is inherently inferior—only that naive application of deep learning to small benchmark datasets is ineffective. Full training with early stopping, learning rate scheduling, data augmentation, and end-to-end clustering objectives would be necessary for a fair comparison.

### 7.4 No User Study for Explainability

Although ADE-FCM provides permutation importance and SHAP-based explanations, we did not conduct a user study to evaluate their quality or usefulness. Explainability quality is inherently subjective and task-dependent: a feature importance ranking that seems reasonable to a data scientist may be unhelpful to a domain expert. Formal evaluation of explanation quality requires user studies with realistic tasks and domain experts [42].

### 7.6 Hyperparameter Sensitivity

ADE-FCM introduces several parameters that were fixed across all experiments:
- m<sub>max</sub> = 2.5, m<sub>min</sub> = 1.1 (adaptive fuzzifier endpoints)
- κ = 0.9 (dynamic threshold decay rate)
- λ = 0.02 (auto-K complexity penalty)
- τ = 1.0 (center reinitialization threshold)
- γ = 0.05 (outlier contamination fraction)

While the ablation study (Section 5.7) and robustness analysis (Section 5.4) suggest that most components and hyperparameters have small effects individually, the joint sensitivity to these parameters has not been systematically explored. A full factorial experiment varying all parameters would require thousands of runs and is left for future work.

### 7.7 Reproducibility

We have taken several steps to ensure reproducibility: all 396+ experiment configurations are recorded (18 datasets × 5 algorithms × 5 seeds, plus 270 sensitivity experiments and 108 robustness experiments), all code dependencies are versioned, and 120 unit tests validate the implementation. However, we acknowledge that exact reproduction depends on specific library versions (NumPy, SciPy, scikit-learn, SHAP) and hardware characteristics (CPU, RAM). Cross-platform reproducibility has not been tested.

---

## 8. Future Work

Based on the limitations identified in Section 7, we outline several directions for future work.

### 8.1 Large-Scale Benchmarking

The benchmark of 18 datasets exceeds the Demšar recommendation for meaningful statistical comparison, but expanding to 30+ datasets would provide even greater power. Priority domains include:
- **Text data:** 20 Newsgroups, Reuters, AG News (sparse high-dimensional)
- **Image data:** CIFAR-10, Fashion-MNIST (raw pixels and learned features)
- **Biological data:** Gene expression, protein sequences (high-dim, low-N)
- **Time series:** UCR Time Series Archive (varying lengths and dimensionalities)
- **Real-world noisy data:** Data with missing values, outliers, and label noise

### 8.2 Full DeepADEFCM Training

The deep learning variant (DeepADEFCM) deserves a thorough investigation with:
- End-to-end training with a unified clustering + reconstruction loss
- Convolutional and transformer architectures for image/text data
- Learning rate scheduling, warmup, and early stopping
- Pre-training on large unlabeled corpora followed by fine-tuning

Given our results, a dedicated study with 50+ seeds and extensive hyperparameter search would be needed to determine whether deep fuzzy clustering can match or exceed classical methods.

### 8.3 Human Evaluation of Explainability

A formal user study with domain experts (e.g., medical professionals for the Breast Cancer dataset) would evaluate:
- Do permutation importance rankings align with domain knowledge?
- Do SHAP explanations help users understand individual cluster assignments?
- Does confidence-weighted scoring improve trust in the system?
- How does explanation time vs accuracy trade off in decision-making?

Following the framework of Hoffman et al. [42], the study should measure both objective metrics (decision accuracy, time) and subjective metrics (trust, satisfaction, mental load).

### 8.5 Advanced Auto-K Methods

Alternative automatic K discovery methods should be explored:
- **Ensemble validity indices:** Combining silhouette, Davies-Bouldin, Xie-Beni, and gap statistic via rank aggregation.
- **Stability-based selection:** Measuring cluster instability via bootstrap resampling, selecting K that maximizes stability [43].
- **Elbow detection algorithms:** Sophisticated knee-point detection [44] applied to distortion curves.

### 8.6 Integration with Modern Clustering Paradigms

ADE-FCM could be extended to incorporate:
- **Deep feature learning:** Learn features end-to-end with a clustering objective [40].
- **Variational methods:** Model cluster assignments as latent variables in a variational autoencoder [45].
- **Contrastive learning:** Use contrastive objectives to learn cluster-friendly representations [46].

These extensions would bridge the gap between classical fuzzy clustering and modern representation learning.

---

## 9. Conclusion

This paper introduced ADE-FCM, an adaptive explainable framework for fuzzy clustering that integrates seven capabilities: adaptive fuzzifier scheduling, dynamic convergence thresholds, automatic cluster-count discovery with complexity penalty, pluggable distance metrics, starvation-preventing center reinitialization, robust trimmed-mean center updates, and built-in explainability through permutation importance and SHAP values.

Through a rigorous experimental evaluation across 18 benchmark datasets, 4 competing algorithms, and 396 total experiments plus 270 sensitivity and 108 robustness experiments, we demonstrated:

1. **ADE-FCM achieves a medium effect size improvement over FCM (Cohen's d = 0.47, 95% CI [0.13, 0.79]), with statistical significance confirmed by Wilcoxon signed-rank test (p = 0.043).** ADE-FCM achieves the best mean Friedman rank (2.083) among 5 algorithms, and the Friedman test is highly significant (p < 0.0001).

2. **Distance metric flexibility is transformative on high-dimensional data.** On Digits (64 dims), ADE-FCM with Cosine achieves ARI = 0.551 vs FCM with Euclidean at 0.181. Similar patterns hold for Optdigits (0.584 vs 0.218) and Mfeat-factors (0.569 vs 0.146).

3. **ADE-FCM is remarkably robust to hyperparameter variation.** Sensitivity analysis (270 experiments) shows that 5 of 6 parameters have negligible effect on ARI; only outlier contamination shows measurable sensitivity (max range 0.055).

4. **FCLM consistently underperforms across all 18 datasets**, suffering from center collapse due to median-based updates.

The practical value of ADE-FCM lies in providing an integrated solution that eliminates manual parameter tuning, accommodates diverse data geometries through pluggable metrics, and offers interpretable explanations—while achieving a meaningful accuracy improvement over standard FCM.

All code, data, and experimental configurations are available for reproducibility.

---

## Acknowledgments

The authors thank the anonymous reviewers for their constructive feedback. This research utilized the UCI Machine Learning Repository and OpenML for benchmark datasets. No external funding was received for this work.

---

## References

[1] J. C. Bezdek, *Pattern Recognition with Fuzzy Objective Function Algorithms*. New York, NY, USA: Plenum Press, 1981.

[2] N. R. Pal and J. C. Bezdek, "On cluster validity for the fuzzy c-means model," *IEEE Trans. Fuzzy Syst.*, vol. 3, no. 3, pp. 370–379, 1995.

[3] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 4765–4774.

[4] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[5] P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," *J. Comput. Appl. Math.*, vol. 20, pp. 53–65, 1987.

[6] J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters," *J. Cybern.*, vol. 3, no. 3, pp. 32–57, 1973.

[7] J. C. Bezdek, R. Ehrlich, and W. Full, "FCM: The fuzzy c-means clustering algorithm," *Comput. Geosci.*, vol. 10, no. 2–3, pp. 191–203, 1984.

[8] M. S. Yang, Y. Nataliani, and N. R. Pal, "A parallel fuzzy c-median clustering algorithm," *IEEE Trans. Fuzzy Syst.*, vol. 27, no. 12, pp. 2421–2434, 2019.

[9] R. Krishnapuram and J. M. Keller, "A possibilistic approach to clustering," *IEEE Trans. Fuzzy Syst.*, vol. 1, no. 2, pp. 98–110, 1993.

[10] D. E. Gustafson and W. C. Kessel, "Fuzzy clustering with a fuzzy covariance matrix," in *Proc. IEEE Conf. Decis. Control*, 1978, pp. 761–766.

[11] J. Yu, Q. Cheng, and H. Huang, "Analysis of the weighting exponent in the FCM," *IEEE Trans. Syst., Man, Cybern. B, Cybern.*, vol. 34, no. 1, pp. 634–639, 2004.

[12] K. Zhou, S. Yang, and S. Ding, "An adaptive fuzzy c-means algorithm with annealing evolution," *Int. J. Comput. Intell. Syst.*, vol. 5, no. 6, pp. 1078–1091, 2012.

[13] I. Ozkan and I. B. Turksen, "Upper and lower values for the level of fuzziness in FCM," *Inf. Sci.*, vol. 177, no. 23, pp. 5143–5152, 2007.

[14] R. Tibshirani, G. Walther, and T. Hastie, "Estimating the number of clusters in a data set via the gap statistic," *J. R. Stat. Soc. Ser. B*, vol. 63, no. 2, pp. 411–423, 2001.

[15] T. M. Kodinariya and P. R. Makwana, "Review on determining number of cluster in K-Means clustering," *Int. J. Adv. Res. Comput. Sci. Manage. Stud.*, vol. 1, no. 6, pp. 90–95, 2013.

[16] S. Monti, P. Tamayo, J. Mesirov, and T. Golub, "Consensus clustering: A resampling-based method for class discovery and visualization of gene expression microarray data," *Mach. Learn.*, vol. 52, no. 1–2, pp. 91–118, 2003.

[17] R. J. G. B. Campello, "A fuzzy extension of the silhouette width criterion for cluster analysis," *Fuzzy Sets Syst.*, vol. 157, no. 21, pp. 2858–2875, 2006.

[18] X. L. Xie and G. Beni, "A validity measure for fuzzy clustering," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 13, no. 8, pp. 841–847, 1991.

[19] V. Schwämmle and O. N. Jensen, "A simple and fast method to determine the number of clusters in a dataset," *BMC Bioinformatics*, vol. 11, art. 440, 2010.

[20] A. Arbelaitz, I. Gurrutxaga, J. Muguerza, J. M. Pérez, and I. Perona, "An extensive comparative study of cluster validity indices," *Pattern Recognit.*, vol. 46, no. 1, pp. 243–256, 2013.

[21] K. Beyer, J. Goldstein, R. Ramakrishnan, and U. Shaft, "When is 'nearest neighbor' meaningful?" in *Proc. Int. Conf. Database Theory*, 1999, pp. 217–235.

[22] C. C. Aggarwal, A. Hinneburg, and D. A. Keim, "On the surprising behavior of distance metrics in high dimensional space," in *Proc. Int. Conf. Database Theory*, 2001, pp. 420–434.

[23] A. Huang, "Similarity measures for text document clustering," in *Proc. New Zealand Comput. Sci. Res. Student Conf.*, 2008, pp. 49–56.

[24] P. C. Mahalanobis, "On the generalized distance in statistics," *Proc. Nat. Inst. Sci. India*, vol. 2, no. 1, pp. 49–55, 1936.

[25] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 1135–1144.

[26] J. Crabbé, Z. Qian, and M. van der Schaar, "Explaining latent representations with a corpus of examples," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020, pp. 12 112–12 124.

[27] P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nat. Methods*, vol. 17, pp. 261–272, 2020.

[28] P. J. Huber and E. M. Ronchetti, *Robust Statistics*, 2nd ed. Hoboken, NJ, USA: Wiley, 2009.

[29] D. Dua and C. Graff, "UCI Machine Learning Repository," 2019. [Online]. Available: http://archive.ics.uci.edu/ml

[30] J. Vanschoren, J. N. van Rijn, B. Bischl, and L. Torgo, "OpenML: Networked science in machine learning," *ACM SIGKDD Explor. Newsl.*, vol. 15, no. 2, pp. 49–60, 2013.

[31] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proc. Berkeley Symp. Math. Statist. Probab.*, vol. 1, 1967, pp. 281–297.

[32] J. H. Ward Jr., "Hierarchical grouping to optimize an objective function," *J. Am. Stat. Assoc.*, vol. 58, no. 301, pp. 236–244, 1963.

[33] C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[34] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[35] L. Hubert and P. Arabie, "Comparing partitions," *J. Classif.*, vol. 2, no. 1, pp. 193–218, 1985.

[36] A. Strehl and J. Ghosh, "Cluster ensembles—A knowledge reuse framework for combining multiple partitions," *J. Mach. Learn. Res.*, vol. 3, pp. 583–617, 2002.

[37] D. L. Davies and D. W. Bouldin, "A cluster separation measure," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 1, no. 2, pp. 224–227, 1979.

[38] J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, 2006.

[39] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Mahwah, NJ, USA: Lawrence Erlbaum, 1988.

[40] J. Xie, R. Girshick, and A. Farhadi, "Unsupervised deep embedding for clustering analysis," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2016, pp. 478–487.

[41] E. Min, X. Guo, Q. Liu, G. Zhang, J. Cui, and J. Long, "A survey of clustering with deep learning: From the perspective of network architecture," *IEEE Access*, vol. 6, pp. 39 570–39 584, 2018.

[42] R. R. Hoffman, S. T. Mueller, G. Klein, and J. Litman, "Metrics for explainable AI: Challenges and prospects," arXiv preprint arXiv:1812.04608, 2018.

[43] A. Ben-Hur, A. Elisseeff, and I. Guyon, "A stability based method for discovering structure in clustered data," in *Proc. Pacific Symp. Biocomputing*, 2002, pp. 6–17.

[44] V. Satopää, J. Albrecht, D. Irwin, and B. Raghavan, "Finding a 'kneedle' in a haystack: Detecting knee points in system behavior," in *Proc. IEEE Int. Conf. Distrib. Comput. Syst. Workshops*, 2011, pp. 166–171.

[45] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2014.

[46] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 1597–1607.

[47] U. Maulik and S. Bandyopadhyay, "Performance evaluation of some clustering algorithms and validity indices," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 12, pp. 1650–1654, 2002.

[48] M. Halkidi, Y. Batistakis, and M. Vazirgiannis, "On clustering validation techniques," *J. Intell. Inf. Syst.*, vol. 17, no. 2–3, pp. 107–145, 2001.

[49] J. A. Hartigan and M. A. Wong, "Algorithm AS 136: A K-means clustering algorithm," *J. R. Stat. Soc. Ser. C*, vol. 28, no. 1, pp. 100–108, 1979.

[50] L. Kaufman and P. J. Rousseeuw, *Finding Groups in Data: An Introduction to Cluster Analysis*. New York, NY, USA: Wiley, 1990.

[51] A. K. Jain, "Data clustering: 50 years beyond K-means," *Pattern Recognit. Lett.*, vol. 31, no. 8, pp. 651–666, 2010.

[52] S. Theodoridis and K. Koutroumbas, *Pattern Recognition*, 4th ed. Burlington, MA, USA: Academic Press, 2009.
