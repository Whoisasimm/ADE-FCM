"""
ADE-FCM: Adaptive Distributed Explainable Fuzzy C-Means
=======================================================
Novel contributions:
1. KMeans++ Initialization          - Improved starting centroids
2. Density-Based Initialization     - High-density region seeding
3. Adaptive Fuzzifier m(t)          - Time-varying fuzzy exponent
4. Confidence Weighted Membership   - Entropy-based confidence scoring
5. Automatic Cluster Discovery      - Consensus silhouette/gap/DB/BC
6. Outlier Robust Membership        - Statistically flagged outliers
7. Early Stopping                   - Patience-based convergence
8. Dynamic Convergence Threshold    - Shrinking epsilon schedule
9. Explainable Clustering (XAI)     - Feature importances + summaries
10. Distributed Spark Optimization  - Broadcast/mapPartitions/treeAggregate
"""

from .ade_fcm import ADEFCM
from .density_init import DensityInitializer, KMeansPlusPlusInitializer
from .adaptive_params import AdaptiveFuzzifier, DynamicThreshold, EarlyStopping
from .auto_cluster import AutomaticClusterDiscovery, ClusterEvaluator
from .spark_ade_fcm import SparkADEFCM
from .xai import (
    explain_clusters,
    feature_importance,
    cluster_summary,
    describe_cluster_natural,
    shap_explain,
)
from .outlier_detector import OutlierDetector

__all__ = [
    "ADEFCM",
    "DensityInitializer",
    "KMeansPlusPlusInitializer",
    "AdaptiveFuzzifier",
    "DynamicThreshold",
    "EarlyStopping",
    "AutomaticClusterDiscovery",
    "ClusterEvaluator",
    "SparkADEFCM",
    "explain_clusters",
    "feature_importance",
    "cluster_summary",
    "describe_cluster_natural",
    "shap_explain",
    "OutlierDetector",
]
