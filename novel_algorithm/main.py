"""
ADE-FCM: Full Pipeline
=======================
Integrates all 10 contributions into a unified pipeline:

1. KMeans++ Initialization
2. Density-Based Initialization
3. Adaptive Fuzzifier m(t)
4. Confidence Weighted Membership
5. Automatic Cluster Discovery
6. Outlier Robust Membership
7. Early Stopping
8. Dynamic Convergence Threshold
9. Explainable Clustering (XAI)
10. Distributed Spark Optimization

Provides high-level `ade_fcm_pipeline()` and `ADEFCMPipeline` class.
"""

import numpy as np
from loguru import logger


def ade_fcm_pipeline(
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
    spark_checkpoint_dir=None,
    explain=True,
    feature_names=None,
):
    """Run the complete ADE-FCM pipeline.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    n_clusters : int or 'auto', default='auto'
    max_iter : int, default=300
    m : float or 'adaptive', default='adaptive'
    epsilon : float or 'dynamic', default='dynamic'
    init_method : {'kmeans++', 'density', 'random'}, default='kmeans++'
    early_stopping_patience : int, default=10
    outlier_threshold : float, default=2.0
    random_state : int, default=42
    verbose : bool, default=True
    use_spark : bool, default=False
        If True, use SparkADEFCM for distributed execution.
    spark_master : str, default='local[*]'
    spark_checkpoint_dir : str or None
    explain : bool, default=True
        If True, generate XAI explanations after fitting.
    feature_names : list of str or None

    Returns
    -------
    result : dict
        Pipeline result with keys:
            - labels
            - centers
            - U (membership)
            - n_iter
            - J_history
            - outlier_mask
            - outlier_scores
            - summaries (if explain=True)
            - descriptions (if explain=True)
            - feature_importances (if explain=True)
            - explanation (if explain=True)
    """
    n, d = X.shape
    logger.info(f"ADE-FCM Pipeline: {n} samples, {d} features")

    if use_spark:
        from .spark_ade_fcm import SparkADEFCM

        model = SparkADEFCM(
            n_clusters=n_clusters,
            max_iter=max_iter,
            m=m,
            epsilon=epsilon,
            init_method=init_method,
            early_stopping_patience=early_stopping_patience,
            outlier_threshold=outlier_threshold,
            random_state=random_state,
            verbose=verbose,
            spark_master=spark_master,
            checkpoint_dir=spark_checkpoint_dir,
        )
    else:
        from .ade_fcm import ADEFCM

        model = ADEFCM(
            n_clusters=n_clusters,
            max_iter=max_iter,
            m=m,
            epsilon=epsilon,
            init_method=init_method,
            early_stopping_patience=early_stopping_patience,
            outlier_threshold=outlier_threshold,
            random_state=random_state,
            verbose=verbose,
        )

    model.fit(X)

    result = {
        "labels": model.labels_,
        "centers": model.centers_,
        "U": model.U_,
        "n_iter": model.n_iter_,
        "n_clusters": model.n_clusters,
        "J_history": model.J_history_,
        "outlier_mask": model.outlier_mask_,
        "outlier_scores": getattr(model, "outlier_scores_", None),
    }

    if explain:
        from .xai import (
            explain_clusters,
            feature_importance,
            cluster_summary,
            describe_cluster_natural,
        )

        # Compute importances
        f_imp = feature_importance(
            X, model.labels_, model.centers_, method="shift"
        )
        result["feature_importances"] = f_imp

        # Cluster summaries
        summaries = cluster_summary(
            X, model.labels_, model.centers_, feature_names=feature_names
        )
        result["summaries"] = summaries

        # Natural language descriptions
        descriptions = [
            describe_cluster_natural(s, feature_names) for s in summaries
        ]
        result["descriptions"] = descriptions

        # Full explanation dict
        result["explanation"] = explain_clusters(
            X,
            model.labels_,
            model.centers_,
            feature_names=feature_names,
            outlier_mask=model.outlier_mask_,
        )

        # Per-outlier detection summary
        if model.outlier_mask_ is not None and model.outlier_mask_.sum() > 0:
            outlier_indices = np.where(model.outlier_mask_)[0]
            result["outlier_indices"] = outlier_indices
            result["outlier_details"] = {
                "count": int(model.outlier_mask_.sum()),
                "ratio": float(model.outlier_mask_.mean()),
                "indices": outlier_indices.tolist(),
            }

    if use_spark:
        model.stop()

    logger.info("ADE-FCM Pipeline complete.")
    return result


class ADEFCMPipeline:
    """Pipeline class for ADE-FCM with persistent state.

    Parameters
    ----------
    n_clusters : int or 'auto', default='auto'
    max_iter : int, default=300
    m : float or 'adaptive', default='adaptive'
    epsilon : float or 'dynamic', default='dynamic'
    init_method : {'kmeans++', 'density', 'random'}, default='kmeans++'
    early_stopping_patience : int, default=10
    outlier_threshold : float, default=2.0
    random_state : int, default=42
    verbose : bool, default=True
    use_spark : bool, default=False
    spark_master : str, default='local[*]'
    spark_checkpoint_dir : str or None
    """

    def __init__(
        self,
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
        spark_checkpoint_dir=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.init_method = init_method
        self.early_stopping_patience = early_stopping_patience
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.use_spark = use_spark
        self.spark_master = spark_master
        self.spark_checkpoint_dir = spark_checkpoint_dir
        self.model_ = None
        self.result_ = None

    def fit(self, X, feature_names=None, explain=True):
        """Fit the pipeline.

        Parameters
        ----------
        X : ndarray
        feature_names : list of str or None
        explain : bool, default=True

        Returns
        -------
        self : ADEFCMPipeline
        """
        self.result_ = ade_fcm_pipeline(
            X,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            m=self.m,
            epsilon=self.epsilon,
            init_method=self.init_method,
            early_stopping_patience=self.early_stopping_patience,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
            verbose=self.verbose,
            use_spark=self.use_spark,
            spark_master=self.spark_master,
            spark_checkpoint_dir=self.spark_checkpoint_dir,
            explain=explain,
            feature_names=feature_names,
        )
        return self

    def predict(self, X):
        """Predict labels for new data.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        labels : ndarray
        """
        if self.use_spark:
            from .spark_ade_fcm import SparkADEFCM
            # Re-create a minimal model with the learned centers
            # This requires storing the model; for now we rely on
            # the user having called fit() and the model being available
            raise NotImplementedError(
                "Predict with SparkADEFCM is not yet supported "
                "via the pipeline. Use ade_fcm_pipeline directly."
            )
        else:
            from .ade_fcm import ADEFCM

            model = ADEFCM(
                n_clusters=self.result_["n_clusters"],
                m=self.m,
                random_state=self.random_state,
                verbose=False,
            )
            model.centers_ = self.result_["centers"]
            return model.predict(X)

    def summary(self):
        """Return a text summary of the pipeline result.

        Returns
        -------
        text : str
        """
        if self.result_ is None:
            return "Pipeline not fitted yet."
        r = self.result_
        lines = [
            "=" * 60,
            "ADE-FCM Pipeline Summary",
            "=" * 60,
            f"Dataset: {r['labels'].shape[0]} samples",
            f"Clusters: {r['n_clusters']}",
            f"Iterations: {r['n_iter']}",
            f"Final objective J: {r['J_history'][-1]:.4f}" if r["J_history"] else "",
            "",
            "Outlier Detection:",
            f"  Method: Weighted distance (threshold={2.0} sigma)",
            f"  Outliers: {r.get('outlier_details', {}).get('count', 'N/A')} "
            f"({r.get('outlier_details', {}).get('ratio', 0)*100:.1f}%)",
            "",
        ]
        if "descriptions" in r and r["descriptions"]:
            lines.append("Cluster Descriptions:")
            lines.append("-" * 40)
            for desc in r["descriptions"]:
                lines.append(desc)
                lines.append("")
        if "explanation" in r:
            exp = r["explanation"]
            if "top_features_global" in exp:
                lines.append("Global Top Features:")
                for name, score in exp["top_features_global"]:
                    lines.append(f"  {name}: {score:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        if self.result_ is None:
            return "ADEFCMPipeline(unfitted)"
        return (
            f"ADEFCMPipeline(n_clusters={self.result_['n_clusters']}, "
            f"n_iter={self.result_['n_iter']}, "
            f"outliers={self.result_.get('outlier_details', {}).get('count', 'N/A')})"
        )


# ---------------------------------------------------------------------------
# Convenience demo runner
# ---------------------------------------------------------------------------
def demo():
    """Run a small demonstration of the ADE-FCM pipeline on synthetic data."""
    logger.info("Running ADE-FCM demo...")
    rng = np.random.RandomState(42)

    # Generate 3 well-separated Gaussian blobs
    n_per_cluster = 100
    X_list = []
    for center in [(0, 0), (5, 5), (10, 0)]:
        X_list.append(
            rng.randn(n_per_cluster, 2) + np.array(center)
        )
    X = np.vstack(X_list)
    logger.info(f"Generated {X.shape[0]} samples with 3 true clusters.")

    # Run pipeline (auto-discover clusters)
    result = ade_fcm_pipeline(
        X,
        n_clusters="auto",
        max_iter=200,
        init_method="kmeans++",
        verbose=True,
        explain=True,
        feature_names=["x", "y"],
    )

    logger.info(f"Discovered K = {result['n_clusters']}")
    logger.info(f"Iterations: {result['n_iter']}")
    logger.info(f"Outliers: {result.get('outlier_details', {}).get('count', 0)}")

    if "descriptions" in result:
        for desc in result["descriptions"]:
            logger.info("\n" + desc)

    return result


if __name__ == "__main__":
    demo()
