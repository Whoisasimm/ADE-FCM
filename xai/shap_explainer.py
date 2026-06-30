"""
SHAP-based explainer for ADE-FCM clustering.
Computes SHAP values for each cluster decision using KernelSHAP.
"""
import numpy as np
from loguru import logger


class ShapExplainer:
    """SHAP-based explanation of cluster membership decisions.

    For each cluster, treats the membership function as a black-box
    model and uses KernelSHAP to attribute feature contributions
    to the predicted membership value.
    """

    def __init__(self, model=None, X=None, feature_names=None, nsamples=500):
        self.model = model
        self.X = np.asarray(X, dtype=np.float64) if X is not None else None
        self.feature_names = feature_names
        self.nsamples = nsamples
        self.shap_values_ = None
        self.base_values_ = None
        self.expected_values_ = None

    def _membership_predictor(self, cluster_id):
        """Return a callable that predicts membership for cluster_id."""
        def predict(X_input):
            X_arr = np.asarray(X_input, dtype=np.float64)
            m_val = getattr(self.model, 'm', 2.0)
            if isinstance(m_val, str):
                m_val = 2.0
            U = self.model._update_membership(X_arr, self.model.centers_, m_val)
            return U[:, cluster_id]
        return predict

    def fit(self, X=None, feature_names=None):
        """Compute SHAP values for all clusters.

        Parameters
        ----------
        X : ndarray, optional
        feature_names : list, optional
        """
        if X is not None:
            self.X = np.asarray(X, dtype=np.float64)
        if feature_names is not None:
            self.feature_names = feature_names
        if self.X is None:
            raise ValueError("No data provided for SHAP explanation.")
        if self.model is None or self.model.centers_ is None:
            raise ValueError("Model must be fitted before SHAP explanation.")

        try:
            import shap
        except ImportError:
            logger.warning("shap not installed; install with: pip install shap")
            raise

        n_clusters = self.model.centers_.shape[0]
        n_samples = self.X.shape[0]
        n_features = self.X.shape[1]
        rng = np.random.RandomState(42)

        self.shap_values_ = []
        self.expected_values_ = []

        bg_size = min(100, n_samples)
        background = self.X[rng.choice(n_samples, bg_size, replace=False)]

        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        for cid in range(n_clusters):
            explainer = shap.KernelExplainer(self._membership_predictor(cid), background)
            nsamp = min(self.nsamples, n_samples * n_features)
            shaps = explainer.shap_values(self.X, nsamples=nsamp)
            self.shap_values_.append(shaps)
            self.expected_values_.append(float(explainer.expected_value))

        return self

    def cluster_shap_summary(self, cluster_id):
        """Aggregate SHAP values for a single cluster.

        Returns dict with normalized mean absolute SHAP per feature.
        """
        shap_mat = self.shap_values_[cluster_id]
        mean_abs = np.mean(np.abs(shap_mat), axis=0)
        total = mean_abs.sum()
        if total > 0:
            mean_abs = mean_abs / total

        feature_imp = {
            self.feature_names[i]: float(mean_abs[i])
            for i in range(len(self.feature_names))
        }
        top = sorted(feature_imp.items(), key=lambda x: -x[1])[:5]
        return {
            "cluster_id": int(cluster_id),
            "mean_abs_shap": feature_imp,
            "top_features": [{"name": n, "importance": v} for n, v in top],
            "expected_value": self.expected_values_[cluster_id],
        }

    def global_shap_summary(self):
        """Aggregate SHAP values across all clusters."""
        return [self.cluster_shap_summary(cid) for cid in range(len(self.shap_values_))]

    def shap_decision_plot_data(self, cluster_id, top_n=10):
        """Return per-sample SHAP values for a beeswarm-style summary."""
        shap_mat = self.shap_values_[cluster_id]
        mean_abs = np.mean(np.abs(shap_mat), axis=0)
        top_idx = np.argsort(-mean_abs)[:top_n]
        return {
            "feature_names": [self.feature_names[i] for i in top_idx],
            "shap_values": shap_mat[:, top_idx].tolist(),
            "feature_values": self.X[:, top_idx].tolist(),
            "expected_value": self.expected_values_[cluster_id],
        }
