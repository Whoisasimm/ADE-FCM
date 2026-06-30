"""
Explainable AI module for ADE-FCM clustering.
Generates cluster summaries, feature importance, and natural language descriptions.
"""
import numpy as np
import json
from loguru import logger


class ClusterExplainer:
    """Explain clustering results with feature importance and NL descriptions."""

    def __init__(self, model=None, X=None, feature_names=None):
        self.model = model
        self.X = X
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])] if X is not None else []
        self.cluster_info_ = None

    def feature_importance(self, cluster_id):
        """Compute feature importance for a specific cluster.
        FI_jf = sum_i u_ij^m * |x_if - v_jf| / sum_f' sum_i u_ij^m * |x_if' - v_jf'|
        """
        if self.model is None or self.X is None:
            return {}
        centers = self.model.centers_
        U = self.model.U_
        m = getattr(self.model, 'm', 2.0)
        if isinstance(m, str):
            m = 2.0

        Um = U ** m
        n_features = self.X.shape[1]
        fi = np.zeros(n_features)

        for f in range(n_features):
            weighted_diff = Um[:, cluster_id] * np.abs(self.X[:, f] - centers[cluster_id, f])
            fi[f] = np.sum(weighted_diff)

        total = np.sum(fi)
        if total > 0:
            fi = fi / total

        return {self.feature_names[f]: float(fi[f]) for f in range(n_features)}

    def cluster_summary(self, cluster_id):
        """Generate statistical summary for a cluster."""
        if self.X is None or self.model is None:
            return {}
        labels = self.model.labels_ if hasattr(self.model, 'labels_') else np.argmax(self.model.U_, axis=1)
        mask = labels == cluster_id
        if not mask.any():
            return {"size": 0, "percentage": 0.0}

        cluster_data = self.X[mask]
        summary = {
            "cluster_id": int(cluster_id),
            "size": int(np.sum(mask)),
            "percentage": float(np.mean(mask) * 100),
            "features": {}
        }
        for i, name in enumerate(self.feature_names):
            summary["features"][name] = {
                "mean": float(np.mean(cluster_data[:, i])),
                "std": float(np.std(cluster_data[:, i])),
                "min": float(np.min(cluster_data[:, i])),
                "max": float(np.max(cluster_data[:, i])),
                "importance": self.feature_importance(cluster_id).get(name, 0)
            }
        return summary

    def global_explanation(self):
        """Generate explanations for all clusters."""
        if self.model is None:
            return []
        n_clusters = self.model.centers_.shape[0]
        return [self.cluster_summary(i) for i in range(n_clusters)]

    def natural_language_description(self, cluster_id):
        """Generate human-readable cluster description."""
        summary = self.cluster_summary(cluster_id)
        if summary.get("size", 0) == 0:
            return f"Cluster {cluster_id} is empty."

        fi = self.feature_importance(cluster_id)
        top_features = sorted(fi.items(), key=lambda x: -x[1])[:3]

        desc = f"Cluster {cluster_id} contains {summary['size']} points "
        desc += f"({summary['percentage']:.1f}% of total). "

        if top_features:
            desc += "Key distinguishing features: "
            desc += ", ".join([f"{name} (importance: {imp:.3f})" for name, imp in top_features])
            desc += ". "

        desc += "Characteristic values: "
        for name, _ in top_features:
            feat = summary['features'].get(name, {})
            desc += f"{name}={feat.get('mean', 0):.3f} "

        return desc

    def generate_report(self, output_path="xai_report.json"):
        """Generate complete XAI report."""
        report = {
            "n_clusters": self.model.centers_.shape[0] if self.model is not None else 0,
            "n_features": len(self.feature_names),
            "n_samples": len(self.X) if self.X is not None else 0,
            "clusters": self.global_explanation(),
            "nl_descriptions": {}
        }
        if self.model is not None:
            n_clusters = self.model.centers_.shape[0]
            for i in range(n_clusters):
                report["nl_descriptions"][f"cluster_{i}"] = self.natural_language_description(i)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"XAI report saved to {output_path}")
        return report
