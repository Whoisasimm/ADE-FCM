"""
XAI Visualization module for ADE-FCM.
Feature importance bar charts, cluster profile radar charts,
parallel coordinate plots, and scatter visualizations.
"""
import numpy as np
from pathlib import Path
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
import seaborn as sns
from math import pi

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
})
sns.set_style("whitegrid")


class XAIVisualizer:
    """Generate XAI visualizations for ADE-FCM clustering results."""

    def __init__(self, model=None, X=None, feature_names=None, output_dir="xai_plots"):
        self.model = model
        self.X = np.asarray(X, dtype=np.float64) if X is not None else None
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.X is not None and self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]

    def plot_feature_importance(self, cluster_id, top_n=10, save=True):
        """Bar chart of feature importance for a specific cluster."""
        from .cluster_explainer import ClusterExplainer
        explainer = ClusterExplainer(self.model, self.X, self.feature_names)
        fi = explainer.feature_importance(cluster_id)
        if not fi:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No importance data", ha="center", va="center")
            return fig

        sorted_items = sorted(fi.items(), key=lambda x: -x[1])[:top_n]
        names, values = zip(*sorted_items)

        fig, ax = plt.subplots()
        colors = sns.color_palette("viridis", len(names))
        bars = ax.barh(range(len(names)), values, color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Cluster {cluster_id} — Feature Importance")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"feature_importance_cluster_{cluster_id}.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def plot_all_feature_importance(self, top_n=10):
        """Generate feature importance plots for all clusters."""
        paths = []
        n_clusters = self.model.centers_.shape[0]
        for cid in range(n_clusters):
            fig = self.plot_feature_importance(cid, top_n, save=True)
            paths.append(str(self.output_dir / f"feature_importance_cluster_{cid}.png"))
        return paths

    def plot_cluster_radar(self, cluster_id, save=True):
        """Radar chart showing cluster profile across features."""
        from .cluster_explainer import ClusterExplainer
        explainer = ClusterExplainer(self.model, self.X, self.feature_names)
        summary = explainer.cluster_summary(cluster_id)

        if summary.get("size", 0) == 0:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.set_title(f"Cluster {cluster_id} is empty")
            return fig

        names = list(summary["features"].keys())
        means = [summary["features"][n]["mean"] for n in names]
        stds = [summary["features"][n]["std"] for n in names]
        num_vars = len(names)

        angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        means += means[:1]
        stds += stds[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.plot(angles, means, "o-", linewidth=2, color="#3498db", label="Mean")
        ax.fill(angles, means, alpha=0.15, color="#3498db")
        ax.errorbar(angles, means, yerr=stds, fmt="none", ecolor="#e74c3c",
                     capsize=3, capthick=1, alpha=0.6, label="±1 Std")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(f"Cluster {cluster_id} Profile  (n={summary['size']})", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()

        if save:
            path = self.output_dir / f"radar_cluster_{cluster_id}.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def plot_all_radars(self):
        """Generate radar charts for all clusters."""
        paths = []
        n_clusters = self.model.centers_.shape[0]
        for cid in range(n_clusters):
            fig = self.plot_cluster_radar(cid, save=True)
            paths.append(str(self.output_dir / f"radar_cluster_{cid}.png"))
        return paths

    def plot_parallel_coordinates(self, save=True):
        """Parallel coordinate plot with clusters color-highlighted."""
        if self.X is None or self.model is None:
            return None

        labels = self.model.labels_ if hasattr(self.model, 'labels_') else np.argmax(self.model.U_, axis=1)
        n_clusters = self.model.centers_.shape[0]
        n_features = self.X.shape[1]
        n_samples = self.X.shape[0]

        if n_features < 2:
            return None

        norm_X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-10)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = sns.color_palette("husl", n_clusters)

        x_vals = np.arange(n_features)

        for cid in range(n_clusters):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_data = norm_X[mask]
            for row in cluster_data:
                ax.plot(x_vals, row, color=colors[cid], alpha=0.08)
            mean_row = cluster_data.mean(axis=0)
            ax.plot(x_vals, mean_row, color=colors[cid], linewidth=3,
                    label=f"Cluster {cid}", zorder=5)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value")
        ax.set_title("Parallel Coordinate Plot by Cluster")
        ax.legend(loc="best")
        plt.tight_layout()

        if save:
            path = self.output_dir / "parallel_coordinates.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def plot_cluster_scatter(self, x_feature=0, y_feature=1, save=True):
        """2D scatter plot of two features colored by cluster."""
        if self.X is None or self.model is None:
            return None

        labels = self.model.labels_ if hasattr(self.model, 'labels_') else np.argmax(self.model.U_, axis=1)
        n_clusters = self.model.centers_.shape[0]
        centers = self.model.centers_
        n_features = self.X.shape[1]

        if n_features <= max(x_feature, y_feature):
            x_feature, y_feature = 0, min(1, n_features - 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("husl", n_clusters)

        for cid in range(n_clusters):
            mask = labels == cid
            ax.scatter(self.X[mask, x_feature], self.X[mask, y_feature],
                       c=[colors[cid]], label=f"Cluster {cid}", alpha=0.5, s=15, edgecolors="none")

        ax.scatter(centers[:, x_feature], centers[:, y_feature],
                   c="black", marker="X", s=200, edgecolors="white", linewidths=1.5,
                   label="Centers", zorder=5)

        xname = self.feature_names[x_feature] if self.feature_names else f"feat_{x_feature}"
        yname = self.feature_names[y_feature] if self.feature_names else f"feat_{y_feature}"
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"Cluster Scatter: {xname} vs {yname}")
        ax.legend(loc="best", markerscale=2)
        plt.tight_layout()

        if save:
            path = self.output_dir / f"scatter_{x_feature}_{y_feature}.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def plot_outlier_analysis(self, save=True):
        """Visualize outlier scores and flagged points."""
        if self.X is None or self.model is None:
            return None
        if not hasattr(self.model, 'outlier_scores_') or self.model.outlier_scores_ is None:
            return None

        scores = self.model.outlier_scores_
        mask = self.model.outlier_mask_ if hasattr(self.model, 'outlier_mask_') else None
        n_features = self.X.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(scores, bins=50, color="#3498db", edgecolor="white", alpha=0.7)
        if mask is not None and mask.any():
            ax.axvline(x=scores[mask].min(), color="#e74c3c", linestyle="--",
                       linewidth=2, label=f"Threshold ({scores[mask].min():.3f})")
        ax.set_xlabel("Outlier Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Outlier Score Distribution")
        ax.legend()

        ax = axes[1]
        if n_features >= 2:
            labels = self.model.labels_ if hasattr(self.model, 'labels_') else np.argmax(self.model.U_, axis=1)
            n_clusters = self.model.centers_.shape[0]
            colors = sns.color_palette("husl", n_clusters)
            for cid in range(n_clusters):
                cl_mask = labels == cid
                ax.scatter(self.X[cl_mask, 0], self.X[cl_mask, 1],
                           c=[colors[cid]], alpha=0.3, s=10, edgecolors="none")
            if mask is not None:
                outlier_idx = np.where(mask)[0]
                ax.scatter(self.X[outlier_idx, 0], self.X[outlier_idx, 1],
                           c="red", marker="x", s=50, linewidths=1.5, label=f"Outliers ({len(outlier_idx)})")
            ax.set_xlabel(self.feature_names[0] if self.feature_names else "feature_0")
            ax.set_ylabel(self.feature_names[1] if self.feature_names else "feature_1")
            ax.set_title("Outlier Detection")
            ax.legend(loc="best")

        plt.tight_layout()

        if save:
            path = self.output_dir / "outlier_analysis.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def plot_membership_heatmap(self, save=True):
        """Heatmap of fuzzy membership matrix (sampled if large)."""
        if self.model is None or self.model.U_ is None:
            return None

        U = self.model.U_
        n_samples = U.shape[0]
        max_show = 5000

        if n_samples > max_show:
            idx = np.random.RandomState(42).choice(n_samples, max_show, replace=False)
            U_show = U[idx]
        else:
            U_show = U

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(U_show.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Cluster")
        ax.set_title(f"Fuzzy Membership Matrix  (showing {U_show.shape[0]}/{n_samples} samples)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Membership")
        plt.tight_layout()

        if save:
            path = self.output_dir / "membership_heatmap.png"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Saved {path}")
            plt.close(fig)
        return fig

    def generate_all_plots(self):
        """Generate all XAI visualizations."""
        paths = {}

        try:
            paths["parallel_coordinates"] = self.plot_parallel_coordinates(save=True)
        except Exception as e:
            logger.warning(f"Parallel coordinates failed: {e}")

        try:
            paths["scatter"] = self.plot_cluster_scatter(0, 1, save=True)
        except Exception as e:
            logger.warning(f"Scatter failed: {e}")

        try:
            paths["outlier"] = self.plot_outlier_analysis(save=True)
        except Exception as e:
            logger.warning(f"Outlier analysis failed: {e}")

        try:
            paths["membership_heatmap"] = self.plot_membership_heatmap(save=True)
        except Exception as e:
            logger.warning(f"Membership heatmap failed: {e}")

        n_clusters = self.model.centers_.shape[0] if self.model is not None else 0
        for cid in range(n_clusters):
            try:
                paths[f"importance_{cid}"] = self.plot_feature_importance(cid, save=True)
            except Exception as e:
                logger.warning(f"Importance plot cluster {cid} failed: {e}")
            try:
                paths[f"radar_{cid}"] = self.plot_cluster_radar(cid, save=True)
            except Exception as e:
                logger.warning(f"Radar cluster {cid} failed: {e}")

        return paths
