"""
Visualization utilities for clustering results, convergence, and comparisons.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Visualizer:
    """Generate plots for clustering analysis and comparison."""

    def __init__(self, output_dir='plots', dpi=150, style='seaborn-v0_8-darkgrid'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('ggplot')
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def _reduce_dims(self, X, n_components=2, method='pca'):
        """Reduce dimensionality for visualization."""
        X = np.asarray(X)
        if X.shape[1] <= n_components:
            return X
        if method == 'pca':
            return PCA(n_components=n_components, random_state=42).fit_transform(X)
        elif method == 'tsne':
            return TSNE(n_components=n_components, random_state=42, perplexity=30).fit_transform(X)
        else:
            return PCA(n_components=n_components, random_state=42).fit_transform(X)

    def plot_clusters_2d(self, X, labels, centers=None, title='Cluster Visualization',
                         save_name='clusters_2d.png', show=False, method='pca'):
        """2D scatter plot of clusters with optional centers."""
        X_2d = self._reduce_dims(X, n_components=2, method=method)
        fig, ax = plt.subplots(figsize=(10, 8))
        n_clusters = len(set(labels)) if centers is None else len(centers)

        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10',
                             s=30, alpha=0.7, edgecolors='w', linewidth=0.3)

        if centers is not None:
            centers_2d = self._reduce_dims(centers, n_components=2, method=method)
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X',
                       s=200, linewidths=2, edgecolors='black', label='Centers')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
        ax.legend()
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_clusters_3d(self, X, labels, centers=None, title='3D Cluster Visualization',
                         save_name='clusters_3d.png', show=False):
        """3D scatter plot of clusters."""
        X_3d = self._reduce_dims(X, n_components=3, method='pca')
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        n_clusters = len(set(labels))

        scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels,
                             cmap='tab10', s=30, alpha=0.7, edgecolors='w', linewidth=0.3)

        if centers is not None:
            centers_3d = self._reduce_dims(centers, n_components=3, method='pca')
            ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2],
                       c='red', marker='X', s=200, linewidths=2, edgecolors='black',
                       label='Centers')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('Cluster')
        ax.legend()
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_membership_heatmap(self, U, title='Fuzzy Membership Matrix',
                                save_name='membership_heatmap.png', show=False):
        """Heatmap of membership matrix (samples x clusters)."""
        n, c = U.shape
        aspect = min(n / max(c, 1), 50)
        fig, ax = plt.subplots(figsize=(max(c * 1.2, 6), min(n * 0.02 + 2, 10)))
        im = ax.imshow(U, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                        extent=[0, c, n, 0])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Sample')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Membership')
        ax.set_xticks(np.arange(c) + 0.5)
        ax.set_xticklabels([f'C{j}' for j in range(c)])
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_convergence(self, J_history, title='Convergence Curve',
                         save_name='convergence.png', show=False, log_scale=False):
        """Plot objective function value over iterations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        iters = np.arange(1, len(J_history) + 1)
        ax.plot(iters, J_history, 'b-o', markersize=4, linewidth=1.5, label='Objective')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Function Value')
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_centers_comparison(self, centers_fcm, centers_fclm, feature_names=None,
                                title='FCM vs FCLM Centers', save_name='centers_comparison.png',
                                show=False):
        """Side-by-side comparison of FCM and FCLM cluster centers."""
        c, d = centers_fcm.shape
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(d)]

        fig, axes = plt.subplots(1, 2, figsize=(max(d * 0.5, 8), max(c * 1.5, 4)),
                                  sharey=True)

        for idx, (centers, label) in enumerate([(centers_fcm, 'FCM'), (centers_fclm, 'FCLM')]):
            ax = axes[idx]
            x = np.arange(d)
            for j in range(c):
                ax.plot(x, centers[j], 'o-', label=f'C{j}', linewidth=2, markersize=6)
            ax.set_title(f'{label} Centers', fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_metrics_comparison(self, metrics_fcm, metrics_fclm, title='FCM vs FCLM Metrics',
                               save_name='metrics_comparison.png', show=False):
        """Grouped bar chart comparing evaluation metrics between algorithms."""
        common_keys = [k for k in metrics_fcm if k in metrics_fclm]
        if not common_keys:
            logger.warning("No common metrics to compare")
            return None

        x = np.arange(len(common_keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(len(common_keys) * 1.5, 8), 6))
        vals_fcm = [metrics_fcm[k] if isinstance(metrics_fcm[k], (int, float)) else 0 for k in common_keys]
        vals_fclm = [metrics_fclm[k] if isinstance(metrics_fclm[k], (int, float)) else 0 for k in common_keys]

        bars1 = ax.bar(x - width / 2, vals_fcm, width, label='FCM', color='steelblue', alpha=0.85)
        bars2 = ax.bar(x + width / 2, vals_fclm, width, label='FCLM', color='firebrick', alpha=0.85)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_keys, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            if height != 0 and not np.isinf(height):
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if height != 0 and not np.isinf(height):
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_silhouette_analysis(self, X, labels, title='Silhouette Analysis',
                                 save_name='silhouette_analysis.png', show=False):
        """Silhouette plot for each cluster."""
        from sklearn.metrics import silhouette_samples, silhouette_score
        X = np.asarray(X)
        labels = np.asarray(labels)
        n_clusters = len(set(labels))
        silhouette_vals = silhouette_samples(X, labels)
        silhouette_avg = silhouette_score(X, labels)

        fig, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10

        for i in range(n_clusters):
            cluster_vals = silhouette_vals[labels == i]
            cluster_vals.sort()
            size = len(cluster_vals)
            y_upper = y_lower + size
            color = cm.tab10(i / max(n_clusters, 1))
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, f'C{i}', fontsize=10)
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color='red', linestyle='--',
                   label=f'Average: {silhouette_avg:.3f}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.legend()
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig, silhouette_avg

    def plot_elbow_curve(self, inertias, max_k, title='Elbow Method',
                         save_name='elbow_curve.png', show=False):
        """Plot elbow/inertia curve for optimal k selection."""
        ks = np.arange(2, max_k + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ks, inertias, 'bo-', markersize=6, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia (SSE)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ks)
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_cluster_sizes(self, labels, title='Cluster Sizes',
                           save_name='cluster_sizes.png', show=False):
        """Bar chart of cluster sizes."""
        labels = np.asarray(labels)
        n_clusters = len(set(labels))
        sizes = [np.sum(labels == i) for i in range(n_clusters)]

        fig, ax = plt.subplots(figsize=(max(n_clusters * 0.8, 6), 5))
        bars = ax.bar(range(n_clusters), sizes, color=[self.colors[i % 10] for i in range(n_clusters)],
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Size')
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])

        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(size), ha='center', va='bottom', fontsize=10)

        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix',
                              save_name='confusion_matrix.png', show=False, normalize=False):
        """Plot confusion matrix comparing true vs predicted labels."""
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

        fig, ax = plt.subplots(figsize=(max(cm.shape[0], 6), max(cm.shape[1], 5)))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels([f'{i}' for i in range(cm.shape[1])])
        ax.set_yticklabels([f'{i}' for i in range(cm.shape[0])])

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046)
        cbar.set_label('Normalized' if normalize else 'Count')
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_pairwise(self, X, labels, feature_names=None, title='Pairwise Feature Plot',
                      save_name='pairwise.png', show=False, max_features=6):
        """Pairwise scatter matrix of features colored by cluster."""
        X = np.asarray(X)
        d = min(X.shape[1], max_features)
        if d < 2:
            logger.warning("Need at least 2 features for pairwise plot")
            return None
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(d)]

        fig, axes = plt.subplots(d, d, figsize=(d * 2.5, d * 2.5))
        n_clusters = len(set(labels))

        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                if i == j:
                    ax.hist(X[:, i], bins=20, color='gray', alpha=0.6, edgecolor='black')
                    ax.set_xlabel(feature_names[i] if i == d - 1 else '')
                else:
                    scatter = ax.scatter(X[:, j], X[:, i], c=labels, cmap='tab10',
                                         s=10, alpha=0.6, edgecolors='w', linewidth=0.1)
                if i == d - 1:
                    ax.set_xlabel(feature_names[j])
                if j == 0:
                    ax.set_ylabel(feature_names[i])
                ax.tick_params(labelsize=6)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_performance_comparison(self, fcm_time, fclm_time, fcm_iters, fclm_iters,
                                    title='FCM vs FCLM Performance',
                                    save_name='performance_comparison.png', show=False):
        """Bar chart comparing execution time and iteration count."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(['FCM', 'FCLM'], [fcm_time, fclm_time],
                color=['steelblue', 'firebrick'], alpha=0.85, edgecolor='black')
        ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate([fcm_time, fclm_time]):
            ax1.text(i, v + max(fcm_time, fclm_time) * 0.01, f'{v:.3f}s',
                     ha='center', va='bottom', fontsize=10)

        ax2.bar(['FCM', 'FCLM'], [fcm_iters, fclm_iters],
                color=['steelblue', 'firebrick'], alpha=0.85, edgecolor='black')
        ax2.set_title('Iterations to Converge', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Iterations')
        ax2.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate([fcm_iters, fclm_iters]):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=10)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        self._save_fig(fig, save_name, show)
        return fig

    def plot_combined_results(self, X, labels_fcm, labels_fclm, centers_fcm, centers_fclm,
                              J_history_fcm, J_history_fclm, metrics_fcm, metrics_fclm,
                              y_true=None, title_prefix='', show=False):
        """Generate comprehensive combined figure with multiple subplots."""
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

        X_2d = self._reduce_dims(X, n_components=2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_fcm, cmap='tab10', s=15, alpha=0.7)
        if centers_fcm is not None:
            c2d = self._reduce_dims(centers_fcm, n_components=2)
            ax1.scatter(c2d[:, 0], c2d[:, 1], c='red', marker='X', s=100, edgecolors='black')
        ax1.set_title(f'{title_prefix}FCM Clusters')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_fclm, cmap='tab10', s=15, alpha=0.7)
        if centers_fclm is not None:
            c2d = self._reduce_dims(centers_fclm, n_components=2)
            ax2.scatter(c2d[:, 0], c2d[:, 1], c='red', marker='X', s=100, edgecolors='black')
        ax2.set_title(f'{title_prefix}FCLM Clusters')

        if y_true is not None:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', s=15, alpha=0.7)
            ax3.set_title(f'{title_prefix}Ground Truth')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(J_history_fcm, 'b-o', markersize=3, label='FCM', linewidth=1)
        ax4.plot(J_history_fclm, 'r-s', markersize=3, label='FCLM', linewidth=1)
        ax4.set_title('Convergence')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Objective')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, :2])
        if metrics_fcm and metrics_fclm:
            common = [k for k in metrics_fcm if k in metrics_fclm][:8]
            x = np.arange(len(common))
            w = 0.35
            v_fcm = [metrics_fcm[k] if isinstance(metrics_fcm[k], (int, float)) else 0 for k in common]
            v_fclm = [metrics_fclm[k] if isinstance(metrics_fclm[k], (int, float)) else 0 for k in common]
            ax5.bar(x - w / 2, v_fcm, w, label='FCM', color='steelblue', alpha=0.85)
            ax5.bar(x + w / 2, v_fclm, w, label='FCLM', color='firebrick', alpha=0.85)
            ax5.set_xticks(x)
            ax5.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
            ax5.set_title('Metrics Comparison')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3, axis='y')

        ax6 = fig.add_subplot(gs[1, 2:])
        sizes_fcm = [np.sum(labels_fcm == i) for i in range(len(set(labels_fcm)))]
        sizes_fclm = [np.sum(labels_fclm == i) for i in range(len(set(labels_fclm)))]
        x = np.arange(max(len(sizes_fcm), len(sizes_fclm)))
        w = 0.35
        if sizes_fcm:
            ax6.bar(x[:len(sizes_fcm)] - w / 2, sizes_fcm, w, label='FCM', color='steelblue', alpha=0.85)
        if sizes_fclm:
            ax6.bar(x[:len(sizes_fclm)] + w / 2, sizes_fclm, w, label='FCLM', color='firebrick', alpha=0.85)
        ax6.set_title('Cluster Sizes')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Count')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        metrics_text = ''
        if metrics_fcm:
            metrics_text += 'FCM:  ' + ' | '.join(f'{k}={v:.4f}' for k, v in list(metrics_fcm.items())[:6])
        if metrics_fclm:
            metrics_text += '\nFCLM: ' + ' | '.join(f'{k}={v:.4f}' for k, v in list(metrics_fclm.items())[:6])
        ax7.text(0.5, 0.5, metrics_text, transform=ax7.transAxes, fontsize=10,
                 ha='center', va='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f'{title_prefix}FCM vs FCLM Comparison', fontsize=16, fontweight='bold')
        self._save_fig(fig, f'{title_prefix.lower().replace(" ", "_")}combined_results.png' if title_prefix else 'combined_results.png', show)
        return fig

    def _save_fig(self, fig, save_name, show):
        """Save figure and optionally display."""
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig)
