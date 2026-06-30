"""
Visualization tools for DeepADEFCM.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DeepADEFCMVisualizer:
    """Visualize latent space, reconstructions, and training progress."""

    def __init__(self, save_dir="./results/deep_ade_fcm"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_latent_space(self, Z, labels, centers=None, epoch=0):
        """Plot 2D/3D latent space with cluster assignments."""
        n_dims = Z.shape[1]
        n_plots = min(3, n_dims)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        dim_pairs = [(0, 1), (0, 2), (1, 2)][:n_plots]
        for idx, (d1, d2) in enumerate(dim_pairs):
            if n_dims <= max(d1, d2):
                continue
            ax = axes[idx]
            for k, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(Z[mask, d1], Z[mask, d2], c=[colors[k]],
                           label=f"Cluster {label}", alpha=0.6, s=20)
            if centers is not None:
                ax.scatter(centers[:, d1], centers[:, d2], c='red',
                           marker='X', s=200, edgecolors='black', linewidths=2, label='Centers')
            ax.set_xlabel(f"Latent dim {d1}")
            ax.set_ylabel(f"Latent dim {d2}")
            ax.set_title(f"Latent Space (dims {d1}-{d2})")
            ax.legend(fontsize=8)

        plt.tight_layout()
        path = self.save_dir / f"latent_space_epoch_{epoch}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def plot_reconstructions(self, X_original, X_reconstructed, n_samples=10):
        """Plot original vs reconstructed samples."""
        idx = np.random.choice(len(X_original), min(n_samples, len(X_original)), replace=False)
        n_features = X_original.shape[1]

        n_cols = len(idx)
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2, 4))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        for i, sample_idx in enumerate(idx):
            axes[0, i].plot(X_original[sample_idx], 'b-', alpha=0.7)
            axes[0, i].set_title(f"Original #{sample_idx}")
            axes[0, i].set_xticks([])

            axes[1, i].plot(X_reconstructed[sample_idx], 'r-', alpha=0.7)
            axes[1, i].set_title("Reconstructed")
            axes[1, i].set_xticks([])

        plt.tight_layout()
        path = self.save_dir / "reconstructions.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def plot_training_history(self, loss_history):
        """Plot reconstruction, clustering, and total loss over epochs."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, (key, color) in zip(axes, [
            ('reconstruction', 'blue'), ('clustering', 'orange'), ('total', 'green')
        ]):
            values = loss_history.get(key, [])
            if values:
                ax.plot(values, color=color, linewidth=2)
                ax.set_title(f"{key.capitalize()} Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.save_dir / "training_history.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def plot_feature_importance(self, explanations, feature_names=None):
        """Plot feature importance for each cluster."""
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(10)]

        n_clusters = len(explanations)
        fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 4))
        if n_clusters == 1:
            axes = [axes]

        for ax, exp in zip(axes, explanations):
            fi = exp['feature_importance']
            sorted_items = sorted(fi.items(), key=lambda x: -x[1])[:10]
            names, values = zip(*sorted_items)
            ax.barh(range(len(names)), values, color='steelblue')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Importance")
            ax.set_title(f"Cluster {exp['cluster_id']}")
            ax.invert_yaxis()

        plt.tight_layout()
        path = self.save_dir / "feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)
