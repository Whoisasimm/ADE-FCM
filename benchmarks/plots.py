from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


class BenchmarkPlotter:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._colors = None

    def _get_colors(self, n):
        if self._colors is None or len(self._colors) < n:
            self._colors = sns.color_palette("husl", n)
        return self._colors[:n]

    def plot_comparison_bar(self, df, metric, title=None):
        fig, ax = plt.subplots()
        pivot = df.pivot_table(
            index="algorithm", columns="dataset", values=metric, aggfunc="mean"
        ).fillna(0)
        pivot.plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Comparison of {metric.replace('_', ' ').title()}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        path = self.output_dir / f"comparison_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_scalability(self, sizes, times, algorithms, title=None):
        fig, ax = plt.subplots()
        times = np.asarray(times)
        if times.ndim == 1:
            times = times[np.newaxis, :]
        colors = self._get_colors(len(algorithms))
        for i, (algo, color) in enumerate(zip(algorithms, colors)):
            ax.plot(sizes, times[i], marker="o", label=algo, color=color, linewidth=2)
        ax.set_xlabel("Dataset Size (n_samples)")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title(title or "Scalability Analysis")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.output_dir / "scalability.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_memory_usage(self, df, title=None):
        if "memory_mb" not in df.columns:
            return None
        fig, ax = plt.subplots()
        pivot = df.pivot_table(
            index="algorithm", columns="dataset", values="memory_mb", aggfunc="mean"
        ).fillna(0)
        pivot.plot(kind="barh", ax=ax, colormap="plasma", width=0.8)
        ax.set_xlabel("Memory Usage (MB)")
        ax.set_title(title or "Memory Usage Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        path = self.output_dir / "memory_usage.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_radar_chart(self, df, metrics, title=None):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        pivot = df.groupby("algorithm")[metrics].mean().fillna(0)
        values = pivot.values
        algo_names = pivot.index.tolist()
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        colors = self._get_colors(len(algo_names))
        for i, algo in enumerate(algo_names):
            v = values[i].tolist()
            v += v[:1]
            ax.plot(angles, v, "o-", linewidth=2, label=algo, color=colors[i])
            ax.fill(angles, v, alpha=0.1, color=colors[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
        ax.set_title(title or "Algorithm Comparison Radar Chart", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        path = self.output_dir / "radar_chart.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_heatmap(self, df, metric, title=None):
        if metric not in df.columns:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot = df.pivot_table(
            index="algorithm", columns="dataset", values=metric, aggfunc="mean"
        ).fillna(0)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": metric.replace("_", " ").title()},
        )
        ax.set_title(title or f"Heatmap of {metric.replace('_', ' ').title()}")
        ax.set_ylabel("Algorithm")
        ax.set_xlabel("Dataset")
        plt.tight_layout()
        path = self.output_dir / f"heatmap_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_all_plots(self, df):
        paths = {}
        metrics = [
            "accuracy", "nmi", "ari", "silhouette_score",
            "davies_bouldin_score", "calinski_harabasz_score",
            "execution_time", "memory_mb",
        ]
        for metric in metrics:
            if metric not in df.columns:
                continue
            if df[metric].isna().all():
                continue
            try:
                paths[f"comparison_{metric}"] = self.plot_comparison_bar(df, metric)
            except Exception as e:
                print(f"  Failed comparison_{metric}: {e}")
            try:
                paths[f"heatmap_{metric}"] = self.plot_heatmap(df, metric)
            except Exception as e:
                print(f"  Failed heatmap_{metric}: {e}")
        try:
            valid_metrics = [m for m in metrics if m in df.columns and not df[m].isna().all()]
            if len(valid_metrics) >= 3:
                paths["radar"] = self.plot_radar_chart(df, valid_metrics[:6])
        except Exception as e:
            print(f"  Failed radar: {e}")
        try:
            paths["memory"] = self.plot_memory_usage(df)
        except Exception as e:
            print(f"  Failed memory: {e}")
        return paths
