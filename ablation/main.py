"""
Ablation Study Runner for ADE-FCM.
Runs ablation experiments on multiple datasets, generates comparative
plots and a comprehensive summary report.
"""
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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


ABLATION_NAMES = {
    "full_ade_fcm": "Full ADE-FCM",
    "without_adaptive_fuzzifier": "w/o Adaptive m(t)",
    "without_auto_k": "w/o Auto K",
    "without_explainability": "w/o XAI",
    "without_outlier_robustness": "w/o Outlier Robustness",
    "without_early_stopping": "w/o Early Stopping",
}


def generate_dataset(name, n_samples, n_features, n_clusters, noise=0.5, random_state=42):
    """Generate a synthetic dataset with known structure."""
    rng = np.random.RandomState(random_state + hash(name) % 10000)
    centers = rng.randn(n_clusters, n_features) * 3
    labels = rng.randint(0, n_clusters, size=n_samples)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        X[i] = centers[labels[i]] + rng.randn(n_features) * noise
    return X, labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ADE-FCM Ablation Study Runner")
    parser.add_argument("--output_dir", type=str, default="ablation_output", help="Output directory")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def ablation_trial(X, n_clusters, random_state):
    """Run a single ablation study on one dataset."""
    from .ablation_study import AblationStudy

    study = AblationStudy(X, n_clusters=n_clusters, random_state=random_state)
    results = study.run_all()
    report = study.generate_report(output_path=None)
    return report


def main(argv=None):
    args = parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    datasets = {
        "synth_easy": {"n_samples": 500, "n_features": 4, "n_clusters": 3, "noise": 0.3},
        "synth_medium": {"n_samples": 1000, "n_features": 8, "n_clusters": 5, "noise": 0.5},
        "synth_hard": {"n_samples": 2000, "n_features": 16, "n_clusters": 8, "noise": 0.8},
        "synth_overlap": {"n_samples": 800, "n_features": 6, "n_clusters": 4, "noise": 1.2},
        "synth_highdim": {"n_samples": 600, "n_features": 32, "n_clusters": 4, "noise": 0.5},
    }

    all_reports = {}

    for dname, dspec in datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dname} ({dspec['n_samples']}x{dspec['n_features']}, K={dspec['n_clusters']})")
        logger.info(f"{'='*60}")

        X, true_labels = generate_dataset(
            name=dname,
            n_samples=dspec["n_samples"],
            n_features=dspec["n_features"],
            n_clusters=dspec["n_clusters"],
            noise=dspec.get("noise", 0.5),
            random_state=args.seed,
        )

        start = time.time()
        report = ablation_trial(X, dspec["n_clusters"], args.seed)
        elapsed = time.time() - start

        all_reports[dname] = {
            "spec": dspec,
            "report": report,
            "elapsed": round(elapsed, 2),
        }

        full = report.get("full_ade_fcm", {})
        logger.info(f"  Full ADE-FCM: Silhouette={full.get('silhouette', 'N/A'):.4f}, "
                    f"DB={full.get('davies_bouldin', 'N/A'):.4f}, "
                    f"Iter={full.get('iterations', 'N/A')}, "
                    f"Time={full.get('time', 'N/A'):.2f}s")

        for ablation_name, metrics in report.get("ablation_results", {}).items():
            if ablation_name == "full_ade_fcm" or "error" in metrics:
                continue
            logger.info(f"  {ABLATION_NAMES.get(ablation_name, ablation_name)}: "
                        f"Silhouette={metrics.get('silhouette', 'N/A'):.4f}, "
                        f"DB={metrics.get('davies_bouldin', 'N/A'):.4f}")

        logger.info(f"  Trial completed in {elapsed:.2f}s")

    _generate_summary_report(all_reports, out_dir, plots_dir)
    _generate_comparison_plots(all_reports, plots_dir)

    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    logger.info(f"Full results saved to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 60)
    return all_reports


def _generate_summary_report(all_reports, out_dir, plots_dir):
    """Generate HTML summary report."""
    import datetime

    rows = []
    for dname, ddata in all_reports.items():
        report = ddata["report"]
        full = report.get("full_ade_fcm", {})
        base_row = {
            "dataset": dname,
            "samples": ddata["spec"]["n_samples"],
            "features": ddata["spec"]["n_features"],
            "true_k": ddata["spec"]["n_clusters"],
        }
        for ablation_name in ABLATION_NAMES:
            metrics = report.get("ablation_results", {}).get(ablation_name, {})
            deg = report.get("degradation", {}).get(ablation_name, {})
            base_row[f"{ablation_name}_silhouette"] = metrics.get("silhouette", None)
            base_row[f"{ablation_name}_davies_bouldin"] = metrics.get("davies_bouldin", None)
            base_row[f"{ablation_name}_time"] = metrics.get("time", None)
            base_row[f"{ablation_name}_iterations"] = metrics.get("iterations", None)
            base_row[f"{ablation_name}_sil_degradation"] = deg.get("silhouette", None)
            base_row[f"{ablation_name}_db_degradation"] = deg.get("davies_bouldin", None)
        rows.append(base_row)

    df = pd.DataFrame(rows)

    csv_path = out_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV summary saved to {csv_path}")

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>ADE-FCM Ablation Study Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        "margin: 40px; background: #f8f9fa; color: #333; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; border-bottom: 2px solid #e74c3c; padding-bottom: 5px; }",
        "h3 { color: #2c3e50; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; "
        "background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }",
        "th, td { padding: 8px 10px; text-align: center; border-bottom: 1px solid #ddd; }",
        "th { background: #e74c3c; color: white; }",
        "tr:hover { background: #f1f1f1; }",
        ".card { background: white; border-radius: 8px; padding: 15px; margin: 15px 0; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".pos { color: #27ae60; font-weight: bold; }",
        ".neg { color: #e74c3c; font-weight: bold; }",
        "img { max-width: 100%; height: auto; margin: 10px 0; "
        "border: 1px solid #ddd; border-radius: 4px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>ADE-FCM Ablation Study Report</h1>",
        f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p>Datasets evaluated: <strong>{len(all_reports)}</strong></p>",
        "<hr>",
    ]

    for dname, ddata in all_reports.items():
        report = ddata["report"]
        full = report.get("full_ade_fcm", {})

        html_parts.append(f'<div class="card">')
        html_parts.append(f"<h2>{dname}</h2>")
        html_parts.append(
            f"<p>Samples: {ddata['spec']['n_samples']} | "
            f"Features: {ddata['spec']['n_features']} | "
            f"True K: {ddata['spec']['n_clusters']} | "
            f"Trial time: {ddata['elapsed']:.2f}s</p>"
        )

        html_parts.append("<h3>Full ADE-FCM Performance</h3>")
        html_parts.append(
            f"<p>Silhouette: <strong>{full.get('silhouette', 'N/A'):.4f}</strong> | "
            f"Davies-Bouldin: <strong>{full.get('davies_bouldin', 'N/A'):.4f}</strong> | "
            f"Iterations: <strong>{full.get('iterations', 'N/A')}</strong> | "
            f"Time: <strong>{full.get('time', 'N/A'):.2f}s</strong></p>"
        )

        html_parts.append("<h3>Ablation Results</h3>")
        html_parts.append(
            "<table><tr><th>Component Removed</th><th>Silhouette</th>"
            "<th>Sil Δ%</th><th>DB</th><th>DB Δ%</th>"
            "<th>Time (s)</th><th>Iterations</th></tr>"
        )

        for ablation_name, display_name in ABLATION_NAMES.items():
            if ablation_name == "full_ade_fcm":
                continue
            metrics = report.get("ablation_results", {}).get(ablation_name, {})
            deg = report.get("degradation", {}).get(ablation_name, {})
            if "error" in metrics:
                html_parts.append(f"<tr><td>{display_name}</td><td colspan='6' style='color:red;'>Error: {metrics['error']}</td></tr>")
                continue

            sil = metrics.get("silhouette", "N/A")
            db = metrics.get("davies_bouldin", "N/A")
            t = metrics.get("time", "N/A")
            it = metrics.get("iterations", "N/A")
            sil_d = deg.get("silhouette", 0)
            db_d = deg.get("davies_bouldin", 0)

            sil_cls = "neg" if sil_d < -5 else "pos"
            db_cls = "pos" if db_d > 5 else "neg"

            html_parts.append(
                f"<tr><td>{display_name}</td>"
                f"<td>{sil:.4f}</td>"
                f"<td class='{sil_cls}'>{sil_d:+.1f}%</td>"
                f"<td>{db:.4f}</td>"
                f"<td class='{db_cls}'>{db_d:+.1f}%</td>"
                f"<td>{t:.2f}</td>"
                f"<td>{it}</td></tr>"
            )

        html_parts.append("</table>")
        html_parts.append("</div>")

    html_parts.append("<h2>Comparison Plots</h2>")
    plot_files = sorted(plots_dir.glob("*.png"))
    for pf in plot_files:
        html_parts.append(f"<h3>{pf.stem.replace('_', ' ').title()}</h3>")
        html_parts.append(f'<img src="plots/{pf.name}" alt="{pf.stem}">')

    html_parts.append("</body></html>")

    html_path = out_dir / "ablation_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    logger.info(f"HTML report saved to {html_path}")


def _generate_comparison_plots(all_reports, plots_dir):
    """Generate comparative bar charts for ablation results."""
    if not all_reports:
        return

    metrics_data = []
    for dname, ddata in all_reports.items():
        report = ddata["report"]
        for ablation_name, display_name in ABLATION_NAMES.items():
            metrics = report.get("ablation_results", {}).get(ablation_name, {})
            if "error" in metrics:
                continue
            metrics_data.append({
                "dataset": dname,
                "ablation": display_name,
                "silhouette": metrics.get("silhouette", 0),
                "davies_bouldin": metrics.get("davies_bouldin", 0),
                "time": metrics.get("time", 0),
                "iterations": metrics.get("iterations", 0),
            })

    if not metrics_data:
        return

    df = pd.DataFrame(metrics_data)

    for metric, ylabel in [
        ("silhouette", "Silhouette Score (↑)"),
        ("davies_bouldin", "Davies-Bouldin Index (↓)"),
        ("time", "Execution Time (s)"),
        ("iterations", "Iterations"),
    ]:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot = df.pivot_table(index="ablation", columns="dataset", values=metric, aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, colormap="viridis", width=0.8, edgecolor="white")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Ablation Study — {ylabel}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            path = plots_dir / f"ablation_{metric}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {path}")
        except Exception as e:
            logger.warning(f"Failed to plot {metric}: {e}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        pivot_sil = df.pivot_table(index="ablation", columns="dataset", values="silhouette", aggfunc="mean")
        sil_drop = (pivot_sil.loc["Full ADE-FCM"] - pivot_sil).drop("Full ADE-FCM")
        sil_drop_pct = (sil_drop / pivot_sil.loc["Full ADE-FCM"].replace(0, np.nan)) * 100
        sil_drop_pct.plot(kind="bar", ax=ax, colormap="RdBu_r", width=0.8, edgecolor="white")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Silhouette Degradation (%)")
        ax.set_title("Silhouette Degradation vs Full ADE-FCM")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        ax = axes[1]
        pivot_db = df.pivot_table(index="ablation", columns="dataset", values="davies_bouldin", aggfunc="mean")
        db_drop = (pivot_db - pivot_db.loc["Full ADE-FCM"]).drop("Full ADE-FCM")
        db_drop_pct = (db_drop / pivot_db.loc["Full ADE-FCM"].replace(0, np.nan)) * 100
        db_drop_pct.plot(kind="bar", ax=ax, colormap="RdBu_r", width=0.8, edgecolor="white")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Davies-Bouldin Degradation (%)")
        ax.set_title("DB Degradation vs Full ADE-FCM")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        path = plots_dir / "ablation_degradation.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
    except Exception as e:
        logger.warning(f"Failed to plot degradation: {e}")

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot_time = df.pivot_table(index="ablation", columns="dataset", values="time", aggfunc="mean")
        pivot_time.plot(kind="barh", ax=ax, colormap="plasma", width=0.8, edgecolor="white")
        ax.set_xlabel("Execution Time (s)")
        ax.set_title("Ablation Study — Execution Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        path = plots_dir / "ablation_time.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
    except Exception as e:
        logger.warning(f"Failed to plot time: {e}")


if __name__ == "__main__":
    main()
