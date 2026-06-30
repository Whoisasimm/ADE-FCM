"""
XAI Report Generator for ADE-FCM.
Generates full XAI report with feature importance, SHAP explanations,
NL descriptions, visualizations, and a self-contained HTML report.
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from loguru import logger


def generate_synthetic_data(n_samples=1000, n_features=8, n_clusters=5, noise=0.5, random_state=42):
    """Generate synthetic labeled data for demonstration."""
    rng = np.random.RandomState(random_state)
    centers = rng.randn(n_clusters, n_features) * 3
    labels = rng.randint(0, n_clusters, size=n_samples)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        X[i] = centers[labels[i]] + rng.randn(n_features) * noise
    return X, labels


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ADE-FCM XAI Report Generator")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (CSV)")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--output_dir", type=str, default="xai_output", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--nsamples", type=int, default=500, help="Synthetic n_samples")
    parser.add_argument("--nfeatures", type=int, default=8, help="Synthetic n_features")
    parser.add_argument("--shap", action="store_true", help="Compute SHAP explanations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def load_data(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    elif ext == ".npy":
        return np.load(path)
    elif ext == ".npz":
        with np.load(path) as npz:
            return npz[list(npz.keys())[0]]
    else:
        raise ValueError(f"Unsupported format: {ext}")


def main(argv=None):
    args = parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if args.data:
        logger.info(f"Loading data from {args.data}")
        X = load_data(args.data)
        feature_names = None
    else:
        logger.info(f"Generating synthetic data ({args.nsamples}x{args.nfeatures})")
        X, true_labels = generate_synthetic_data(
            n_samples=args.nsamples, n_features=args.nfeatures,
            n_clusters=args.n_clusters, random_state=args.seed
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    logger.info(f"Data shape: {X.shape}")

    from novel_algorithm.ade_fcm import ADEFCM

    model = ADEFCM(
        n_clusters=args.n_clusters,
        m="adaptive",
        epsilon="dynamic",
        init_method="kmeans++",
        random_state=args.seed,
        verbose=True,
    )
    model.fit(X)
    logger.info(f"Fitted ADE-FCM: {model.n_clusters} clusters, {model.n_iter_} iterations")

    from .cluster_explainer import ClusterExplainer
    explainer = ClusterExplainer(model, X, feature_names)

    logger.info("Generating cluster explanations...")
    report = explainer.generate_report(str(out_dir / "xai_report.json"))

    nl_descriptions = {}
    for i in range(model.n_clusters):
        desc = explainer.natural_language_description(i)
        nl_descriptions[f"cluster_{i}"] = desc
        logger.info(desc)

    shap_data = None
    if args.shap:
        logger.info("Computing SHAP explanations (this may take a while)...")
        try:
            from .shap_explainer import ShapExplainer
            shaper = ShapExplainer(model, X, feature_names, nsamples=200)
            shaper.fit()
            shap_data = shaper.global_shap_summary()
            logger.info("SHAP explanations computed.")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    from .visualizer import XAIVisualizer
    viz = XAIVisualizer(model, X, feature_names, output_dir=str(plots_dir))
    logger.info("Generating visualizations...")
    plot_paths = viz.generate_all_plots()

    html_path = out_dir / "xai_report.html"
    _generate_html(html_path, report, nl_descriptions, shap_data, plot_paths)
    logger.info(f"HTML report saved to {html_path}")

    result = {
        "report": report,
        "nl_descriptions": nl_descriptions,
        "shap": shap_data,
        "plots": {k: str(v) for k, v in plot_paths.items() if v is not None},
        "html_report": str(html_path),
    }

    with open(out_dir / "xai_complete.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("XAI pipeline complete.")
    return result


def _generate_html(html_path, report, nl_descriptions, shap_data, plot_paths):
    """Generate a self-contained HTML report."""
    import datetime

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>ADE-FCM XAI Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        "margin: 40px; background: #f8f9fa; color: #333; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }",
        "h3 { color: #2c3e50; }",
        ".summary { background: #fff; padding: 20px; border-radius: 8px; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; "
        "background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }",
        "th { background: #3498db; color: white; }",
        "tr:hover { background: #f1f1f1; }",
        ".cluster-card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".badge { display: inline-block; background: #3498db; color: white; "
        "padding: 4px 12px; border-radius: 12px; font-size: 12px; }",
        ".nl-desc { background: #eef; padding: 12px; border-left: 4px solid #3498db; "
        "border-radius: 4px; margin: 10px 0; }",
        "img { max-width: 100%; height: auto; margin: 10px 0; "
        "border: 1px solid #ddd; border-radius: 4px; }",
        ".plot-grid { display: flex; flex-wrap: wrap; gap: 20px; }",
        ".plot-grid img { flex: 1 1 45%; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>ADE-FCM XAI Report</h1>",
        f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]

    html_parts.append('<div class="summary">')
    html_parts.append(f"<h2>Summary</h2>")
    html_parts.append(f"<p>Clusters: <strong>{report['n_clusters']}</strong></p>")
    html_parts.append(f"<p>Features: <strong>{report['n_features']}</strong></p>")
    html_parts.append(f"<p>Samples: <strong>{report['n_samples']}</strong></p>")
    html_parts.append("</div>")

    html_parts.append("<h2>Natural Language Descriptions</h2>")
    for key, desc in nl_descriptions.items():
        html_parts.append(f'<div class="nl-desc"><strong>{key}:</strong> {desc}</div>')

    html_parts.append("<h2>Cluster Details</h2>")
    for cluster in report.get("clusters", []):
        cid = cluster["cluster_id"]
        html_parts.append(
            f'<div class="cluster-card">'
            f'<h3>Cluster {cid} <span class="badge">'
            f'{cluster["size"]} points ({cluster["percentage"]:.1f}%)'
            f'</span></h3>'
        )

        html_parts.append(f"<p><em>{nl_descriptions.get(f'cluster_{cid}', '')}</em></p>")

        html_parts.append("<h4>Feature Statistics</h4>")
        html_parts.append(
            "<table><tr><th>Feature</th><th>Mean</th><th>Std</th>"
            "<th>Min</th><th>Max</th><th>Importance</th></tr>"
        )
        for fname, fstats in cluster.get("features", {}).items():
            html_parts.append(
                f"<tr><td>{fname}</td>"
                f"<td>{fstats['mean']:.4f}</td>"
                f"<td>{fstats['std']:.4f}</td>"
                f"<td>{fstats['min']:.4f}</td>"
                f"<td>{fstats['max']:.4f}</td>"
                f"<td>{fstats['importance']:.4f}</td></tr>"
            )
        html_parts.append("</table>")
        html_parts.append("</div>")

    if shap_data:
        html_parts.append("<h2>SHAP Feature Importance</h2>")
        for s in shap_data:
            html_parts.append(f'<div class="cluster-card">')
            html_parts.append(f"<h3>Cluster {s['cluster_id']} SHAP</h3>")
            html_parts.append(f"<p>Base value (expected output): {s['expected_value']:.4f}</p>")
            html_parts.append("<table><tr><th>Feature</th><th>Mean |SHAP|</th></tr>")
            for item in s["top_features"]:
                html_parts.append(f"<tr><td>{item['name']}</td><td>{item['importance']:.4f}</td></tr>")
            html_parts.append("</table>")
            html_parts.append("</div>")

    img_map = {
        "parallel_coordinates": "Parallel Coordinate Plot",
        "scatter": "Cluster Scatter Plot",
        "outlier": "Outlier Analysis",
        "membership_heatmap": "Membership Heatmap",
    }

    html_parts.append("<h2>Visualizations</h2>")
    for key, title in img_map.items():
        if key in plot_paths and plot_paths[key] is not None:
            rel = Path(plot_paths[key]).name
            html_parts.append(f"<h3>{title}</h3>")
            html_parts.append(f'<img src="plots/{rel}" alt="{title}">')

    n_clusters = report["n_clusters"]
    for cid in range(n_clusters):
        for ptype in ["importance", "radar"]:
            key = f"{ptype}_{cid}"
            if key in plot_paths and plot_paths[key] is not None:
                rel = Path(plot_paths[key]).name
                label = f"Cluster {cid} {'Feature Importance' if ptype == 'importance' else 'Radar Profile'}"
                html_parts.append(f"<h3>{label}</h3>")
                html_parts.append(f'<img src="plots/{rel}" alt="{label}">')

    html_parts.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


if __name__ == "__main__":
    main()
