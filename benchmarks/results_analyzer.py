from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class ResultsAnalyzer:
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else Path("results")

    def load_results(self, path=None):
        if path is None:
            path = self.results_dir / "benchmark_results.csv"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            df = pd.read_json(path, orient="records")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return df

    def rank_algorithms(self, df, metric, ascending=False):
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        pivot = df.pivot_table(
            index="algorithm", columns="dataset", values=metric, aggfunc="mean"
        )
        ranks = pivot.rank(ascending=ascending)
        mean_rank = ranks.mean(axis=1).sort_values()
        result = pd.DataFrame({
            "mean_value": pivot.mean(axis=1).round(4),
            "std_value": pivot.std(axis=1).round(4),
            "mean_rank": mean_rank.round(2),
        }).sort_values("mean_rank")
        result["rank"] = range(1, len(result) + 1)
        return result

    def statistical_significance(self, df, metric, method="wilcoxon"):
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        pivot = df.pivot_table(
            index=["dataset", "algorithm"],
            values=metric,
            aggfunc="first",
        ).reset_index()

        algorithms = pivot["algorithm"].unique()
        n = len(algorithms)
        results = {}
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = algorithms[i], algorithms[j]
                v1 = pivot[pivot["algorithm"] == a1].set_index("dataset")[metric]
                v2 = pivot[pivot["algorithm"] == a2].set_index("dataset")[metric]
                common = v1.index.intersection(v2.index)
                if len(common) < 3:
                    continue
                x, y = v1[common].values, v2[common].values
                diff = x - y
                if np.all(diff == 0):
                    continue
                try:
                    if method == "wilcoxon":
                        stat, pval = scipy_stats.wilcoxon(x, y, alternative="two-sided")
                    elif method == "ttest":
                        stat, pval = scipy_stats.ttest_rel(x, y)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                except (ValueError, RuntimeError):
                    continue
                results[f"{a1} vs {a2}"] = {
                    "statistic": round(stat, 4),
                    "p_value": round(pval, 6),
                    "significant": pval < 0.05,
                    "better": a1 if np.mean(diff) > 0 else a2,
                }
        return pd.DataFrame.from_dict(results, orient="index")

    def generate_latex_table(self, df, caption="Benchmark Results", label="tab:benchmark"):
        latex = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}"]
        latex.append(r"\label{" + label + "}")
        latex.append(r"\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}")
        latex.append(r"\toprule")
        cols = df.columns.tolist()
        header = " & ".join(r"\textbf{" + c.replace("_", " ").title() + "}" for c in cols)
        latex.append(header + r" \\")
        latex.append(r"\midrule")
        for _, row in df.iterrows():
            row_str = " & ".join(
                f"{v:.3f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                else str(v)
                for v in row.values
            )
            latex.append(row_str + r" \\")
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        return "\n".join(latex)

    def generate_report(self, df, output_path=None):
        report_lines = []
        report_lines.append("# Benchmark Report\n")
        report_lines.append(
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        report_lines.append("## Summary\n")
        report_lines.append(
            f"- Algorithms tested: {df['algorithm'].nunique()}\n"
        )
        report_lines.append(
            f"- Datasets used: {df['dataset'].nunique()}\n"
        )
        report_lines.append(
            f"- Total runs: {len(df)}\n"
        )

        report_lines.append("## Per-Algorithm Performance\n")
        metric_cols = [
            c for c in df.columns
            if c not in ("algorithm", "dataset", "error", "n_clusters")
        ]
        avg = df.groupby("algorithm")[metric_cols].mean().round(4)
        report_lines.append(avg.to_markdown())
        report_lines.append("")

        report_lines.append("## Ranking by Silhouette Score\n")
        try:
            ranking = self.rank_algorithms(df, "silhouette_score", ascending=False)
            report_lines.append(ranking.to_markdown())
            report_lines.append("")
        except Exception:
            pass

        report_lines.append("## Statistical Significance (Wilcoxon)\n")
        try:
            sig = self.statistical_significance(df, "silhouette_score")
            report_lines.append(sig.to_markdown())
            report_lines.append("")
        except Exception:
            pass

        report_lines.append("## Dataset-wise Comparison\n")
        for ds in sorted(df["dataset"].unique()):
            ds_df = df[df["dataset"] == ds][["algorithm"] + metric_cols]
            ds_df = ds_df.set_index("algorithm").round(4)
            report_lines.append(f"### {ds}\n")
            report_lines.append(ds_df.to_markdown())
            report_lines.append("")

        report_str = "\n".join(report_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_str, encoding="utf-8")

        return report_str
