"""
Phase 7A: Publication Validation
=================================
7 datasets x 6 algorithms x 10 seeds
Metrics: ARI, NMI, Silhouette, Davies-Bouldin, Runtime
Statistics: Friedman, Nemenyi, Effect Sizes, CD Diagram
"""
import warnings, time, json, os, sys, itertools
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Dataset loaders ─────────────────────────────────────────────────────────
def load_dataset(name):
    if name == 'iris':
        from sklearn.datasets import load_iris
        d = load_iris(); return d.data, d.target
    elif name == 'wine':
        from sklearn.datasets import load_wine
        d = load_wine(); return d.data, d.target
    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(); return d.data, d.target
    elif name == 'digits':
        from sklearn.datasets import load_digits
        d = load_digits(); return d.data, d.target
    elif name == 'glass':
        from sklearn.datasets import fetch_openml
        d = fetch_openml('glass', version=1, parser='auto')
        X = d.data.astype(float).values
        # Target has string labels like 'build wind float' - encode as ints
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(d.target.values.astype(str))
        return X, y
    elif name == 'seeds':
        from sklearn.datasets import fetch_openml
        d = fetch_openml('seeds', version=1, parser='auto')
        return d.data.astype(float).values, d.target.astype(int).values
    elif name == 'sonar':
        from sklearn.datasets import fetch_openml
        d = fetch_openml('sonar', version=1, parser='auto')
        X = d.data.astype(float).values
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(d.target.values.astype(str))
        return X, y
    raise ValueError(f"Unknown dataset: {name}")

DATASETS = ['iris', 'wine', 'breast_cancer', 'digits', 'glass', 'seeds', 'sonar']
SEEDS = list(range(42, 52))

# ── Inline FCM / FCLM (self-contained, no external deps) ─────────────────────
def _defuzzify(U):
    return np.argmax(U, axis=1)

def run_fcm_inline(X, n_clusters, seed):
    n, d = X.shape
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, n_clusters, replace=False)]
    m_val = 2.0
    epsilon = 1e-5
    max_iter = 200
    U_old = None
    t0 = time.time()
    for _ in range(max_iter):
        dists = np.zeros((n, n_clusters))
        for j in range(n_clusters):
            diff = X - centers[j]
            dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        inv = 2.0 / (m_val - 1)
        U = np.zeros((n, n_clusters))
        for i in range(n):
            for j in range(n_clusters):
                denom = np.sum((dists[i, j] / dists[i, :]) ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0
        if U_old is not None:
            if np.linalg.norm(U - U_old, 'fro') < epsilon:
                break
        U_old = U
        Um = U ** m_val
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        centers = (numerator / denominator).T
    t = time.time() - t0
    return _defuzzify(U), t

def run_fclm_inline(X, n_clusters, seed):
    n, d = X.shape
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, n_clusters, replace=False)]
    m_val = 2.0
    epsilon = 1e-5
    max_iter = 200
    U_old = None
    t0 = time.time()
    for _ in range(max_iter):
        dists = np.zeros((n, n_clusters))
        for j in range(n_clusters):
            diff = X - centers[j]
            dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        inv = 2.0 / (m_val - 1)
        U = np.zeros((n, n_clusters))
        for i in range(n):
            for j in range(n_clusters):
                denom = np.sum((dists[i, j] / dists[i, :]) ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0
        if U_old is not None:
            if np.linalg.norm(U - U_old, 'fro') < epsilon:
                break
        U_old = U
        Um = U ** m_val
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        wcenters = (numerator / denominator).T
        centers = np.zeros_like(wcenters)
        for j in range(n_clusters):
            diff = X - wcenters[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            wd = dists_j * U[:, j]
            centers[j] = X[np.argsort(wd)[n // 2]]
    t = time.time() - t0
    return _defuzzify(U), t

# ── Algorithm wrappers ───────────────────────────────────────────────────────
def run_ade_fcm(X, n_clusters, seed, metric='cosine'):
    from novel_algorithm.ade_fcm import ADEFCM
    m = ADEFCM(
        n_clusters=n_clusters, max_iter=200, m='adaptive', epsilon='dynamic',
        outlier_contamination=0, center_reinit_threshold=1.0,
        metric=metric, compute_xai=False, random_state=seed, verbose=False
    )
    t0 = time.time()
    m.fit(X)
    t = time.time() - t0
    return m.labels_, t

def run_deep_ade_fcm(X, n_clusters, seed):
    from deep_ade_fcm.deep_ade_fcm import DeepADEFCM
    m = DeepADEFCM(
        input_dim=X.shape[1], latent_dim=min(10, X.shape[1]),
        n_clusters=n_clusters, lambda_cluster=0.5,
        ae_epochs=15, joint_epochs=10, batch_size=256,
        random_state=seed, verbose=False
    )
    t0 = time.time()
    m.fit(X)
    t = time.time() - t0
    return m.labels_, t

def run_fcm(X, n_clusters, seed):
    return run_fcm_inline(X, n_clusters, seed)

def run_fclm(X, n_clusters, seed):
    return run_fclm_inline(X, n_clusters, seed)

def run_kmeans(X, n_clusters, seed):
    m = KMeans(n_clusters=n_clusters, n_init=1, max_iter=300, random_state=seed)
    t0 = time.time()
    labels = m.fit_predict(X)
    t = time.time() - t0
    return labels, t

def run_agglomerative(X, n_clusters, seed):
    m = AgglomerativeClustering(n_clusters=n_clusters)
    t0 = time.time()
    labels = m.fit_predict(X)
    t = time.time() - t0
    return labels, t

ALGORITHMS = {
    'ADE-FCM': run_ade_fcm,
    'DeepADEFCM': run_deep_ade_fcm,
    'FCM': run_fcm,
    'FCLM': run_fclm,
    'KMeans': run_kmeans,
    'Agglomerative': run_agglomerative,
}

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(X, y_true, labels):
    n_eff = len(set(labels))
    ari = adjusted_rand_score(y_true, labels) if n_eff > 1 else 0.0
    nmi = normalized_mutual_info_score(y_true, labels) if n_eff > 1 else 0.0
    sil = silhouette_score(X, labels) if n_eff > 1 else 0.0
    db = davies_bouldin_score(X, labels) if n_eff > 1 else 0.0
    return {'ari': ari, 'nmi': nmi, 'silhouette': sil, 'davies_bouldin': db}

# ── Statistical tests ────────────────────────────────────────────────────────
def friedman_test(data):
    ranks = np.array([stats.rankdata(-row) for row in data])
    n, k = ranks.shape
    mean_ranks = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4)
    f_stat = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_value = 1 - stats.f.cdf(f_stat, k - 1, (n - 1) * (k - 1))
    return chi2, f_stat, p_value, mean_ranks, ranks

def nemenyi_test(mean_ranks, n_datasets, alpha=0.05):
    k = len(mean_ranks)
    q_alpha = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    q = q_alpha.get(k, 2.850)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_datasets))
    pvals = {}
    for i, j in itertools.combinations(range(k), 2):
        z = abs(mean_ranks[i] - mean_ranks[j]) / np.sqrt(k * (k + 1) / (6 * n_datasets))
        p = 2 * (1 - stats.norm.cdf(z))
        pvals[(i, j)] = p
    return cd, pvals

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 7A: PUBLICATION VALIDATION")
    print("=" * 70)

    results = {}

    for ds_name in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 70}")

        X, y = load_dataset(ds_name)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_clusters = len(np.unique(y))

        print(f"  Shape: {X.shape}, Clusters: {n_clusters}")
        results[ds_name] = {}
        sys.stdout.flush()

        for algo_name, algo_fn in ALGORITHMS.items():
            print(f"  Algorithm: {algo_name}...", end=' ', flush=True)
            results[ds_name][algo_name] = []

            for seed in SEEDS:
                try:
                    labels, runtime = algo_fn(X_scaled, n_clusters, seed)
                    metrics = compute_metrics(X_scaled, y, labels)
                    metrics['runtime'] = runtime
                    metrics['seed'] = seed
                    results[ds_name][algo_name].append(metrics)
                except Exception as e:
                    print(f"\n    ERROR seed={seed}: {e}")
                    results[ds_name][algo_name].append({
                        'ari': np.nan, 'nmi': np.nan, 'silhouette': np.nan,
                        'davies_bouldin': np.nan, 'runtime': np.nan, 'seed': seed
                    })

            if len(results[ds_name][algo_name]) == 10:
                aris = [r['ari'] for r in results[ds_name][algo_name] if not np.isnan(r['ari'])]
                print(f"ARI={np.mean(aris):.4f}±{np.std(aris):.4f} (n={len(aris)})", flush=True)
            else:
                print(f"{len(results[ds_name][algo_name])} seeds")

    # ── Save raw results for reproducibility ──────────────────────────────
    raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'phase7a_raw_results.json')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    # Convert to serializable format
    serializable = {}
    for ds_name in DATASETS:
        serializable[ds_name] = {}
        for algo_name in ALGORITHMS:
            serializable[ds_name][algo_name] = [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}
                for m in results[ds_name][algo_name]
            ]
    with open(raw_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # ── Aggregate into tables ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING TABLES AND STATISTICS")
    print("=" * 70)

    metrics_names = ['ari', 'nmi', 'silhouette', 'davies_bouldin', 'runtime']
    table_data = {}
    for metric in metrics_names:
        table_data[metric] = {}
        for ds_name in DATASETS:
            table_data[metric][ds_name] = {}
            for algo_name in ALGORITHMS:
                vals = [r[metric] for r in results[ds_name][algo_name] if not np.isnan(r[metric])]
                table_data[metric][ds_name][algo_name] = (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)

    algo_names = list(ALGORITHMS.keys())
    n_datasets = len(DATASETS)
    n_algorithms = len(algo_names)

    ranking_matrix = np.zeros((n_datasets, n_algorithms))
    for i, ds_name in enumerate(DATASETS):
        for j, algo_name in enumerate(algo_names):
            v = table_data['ari'][ds_name][algo_name][0]
            ranking_matrix[i, j] = v if not np.isnan(v) else -1

    chi2, f_stat, p_friedman, mean_ranks, ranks = friedman_test(ranking_matrix)
    cd, nemenyi_pvals = nemenyi_test(mean_ranks, n_datasets)

    cohens_d = {}
    for i, j in itertools.combinations(range(n_algorithms), 2):
        a_vals = np.array([table_data['ari'][ds][algo_names[i]][0] for ds in DATASETS])
        b_vals = np.array([table_data['ari'][ds][algo_names[j]][0] for ds in DATASETS])
        a_vals = a_vals[~np.isnan(a_vals)]
        b_vals = b_vals[~np.isnan(b_vals)]
        if len(a_vals) > 0 and len(b_vals) > 0:
            pooled = np.sqrt((np.std(a_vals) ** 2 + np.std(b_vals) ** 2) / 2)
            d = (np.mean(a_vals) - np.mean(b_vals)) / pooled if pooled > 0 else 0
        else:
            d = 0
        cohens_d[(algo_names[i], algo_names[j])] = d

    # ── Generate report ───────────────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'PHASE7A_VALIDATION_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 7A: Publication Validation Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Datasets**: {', '.join(DATASETS)}  \n")
        f.write(f"**Algorithms**: {', '.join(ALGORITHMS.keys())}  \n")
        f.write(f"**Seeds**: {SEEDS[0]}--{SEEDS[-1]} ({len(SEEDS)} seeds)  \n")
        f.write(f"**Total experiments**: {len(DATASETS) * len(ALGORITHMS) * len(SEEDS)}\n\n")

        # Dataset Summary
        f.write("## Dataset Summary\n\n")
        f.write("| Dataset | Samples | Features | Classes |\n")
        f.write("|---------|---------|----------|--------|\n")
        for ds_name in DATASETS:
            Xd, yd = load_dataset(ds_name)
            f.write(f"| {ds_name} | {Xd.shape[0]} | {Xd.shape[1]} | {len(np.unique(yd))} |\n")
        f.write("\n")

        # Primary Metric: ARI
        f.write("## Adjusted Rand Index (ARI)\n\n")
        f.write("| Dataset | " + " | ".join(f"{a}" for a in algo_names) + " |\n")
        f.write("|---------|" + "|".join("-" * max(len(a), 8) for a in algo_names) + "|\n")
        for ds_name in DATASETS:
            row = f"| {ds_name} "
            for a in algo_names:
                mu, sd = table_data['ari'][ds_name][a]
                row += f" | {mu:.4f}+-{sd:.4f}" if not np.isnan(mu) else " | ---"
            f.write(row + " |\n")
        f.write("\n")

        # Secondary Metrics
        for metric in ['nmi', 'silhouette', 'davies_bouldin', 'runtime']:
            label = metric.replace('_', ' ').title()
            f.write(f"## {label}\n\n")
            f.write("| Dataset | " + " | ".join(f"{a}" for a in algo_names) + " |\n")
            f.write("|---------|" + "|".join("-" * max(len(a), 8) for a in algo_names) + "|\n")
            for ds_name in DATASETS:
                row = f"| {ds_name} "
                for a in algo_names:
                    mu, sd = table_data[metric][ds_name][a]
                    row += f" | {mu:.4f}+-{sd:.4f}" if not np.isnan(mu) else " | ---"
                f.write(row + " |\n")
            f.write("\n")

        # Friedman Test
        f.write("## Friedman Test (Primary: ARI)\n\n")
        f.write(f"- Chi2 = {chi2:.4f}\n")
        f.write(f"- F-statistic = {f_stat:.4f}\n")
        f.write(f"- p-value = {p_friedman:.6f}\n\n")
        if p_friedman < 0.05:
            f.write("> Significant differences detected among algorithms (p < 0.05).\n\n")
        else:
            f.write("> No significant differences detected (p >= 0.05).\n\n")

        f.write("### Average Ranks\n\n")
        f.write("| Algorithm | Mean Rank |\n")
        f.write("|-----------|-----------|\n")
        sorted_idx = np.argsort(mean_ranks)
        for idx in sorted_idx:
            f.write(f"| {algo_names[idx]} | {mean_ranks[idx]:.2f} |\n")
        f.write("\n")

        # Nemenyi Post-Hoc
        f.write("## Nemenyi Post-Hoc Test\n\n")
        f.write(f"Critical Difference (CD) at alpha=0.05: **{cd:.4f}**\n\n")
        f.write("| Algorithm A | Algorithm B | p-value | Significant?\n")
        f.write("|-------------|-------------|---------|------------|\n")
        for (i, j), p in sorted(nemenyi_pvals.items(), key=lambda x: x[1]):
            sig = "YES" if p < 0.05 else "NO"
            f.write(f"| {algo_names[i]} | {algo_names[j]} | {p:.6f} | {sig} |\n")
        f.write("\n")

        # Effect Sizes
        f.write("## Effect Sizes (Cohen's d on ARI)\n\n")
        f.write("| Algorithm A | Algorithm B | Cohen's d | Interpretation |\n")
        f.write("|-------------|-------------|-----------|----------------|\n")
        for (a, b), d in sorted(cohens_d.items(), key=lambda x: abs(x[1]), reverse=True):
            interp = "Large" if abs(d) >= 0.8 else ("Medium" if abs(d) >= 0.5 else ("Small" if abs(d) >= 0.2 else "Negligible"))
            f.write(f"| {a} | {b} | {d:.4f} | {interp} |\n")
        f.write("\n")

        # CD Diagram
        f.write("## Critical Difference Diagram\n\n")
        f.write("```\n")
        order = np.argsort(mean_ranks)
        f.write(f"CD = {cd:.2f}\n\n")
        min_r, max_r = mean_ranks.min(), mean_ranks.max()
        rng_r = max_r - min_r if max_r > min_r else 1

        def rank_to_pos(r):
            return int((r - min_r) / rng_r * 58)

        axis_len = 60
        f.write("Ranks:  ")
        for i in range(11):
            tick = min_r + i * rng_r / 10
            f.write(f"{tick:5.1f} ")
        f.write("\n")
        f.write("        " + "-" * axis_len + "\n")

        for idx in order:
            name = algo_names[idx]
            r = mean_ranks[idx]
            pos = rank_to_pos(r)
            f.write(f"        {' ' * pos}  {name} ({r:.2f})\n")

        best_r = mean_ranks[order[0]]
        cd_left = max(min_r, best_r - cd)
        cd_right = min(max_r, best_r + cd)
        p_left = rank_to_pos(cd_left)
        p_right = rank_to_pos(cd_right)
        f.write(f"        {' ' * p_left}{'>' * (p_right - p_left)}{' ' * (axis_len - p_right)}\n")
        f.write(f"        CD={cd:.2f} interval centered on best rank\n")
        f.write("```\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"**Best overall ARI**: {algo_names[sorted_idx[0]]} (mean rank {mean_ranks[sorted_idx[0]]:.2f})\n\n")
        f.write(f"**Significant differences detected**: {'Yes' if p_friedman < 0.05 else 'No'}\n\n")

        f.write("### Per-Dataset Winners (ARI)\n\n")
        f.write("| Dataset | Winner | ARI |\n")
        f.write("|---------|--------|-----|\n")
        for ds_name in DATASETS:
            best_algo = max(algo_names, key=lambda a: table_data['ari'][ds_name][a][0] if not np.isnan(table_data['ari'][ds_name][a][0]) else -1)
            best_val = table_data['ari'][ds_name][best_algo][0]
            f.write(f"| {ds_name} | {best_algo} | {best_val:.4f} |\n")
        f.write("\n")

    print(f"\n\nReport written to: {report_path}")
    return results, table_data, mean_ranks, cd

if __name__ == '__main__':
    main()
