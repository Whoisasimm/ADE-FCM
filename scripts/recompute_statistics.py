"""Recompute statistics with expanded 18-dataset benchmark."""
import sys, os, json, warnings
import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

with open(os.path.join(BASE, 'results', 'expanded_benchmarks.json')) as f:
    data = json.load(f)

# Build ARI matrix: datasets × algorithms
algorithms = ['ADE-FCM', 'FCM', 'KMeans', 'Agglomerative', 'FCLM']
ari_matrix = {}
for ds_name, ds_data in data.items():
    results = ds_data.get('results', {})
    row = {}
    for algo in algorithms:
        records = results.get(algo, [])
        ari_vals = [r['ari'] for r in records if r.get('ari') is not None]
        if ari_vals:
            row[algo] = np.mean(ari_vals)
    if row:
        ari_matrix[ds_name] = row

print(f'Loaded {len(ari_matrix)} datasets with ARI data')

# Build table
print(f'\n{"Dataset":20s}', end='')
for a in algorithms:
    print(f'{a:15s}', end='')
print()
print('-' * (20 + 15 * len(algorithms)))
for ds_name, row in ari_matrix.items():
    print(f'{ds_name:20s}', end='')
    for a in algorithms:
        v = row.get(a, -1)
        print(f'{v:<15.4f}', end='')
    print()

# --- Cohen's d: ADE-FCM vs FCM ---
ade_aris = np.array([ari_matrix[d]['ADE-FCM'] for d in ari_matrix if 'ADE-FCM' in ari_matrix[d] and 'FCM' in ari_matrix[d]])
fcm_aris = np.array([ari_matrix[d]['FCM'] for d in ari_matrix if 'ADE-FCM' in ari_matrix[d] and 'FCM' in ari_matrix[d]])
n_pairs = len(ade_aris)
diff = ade_aris - fcm_aris
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

print(f'\n=== ADE-FCM vs FCM (n={n_pairs} datasets) ===')
print(f'  ADE-FCM mean ARI: {np.mean(ade_aris):.4f} ± {np.std(ade_aris, ddof=1):.4f}')
print(f'  FCM mean ARI:     {np.mean(fcm_aris):.4f} ± {np.std(fcm_aris, ddof=1):.4f}')
print(f'  Mean difference:  {mean_diff:.4f}')
print(f'  Std difference:   {std_diff:.4f}')
print(f'  Cohen\'s d:        {cohens_d:.4f}')

# Bootstrap CI for Cohen's d
rng = np.random.RandomState(42)
n_bootstrap = 10000
d_samples = []
for _ in range(n_bootstrap):
    idx = rng.choice(n_pairs, n_pairs, replace=True)
    d_boot = np.mean(diff[idx]) / max(np.std(diff[idx], ddof=1), 1e-10)
    d_samples.append(d_boot)
d_ci_low, d_ci_high = np.percentile(d_samples, [2.5, 97.5])
print(f'  Cohen\'s d 95% CI: [{d_ci_low:.4f}, {d_ci_high:.4f}]')

# Bootstrap CI for ARI means
ade_samples = []
fcm_samples = []
for _ in range(n_bootstrap):
    idx = rng.choice(len(ade_aris), len(ade_aris), replace=True)
    ade_samples.append(np.mean(ade_aris[idx]))
    fcm_samples.append(np.mean(fcm_aris[idx]))
ade_ci = np.percentile(ade_samples, [2.5, 97.5])
fcm_ci = np.percentile(fcm_samples, [2.5, 97.5])
print(f'  ADE-FCM ARI 95% CI: [{ade_ci[0]:.4f}, {ade_ci[1]:.4f}]')
print(f'  FCM ARI 95% CI:     [{fcm_ci[0]:.4f}, {fcm_ci[1]:.4f}]')

# --- Paired t-test across datasets ---
print(f'\n=== Paired test across datasets (ADE-FCM vs FCM, n={n_pairs} pairs) ===')
t_stat, t_p = sp_stats.ttest_rel(ade_aris, fcm_aris)
try:
    w_stat, w_p = sp_stats.wilcoxon(ade_aris, fcm_aris, alternative='two-sided')
except:
    w_stat, w_p = 0, 1.0
print(f'  Paired t-test:  t={t_stat:.4f}, p={t_p:.6f}')
print(f'  Wilcoxon test:  W={w_stat:.0f}, p={w_p:.6f}')
print(f'  Mean diff: {mean_diff:.4f} (ADE-FCM - FCM)')
print(f'  Significant: {"YES" if t_p < 0.05 else "NO"} (t-test), {"YES" if w_p < 0.05 else "NO"} (Wilcoxon)')

# --- Friedman test ---
algo_list = [a for a in algorithms if all(a in ari_matrix[d] for d in ari_matrix)]
if len(algo_list) >= 3:
    friedman_data = []
    for ds_name in sorted(ari_matrix.keys()):
        row = [ari_matrix[ds_name][a] for a in algo_list]
        friedman_data.append(row)
    friedman_data = np.array(friedman_data)
    try:
        f_stat, f_p = sp_stats.friedmanchisquare(*[friedman_data[:, i] for i in range(len(algo_list))])
        print(f'\n=== Friedman Test ===')
        print(f'  Algorithms: {algo_list}')
        print(f'  Friedman stat = {f_stat:.4f}, p = {f_p:.6f}')
        print(f'  Significant: {"YES" if f_p < 0.05 else "NO"}')

        # Nemenyi post-hoc if significant
        if f_p < 0.05:
            from scipy.stats import rankdata
            import itertools
            ranks = np.array([rankdata(-row) for row in friedman_data])
            mean_ranks = ranks.mean(axis=0)
            print('\n  Mean ranks:')
            for i, a in enumerate(algo_list):
                print(f'    {a:15s}: {mean_ranks[i]:.3f}')

            q_alpha = 2.569  # critical value for 5 algorithms at alpha=0.05
            cd = q_alpha * np.sqrt(len(algo_list) * (len(algo_list) + 1) / (6 * len(ari_matrix)))
            print(f'  Critical difference (Nemenyi): {cd:.4f}')
            print('\n  Pairwise comparisons:')
            for i, j in itertools.combinations(range(len(algo_list)), 2):
                diff_r = abs(mean_ranks[i] - mean_ranks[j])
                print(f'    {algo_list[i]:15s} vs {algo_list[j]:15s}: rank diff={diff_r:.3f} {"SIG" if diff_r > cd else "n.s."}')
    except Exception as e:
        print(f'  Friedman test failed: {e}')

# --- Ranking ---
print(f'\n=== Algorithm Rankings (mean ARI across {len(ari_matrix)} datasets) ===')
ranked = sorted([(np.mean([ari_matrix[d][a] for d in ari_matrix if a in ari_matrix[d]]), a) for a in algorithms], reverse=True)
for i, (m, a) in enumerate(ranked):
    print(f'  {i+1}. {a:15s}: {m:.4f}')

# Save results
friedman_stat = locals().get('f_stat')
friedman_p = locals().get('f_p')
stats = {
    'n_datasets': len(ari_matrix),
    'ade_fcm_mean_ari': float(np.mean(ade_aris)),
    'fcm_mean_ari': float(np.mean(fcm_aris)),
    'cohens_d': float(cohens_d),
    'cohens_d_ci_95': [float(d_ci_low), float(d_ci_high)],
    'ade_ari_ci_95': [float(ade_ci[0]), float(ade_ci[1])],
    'fcm_ari_ci_95': [float(fcm_ci[0]), float(fcm_ci[1])],
    'paired_t_test_stat': float(t_stat),
    'paired_t_test_p': float(t_p),
    'wilcoxon_stat': float(w_stat),
    'wilcoxon_p': float(w_p),
    'friedman_stat': float(friedman_stat) if friedman_stat is not None else None,
    'friedman_p': float(friedman_p) if friedman_p is not None else None,
    'ranking': [a for _, a in ranked],
}
with open(os.path.join(BASE, 'results', 'statistics_summary.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(f'\nSaved statistics_summary.json')
