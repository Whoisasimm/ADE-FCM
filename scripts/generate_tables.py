"""Generate publication tables in CSV and LaTeX format from raw experimental data."""
import json, os
import numpy as np
from scipy.stats import friedmanchisquare, rankdata

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base, 'results', 'phase7a_raw_results.json')) as f:
    data = json.load(f)

datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'glass', 'seeds', 'sonar']
algos = ['ADE-FCM', 'DeepADEFCM', 'FCM', 'FCLM', 'KMeans', 'Agglomerative']
metrics = ['ari', 'nmi', 'silhouette', 'davies_bouldin', 'runtime']
metric_labels = {'ari': 'ARI', 'nmi': 'NMI', 'silhouette': 'Silhouette',
                 'davies_bouldin': 'Davies-Bouldin', 'runtime': 'Runtime (s)'}
outdir = os.path.join(base, 'paper', 'tables')
os.makedirs(outdir, exist_ok=True)

def get_stats(ds, algo, metric):
    vals = [r[metric] for r in data[ds][algo]
            if r[metric] is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1))

def esc(s):
    return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')

def cohens_d_pooled(vals_a, vals_b):
    """Compute Cohen's d using pooled standard deviation across all observations."""
    m1, m2 = np.mean(vals_a), np.mean(vals_b)
    n1, n2 = len(vals_a), len(vals_b)
    s1, s2 = np.std(vals_a, ddof=1), np.std(vals_b, ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if sp == 0:
        return 0.0
    return (m1 - m2) / sp

# --- Per-metric result tables ---
for metric in metrics:
    ml = metric_labels[metric]
    csv_lines = ['Dataset,' + ','.join(algos)]
    tex_lines = [
        '\\begin{table}[ht]', '\\centering',
        '\\caption{' + ml + ' Results for All Datasets and Algorithms.}',
        '\\label{tab:' + metric + '}',
        '\\begin{tabular}{l' + 'c' * len(algos) + '}', '\\toprule',
        'Dataset & ' + ' & '.join(algos) + ' \\\\', '\\midrule',
    ]
    for ds in datasets:
        row_csv = [ds]
        row_tex = [ds]
        for a in algos:
            mu, sd = get_stats(ds, a, metric)
            if mu is None:
                row_csv.append('---')
                row_tex.append('---')
            else:
                row_csv.append(f'{mu:.4f}+-{sd:.4f}')
                row_tex.append(f'${mu:.4f}\\pm{sd:.4f}$')
        csv_lines.append(','.join(row_csv))
        tex_lines.append(' & '.join(row_tex) + ' \\\\')
    tex_lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(outdir, f'{metric}_results.csv'), 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    with open(os.path.join(outdir, f'{metric}_results.tex'), 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')

# --- Rankings (by ARI, higher=better, tied=mean rank) ---
# Build per-dataset mean ARI matrix
ari_matrix = np.zeros((len(datasets), len(algos)))
for i, ds in enumerate(datasets):
    for j, algo in enumerate(algos):
        mu, _ = get_stats(ds, algo, 'ari')
        ari_matrix[i, j] = mu if mu is not None else -np.inf

# Rank each row (higher ARI -> rank 1)
rank_matrix = np.array([rankdata(-row, method='average') for row in ari_matrix])
mean_ranks = {algos[j]: float(np.mean(rank_matrix[:, j])) for j in range(len(algos))}

sorted_algos = sorted(algos, key=lambda a: mean_ranks[a])
with open(os.path.join(outdir, 'algorithm_rankings.csv'), 'w') as f:
    f.write('Algorithm,MeanRank\n')
    for a in sorted_algos:
        f.write(f'{a},{mean_ranks[a]:.2f}\n')

with open(os.path.join(outdir, 'algorithm_rankings.tex'), 'w') as f:
    f.write('\\begin{table}[ht]\n\\centering\n')
    f.write('\\caption{Mean Algorithm Rankings Across All Datasets (ARI).}\n')
    f.write('\\label{tab:rankings}\n')
    f.write('\\begin{tabular}{lc}\n\\toprule\nAlgorithm & Mean Rank \\\\\n\\midrule\n')
    for a in sorted_algos:
        f.write(f'{esc(a)} & {mean_ranks[a]:.2f} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# --- Friedman test ---
friedman_cols = [rank_matrix[:, j] for j in range(len(algos))]
stat, p_friedman = friedmanchisquare(*friedman_cols)

# --- Effect sizes (Cohen's d, pooled) on ARI ---
effect_pairs = []
for i in range(len(algos)):
    for j in range(i+1, len(algos)):
        a, b = algos[i], algos[j]
        all_a, all_b = [], []
        for ds in datasets:
            mu_a, _ = get_stats(ds, a, 'ari')
            mu_b, _ = get_stats(ds, b, 'ari')
            if mu_a is not None and mu_b is not None:
                all_a.extend([r['ari'] for r in data[ds][a]])
                all_b.extend([r['ari'] for r in data[ds][b]])
        if all_a and all_b:
            d = cohens_d_pooled(all_a, all_b)
            if abs(d) >= 0.8:
                interp = 'Large'
            elif abs(d) >= 0.5:
                interp = 'Medium'
            elif abs(d) >= 0.2:
                interp = 'Small'
            else:
                interp = 'Negligible'
            effect_pairs.append((a, b, d, interp))
    effect_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

with open(os.path.join(outdir, 'effect_sizes.csv'), 'w') as f:
    f.write('Algorithm_A,Algorithm_B,Cohens_d,Interpretation\n')
    for a, b, d, i in effect_pairs:
        f.write(f'{a},{b},{d:.4f},{i}\n')

with open(os.path.join(outdir, 'effect_sizes.tex'), 'w') as f:
    f.write('\\begin{table}[ht]\n\\centering\n')
    f.write('\\caption{Cohen\\\'s d Effect Sizes for Pairwise Algorithm Comparisons on ARI (pooled standard deviation).}\n')
    f.write('\\label{tab:effects}\n')
    f.write('\\begin{tabular}{lcl}\n\\toprule\nComparison & d & Interpretation \\\\\n\\midrule\n')
    for a, b, d, i in effect_pairs:
        f.write(f'{esc(a)} vs {esc(b)} & ${d:.2f}$ & {i} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# --- Nemenyi post-hoc test ---
# Critical difference for Nemenyi: CD = q_alpha * sqrt(k(k+1)/(6N))
# where q_alpha for k=6, alpha=0.05 is approximately 2.850 (from Demsar 2006)
k = len(algos)
n = len(datasets)
q_alpha = 2.850  # for k=6 at alpha=0.05
cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

# Pairwise Nemenyi: compare mean rank differences to CD
nemenyi_pairs = []
for i in range(len(algos)):
    for j in range(i+1, len(algos)):
        a, b = algos[i], algos[j]
        rd = abs(mean_ranks[a] - mean_ranks[b])
        significant = rd > cd
        # Approximate p-value using the studentized range distribution
        # Use the fact that Nemenyi is based on the studentized range
        from scipy.stats import studentized_range
        # The test statistic: z = rd / sqrt(k(k+1)/(6n))
        z = rd / np.sqrt(k * (k + 1) / (6 * n))
        # p-value from studentized range with k groups and inf df
        p = 1.0 - studentized_range.cdf(z * np.sqrt(2), k, np.inf)
        nemenyi_pairs.append((a, b, p, significant))

nemenyi_pairs.sort(key=lambda x: x[2])

with open(os.path.join(outdir, 'nemenyi_results.csv'), 'w') as f:
    f.write('Algorithm_A,Algorithm_B,p_value,Significant\n')
    for a, b, p, s in nemenyi_pairs:
        f.write(f'{a},{b},{p:.4f},{s}\n')

# --- Dataset summary ---
ds_info = {
    'iris': (150, 4, 3), 'wine': (178, 13, 3), 'breast_cancer': (569, 30, 2),
    'digits': (1797, 64, 10), 'glass': (214, 9, 6), 'seeds': (210, 7, 3),
    'sonar': (208, 60, 2)
}
csv_lines = ['Dataset,Samples,Features,Classes']
tex_lines = [
    '\\begin{table}[ht]', '\\centering',
    '\\caption{Benchmark Dataset Summary.}', '\\label{tab:datasets}',
    '\\begin{tabular}{lccc}', '\\toprule',
    'Dataset & Samples & Features & Classes \\\\', '\\midrule',
]
for ds, (n, d, c) in ds_info.items():
    csv_lines.append(f'{ds},{n},{d},{c}')
    tex_lines.append(f'{ds} & {n} & {d} & {c} \\\\')
tex_lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']

with open(os.path.join(outdir, 'dataset_summary.csv'), 'w') as f:
    f.write('\n'.join(csv_lines) + '\n')
with open(os.path.join(outdir, 'dataset_summary.tex'), 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')

print(f'Friedman chi2 = {stat:.3f}, p = {p_friedman:.3f}')
print(f'Nemenyi CD (alpha=0.05) = {cd:.3f}')
print('Tables generated in paper/tables/')
for fname in sorted(os.listdir(outdir)):
    print(f'  {fname}')
