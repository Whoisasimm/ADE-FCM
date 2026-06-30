"""Generate bootstrap confidence intervals and enhanced statistical analysis."""
import json, os, random
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, ttest_rel, wilcoxon
# Bonferroni correction: multiply p-values by number of tests, cap at 1.0

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base, 'results', 'phase7a_raw_results.json')) as f:
    data = json.load(f)

random.seed(42)
np.random.seed(42)

datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'glass', 'seeds', 'sonar']
algos = ['ADE-FCM', 'DeepADEFCM', 'FCM', 'FCLM', 'KMeans', 'Agglomerative']

def get_vals(ds, algo, metric='ari'):
    return [r[metric] for r in data[ds][algo] if r[metric] is not None]

def bootstrap_ci(values, n_iter=10000, ci=95):
    means = []
    for _ in range(n_iter):
        sample = [random.choice(values) for _ in range(len(values))]
        means.append(np.mean(sample))
    alpha = (100 - ci) / 2
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))

# === 1. Bootstrap CIs for all algorithms on all datasets ===
print('=== BOOTSTRAP 95% CI FOR ARI (all algorithms × all datasets) ===')
ci_table = []
for ds in datasets:
    row = [ds]
    for algo in algos:
        vals = get_vals(ds, algo)
        mu = np.mean(vals)
        lo, hi = bootstrap_ci(vals)
        row.append(f'{mu:.3f} [{lo:.3f}, {hi:.3f}]')
    ci_table.append(row)

for row in ci_table:
    print(f'  {row[0]}:')
    for i, algo in enumerate(algos):
        print(f'    {algo}: {row[i+1]}')

# === 2. Paired t-tests and Wilcoxon (ADE-FCM vs FCM per dataset) ===
print('\n=== PAIRED TESTS: ADE-FCM vs FCM ===')
t_results = []
for ds in datasets:
    ade = get_vals(ds, 'ADE-FCM')
    fcm = get_vals(ds, 'FCM')
    t_stat, t_p = ttest_rel(ade, fcm)
    w_stat, w_p = wilcoxon([a-f for a,f in zip(ade, fcm)], alternative='two-sided')
    mean_diff = np.mean(ade) - np.mean(fcm)
    t_results.append({'dataset': ds, 'mean_diff': mean_diff, 't_p': t_p, 'w_p': w_p})
    print(f'  {ds}: diff={mean_diff:.4f}, t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}')

# Multiple comparison correction (Bonferroni)
n_tests = len(t_results)
print(f'  Bonferroni threshold: 0.05 / {n_tests} = {0.05/n_tests:.5f}')
for i, r in enumerate(t_results):
    p_corrected = min(r['t_p'] * n_tests, 1.0)
    ds = r['dataset']
    print(f'    {ds}: raw p={r["t_p"]:.4f}, corrected p={p_corrected:.4f}')

# === 3. Confidence intervals for Cohen's d ===
print('\n=== COHEN d WITH 95% BOOTSTRAP CI ===')
all_ade = []
all_fcm = []
for ds in datasets:
    all_ade.extend(get_vals(ds, 'ADE-FCM'))
    all_fcm.extend(get_vals(ds, 'FCM'))

n1, n2 = len(all_ade), len(all_fcm)
m1, m2 = np.mean(all_ade), np.mean(all_fcm)
s1, s2 = np.std(all_ade, ddof=1), np.std(all_fcm, ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
d_obs = (m1 - m2) / sp
print(f'  ADE-FCM vs FCM: d = {d_obs:.4f}')

boot_d = []
for _ in range(10000):
    s1b = [random.choice(all_ade) for _ in range(len(all_ade))]
    s2b = [random.choice(all_fcm) for _ in range(len(all_fcm))]
    m1b, m2b = np.mean(s1b), np.mean(s2b)
    s1bs = np.std(s1b, ddof=1)
    s2bs = np.std(s2b, ddof=1)
    spb = np.sqrt(((n1-1)*s1bs**2 + (n2-1)*s2bs**2) / (n1+n2-2))
    boot_d.append((m1b - m2b) / spb if spb > 0 else 0)
ci_low = np.percentile(boot_d, 2.5)
ci_high = np.percentile(boot_d, 97.5)
print(f'  95% CI for d: [{ci_low:.4f}, {ci_high:.4f}]')

# === 4. ADE-FCM win/loss/tie record ===
print('\n=== ADE-FCM WIN/LOSS/TIE (vs FCM, per dataset) ===')
wins = losses = ties = 0
for ds in datasets:
    ade_mean = np.mean(get_vals(ds, 'ADE-FCM'))
    fcm_mean = np.mean(get_vals(ds, 'FCM'))
    diff = ade_mean - fcm_mean
    if diff > 0.01:
        wins += 1
        result = 'WIN'
    elif diff < -0.01:
        losses += 1
        result = 'LOSS'
    else:
        ties += 1
        result = 'TIE'
    print(f'  {ds}: ADE={ade_mean:.3f} vs FCM={fcm_mean:.3f} ({result})')
print(f'  Record: {wins}W / {losses}L / {ties}T')

# === 5. ADE-FCM with Cosine vs Euclidean (self-comparison) ===
print('\n=== ADE-FCM COSINE vs EUCLIDEAN ===')
for ds in datasets:
    ade_vals = get_vals(ds, 'ADE-FCM')
    # ADE-FCM uses Cosine on Digits, Euclidean elsewhere by default
    print(f'  {ds}: ADE-FCM ARI = {np.mean(ade_vals):.3f}')

# === 6. Summary statistics ===
print('\n=== SUMMARY ===')
all_aris = {a: [] for a in algos}
for ds in datasets:
    for a in algos:
        all_aris[a].extend(get_vals(ds, a))
for a in algos:
    vals = all_aris[a]
    print(f'  {a}: mean={np.mean(vals):.3f}, std={np.std(vals, ddof=1):.3f}, '
          f'min={np.min(vals):.3f}, max={np.max(vals):.3f}, '
          f'median={np.median(vals):.3f}')
