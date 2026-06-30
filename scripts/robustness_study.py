"""Robustness study: noise, outliers, missing values on 3 representative datasets."""
import sys, os, json, time, warnings
import numpy as np
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

def run_ade_fcm(X, n_clusters, seed):
    from novel_algorithm.ade_fcm import ADEFCM
    metric = 'cosine' if X.shape[1] >= 30 else 'euclidean'
    m = ADEFCM(n_clusters=n_clusters, max_iter=100, m='adaptive', epsilon='dynamic',
               metric=metric, compute_xai=False, random_state=seed, verbose=False)
    t0 = time.time()
    m.fit(X)
    return m.labels_, time.time() - t0

def run_fcm(X, n_clusters, seed):
    from scripts.phase7a_validation import run_fcm as _run_fcm
    return _run_fcm(X, n_clusters, seed)

# Datasets: small (iris), medium (wine), large (optdigits)
datasets_info = {
    'iris': (sk_datasets.load_iris, 3),
    'wine': (sk_datasets.load_wine, 3),
    'digits': (sk_datasets.load_digits, 10),
}

SEEDS = [42, 43, 44]
results = {}

for ds_name, (loader, n_clusters) in datasets_info.items():
    X_raw, y = loader(return_X_y=True)
    X_base = StandardScaler().fit_transform(X_raw)
    n, d = X_base.shape

    print(f'\n=== {ds_name} ({n}x{d}, K={n_clusters}) ===')
    ds_results = {}

    # --- Clean baseline ---
    for algo_name, runner in [('ADE-FCM', run_ade_fcm), ('FCM', run_fcm)]:
        records = []
        for seed in SEEDS:
            labels, runtime = runner(X_base, n_clusters, seed)
            ari = adjusted_rand_score(y, labels) if len(set(labels)) > 1 else 0.0
            records.append({'ari': ari, 'runtime': round(runtime, 4), 'seed': seed})
        mean_ari = np.mean([r['ari'] for r in records])
        print(f'  Clean {algo_name:10s}: ARI={mean_ari:.4f}')
        ds_results[f'clean_{algo_name.lower().replace("-","_")}'] = records

    # --- 1. Label noise: flip 5%, 10%, 20% of labels ---
    for noise_pct in [5, 10, 20]:
        X_noise = X_base.copy()
        rng = np.random.RandomState(42)
        n_flip = int(n * noise_pct / 100)
        flip_idx = rng.choice(n, n_flip, replace=False)
        X_noise[flip_idx] += rng.randn(n_flip, d) * 0.5  # add gaussian noise to features

        for algo_name, runner in [('ADE-FCM', run_ade_fcm), ('FCM', run_fcm)]:
            records = []
            for seed in SEEDS:
                try:
                    labels, runtime = runner(X_noise, n_clusters, seed)
                    ari = adjusted_rand_score(y, labels) if len(set(labels)) > 1 else 0.0
                    records.append({'ari': ari, 'runtime': round(runtime, 4), 'seed': seed})
                except Exception as e:
                    records.append({'ari': None, 'error': str(e), 'seed': seed})
            ari_vals = [r['ari'] for r in records if r['ari'] is not None]
            if ari_vals:
                print(f'  Noise{noise_pct}% {algo_name:10s}: ARI={np.mean(ari_vals):.4f}')
            ds_results[f'noise_{noise_pct}_{algo_name.lower().replace("-","_")}'] = records

    # --- 2. Outliers: replace 5%, 10% of points with extreme values ---
    for outlier_pct in [5, 10]:
        X_out = X_base.copy()
        rng = np.random.RandomState(42)
        n_out = int(n * outlier_pct / 100)
        out_idx = rng.choice(n, n_out, replace=False)
        X_out[out_idx] = rng.randn(n_out, d) * 10  # extreme outliers

        for algo_name, runner in [('ADE-FCM', run_ade_fcm), ('FCM', run_fcm)]:
            records = []
            for seed in SEEDS:
                try:
                    labels, runtime = runner(X_out, n_clusters, seed)
                    ari = adjusted_rand_score(y, labels) if len(set(labels)) > 1 else 0.0
                    records.append({'ari': ari, 'runtime': round(runtime, 4), 'seed': seed})
                except Exception as e:
                    records.append({'ari': None, 'error': str(e), 'seed': seed})
            ari_vals = [r['ari'] for r in records if r['ari'] is not None]
            if ari_vals:
                print(f'  Outlier{outlier_pct}% {algo_name:10s}: ARI={np.mean(ari_vals):.4f}')
            ds_results[f'outlier_{outlier_pct}_{algo_name.lower().replace("-","_")}'] = records

    # --- 3. Missing values: zero out 5%, 10%, 20% of entries ---
    for miss_pct in [5, 10, 20]:
        X_miss = X_base.copy()
        rng = np.random.RandomState(42)
        n_miss = int(n * d * miss_pct / 100)
        miss_idx = rng.choice(n * d, n_miss, replace=False)
        rows, cols = np.unravel_index(miss_idx, (n, d))
        X_miss[rows, cols] = 0.0  # impute missing as 0 (mean after standardization)

        for algo_name, runner in [('ADE-FCM', run_ade_fcm), ('FCM', run_fcm)]:
            records = []
            for seed in SEEDS:
                try:
                    labels, runtime = runner(X_miss, n_clusters, seed)
                    ari = adjusted_rand_score(y, labels) if len(set(labels)) > 1 else 0.0
                    records.append({'ari': ari, 'runtime': round(runtime, 4), 'seed': seed})
                except Exception as e:
                    records.append({'ari': None, 'error': str(e), 'seed': seed})
            ari_vals = [r['ari'] for r in records if r['ari'] is not None]
            if ari_vals:
                print(f'  Missing{miss_pct}% {algo_name:10s}: ARI={np.mean(ari_vals):.4f}')
            ds_results[f'missing_{miss_pct}_{algo_name.lower().replace("-","_")}'] = records

    results[ds_name] = ds_results

outpath = os.path.join(BASE, 'results', 'robustness_study.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: {outpath} ({os.path.getsize(outpath)} bytes)')

# Summary table
print('\n\n=== ROBUSTNESS SUMMARY TABLE ===')
print(f'{"Dataset":12s} {"Condition":18s} {"ADE-FCM ARI":12s} {"FCM ARI":12s} {"Drop":8s}')
print('-' * 62)
for ds_name in datasets_info:
    clean_ade = np.mean([r['ari'] for r in results[ds_name]['clean_ade_fcm']])
    clean_fcm = np.mean([r['ari'] for r in results[ds_name]['clean_fcm']])
    for cond in ['noise_10', 'outlier_10', 'missing_10']:
        ade_key = f'{cond}_ade_fcm'
        fcm_key = f'{cond}_fcm'
        if ade_key in results[ds_name]:
            ade_ari = np.mean([r['ari'] for r in results[ds_name][ade_key] if r['ari'] is not None])
            fcm_ari = np.mean([r['ari'] for r in results[ds_name][fcm_key] if r['ari'] is not None])
            ade_drop = (ade_ari - clean_ade) / max(abs(clean_ade), 1e-10) * 100
            fcm_drop = (fcm_ari - clean_fcm) / max(abs(clean_fcm), 1e-10) * 100
            cond_label = cond.replace('_', ' ').title()
            print(f'{ds_name:12s} {cond_label:18s} {ade_ari:.4f}       {fcm_ari:.4f}       {ade_drop:+.1f}%')
