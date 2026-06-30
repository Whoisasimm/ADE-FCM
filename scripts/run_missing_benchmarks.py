"""Run missing OpenML benchmarks (no DeepADEFCM)."""
import sys, os, json, time, warnings
import numpy as np
from sklearn import datasets as sk_datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from scripts.phase7a_validation import run_fcm as _run_fcm
from scripts.phase7a_validation import run_fclm as _run_fclm
from scripts.phase7a_validation import run_kmeans as _run_kmeans
from scripts.phase7a_validation import run_agglomerative as _run_aggl

def run_ade_fcm_fast(X, n_clusters, seed, max_iter=100):
    from novel_algorithm.ade_fcm import ADEFCM
    metric = 'cosine' if X.shape[1] >= 30 else 'euclidean'
    m = ADEFCM(n_clusters=n_clusters, max_iter=max_iter, m='adaptive', epsilon='dynamic',
               metric=metric, compute_xai=False, random_state=seed, verbose=False)
    t0 = time.time()
    m.fit(X)
    return m.labels_, time.time() - t0

SEEDS = [42, 43, 44, 45, 46]
runners = {
    'ADE-FCM': run_ade_fcm_fast,
    'FCM': _run_fcm,
    'KMeans': _run_kmeans,
    'Agglomerative': _run_aggl,
    'FCLM': _run_fclm,
}
deterministic = {'FCM', 'Agglomerative'}

def compute_metrics(X, y_true, labels):
    n_eff = len(set(labels))
    return {
        'ari': adjusted_rand_score(y_true, labels) if n_eff > 1 else 0.0,
        'nmi': normalized_mutual_info_score(y_true, labels) if n_eff > 1 else 0.0,
        'silhouette': silhouette_score(X, labels) if n_eff > 1 else -1.0,
        'davies_bouldin': davies_bouldin_score(X, labels) if n_eff > 1 else -1.0,
    }

openml_names = ['glass', 'seeds', 'sonar', 'ecoli', 'yeast', 'vehicle', 'segment', 'optdigits', 'mfeat-factors']

all_results = {}
total_exp = 0
t_start = time.time()

for name in openml_names:
    try:
        data = fetch_openml(name=name, version=1, parser='auto', as_frame=False)
        X, y = np.array(data.data, dtype=float), np.array(data.target)
        if y.dtype.kind == 'O':
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = y.astype(int)
        k = len(np.unique(y))
        X = StandardScaler().fit_transform(X)
        print(f'--- {name}: {X.shape}, K={k} ---')

        ds_results = {}
        for algo, runner in runners.items():
            seeds = [42] if algo in deterministic else SEEDS
            records = []
            for seed in seeds:
                try:
                    labels, runtime = runner(X, k, seed)
                    metrics = compute_metrics(X, y, labels)
                    metrics['runtime'] = runtime
                    metrics['seed'] = seed
                    records.append(metrics)
                except Exception as e:
                    records.append({'seed': seed, 'ari': None, 'nmi': None, 'silhouette': None, 'davies_bouldin': None, 'runtime': 0, 'error': str(e)})
                total_exp += 1
            ari_vals = [r['ari'] for r in records if r['ari'] is not None]
            if ari_vals:
                print(f'  {algo:15s}: ARI={np.mean(ari_vals):.3f}')
            ds_results[algo] = records
        all_results[name] = {
            'n_samples': X.shape[0], 'n_features': X.shape[1], 'n_clusters': k,
            'results': ds_results,
        }
    except Exception as e:
        print(f'[ERR] {name}: {e}')

elapsed = time.time() - t_start
print(f'\nDONE: {total_exp} experiments in {elapsed:.0f}s')

outpath = os.path.join(BASE, 'results', 'expanded_benchmarks_missing.json')
with open(outpath, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to {outpath} ({os.path.getsize(outpath)} bytes)')
