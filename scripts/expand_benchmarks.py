"""Expanded benchmark: 20 datasets × 6 algorithms × 5 seeds.
Reuses algorithm runners from phase7a_validation.py.
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import datasets as sk_datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Reuse algorithm runners from phase7a validation
from scripts.phase7a_validation import run_ade_fcm, run_fcm, run_fclm, run_kmeans, run_agglomerative, run_deep_ade_fcm

OUTPUT = os.path.join(BASE, 'results', 'expanded_benchmarks.json')
SEEDS = [42, 43, 44, 45, 46]

def compute_metrics(X, y_true, labels):
    n_eff = len(set(labels))
    return {
        'ari': adjusted_rand_score(y_true, labels) if n_eff > 1 else 0.0,
        'nmi': normalized_mutual_info_score(y_true, labels) if n_eff > 1 else 0.0,
        'silhouette': silhouette_score(X, labels) if n_eff > 1 else -1.0,
        'davies_bouldin': davies_bouldin_score(X, labels) if n_eff > 1 else -1.0,
    }

def load_dataset_by_name(name):
    """Returns (X, y, n_clusters). Handles sklearn built-in, OpenML, and synthetic."""
    X, y = None, None
    
    # sklearn built-in
    builtin_loader = getattr(sk_datasets, f'load_{name}', None)
    if builtin_loader:
        X, y = builtin_loader(return_X_y=True)
        return X, y, len(np.unique(y))
    
    # OpenML fetch
    if name in ('glass', 'seeds', 'sonar', 'ecoli', 'yeast', 'vehicle', 'segment',
                'satimage', 'optdigits', 'mfeat-factors', 'pendigits'):
        data = fetch_openml(name=name, version=1, parser='auto', as_frame=False)
        X = data.data
        y = data.target
        if isinstance(y, np.ndarray) and y.dtype.kind == 'O':
            y = LabelEncoder().fit_transform(y.astype(str))
        return X, y, len(np.unique(y))
    
    # Synthetic classification
    if name.startswith('synth_'):
        cfg = {
            'synth_clean':      (1000, 20, 5, 0.0, None),
            'synth_noisy':      (1000, 20, 5, 0.3, None),
            'synth_imbalanced': (1000, 20, 5, 0.1, [0.7] + [0.3/4]*4),
            'synth_highdim':    (500, 100, 5, 0.1, None),
            'synth_overlap':    (1000, 10, 5, 0.5, None),
        }
        if name in cfg:
            n, d, k, flip, w = cfg[name]
            X, y = sk_datasets.make_classification(
                n_samples=n, n_features=d, n_informative=max(d//2, 2),
                n_redundant=max(d//4, 1), n_clusters_per_class=1,
                n_classes=k, flip_y=flip, weights=w, random_state=42,
            )
            return X, y, len(np.unique(y))
    
    raise ValueError(f'Unknown dataset: {name}')

def get_all_datasets():
    """Return list of (name, X, y, n_clusters, desc)."""
    names = [
        # Original 7
        'iris', 'wine', 'breast_cancer', 'digits', 'glass', 'seeds', 'sonar',
        # New UCI
        'ecoli', 'yeast', 'vehicle', 'segment', 'optdigits', 'mfeat-factors',
        # Synthetic
        'synth_clean', 'synth_noisy', 'synth_imbalanced', 'synth_highdim', 'synth_overlap',
    ]
    datasets = []
    for name in names:
        try:
            X, y, k = load_dataset_by_name(name)
            # Subsample large datasets
            if X.shape[0] > 5000:
                X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
            X = StandardScaler().fit_transform(X)
            datasets.append((name, X, y, k, f'{X.shape[0]}×{X.shape[1]}, K={k}'))
            print(f'  [OK] {name}: {X.shape}')
        except Exception as e:
            print(f'  [ERR] {name}: {e}')
    return datasets

def main():
    print('=' * 60)
    print('EXPANDED BENCHMARKS')
    print('=' * 60)
    
    datasets = get_all_datasets()
    print(f'\nLoaded {len(datasets)} datasets')
    
    runners = {
        'ADE-FCM': run_ade_fcm,
        'FCM': run_fcm,
        'KMeans': run_kmeans,
        'Agglomerative': run_agglomerative,
        'DeepADEFCM': run_deep_ade_fcm,
        'FCLM': run_fclm,
    }
    deterministic = {'FCM', 'Agglomerative'}
    
    all_results = {}
    total_experiments = 0
    t_start = time.time()
    
    for dname, X, y, n_clusters, desc in datasets:
        print(f'\n--- {dname} ({desc}) ---')
        ds_results = {}
        
        for algo_name, runner in runners.items():
            seeds = [42] if algo_name in deterministic else SEEDS
            records = []
            
            for seed in seeds:
                try:
                    labels, runtime = runner(X, n_clusters, seed)
                    metrics = compute_metrics(X, y, labels)
                    metrics['runtime'] = runtime
                    metrics['seed'] = seed
                    records.append(metrics)
                except Exception as e:
                    records.append({'seed': seed, 'ari': None, 'nmi': None, 'silhouette': None, 'davies_bouldin': None, 'runtime': 0, 'error': str(e)})
                total_experiments += 1
            
            ari_vals = [r['ari'] for r in records if r['ari'] is not None]
            if ari_vals:
                print(f'  {algo_name:15s}: ARI={np.mean(ari_vals):.3f}±{np.std(ari_vals, ddof=1):.3f}')
                ds_results[algo_name] = records
            else:
                print(f'  {algo_name:15s}: FAILED')
                ds_results[algo_name] = records
        
        all_results[dname] = {
            'description': desc,
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_clusters': int(n_clusters),
            'results': ds_results,
        }
    
    elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'DONE: {total_experiments} experiments in {elapsed:.0f}s')
    
    with open(OUTPUT, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved to {OUTPUT} ({os.path.getsize(OUTPUT)} bytes)')

if __name__ == '__main__':
    main()
