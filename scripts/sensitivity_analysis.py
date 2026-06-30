"""Hyperparameter sensitivity analysis for ADE-FCM.
Sweeps 6 parameters x 3 values x 3 datasets x 5 seeds = 270 runs.
"""
import sys, os, copy, json, time
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from novel_algorithm.ade_fcm import ADEFCM

def load_dataset(name):
    loaders = {
        'iris': lambda: datasets.load_iris(return_X_y=True),
        'wine': lambda: datasets.load_wine(return_X_y=True),
        'digits': lambda: datasets.load_digits(return_X_y=True),
    }
    X, y = loaders[name]()
    X = StandardScaler().fit_transform(X)
    return X, y

datasets_list = ['iris', 'wine', 'digits']

default_params = {
    'n_clusters': 3, 'max_iter': 100, 'm': 'adaptive', 'epsilon': 'dynamic',
    'init_method': 'kmeans++', 'metric': 'euclidean', 'compute_xai': False,
    'outlier_contamination': 0.05, 'center_reinit_threshold': 1.0,
    'm_min': 1.1, 'm_max': 2.5, 'alpha': 3.0, 'eps_0': 1e-3, 'beta': 5.0,
}

sweeps = {
    'm_max': [2.0, 3.0, 3.5],
    'm_min': [1.01, 1.3, 1.5],
    'beta': [2.0, 5.0, 10.0],
    'alpha': [1.0, 3.0, 5.0],
    'center_reinit_threshold': [0.5, 2.0, 5.0],
    'outlier_contamination': [0.0, 0.1, 0.2],
}

n_seeds = 5
seeds = [42, 43, 44, 45, 46]
results = []
total = len(datasets_list) * len(sweeps) * 3 * n_seeds
done = 0

for ds_name in datasets_list:
    X, y_true = load_dataset(ds_name)
    actual_k = len(np.unique(y_true))
    for param_name, param_values in sweeps.items():
        for val in param_values:
            for seed in seeds:
                params = copy.deepcopy(default_params)
                params[param_name] = val
                params['random_state'] = seed
                params['n_clusters'] = actual_k
                params['metric'] = 'cosine' if ds_name == 'digits' else 'euclidean'
                model = ADEFCM(**params)
                t0 = time.time()
                try:
                    labels = model.fit_predict(X)
                    runtime = time.time() - t0
                    ari = adjusted_rand_score(y_true, labels)
                    nmi_val = normalized_mutual_info_score(y_true, labels)
                    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1.0
                    results.append({
                        'dataset': ds_name, 'parameter': param_name,
                        'value': val, 'seed': seed,
                        'ari': float(ari), 'nmi': float(nmi_val),
                        'silhouette': float(sil), 'runtime': float(runtime),
                    })
                except Exception as e:
                    results.append({
                        'dataset': ds_name, 'parameter': param_name,
                        'value': val, 'seed': seed,
                        'ari': None, 'nmi': None,
                        'silhouette': None, 'runtime': None, 'error': str(e),
                    })
                done += 1
                if done % 30 == 0:
                    print(f'Progress: {done}/{total}')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outpath = os.path.join(PROJECT_ROOT, 'results', 'sensitivity_analysis.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nDone. {len(results)} experiments saved to {outpath}')
