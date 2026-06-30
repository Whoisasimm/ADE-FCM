"""Analyze hyperparameter sensitivity results and generate paper section."""
import json, os
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base, 'results', 'sensitivity_analysis.json')) as f:
    data = json.load(f)

params = ['m_max', 'm_min', 'beta', 'alpha', 'center_reinit_threshold', 'outlier_contamination']
datasets = ['iris', 'wine', 'digits']

# Group by (dataset, parameter, value)
groups = {}
for r in data:
    if r['ari'] is None:
        continue
    key = (r['dataset'], r['parameter'], r['value'])
    groups.setdefault(key, []).append(r['ari'])

print('=== SENSITIVITY ANALYSIS RESULTS ===')
print()

for ds in datasets:
    print(f'Dataset: {ds}')
    for p in params:
        vals = sorted(set(r['value'] for r in data if r['dataset'] == ds and r['parameter'] == p))
        means = []
        for v in vals:
            key = (ds, p, v)
            if key in groups:
                arr = groups[key]
                means.append(np.mean(arr))
        if len(means) >= 2:
            # Sensitivity = max - min
            sensitivity = max(means) - min(means)
            baseline = means[1]  # middle value is default
            rel_sens = sensitivity / abs(baseline) if abs(baseline) > 1e-10 else 0
            print(f'  {p}: vals={vals}, means={[f"{m:.3f}" for m in means]}, '
                  f'range={sensitivity:.4f}, rel={rel_sens:.2%}')
        else:
            print(f'  {p}: insufficient data')
    print()

# Sensitivity ranking across all datasets
print('=== OVERALL SENSITIVITY RANKING ===')
param_sensitivity = {}
for p in params:
    all_ranges = []
    for ds in datasets:
        vals = sorted(set(r['value'] for r in data if r['dataset'] == ds and r['parameter'] == p))
        means = []
        for v in vals:
            key = (ds, p, v)
            if key in groups:
                means.append(np.mean(groups[key]))
        if len(means) >= 2:
            all_ranges.append(max(means) - min(means))
    if all_ranges:
        param_sensitivity[p] = np.mean(all_ranges)

ranked = sorted(param_sensitivity.items(), key=lambda x: x[1], reverse=True)
for i, (p, s) in enumerate(ranked):
    print(f'  {i+1}. {p}: mean ARI range = {s:.4f}')

# === Generate summary table ===
print()
print('=== HYPERPARAMETER SENSITIVITY TABLE ===')
print(f'{"Parameter":<30} {"Default":<10} {"Range Tested":<20} {"Max ARI Range":<15} {"Sensitivity":<12}')
print('-' * 90)
for p, s in ranked:
    default_vals = {
        'm_max': 2.5, 'm_min': 1.1, 'beta': 5.0, 'alpha': 3.0,
        'center_reinit_threshold': 1.0, 'outlier_contamination': 0.05
    }
    range_strs = {
        'm_max': '[2.0, 3.5]', 'm_min': '[1.01, 1.5]', 'beta': '[2.0, 10.0]',
        'alpha': '[1.0, 5.0]', 'center_reinit_threshold': '[0.5, 5.0]',
        'outlier_contamination': '[0.0, 0.2]'
    }
    sensitivity_label = 'High' if s > 0.05 else 'Moderate' if s > 0.02 else 'Low'
    print(f'{p:<30} {str(default_vals.get(p, "N/A")):<10} {range_strs.get(p, ""):<20} {s:<15.4f} {sensitivity_label:<12}')

# ARI vs default for each parameter
print()
print('=== PARAMETER EFFECT ON ARI (mean across datasets) ===')
for p in params:
    print(f'\n  {p}:')
    all_vals = sorted(set(r['value'] for r in data if r['parameter'] == p))
    for v in all_vals:
        ari_vals = [r['ari'] for r in data if r['parameter'] == p and r['value'] == v and r['ari'] is not None]
        if ari_vals:
            print(f'    = {v}: ARI = {np.mean(ari_vals):.3f} +/- {np.std(ari_vals, ddof=1):.3f}')
