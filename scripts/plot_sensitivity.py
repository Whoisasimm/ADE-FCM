"""Generate sensitivity analysis figures."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base, 'results', 'sensitivity_analysis.json')) as f:
    data = json.load(f)

params = ['m_max', 'm_min', 'beta', 'alpha', 'center_reinit_threshold', 'outlier_contamination']
param_labels = [r'$m_{\max}$', r'$m_{\min}$', r'$\beta$', r'$\alpha$', 'Reinit Threshold', 'Outlier Contam.']
datasets = ['iris', 'wine', 'digits']
colors = ['#2196F3', '#FF5722', '#4CAF50']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (p, plabel) in enumerate(zip(params, param_labels)):
    ax = axes[idx]
    for di, ds in enumerate(datasets):
        vals = sorted(set(r['value'] for r in data if r['dataset'] == ds and r['parameter'] == p))
        means, stds = [], []
        for v in vals:
            ari_vals = [r['ari'] for r in data if r['dataset'] == ds and r['parameter'] == p and r['value'] == v and r['ari'] is not None]
            if ari_vals:
                means.append(np.mean(ari_vals))
                stds.append(np.std(ari_vals, ddof=1))
        if means:
            ax.errorbar(vals[:len(means)], means, yerr=stds, fmt='o-', color=colors[di], label=ds.capitalize(), capsize=3)
    ax.set_xlabel(plabel, fontsize=11)
    ax.set_ylabel('ARI', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if p == 'beta':
        ax.set_xticks([2, 5, 10])
    elif p == 'outlier_contamination':
        ax.set_xticks([0.0, 0.1, 0.2])

plt.suptitle('ADE-FCM Hyperparameter Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
outpath = os.path.join(base, 'docs', 'images', 'sensitivity_analysis.png')
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {outpath}')
print(f'Size: {os.path.getsize(outpath)} bytes')
