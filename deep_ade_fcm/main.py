"""
DeepADEFCM: Full pipeline for benchmarking against standard ADE-FCM.
Generates synthetic data, trains models, evaluates, and visualizes results.
"""
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_ade_fcm.deep_ade_fcm import DeepADEFCM
from deep_ade_fcm.visualization import DeepADEFCMVisualizer
from novel_algorithm.ade_fcm import ADEFCM


def generate_synthetic_datasets(random_state=42):
    """Generate blob, moon, circle, and high-dimensional datasets."""
    rng = np.random.RandomState(random_state)
    datasets = {}

    X, y = make_blobs(n_samples=500, n_features=10, centers=4, random_state=random_state, cluster_std=1.2)
    datasets['blobs'] = (StandardScaler().fit_transform(X), y)

    X, y = make_moons(n_samples=400, noise=0.08, random_state=random_state)
    datasets['moons'] = (StandardScaler().fit_transform(X), y)

    X, y = make_circles(n_samples=400, noise=0.06, factor=0.5, random_state=random_state)
    datasets['circles'] = (StandardScaler().fit_transform(X), y)

    n_high = 600
    X_high = rng.randn(n_high, 50)
    true_centers = rng.randn(5, 50) * 3
    labels_high = np.zeros(n_high, dtype=int)
    for k in range(5):
        mask = (k * n_high // 5 <= np.arange(n_high)) & (np.arange(n_high) < (k + 1) * n_high // 5)
        labels_high[mask] = k
        X_high[mask] += true_centers[k]
    datasets['high_dim'] = (StandardScaler().fit_transform(X_high), labels_high)

    return datasets


def evaluate_model(X, y_true, model, model_name):
    """Compute evaluation metrics for a fitted model."""
    Z = model.get_latent_representation() if hasattr(model, 'get_latent_representation') else model.transform(X)
    labels = model.labels_
    n_clusters = len(set(labels))

    sil = silhouette_score(Z, labels) if n_clusters > 1 else -1.0
    ari = adjusted_rand_score(y_true, labels) if y_true is not None else None
    recon_err = model._compute_reconstruction_error(X) if hasattr(model, '_compute_reconstruction_error') else None

    metrics = {
        'model': model_name,
        'n_clusters_found': n_clusters,
        'silhouette': round(float(sil), 4),
        'ari': round(float(ari), 4) if ari is not None else None,
        'reconstruction_error': round(float(recon_err), 6) if recon_err is not None else None,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'latent_dim': model.latent_dim if hasattr(model, 'latent_dim') else 'N/A',
    }
    logger.info(f"{model_name}: Sil={metrics['silhouette']}, ARI={metrics['ari']}, "
                f"ReconErr={metrics['reconstruction_error']}")
    return metrics


def run_pipeline(datasets, save_dir, n_runs=2):
    """Run full pipeline across all datasets."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis = DeepADEFCMVisualizer(save_dir=save_dir)
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        logger.info(f"{'=' * 60}")
        logger.info(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
        logger.info(f"{'=' * 60}")

        dataset_results = []

        # Standard ADE-FCM baseline
        for run in range(n_runs):
            seed = 42 + run
            ade = ADEFCM(n_clusters='auto', max_iter=100, init_method='kmeans++',
                         random_state=seed, verbose=False)
            ade.fit(X)
            labels = ade.labels_
            Z = X
            sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else -1.0
            ari = adjusted_rand_score(y, labels)
            dataset_results.append({
                'run': run, 'model': 'ADE-FCM',
                'silhouette': round(float(sil), 4), 'ari': round(float(ari), 4),
                'n_clusters': len(set(labels))
            })
            logger.info(f"  ADE-FCM run {run + 1}: Sil={sil:.4f}, ARI={ari:.4f}, K={len(set(labels))}")

        # DeepADEFCM (multiple configs)
        latent_dims = [2, 5] if X.shape[1] <= 10 else [5, 10]
        for latent_dim in latent_dims:
            for run in range(n_runs):
                seed = 42 + run
                try:
                    model = DeepADEFCM(
                        input_dim=X.shape[1],
                        latent_dim=latent_dim,
                        n_clusters='auto',
                        lambda_cluster=0.3,
                        ae_epochs=30,
                        joint_epochs=30,
                        batch_size=64,
                        random_state=seed,
                        verbose=False
                    )
                    model.fit(X)
                    metrics = evaluate_model(X, y, model, f"DeepADEFCM (latent={latent_dim})")
                    metrics['run'] = run
                    metrics['latent_dim'] = latent_dim
                    dataset_results.append(metrics)

                    # Generate plots for first run
                    if run == 0 and len(dataset_results) <= 4:
                        Z = model.get_latent_representation()
                        vis.plot_latent_space(Z, model.labels_, model.fcm_model.centers_,
                                              epoch=0)
                        vis.plot_training_history(model.loss_history_)

                        if hasattr(model, 'get_explanation'):
                            explanations = []
                            for cid in range(model.n_clusters):
                                try:
                                    explanations.append(model.get_explanation(cid))
                                except Exception:
                                    pass
                            if explanations:
                                vis.plot_feature_importance(explanations)

                except Exception as e:
                    logger.error(f"DeepADEFCM (latent={latent_dim}) run {run + 1} failed: {e}")
                    dataset_results.append({
                        'run': run, 'model': f'DeepADEFCM (latent={latent_dim})',
                        'silhouette': None, 'ari': None, 'reconstruction_error': None,
                        'error': str(e)
                    })

        all_results[dataset_name] = dataset_results

        # Save per-dataset results
        result_path = save_dir / f"results_{dataset_name}.json"
        with open(result_path, 'w') as f:
            json.dump(dataset_results, f, indent=2)
        logger.info(f"Saved results to {result_path}")

    return all_results


def print_summary(all_results):
    """Print a summary table of all results."""
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY: Best performing configuration per dataset")
    logger.info(f"{'=' * 80}")
    header = f"{'Dataset':<15} {'Model':<30} {'Silhouette':<12} {'ARI':<12} {'K':<6}"
    logger.info(header)
    logger.info('-' * 80)

    for dataset_name, results in all_results.items():
        best_entry = max((r for r in results if r.get('silhouette') is not None),
                         key=lambda r: r['silhouette'], default=None)
        if best_entry:
            logger.info(f"{dataset_name:<15} {best_entry['model']:<30} "
                        f"{best_entry['silhouette']:<12} {best_entry.get('ari', 'N/A'):<12} "
                        f"{best_entry.get('n_clusters', best_entry.get('n_clusters_found', '?')):<6}")
        # Show all entries
        for r in results:
            if r.get('model') and r.get('silhouette') is not None:
                logger.info(f"{'':15} {r['model']:<30} {r['silhouette']:<12} "
                            f"{r.get('ari', 'N/A'):<12} {r.get('n_clusters', r.get('n_clusters_found', '?')):<6}")


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

    save_dir = Path(__file__).resolve().parent.parent / "results" / "deep_ade_fcm"
    logger.info(f"Results directory: {save_dir}")

    datasets = generate_synthetic_datasets(random_state=42)
    all_results = run_pipeline(datasets, save_dir, n_runs=2)

    # Global results
    global_path = save_dir / "all_results.json"
    with open(global_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {global_path}")

    print_summary(all_results)

    # Aggregate metrics
    sil_values = [r['silhouette'] for results in all_results.values()
                  for r in results if r.get('silhouette') is not None]
    logger.info(f"\nOverall: Mean Silhouette={np.mean(sil_values):.4f}, "
                f"Std={np.std(sil_values):.4f} across {len(sil_values)} evaluations")
    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()
