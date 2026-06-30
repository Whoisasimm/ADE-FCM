"""
Main entry point for baseline FCM / FCLM clustering experiments.
Replicates experiments from the paper "Parallel Fuzzy C-Median Clustering
Algorithm Using Spark for Big Data".

Usage:
    python main.py --dataset iris --algorithms fcm fclm --visualize
    python main.py --synthetic --n-samples 5000 --n-features 20 --n-clusters 8
    python main.py --benchmark --algorithms fcm fclm --visualize
    python main.py --weblog --filepath data/weblog.csv --preprocess
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from data_loader import DataLoader
from preprocessing import Preprocessor
from membership_update import MembershipUpdater
from cluster_update import ClusterUpdater
from objective_function import ObjectiveFunction
from convergence import ConvergenceChecker
from evaluation import Evaluator
from utils import (
    setup_logging, set_random_seed, save_results, save_checkpoint,
    format_elapsed, defuzzify, compute_cluster_sizes, make_timestamp,
    compute_optimal_clusters, compute_purity, ensure_dir
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FCM / FCLM Baseline Clustering Experiments'
    )

    # Dataset selection
    parser.add_argument('--dataset', type=str, default='iris',
                        choices=['iris', 'wine', 'digits', 'breast_cancer',
                                 'synthetic', 'weblog'],
                        help='Dataset to use')
    parser.add_argument('--filepath', type=str, default=None,
                        help='Path to weblog data file')

    # Synthetic data params
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--n-features', type=int, default=10)
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.05)

    # Algorithm params
    parser.add_argument('--algorithms', nargs='+', default=['fcm', 'fclm'],
                        choices=['fcm', 'fclm', 'both'],
                        help='Algorithms to run')
    parser.add_argument('--m', type=float, default=2.0,
                        help='Fuzziness exponent')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum iterations')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                        help='Convergence threshold')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for statistics')

    # Preprocessing (weblog)
    parser.add_argument('--preprocess', action='store_true',
                        help='Run weblog preprocessing pipeline')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Session timeout in minutes')
    parser.add_argument('--min-support', type=int, default=1,
                        help='Minimum page support')

    # Spark
    parser.add_argument('--spark', action='store_true',
                        help='Use Spark for parallel execution')
    parser.add_argument('--spark-master', type=str, default='local[*]')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    return parser.parse_args()


def run_fcm(X, n_clusters, m=2.0, max_iter=100, epsilon=1e-5, seed=42,
            verbose=True):
    """Run serial FCM algorithm.

    Returns (centers, U, labels, J_history, n_iter).
    """
    n, d = X.shape
    rng = np.random.RandomState(seed)

    centers = X[rng.choice(n, n_clusters, replace=False)]
    U_old = None
    J_history = []

    for iteration in range(max_iter):
        dists = np.zeros((n, n_clusters))
        for j in range(n_clusters):
            diff = X - centers[j]
            dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)

        inv = 2.0 / (m - 1)
        U = np.zeros((n, n_clusters))
        for i in range(n):
            for j in range(n_clusters):
                denom = np.sum((dists[i, j] / dists[i, :]) ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0

        if U_old is not None:
            change = np.linalg.norm(U - U_old, 'fro')
            if change < epsilon:
                if verbose:
                    logger.info(f"FCM converged at iteration {iteration + 1}")
                U_old = U
                break

        U_old = U

        Um = U ** m
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        centers = (numerator / denominator).T

        J = 0.0
        for j in range(n_clusters):
            diff = X - centers[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            J += np.sum(U[:, j] ** m * dists_j)
        J_history.append(J)

    labels = defuzzify(U)
    return centers, U, labels, J_history, iteration + 1


def run_fclm(X, n_clusters, m=2.0, max_iter=100, epsilon=1e-5, seed=42,
             verbose=True):
    """Run serial FCLM (median-based) algorithm.

    Returns (centers, U, labels, J_history, n_iter).
    """
    n, d = X.shape
    rng = np.random.RandomState(seed)

    centers = X[rng.choice(n, n_clusters, replace=False)]
    U_old = None
    J_history = []

    for iteration in range(max_iter):
        dists = np.zeros((n, n_clusters))
        for j in range(n_clusters):
            diff = X - centers[j]
            dists[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)

        inv = 2.0 / (m - 1)
        U = np.zeros((n, n_clusters))
        for i in range(n):
            for j in range(n_clusters):
                denom = np.sum((dists[i, j] / dists[i, :]) ** inv)
                U[i, j] = 1.0 / denom if denom > 0 else 0.0

        if U_old is not None:
            change = np.linalg.norm(U - U_old, 'fro')
            if change < epsilon:
                if verbose:
                    logger.info(f"FCLM converged at iteration {iteration + 1}")
                U_old = U
                break

        U_old = U

        Um = U ** m
        numerator = X.T @ Um
        denominator = np.maximum(Um.sum(axis=0), 1e-10)
        weighted_centers = (numerator / denominator).T

        for j in range(n_clusters):
            diff = X - weighted_centers[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            weighted_dists = dists_j * U[:, j]
            median_idx = np.argsort(weighted_dists)[n // 2]
            centers[j] = X[median_idx]

        J = 0.0
        for j in range(n_clusters):
            diff = X - centers[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            J += np.sum(U[:, j] ** m * dists_j)
        J_history.append(J)

    labels = defuzzify(U)
    return centers, U, labels, J_history, iteration + 1


def run_spark_fcm(spark, X_df, feature_cols, n_clusters, **kwargs):
    """Run FCM using Spark engine."""
    from spark_engine import SparkFCMEngine
    engine = SparkFCMEngine(
        spark=spark, n_clusters=n_clusters,
        max_iter=kwargs.get('max_iter', 100),
        m=kwargs.get('m', 2.0),
        epsilon=kwargs.get('epsilon', 1e-5)
    )
    result = engine.fit(X_df, feature_cols)
    labels = engine.predict(X_df, feature_cols)
    return result, labels, engine


def run_spark_fclm(spark, X_df, feature_cols, n_clusters, **kwargs):
    """Run FCLM using Spark engine."""
    from spark_engine import SparkFCLMEngine
    engine = SparkFCLMEngine(
        spark=spark, n_clusters=n_clusters,
        max_iter=kwargs.get('max_iter', 100),
        m=kwargs.get('m', 2.0),
        epsilon=kwargs.get('epsilon', 1e-5)
    )
    result = engine.fit(X_df, feature_cols)
    labels = engine.predict(X_df, feature_cols)
    return result, labels, engine


def run_single_experiment(X, y_true, n_clusters, m, max_iter, epsilon, seed,
                          use_spark=False, spark=None, X_df=None,
                          feature_cols=None, verbose=True):
    """Run a single experiment comparing FCM and FCLM."""
    results = {}
    labels_dict = {}
    centers_dict = {}
    U_dict = {}
    J_dict = {}

    if use_spark and spark is not None and X_df is not None:
        logger.info("Running Spark FCM...")
        (_, centers_fcm, U_fcm, J_fcm), labels_fcm, _ = run_spark_fcm(
            spark, X_df, feature_cols, n_clusters, m=m,
            max_iter=max_iter, epsilon=epsilon
        )
        results['fcm_n_iter'] = len(J_fcm)
        labels_dict['fcm'] = labels_fcm
        centers_dict['fcm'] = centers_fcm
        U_dict['fcm'] = U_fcm
        J_dict['fcm'] = J_fcm

        logger.info("Running Spark FCLM...")
        (_, centers_fclm, U_fclm, J_fclm), labels_fclm, _ = run_spark_fclm(
            spark, X_df, feature_cols, n_clusters, m=m,
            max_iter=max_iter, epsilon=epsilon
        )
        results['fclm_n_iter'] = len(J_fclm)
        labels_dict['fclm'] = labels_fclm
        centers_dict['fclm'] = centers_fclm
        U_dict['fclm'] = U_fclm
        J_dict['fclm'] = J_fclm
    else:
        logger.info("Running serial FCM...")
        centers_fcm, U_fcm, labels_fcm, J_fcm, n_iter_fcm = run_fcm(
            X, n_clusters, m=m, max_iter=max_iter, epsilon=epsilon,
            seed=seed, verbose=verbose
        )
        results['fcm_n_iter'] = n_iter_fcm
        labels_dict['fcm'] = labels_fcm
        centers_dict['fcm'] = centers_fcm
        U_dict['fcm'] = U_fcm
        J_dict['fcm'] = J_fcm

        logger.info("Running serial FCLM...")
        centers_fclm, U_fclm, labels_fclm, J_fclm, n_iter_fclm = run_fclm(
            X, n_clusters, m=m, max_iter=max_iter, epsilon=epsilon,
            seed=seed, verbose=verbose
        )
        results['fclm_n_iter'] = n_iter_fclm
        labels_dict['fclm'] = labels_fclm
        centers_dict['fclm'] = centers_fclm
        U_dict['fclm'] = U_fclm
        J_dict['fclm'] = J_fclm

    for algo in ['fcm', 'fclm']:
        metrics = Evaluator.compute_all(X, labels_dict[algo], y_true)
        for k, v in metrics.items():
            results[f'{algo}_{k}'] = v
        sizes = compute_cluster_sizes(labels_dict[algo], n_clusters)
        for i, s in enumerate(sizes):
            results[f'{algo}_cluster_{i}_size'] = int(s)
        if y_true is not None:
            purity = compute_purity(y_true, labels_dict[algo])
            results[f'{algo}_purity'] = purity

    return results, labels_dict, centers_dict, U_dict, J_dict


def benchmark_datasets(args):
    """Run experiments on all sklearn benchmark datasets."""
    loader = DataLoader()
    all_results = {}
    summary_rows = []

    datasets = ['iris', 'wine', 'digits', 'breast_cancer']
    for name in datasets:
        logger.info(f"{'=' * 60}")
        logger.info(f"Benchmark: {name}")
        logger.info(f"{'=' * 60}")

        X, y_true = loader.load_benchmark_dataset(name)
        n_clusters = len(set(y_true))

        X_scaled = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-10)

        run_results = {}
        all_labels = {}
        all_centers = {}
        all_U = {}
        all_J = {}

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            logger.info(f"  Run {run_idx + 1}/{args.runs} (seed={seed})")
            results, labels, centers, U, J = run_single_experiment(
                X_scaled, y_true, n_clusters, args.m, args.max_iter,
                args.epsilon, seed, verbose=False
            )
            for k, v in results.items():
                if k not in run_results:
                    run_results[k] = []
                run_results[k].append(v)
            if run_idx == 0:
                all_labels = labels
                all_centers = centers
                all_U = U
                all_J = J

        avg_results = {}
        for k, vals in run_results.items():
            avg_results[k] = np.mean(vals)
            avg_results[f'{k}_std'] = np.std(vals)

        all_results[name] = {
            'avg': avg_results,
            'labels': all_labels,
            'centers': all_centers,
            'U': all_U,
            'J': all_J,
            'X': X_scaled,
            'y_true': y_true
        }

        row = {'dataset': name, 'n_samples': X.shape[0], 'n_features': X.shape[1],
               'n_clusters': n_clusters}
        for algo in ['fcm', 'fclm']:
            for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz',
                           'adjusted_rand_index', 'normalized_mutual_info',
                           'rand_index', 'purity', 'n_iter']:
                key = f'{algo}_{metric}'
                if key in avg_results:
                    row[key] = f'{avg_results[key]:.4f}'
                    key_std = f'{key}_std'
                    if key_std in avg_results:
                        row[key_std] = f'{avg_results[key_std]:.4f}'
        summary_rows.append(row)

        if args.visualize:
            from visualization import Visualizer
            vis = Visualizer(output_dir=Path(args.output_dir) / 'plots' / name)

            vis.plot_clusters_2d(X_scaled, all_labels['fcm'], centers=all_centers['fcm'],
                                 title=f'{name} - FCM Clusters',
                                 save_name=f'{name}_fcm_clusters.png')
            vis.plot_clusters_2d(X_scaled, all_labels['fclm'], centers=all_centers['fclm'],
                                 title=f'{name} - FCLM Clusters',
                                 save_name=f'{name}_fclm_clusters.png')

            if all_J.get('fcm'):
                vis.plot_convergence(all_J['fcm'], title=f'{name} - FCM Convergence',
                                     save_name=f'{name}_fcm_convergence.png')
            if all_J.get('fclm'):
                vis.plot_convergence(all_J['fclm'], title=f'{name} - FCLM Convergence',
                                     save_name=f'{name}_fclm_convergence.png')

            fcm_metrics = {k: v for k, v in avg_results.items() if k.startswith('fcm_') and 'std' not in k}
            fclm_metrics = {k: v for k, v in avg_results.items() if k.startswith('fclm_') and 'std' not in k}
            vis.plot_metrics_comparison(
                {k.replace('fcm_', ''): v for k, v in fcm_metrics.items()},
                {k.replace('fclm_', ''): v for k, v in fclm_metrics.items()},
                title=f'{name} - FCM vs FCLM',
                save_name=f'{name}_metrics_comparison.png'
            )

            vis.plot_confusion_matrix(y_true, all_labels['fcm'],
                                       title=f'{name} - FCM Confusion Matrix',
                                       save_name=f'{name}_fcm_confusion.png')
            vis.plot_confusion_matrix(y_true, all_labels['fclm'],
                                       title=f'{name} - FCLM Confusion Matrix',
                                       save_name=f'{name}_fclm_confusion.png')

            vis.plot_combined_results(
                X_scaled, all_labels['fcm'], all_labels['fclm'],
                all_centers['fcm'], all_centers['fclm'],
                all_J.get('fcm', []), all_J.get('fclm', []),
                fcm_metrics, fclm_metrics, y_true,
                title_prefix=f'{name} ', show=False
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.output_dir) / 'benchmark_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Benchmark summary saved to {summary_path}")

    return all_results, summary_df


def run_synthetic_experiment(args):
    """Run experiment on synthetic data."""
    loader = DataLoader()
    logger.info("Generating synthetic data...")

    X, y_true = loader.load_synthetic_data(
        n_samples=args.n_samples, n_features=args.n_features,
        n_clusters=args.n_clusters, random_state=args.seed, noise=args.noise
    )

    X_scaled = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-10)

    logger.info(f"Running algorithms on synthetic data ({X.shape})...")
    results, labels, centers, U, J = run_single_experiment(
        X_scaled, y_true, args.n_clusters, args.m, args.max_iter,
        args.epsilon, args.seed
    )

    logger.info("\n" + "=" * 50)
    logger.info("SYNTHETIC DATA RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.visualize:
        from visualization import Visualizer
        vis = Visualizer(output_dir=Path(args.output_dir) / 'plots' / 'synthetic')

        vis.plot_clusters_2d(X_scaled, labels['fcm'], centers=centers['fcm'],
                             title='Synthetic - FCM Clusters',
                             save_name='synthetic_fcm_clusters.png')
        vis.plot_clusters_2d(X_scaled, labels['fclm'], centers=centers['fclm'],
                             title='Synthetic - FCLM Clusters',
                             save_name='synthetic_fclm_clusters.png')

        vis.plot_clusters_2d(X_scaled, y_true,
                             title='Synthetic - Ground Truth',
                             save_name='synthetic_ground_truth.png')

        if J.get('fcm'):
            vis.plot_convergence(J['fcm'], title='Synthetic - FCM Convergence',
                                 save_name='synthetic_fcm_convergence.png')
        if J.get('fclm'):
            vis.plot_convergence(J['fclm'], title='Synthetic - FCLM Convergence',
                                 save_name='synthetic_fclm_convergence.png')

        vis.plot_cluster_sizes(labels['fcm'], title='Synthetic - FCM Cluster Sizes',
                                save_name='synthetic_fcm_sizes.png')
        vis.plot_cluster_sizes(labels['fclm'], title='Synthetic - FCLM Cluster Sizes',
                                save_name='synthetic_fclm_sizes.png')

        vis.plot_confusion_matrix(y_true, labels['fcm'],
                                   title='Synthetic - FCM Confusion Matrix',
                                   save_name='synthetic_fcm_confusion.png')
        vis.plot_confusion_matrix(y_true, labels['fclm'],
                                   title='Synthetic - FCLM Confusion Matrix',
                                   save_name='synthetic_fclm_confusion.png')

        fcm_metrics = {k.replace('fcm_', ''): v for k, v in results.items() if k.startswith('fcm_')}
        fclm_metrics = {k.replace('fclm_', ''): v for k, v in results.items() if k.startswith('fclm_')}
        vis.plot_combined_results(
            X_scaled, labels['fcm'], labels['fclm'],
            centers['fcm'], centers['fclm'],
            J.get('fcm', []), J.get('fclm', []),
            fcm_metrics, fclm_metrics, y_true,
            title_prefix='Synthetic ', show=False
        )

    save_results(results, Path(args.output_dir) / 'synthetic_results.json')
    return results


def run_weblog_pipeline(args):
    """Run weblog preprocessing and clustering pipeline."""
    if not args.filepath:
        logger.error("No weblog file specified. Use --filepath")
        return None

    logger.info("=" * 60)
    logger.info("WEBLOG PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    loader = DataLoader()
    df = loader.load_weblog_data(args.filepath)

    preprocessor = Preprocessor()
    df = preprocessor.clean_weblog_data(df)
    df = preprocessor.identify_users(df)
    df = preprocessor.identify_sessions(df, timeout_minutes=args.timeout)
    df = preprocessor.reduce_dimensions(df, min_support=args.min_support)
    df = preprocessor.assign_session_weights(df)
    matrix, feature_names = preprocessor.build_session_matrix(df)
    matrix_norm = preprocessor.normalize(matrix)

    logger.info(f"Session matrix shape: {matrix_norm.shape}")
    logger.info(f"Number of features (pages): {len(feature_names)}")

    X = matrix_norm.toarray() if hasattr(matrix_norm, 'toarray') else np.asarray(matrix_norm)

    if X.shape[1] > 100:
        logger.info(f"Large feature space ({X.shape[1]}), applying PCA...")
        from sklearn.decomposition import PCA
        n_components = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=args.seed)
        X = pca.fit_transform(X)
        logger.info(f"Reduced to {X.shape[1]} dimensions ({pca.explained_variance_ratio_.sum():.3f} variance)")
        feature_names = [f'PC{i}' for i in range(X.shape[1])]

    optimal_k, inertias, silhouettes = compute_optimal_clusters(
        X, max_clusters=min(15, X.shape[0] - 1), method='silhouette',
        random_state=args.seed
    )
    logger.info(f"Optimal clusters: {optimal_k}")

    n_clusters = min(optimal_k, args.n_clusters)

    logger.info(f"Clustering with {n_clusters} clusters...")
    results, labels, centers, U, J = run_single_experiment(
        X, None, n_clusters, args.m, args.max_iter,
        args.epsilon, args.seed
    )

    logger.info("\n" + "=" * 50)
    logger.info("WEBLOG CLUSTERING RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    session_profile = pd.DataFrame({
        'session_id': range(len(labels['fcm'])),
        'fcm_cluster': labels['fcm'],
        'fclm_cluster': labels['fclm'],
    })
    session_profile.to_csv(Path(args.output_dir) / 'session_clusters.csv', index=False)
    logger.info(f"Session cluster assignments saved to {Path(args.output_dir) / 'session_clusters.csv'}")

    if args.visualize:
        from visualization import Visualizer
        vis = Visualizer(output_dir=Path(args.output_dir) / 'plots' / 'weblog')

        vis.plot_clusters_2d(X, labels['fcm'], centers=centers['fcm'],
                             title='Weblog - FCM Clusters',
                             save_name='weblog_fcm_clusters.png')
        vis.plot_clusters_2d(X, labels['fclm'], centers=centers['fclm'],
                             title='Weblog - FCLM Clusters',
                             save_name='weblog_fclm_clusters.png')

        if J.get('fcm'):
            vis.plot_convergence(J['fcm'], title='Weblog - FCM Convergence',
                                 save_name='weblog_fcm_convergence.png')
        if J.get('fclm'):
            vis.plot_convergence(J['fclm'], title='Weblog - FCLM Convergence',
                                 save_name='weblog_fclm_convergence.png')

        vis.plot_elbow_curve(inertias, min(15, X.shape[0] - 1),
                              title='Weblog - Elbow Curve',
                              save_name='weblog_elbow.png')

        fcm_metrics = {k.replace('fcm_', ''): v for k, v in results.items() if k.startswith('fcm_')}
        fclm_metrics = {k.replace('fclm_', ''): v for k, v in results.items() if k.startswith('fclm_')}
        vis.plot_combined_results(
            X, labels['fcm'], labels['fclm'],
            centers['fcm'], centers['fclm'],
            J.get('fcm', []), J.get('fclm', []),
            fcm_metrics, fclm_metrics, None,
            title_prefix='Weblog ', show=False
        )

    save_results(results, Path(args.output_dir) / 'weblog_results.json')
    return results


def run_single_dataset(args):
    """Run experiment on a single benchmark dataset."""
    loader = DataLoader()
    X, y_true = loader.load_benchmark_dataset(args.dataset)
    n_clusters = len(set(y_true))

    X_scaled = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-10)

    logger.info(f"Dataset: {args.dataset}, shape={X.shape}, clusters={n_clusters}")
    results, labels, centers, U, J = run_single_experiment(
        X_scaled, y_true, n_clusters, args.m, args.max_iter,
        args.epsilon, args.seed
    )

    logger.info("\n" + "=" * 50)
    logger.info(f"{args.dataset.upper()} RESULTS")
    logger.info("=" * 50)
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.visualize:
        from visualization import Visualizer
        vis = Visualizer(output_dir=Path(args.output_dir) / 'plots' / args.dataset)

        vis.plot_clusters_2d(X_scaled, labels['fcm'], centers=centers['fcm'],
                             title=f'{args.dataset} - FCM Clusters',
                             save_name=f'{args.dataset}_fcm_clusters.png')
        vis.plot_clusters_2d(X_scaled, labels['fclm'], centers=centers['fclm'],
                             title=f'{args.dataset} - FCLM Clusters',
                             save_name=f'{args.dataset}_fclm_clusters.png')

        if U.get('fcm') is not None:
            vis.plot_membership_heatmap(U['fcm'], title=f'{args.dataset} - FCM Membership',
                                        save_name=f'{args.dataset}_fcm_membership.png')

        if J.get('fcm'):
            vis.plot_convergence(J['fcm'], title=f'{args.dataset} - FCM Convergence',
                                 save_name=f'{args.dataset}_fcm_convergence.png')
        if J.get('fclm'):
            vis.plot_convergence(J['fclm'], title=f'{args.dataset} - FCLM Convergence',
                                 save_name=f'{args.dataset}_fclm_convergence.png')

        fcm_metrics = {k.replace('fcm_', ''): v for k, v in results.items() if k.startswith('fcm_')}
        fclm_metrics = {k.replace('fclm_', ''): v for k, v in results.items() if k.startswith('fclm_')}
        vis.plot_metrics_comparison(fcm_metrics, fclm_metrics,
                                    title=f'{args.dataset} - FCM vs FCLM',
                                    save_name=f'{args.dataset}_metrics_comparison.png')

        vis.plot_cluster_sizes(labels['fcm'], title=f'{args.dataset} - FCM Sizes',
                                save_name=f'{args.dataset}_fcm_sizes.png')
        vis.plot_cluster_sizes(labels['fclm'], title=f'{args.dataset} - FCLM Sizes',
                                save_name=f'{args.dataset}_fclm_sizes.png')

        vis.plot_combined_results(
            X_scaled, labels['fcm'], labels['fclm'],
            centers['fcm'], centers['fclm'],
            J.get('fcm', []), J.get('fclm', []),
            fcm_metrics, fclm_metrics, y_true,
            title_prefix=f'{args.dataset} ', show=False
        )

        try:
            vis.plot_silhouette_analysis(X_scaled, labels['fcm'],
                                          title=f'{args.dataset} - FCM Silhouette',
                                          save_name=f'{args.dataset}_fcm_silhouette.png')
            vis.plot_silhouette_analysis(X_scaled, labels['fclm'],
                                          title=f'{args.dataset} - FCLM Silhouette',
                                          save_name=f'{args.dataset}_fclm_silhouette.png')
        except Exception as e:
            logger.warning(f"Silhouette plot failed: {e}")

    save_results(results, Path(args.output_dir) / f'{args.dataset}_results.json')
    return results


def main():
    """Main entry point."""
    args = parse_args()

    output_dir = ensure_dir(args.output_dir)
    vis_dir = ensure_dir(Path(args.output_dir) / 'plots')

    setup_logging(log_file=args.log_file, level=args.log_level)
    set_random_seed(args.seed)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Fuzziness m={args.m}, max_iter={args.max_iter}, epsilon={args.epsilon}")

    if args.synthetic or args.dataset == 'synthetic':
        run_synthetic_experiment(args)
    elif args.dataset == 'weblog':
        run_weblog_pipeline(args)
    elif args.dataset in ['iris', 'wine', 'digits', 'breast_cancer']:
        run_single_dataset(args)
    else:
        benchmark_datasets(args)

    logger.info("All experiments complete.")


if __name__ == '__main__':
    main()
