"""
ADE-FCM: Adaptive Distributed Explainable Fuzzy C-Means
Top-level entry point.
"""
import argparse
import sys
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="ADE-FCM: Clustering for Big Data")
    parser.add_argument("--mode", choices=["train", "eval", "benchmark", "stream", "xai", "ablation", "gpu", "demo"],
                       default="demo", help="Execution mode")
    parser.add_argument("--algorithm", default="ade-fcm", help="Algorithm to use")
    parser.add_argument("--data", default=None, help="Data path")
    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations")
    parser.add_argument("--spark", action="store_true", help="Use Spark")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--auto-k", action="store_true", help="Auto-discover K")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "stream":
        run_stream(args)
    elif args.mode == "xai":
        run_xai(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "gpu":
        run_gpu(args)


def run_demo(args):
    """Run a quick demo on synthetic data."""
    import numpy as np
    from sklearn.datasets import make_blobs
    from novel_algorithm.ade_fcm import ADEFCM

    logger.info("Running ADE-FCM demo on synthetic data...")
    X, y = make_blobs(n_samples=500, n_features=5, centers=4, random_state=42)

    model = ADEFCM(n_clusters=4, init_method='kmeans++', m='adaptive',
                   epsilon='dynamic', random_state=42, verbose=True)
    model.fit(X)

    from sklearn.metrics import silhouette_score, adjusted_rand_score
    sil = silhouette_score(X, model.labels_)
    ari = adjusted_rand_score(y, model.labels_)

    logger.info(f"Results: Silhouette={sil:.4f}, ARI={ari:.4f}, Iterations={model.n_iter_}")
    logger.info(f"Outliers detected: {model.outlier_mask_.sum() if model.outlier_mask_ is not None else 0}")

    from xai.cluster_explainer import ClusterExplainer
    explainer = ClusterExplainer(model, X, feature_names=[f"f{i}" for i in range(X.shape[1])])
    for i in range(4):
        desc = explainer.natural_language_description(i)
        logger.info(f"  {desc}")


# Other functions are stubs that import from appropriate modules
def run_train(args):
    pass


def run_eval(args):
    pass


def run_benchmark(args):
    from benchmarks.main import run_benchmarks
    run_benchmarks()


def run_stream(args):
    from streaming.main import main as stream_main
    stream_main()


def run_xai(args):
    from xai.main import main as xai_main
    xai_main()


def run_ablation(args):
    from ablation.main import main as ablation_main
    ablation_main()


def run_gpu(args):
    from gpu.main import main as gpu_main
    gpu_main()


if __name__ == "__main__":
    main()
