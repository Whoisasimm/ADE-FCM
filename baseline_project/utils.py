"""
Utility functions for the FCM/FCLM baseline project.
"""
import os
import sys
import json
import yaml
import pickle
import random
import hashlib
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def setup_logging(log_file=None, level="INFO"):
    """Configure loguru logging with optional file output."""
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_path), format=fmt, level=level, rotation="10 MB")
    logger.info(f"Logging initialized at level {level}")
    return logger


def set_random_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and python hash."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"Random seed set to {seed}")


def save_checkpoint(state, filepath):
    """Save algorithm state (centers, U, J_history) to pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath):
    """Load algorithm state from pickle."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"Checkpoint not found: {filepath}")
        return None
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    logger.info(f"Checkpoint loaded from {filepath}")
    return state


def save_results(results, filepath):
    """Save evaluation results to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            serializable[k] = [float(x) if isinstance(x, (np.floating,)) else x for x in v]
        else:
            serializable[k] = v
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {filepath}")


def load_results(filepath):
    """Load evaluation results from JSON."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"Results file not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        results = json.load(f)
    logger.info(f"Results loaded from {filepath}")
    return results


def save_config(config, filepath):
    """Save configuration to YAML."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {filepath}")


def load_config(filepath):
    """Load configuration from YAML."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"Config file not found: {filepath}")
        return {}
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {filepath}")
    return config


def save_dataframe(df, filepath, fmt='parquet'):
    """Save DataFrame to disk in specified format."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == 'parquet':
        df.to_parquet(filepath.with_suffix('.parquet'))
    elif fmt == 'csv':
        df.to_csv(filepath.with_suffix('.csv'), index=False)
    elif fmt == 'feather':
        df.to_feather(filepath.with_suffix('.feather'))
    else:
        df.to_pickle(filepath.with_suffix('.pkl'))
    logger.info(f"DataFrame saved ({len(df)} rows) to {filepath}")


def load_dataframe(filepath):
    """Load DataFrame from disk, auto-detect format."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext == '.parquet':
        df = pd.read_parquet(filepath)
    elif ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext == '.feather':
        df = pd.read_feather(filepath)
    elif ext == '.pkl':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    logger.info(f"DataFrame loaded ({len(df)} rows) from {filepath}")
    return df


def format_elapsed(seconds):
    """Format elapsed time as human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs:.1f}s"


def compute_cluster_sizes(labels, n_clusters=None):
    """Compute size of each cluster from labels."""
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = len(set(labels))
    sizes = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        sizes[i] = int(np.sum(labels == i))
    return sizes


def defuzzify(U):
    """Convert fuzzy membership matrix to hard labels (argmax)."""
    return np.argmax(U, axis=1)


def compute_confusion_matrix(y_true, y_pred, n_classes=None):
    """Compute confusion matrix between true and predicted labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_purity(y_true, y_pred):
    """Compute cluster purity (max accuracy per cluster)."""
    cm = compute_confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def cluster_agreement(labels1, labels2):
    """Fraction of point pairs that agree in both labelings (same cluster)."""
    n = len(labels1)
    count = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            agree = (labels1[i] == labels1[j]) == (labels2[i] == labels2[j])
            count += agree
            total += 1
    return count / total if total > 0 else 1.0


def make_timestamp():
    """Return current timestamp string for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    """Ensure directory exists, create if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def dict_hash(d):
    """Create a deterministic hash of a dictionary for caching."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:12]


def compute_optimal_clusters(X, max_clusters=10, method='elbow', random_state=42):
    """Estimate optimal number of clusters using elbow or silhouette method."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(X)
    inertias = []
    silhouettes = []

    for k in range(2, max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if len(set(labels)) > 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(0.0)

    if method == 'elbow':
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        optimal = np.argmax(diffs2) + 2 if len(diffs2) > 0 else 2
    elif method == 'silhouette':
        optimal = np.argmax(silhouettes) + 2
    else:
        optimal = 2

    return optimal, inertias, silhouettes
