"""
Distance metrics for clustering algorithms.
All functions support 1D and 2D numpy arrays.
"""
import numpy as np


def euclidean_distance(x, y):
    """Euclidean (L2) distance between vectors x and y.

    For 1D arrays: returns a scalar.
    For 2D arrays: returns pairwise distances (n1 x n2) if both are 2D,
                   or an array of distances if one is 1D.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        return np.sqrt(np.sum((x - y) ** 2))

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Pairwise: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x*y^T
    x2 = np.sum(x ** 2, axis=1, keepdims=True)
    y2 = np.sum(y ** 2, axis=1, keepdims=True).T
    dists = np.sqrt(np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0))
    return dists.squeeze() if dists.shape[0] == 1 and len(dists.shape) > 1 else dists


def manhattan_distance(x, y):
    """Manhattan (L1) distance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        return np.sum(np.abs(x - y))

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n1, n2 = x.shape[0], y.shape[0]
    dists = np.zeros((n1, n2))
    for i in range(n1):
        dists[i] = np.sum(np.abs(x[i] - y), axis=1)

    return dists.squeeze()


def chebyshev_distance(x, y):
    """Chebyshev (L_inf) distance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        return np.max(np.abs(x - y))

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n1, n2 = x.shape[0], y.shape[0]
    dists = np.zeros((n1, n2))
    for i in range(n1):
        dists[i] = np.max(np.abs(x[i] - y), axis=1)

    return dists.squeeze()


def cosine_similarity(x, y):
    """Cosine similarity (1 - cos(theta)). Returns 0 for identical, 2 for opposite."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx == 0 or ny == 0:
            return 1.0
        return 1.0 - np.dot(x, y) / (nx * ny)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    xn = np.linalg.norm(x, axis=1, keepdims=True)
    yn = np.linalg.norm(y, axis=1, keepdims=True).T
    xn = np.maximum(xn, 1e-10)
    yn = np.maximum(yn, 1e-10)
    similarity = (x @ y.T) / (xn * yn)
    similarity = np.clip(similarity, -1.0, 1.0)
    return (1.0 - similarity).squeeze()


def squared_euclidean(x, y):
    """Squared Euclidean distance."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 and y.ndim == 1:
        return np.sum((x - y) ** 2)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    x2 = np.sum(x ** 2, axis=1, keepdims=True)
    y2 = np.sum(y ** 2, axis=1, keepdims=True).T
    dists = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
    return dists.squeeze()


def minkowski_distance(x, y, p=3):
    """Minkowski distance of order p.

    When p=1: Manhattan, p=2: Euclidean, p=inf: Chebyshev.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if np.isinf(p):
        return chebyshev_distance(x, y)

    if x.ndim == 1 and y.ndim == 1:
        return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    n1, n2 = x.shape[0], y.shape[0]
    dists = np.zeros((n1, n2))
    for i in range(n1):
        dists[i] = np.sum(np.abs(x[i] - y) ** p, axis=1) ** (1.0 / p)

    return dists.squeeze()


def pairwise_distance_matrix(X, metric='euclidean', **kwargs):
    """Compute pairwise distance matrix for a set of points.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    metric : str or callable
        'euclidean', 'manhattan', 'chebyshev', 'cosine', 'squared_euclidean', 'minkowski'
    **kwargs : additional arguments (e.g., p for minkowski)

    Returns
    -------
    dists : ndarray, shape (n_samples, n_samples)
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    metrics = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'chebyshev': chebyshev_distance,
        'cosine': cosine_similarity,
        'squared_euclidean': squared_euclidean,
        'minkowski': lambda x, y: minkowski_distance(x, y, kwargs.get('p', 3)),
    }

    if isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {list(metrics.keys())}")
        dist_func = metrics[metric]
    else:
        dist_func = metric

    dists = np.zeros((n, n))
    for i in range(n):
        dists[i] = dist_func(X[i], X)

    return dists


def point_to_centers_distance(X, centers, metric='euclidean', **kwargs):
    """Compute distance from each point to each center.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    centers : ndarray, shape (n_clusters, n_features)
    metric : str or callable

    Returns
    -------
    dists : ndarray, shape (n_samples, n_clusters)
    """
    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    metrics = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'chebyshev': chebyshev_distance,
        'cosine': cosine_similarity,
        'squared_euclidean': squared_euclidean,
        'minkowski': lambda x, y: minkowski_distance(x, y, kwargs.get('p', 3)),
    }

    if isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(f"Unknown metric '{metric}'")
        dist_func = metrics[metric]
    else:
        dist_func = metric

    return dist_func(X, centers)
