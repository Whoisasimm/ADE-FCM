"""
Explainable AI Module for ADE-FCM
===================================
Implements Contribution 9: Explainable Clustering (XAI).

Provides:
  - Feature importance per cluster (mean shift, Fisher score, permutation)
  - Cluster summaries (size, center, radius, top features)
  - Natural language descriptions of clusters
  - SHAP-style (approximate) explanations
  - Global explanation of the full clustering result
"""

import numpy as np
from scipy.spatial.distance import cdist
from loguru import logger


def feature_importance(X, labels, centers, method="permutation", U=None):
    """Compute per-cluster feature importance scores.

    Uses permutation-based importance by default: shuffling a feature and
    measuring the increase in within-cluster dispersion. For fuzzy clustering,
    also supports membership-weighted dispersion when U is provided.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_clusters, n_features)
    method : {'shift', 'fisher', 'permutation', 'fuzzy_permutation'}, default='permutation'
        Importance method.
    U : ndarray of shape (n_samples, n_clusters) or None
        Fuzzy membership matrix. Required for 'fuzzy_permutation'.

    Returns
    -------
    importances : ndarray of shape (n_clusters, n_features)
        Higher values indicate more important features for each cluster.
    """
    n_clusters = centers.shape[0]
    n_features = X.shape[1]

    if method == "shift":
        global_center = X.mean(axis=0)
        global_std = X.std(axis=0) + 1e-10
        importances = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            importances[k] = np.abs(centers[k] - global_center) / global_std
        return importances

    elif method == "fisher":
        global_mean = X.mean(axis=0)
        between = np.zeros(n_features)
        within = np.zeros(n_features)
        for k in range(n_clusters):
            mask = labels == k
            n_k = mask.sum()
            if n_k < 2:
                continue
            diff = centers[k] - global_mean
            between += n_k * diff ** 2
            cluster_var = X[mask].var(axis=0)
            within += n_k * cluster_var
        within = np.maximum(within, 1e-10)
        return np.tile(between / within, (n_clusters, 1))

    elif method == "permutation":
        base_disp = _total_within_disp(X, labels, centers)
        importances = np.zeros((n_clusters, n_features))
        rng = np.random.RandomState(42)
        for f in range(n_features):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, f])
            perm_labels = labels.copy()
            perm_centers = centers.copy()
            for k in range(n_clusters):
                mask = perm_labels == k
                if mask.sum() > 0:
                    perm_centers[k] = X_perm[mask].mean(axis=0)
            perm_disp = _total_within_disp(X_perm, perm_labels, perm_centers)
            imp = (perm_disp - base_disp) / max(base_disp, 1e-10)
            importances[:, f] = max(imp, 0)
        return importances

    elif method == "fuzzy_permutation":
        if U is None:
            raise ValueError("U (fuzzy membership matrix) required for fuzzy_permutation")
        base_disp = _fuzzy_within_disp(X, U, centers)
        importances = np.zeros((n_clusters, n_features))
        rng = np.random.RandomState(42)
        for f in range(n_features):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, f])
            perm_labels = labels.copy()
            perm_centers = centers.copy()
            for k in range(n_clusters):
                mask = perm_labels == k
                if mask.sum() > 0:
                    perm_centers[k] = X_perm[mask].mean(axis=0)
            perm_disp = _fuzzy_within_disp(X_perm, U, perm_centers)
            imp = (perm_disp - base_disp) / max(base_disp, 1e-10)
            importances[:, f] = max(imp, 0)
        return importances

    else:
        raise ValueError(f"Unknown importance method: {method}")


def _fuzzy_within_disp(X, U, centers):
    """Fuzzy within-cluster dispersion: sum_k sum_i u_ik * ||x_i - v_k||^2."""
    n_clusters = centers.shape[0]
    disp = 0.0
    for k in range(n_clusters):
        diff = X - centers[k]
        dists = np.sum(diff ** 2, axis=1)
        disp += np.sum(U[:, k] * dists)
    return disp


def _total_within_disp(X, labels, centers):
    """Total within-cluster dispersion (sum of squared distances)."""
    disp = 0.0
    for k in range(len(centers)):
        mask = labels == k
        if mask.sum() > 0:
            diff = X[mask] - centers[k]
            disp += np.sum(diff ** 2)
    return disp


def cluster_summary(X, labels, centers, feature_names=None):
    """Generate a summary dict for each cluster.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_clusters, n_features)
    feature_names : list of str or None

    Returns
    -------
    summaries : list of dict
        Each dict contains:
            - cluster_id
            - size (int)
            - proportion (float)
            - center (ndarray)
            - radius (float)  -- mean distance to center
            - max_distance (float)
            - top_features (list of (name, score) tuples)
            - feature_importances (ndarray)
    """
    n_clusters = centers.shape[0]
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    importances = feature_importance(X, labels, centers, method="permutation")
    summaries = []

    for k in range(n_clusters):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k == 0:
            summaries.append({
                "cluster_id": k,
                "size": 0,
                "proportion": 0.0,
                "center": centers[k],
                "radius": 0.0,
                "max_distance": 0.0,
                "top_features": [],
                "feature_importances": importances[k],
            })
            continue

        points = X[mask]
        diff = points - centers[k]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))

        # Top features by importance
        f_imp = importances[k]
        top_idx = np.argsort(f_imp)[::-1][:5]
        top_features = [(feature_names[i], float(f_imp[i])) for i in top_idx]

        summaries.append({
            "cluster_id": k,
            "size": n_k,
            "proportion": float(n_k / X.shape[0]),
            "center": centers[k],
            "radius": float(np.mean(dists)),
            "max_distance": float(np.max(dists)),
            "top_features": top_features,
            "feature_importances": importances[k],
        })

    return summaries


def describe_cluster_natural(summary, feature_names=None):
    """Generate a human-readable natural-language description of a cluster.

    Parameters
    ----------
    summary : dict
        Cluster summary from cluster_summary().
    feature_names : list of str or None

    Returns
    -------
    description : str
    """
    k = summary["cluster_id"]
    size = summary["size"]
    prop = summary["proportion"] * 100
    radius = summary["radius"]
    top = summary["top_features"][:3]

    lines = [f"Cluster {k}:"]
    lines.append(f"  Size: {size} points ({prop:.1f}% of data)")
    lines.append(f"  Radius (mean distance to center): {radius:.3f}")

    if top:
        feat_str = ", ".join([f"{name} (score={score:.3f})" for name, score in top])
        lines.append(f"  Top distinctive features: {feat_str}")

    return "\n".join(lines)


def shap_explain(X, labels, centers, sample_idx=0):
    """Approximate SHAP-style explanation for a single point.

    Uses feature marginalization as a proxy for Shapley values:
    phi_f = E[X_f | cluster] - E[X_f | global]

    This gives the contribution of each feature to the point
    being assigned to its cluster vs. the global average.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_clusters, n_features)
    sample_idx : int
        Index of the point to explain.

    Returns
    -------
    shap_values : ndarray of shape (n_features,)
    explanation : str
    """
    n_features = X.shape[1]
    point = X[sample_idx]
    cluster_id = labels[sample_idx]
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0) + 1e-10

    # Cluster conditional mean
    mask = labels == cluster_id
    if mask.sum() > 0:
        cluster_mean = X[mask].mean(axis=0)
    else:
        cluster_mean = centers[cluster_id]

    # shap_value = (point - global_mean) * (cluster_mean - global_mean) / global_std^2
    shap_values = (point - global_mean) * (cluster_mean - global_mean) / (global_std ** 2)

    # Generate explanation text
    pos_features = []
    neg_features = []
    for f in range(n_features):
        if shap_values[f] > 0:
            pos_features.append(f)
        elif shap_values[f] < 0:
            neg_features.append(f)

    pos_str = ", ".join(
        [f"f{f} ({shap_values[f]:+.3f})" for f in pos_features[:5]]
    )
    neg_str = ", ".join(
        [f"f{f} ({shap_values[f]:+.3f})" for f in neg_features[:5]]
    )

    expl_lines = [
        f"Explanation for point {sample_idx} (assigned to cluster {cluster_id}):",
    ]
    if pos_str:
        expl_lines.append(f"  Features pushing toward this cluster: {pos_str}")
    if neg_str:
        expl_lines.append(f"  Features pushing away from this cluster: {neg_str}")
    if not pos_str and not neg_str:
        expl_lines.append("  No strong feature contributions.")

    explanation = "\n".join(expl_lines)
    return shap_values, explanation


def explain_clusters(X, labels, centers, feature_names=None, outlier_mask=None):
    """Full XAI explanation of the clustering result.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centers : ndarray of shape (n_clusters, n_features)
    feature_names : list of str or None
    outlier_mask : ndarray of bool or None

    Returns
    -------
    explanation : dict
        Contains:
            - n_clusters
            - n_samples
            - n_outliers (int)
            - summaries (list of dict)
            - descriptions (list of str)
            - global_feature_importance (ndarray)
    """
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    n_outliers = int(outlier_mask.sum()) if outlier_mask is not None else 0

    summaries = cluster_summary(X, labels, centers, feature_names)
    descriptions = [
        describe_cluster_natural(s, feature_names) for s in summaries
    ]

    # Global feature importance: mean absolute shift across all clusters
    importances = feature_importance(X, labels, centers, method="permutation")
    global_fi = np.mean(importances, axis=0)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    global_top = [
        (feature_names[i], float(global_fi[i]))
        for i in np.argsort(global_fi)[::-1][:5]
    ]

    return {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "n_outliers": n_outliers,
        "summaries": summaries,
        "descriptions": descriptions,
        "global_feature_importance": global_fi,
        "top_features_global": global_top,
    }
