"""
Density-Based Initialization for ADE-FCM
=========================================
Implements Contribution 2: Density-Based Initialization.

Selects initial cluster centers from high-density regions using a
Gaussian kernel density estimate. Provides both a standalone
DensityInitializer class and a KMeansPlusPlusInitializer class.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from loguru import logger


class DensityInitializer:
    """Density-based center initialization for fuzzy clustering.

    Computes a local density estimate for each point using a Gaussian
    kernel with an adaptive cutoff distance. Centers are selected from
    the highest-density points with a spread-enforcing heuristic.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    density_percentile : float, default=90.0
        Percentile threshold for high-density candidate selection.
    subsample_size : int, default=1000
        Maximum points used for pairwise distance calculation.
    random_state : int, default=42
        Random seed.
    """

    def __init__(
        self,
        n_clusters,
        density_percentile=90.0,
        subsample_size=1000,
        random_state=42,
    ):
        self.n_clusters = n_clusters
        self.density_percentile = density_percentile
        self.subsample_size = subsample_size
        self.random_state = random_state

    def _compute_cutoff_distance(self, X):
        """Compute cutoff distance d_c as the 2nd percentile of pairwise distances.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        d_c : float
        """
        n = X.shape[0]
        if n > self.subsample_size:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n, self.subsample_size, replace=False)
            sample = X[idx]
        else:
            sample = X
        # Use a subset for O(n^2) to stay tractable
        max_pts = min(len(sample), 200)
        if max_pts < 2:
            return 1.0
        sub = sample[:max_pts]
        dists = pdist(sub, metric="euclidean")
        if len(dists) == 0:
            return 1.0
        return max(np.percentile(dists, 2), 1e-10)

    def _local_density(self, X, d_c):
        """Compute Gaussian-kernel local density for each point.

        rho_i = sum_j exp(-(||x_i - x_j|| / d_c)^2)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        d_c : float

        Returns
        -------
        density : ndarray of shape (n_samples,)
        """
        n = X.shape[0]
        density = np.zeros(n)
        # Process in chunks for memory efficiency
        chunk = 500
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            batch = X[start:end]
            # Pairwise squared distances between batch and all points
            diff = batch[:, np.newaxis, :] - X[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=2)
            density[start:end] = np.sum(np.exp(-sq_dists / (d_c ** 2)), axis=1)
        return density

    def initialize(self, X):
        """Select initial cluster centers from high-density regions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        logger.info(
            f"DensityInitializer: computing cutoff distance from {n} points"
        )
        d_c = self._compute_cutoff_distance(X)

        logger.info("DensityInitializer: computing local densities")
        density = self._local_density(X, d_c)

        # Candidate pool: top density_percentile %
        threshold = np.percentile(density, self.density_percentile)
        candidates = np.where(density >= threshold)[0]
        if len(candidates) < self.n_clusters:
            # Fallback: take the absolute highest density points
            candidates = np.argsort(density)[-self.n_clusters:]

        # Select centers ensuring spatial spread
        selected = []
        rng.shuffle(candidates)
        # Pick first candidate
        selected.append(candidates[0])
        used = {candidates[0]}

        while len(selected) < self.n_clusters and len(used) < len(candidates):
            best_c = None
            best_min_dist = -1.0
            for c in candidates:
                if c in used:
                    continue
                # Distance to nearest already-selected center
                c_val = X[c]
                min_dist = min(np.sqrt(np.sum((X[s] - c_val) ** 2)) for s in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_c = c
            if best_c is not None:
                selected.append(best_c)
                used.add(best_c)
            else:
                break

        # If still short, pick randomly from unselected candidates
        remaining = [c for c in candidates if c not in used]
        rng.shuffle(remaining)
        while len(selected) < self.n_clusters and remaining:
            selected.append(remaining.pop())

        logger.info(
            f"DensityInitializer: selected {len(selected)} centers "
            f"(requested {self.n_clusters})"
        )
        return X[np.array(selected)].copy()


class KMeansPlusPlusInitializer:
    """KMeans++ initialization (Arthur & Vassilvitskii 2007).

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    random_state : int, default=42
        Random seed.
    """

    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def initialize(self, X):
        """Select centers using KMeans++ probabilistic seeding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        k = min(self.n_clusters, n)

        centers = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.array(
                [min(np.sum((x - c) ** 2) for c in centers) for x in X]
            )
            probs = dists / np.sum(dists)
            next_idx = rng.choice(n, p=probs)
            centers.append(X[next_idx])

        return np.array(centers)


class RandomInitializer:
    """Simple random center initializer.

    Parameters
    ----------
    n_clusters : int
    random_state : int, default=42
    """

    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def initialize(self, X):
        """Select random points as centers.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        rng = np.random.RandomState(self.random_state)
        k = min(self.n_clusters, X.shape[0])
        idx = rng.choice(X.shape[0], k, replace=False)
        return X[idx].copy()
