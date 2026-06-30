"""
Adaptive Parameters for ADE-FCM
=================================
Implements:
  - Contribution 3: Adaptive Fuzzifier m(t)
  - Contribution 7: Early Stopping
  - Contribution 8: Dynamic Convergence Threshold
"""

import numpy as np
from loguru import logger


class AdaptiveFuzzifier:
    """Time-varying fuzzifier exponent m(t).

    Formula:
        m(t) = m_min + (m_max - m_min) * exp(-alpha * t / T)

    where t is the current iteration and T is the total iterations.

    High m values early promote exploration (soft partitions);
    low m values later encourage crisp partitions.

    Parameters
    ----------
    m_min : float, default=1.1
        Minimum fuzzifier (asymptote, > 1).
    m_max : float, default=2.5
        Maximum fuzzifier (starting value).
    alpha : float, default=3.0
        Decay rate. Higher alpha -> faster decay to m_min.
    max_iter : int, default=300
        Maximum number of iterations (T).
    """

    def __init__(self, m_min=1.1, m_max=2.5, alpha=3.0, max_iter=300):
        self.m_min = m_min
        self.m_max = m_max
        self.alpha = alpha
        self.max_iter = max_iter

    def __call__(self, t):
        """Compute m(t) for iteration t (0-indexed).

        Parameters
        ----------
        t : int

        Returns
        -------
        m_t : float
        """
        ratio = t / max(self.max_iter - 1, 1)
        return self.m_min + (self.m_max - self.m_min) * np.exp(-self.alpha * ratio)

    def schedule(self):
        """Pre-compute the full m(t) schedule.

        Returns
        -------
        schedule : ndarray of shape (max_iter,)
        """
        return np.array([self(t) for t in range(self.max_iter)])

    def summary(self):
        """Return a dict describing the schedule."""
        sched = self.schedule()
        return {
            "m_min": self.m_min,
            "m_max": self.m_max,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "m_start": float(sched[0]),
            "m_end": float(sched[-1]),
            "m_mean": float(sched.mean()),
        }


class DynamicThreshold:
    """Time-varying convergence threshold epsilon(t).

    Formula:
        epsilon(t) = eps_0 * exp(-beta * t / T)

    where t is the current iteration and T is the total iterations.

    Starts loose to allow early movement; shrinks exponentially
    to force precise convergence later.

    Parameters
    ----------
    eps_0 : float, default=1e-3
        Initial threshold.
    beta : float, default=5.0
        Decay rate. Higher beta -> faster tightening.
    max_iter : int, default=300
        Maximum iterations (T).
    min_eps : float, default=1e-8
        Floor value to prevent underflow.
    """

    def __init__(self, eps_0=1e-3, beta=5.0, max_iter=300, min_eps=1e-8):
        self.eps_0 = eps_0
        self.beta = beta
        self.max_iter = max_iter
        self.min_eps = min_eps

    def __call__(self, t):
        """Compute epsilon(t) for iteration t (0-indexed).

        Parameters
        ----------
        t : int

        Returns
        -------
        eps_t : float
        """
        ratio = t / max(self.max_iter - 1, 1)
        eps_t = self.eps_0 * np.exp(-self.beta * ratio)
        return max(eps_t, self.min_eps)

    def schedule(self):
        """Pre-compute the full epsilon(t) schedule.

        Returns
        -------
        schedule : ndarray of shape (max_iter,)
        """
        return np.array([self(t) for t in range(self.max_iter)])

    def summary(self):
        sched = self.schedule()
        return {
            "eps_0": self.eps_0,
            "beta": self.beta,
            "max_iter": self.max_iter,
            "min_eps": self.min_eps,
            "eps_start": float(sched[0]),
            "eps_end": float(sched[-1]),
        }


class EarlyStopping:
    """Patience-based early stopping for iterative clustering.

    Tracks a convergence metric (e.g. Frobenius norm of membership
    change) and stops after `patience` consecutive iterations where
    the metric is below a given threshold.

    Parameters
    ----------
    patience : int, default=10
        Number of consecutive "converged" iterations before stopping.
    min_delta : float, default=0.0
        Minimum absolute change to reset patience (not currently used
        as threshold comparison is delegated to the caller).
    verbose : bool, default=True
    """

    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self._counter = 0
        self._best_metric = float("inf")
        self._stopped_iteration = None
        self.convergence_history = []

    def reset(self):
        """Reset the stopping state."""
        self._counter = 0
        self._best_metric = float("inf")
        self._stopped_iteration = None
        self.convergence_history = []

    def check(self, change, threshold, iteration):
        """Check whether to stop.

        Parameters
        ----------
        change : float
            Convergence metric at this iteration.
        threshold : float
            Convergence threshold.
        iteration : int
            Current iteration number (for logging).

        Returns
        -------
        stop : bool
            Whether training should stop.
        """
        self.convergence_history.append(change)
        if change < threshold:
            self._counter += 1
            if self.verbose and self._counter == 1:
                logger.debug(
                    f"EarlyStopping: change={change:.2e} < threshold={threshold:.2e} "
                    f"at iter {iteration} (counter={self._counter})"
                )
            if self._counter >= self.patience:
                self._stopped_iteration = iteration
                if self.verbose:
                    logger.info(
                        f"EarlyStopping triggered at iteration {iteration} "
                        f"after {self.patience} consecutive converged steps"
                    )
                return True
        else:
            if self._counter > 0:
                self._counter = 0
        return False

    @property
    def stopped(self):
        return self._stopped_iteration is not None

    def summary(self):
        return {
            "patience": self.patience,
            "counter": self._counter,
            "stopped_at": self._stopped_iteration,
            "final_change": self.convergence_history[-1] if self.convergence_history else None,
        }
