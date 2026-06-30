"""
Spark ADE-FCM: Distributed ADE-FCM via PySpark
================================================
Implements Contribution 10: Distributed Spark Optimization.

Provides a scalable implementation of ADE-FCM using PySpark's
broadcast variables, mapPartitions for local computation, and
treeAggregate for efficient distributed statistics.

Features:
  - Broadcast centers and parameters to all workers
  - mapPartitions for local membership/center updates
  - treeAggregate for distributed objective computation
  - Checkpointing for fault tolerance
  - Adaptive parallelism
"""

import numpy as np
from loguru import logger

try:
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False


class SparkADEFCM:
    """Distributed ADE-FCM using PySpark.

    Parameters
    ----------
    n_clusters : int or 'auto', default='auto'
    max_iter : int, default=300
    m : float or 'adaptive', default='adaptive'
    epsilon : float or 'dynamic', default='dynamic'
    init_method : str, default='kmeans++'
    early_stopping_patience : int, default=10
    outlier_threshold : float, default=2.0
    random_state : int, default=42
    verbose : bool, default=True
    spark_master : str, default='local[*]'
        Spark master URL.
    checkpoint_dir : str or None, default=None
        Directory for Spark checkpointing (HDFS or local path).
    num_partitions : int or None, default=None
        Number of partitions. If None, inferred from cluster.
    """

    def __init__(
        self,
        n_clusters="auto",
        max_iter=300,
        m="adaptive",
        epsilon="dynamic",
        init_method="kmeans++",
        early_stopping_patience=10,
        outlier_threshold=2.0,
        random_state=42,
        verbose=True,
        spark_master="local[*]",
        checkpoint_dir=None,
        num_partitions=None,
    ):
        if not _SPARK_AVAILABLE:
            raise ImportError(
                "PySpark is not installed. Install with: pip install pyspark"
            )
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.init_method = init_method
        self.early_stopping_patience = early_stopping_patience
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.spark_master = spark_master
        self.checkpoint_dir = checkpoint_dir
        self.num_partitions = num_partitions
        self.centers_ = None
        self.U_ = None
        self.labels_ = None
        self.J_history_ = []
        self.n_iter_ = 0
        self.outlier_mask_ = None
        self.spark = None
        self.sc = None

    def _init_spark(self):
        """Initialize SparkSession and SparkContext."""
        conf = (
            SparkConf()
            .setMaster(self.spark_master)
            .setAppName("ADE-FCM")
            .set("spark.sql.adaptive.enabled", "true")
            .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )
        self.spark = (
            SparkSession.builder.config(conf=conf)
            .getOrCreate()
        )
        self.sc = self.spark.sparkContext
        if self.checkpoint_dir:
            self.sc.setCheckpointDir(self.checkpoint_dir)
            if self.verbose:
                logger.info(f"Spark checkpoint dir: {self.checkpoint_dir}")
        if self.verbose:
            logger.info(
                f"Spark initialized: master={self.spark_master}, "
                f"defaultParallelism={self.sc.defaultParallelism}"
            )

    @staticmethod
    def _compute_membership_partition(iterator, centers_bc, m):
        """Compute membership matrix for a partition (runs on workers).

        Parameters
        ----------
        iterator : iterable of ndarray
        centers_bc : Broadcast variable
        m : float

        Yields
        ------
        U_part : ndarray
        """
        from .ade_fcm import ADEFCM

        centers = centers_bc.value
        # We need a temporary ADEFCM instance to reuse membership logic
        # but we do membership directly here
        X_part = np.array(list(iterator))
        if len(X_part) == 0:
            return
        n_local = X_part.shape[0]
        c = centers.shape[0]
        dist = np.zeros((n_local, c))
        for j in range(c):
            diff = X_part - centers[j]
            dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dist = np.maximum(dist, 1e-10)
        inv = 2.0 / (m - 1) if m > 1 else 2.0 / 0.1
        U_part = np.zeros((n_local, c))
        for i in range(n_local):
            for j in range(c):
                ratios = dist[i, j] / dist[i, :]
                denom = np.sum(ratios ** inv)
                U_part[i, j] = 1.0 / denom if denom > 1e-10 else 0.0
        yield U_part

    @staticmethod
    def _compute_partial_stats(iterator, centers_bc, m):
        """Compute per-partition center numerator and denominator.

        Yields
        ------
        (numerator_sum, denominator_sum, local_n) : tuple
        """
        centers = centers_bc.value
        X_part = np.array(list(iterator))
        if len(X_part) == 0:
            return

        n_local = X_part.shape[0]
        c = centers.shape[0]
        # Compute distances
        dist = np.zeros((n_local, c))
        for j in range(c):
            diff = X_part - centers[j]
            dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
        dist = np.maximum(dist, 1e-10)
        inv = 2.0 / (m - 1) if m > 1 else 2.0 / 0.1

        U_part = np.zeros((n_local, c))
        for i in range(n_local):
            for j in range(c):
                ratios = dist[i, j] / dist[i, :]
                denom = np.sum(ratios ** inv)
                U_part[i, j] = 1.0 / denom if denom > 1e-10 else 0.0

        Um = U_part ** m
        numerator = X_part.T @ Um
        denominator = Um.sum(axis=0)
        yield (numerator, denominator, n_local)

    def _distributed_center_update(self, rdd, centers, m):
        """Distributed center update using mapPartitions + treeAggregate.

        Each partition computes local numerators and denominators,
        which are aggregated via treeAggregate.

        Parameters
        ----------
        rdd : RDD
        centers : ndarray
        m : float

        Returns
        -------
        new_centers : ndarray
        """
        centers_bc = self.sc.broadcast(centers)

        stats = rdd.mapPartitions(
            lambda it: self._compute_partial_stats(it, centers_bc, m)
        )

        # treeAggregate with zero-value
        zero = (
            np.zeros((centers.shape[1], centers.shape[0])),  # numerator sum
            np.zeros(centers.shape[0]),                       # denominator sum
            0,                                                 # total count
        )

        def seq_op(acc, elem):
            num_sum, den_sum, n_sum = acc
            e_num, e_den, e_n = elem
            return (num_sum + e_num, den_sum + e_den, n_sum + e_n)

        def comb_op(a, b):
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        num_total, den_total, _ = stats.treeAggregate(zero, seq_op, comb_op, depth=2)

        centers_bc.destroy()
        new_centers = (num_total / np.maximum(den_total, 1e-10)).T
        return new_centers

    def _distributed_objective(self, rdd, centers, U, m):
        """Compute the fuzzy objective J distributedly.

        Parameters
        ----------
        rdd : RDD
        centers : ndarray
        U : ndarray
        m : float

        Returns
        -------
        J : float
        """
        centers_bc = self.sc.broadcast(centers)
        # We broadcast U as well (or compute from data)
        # For efficiency we compute J per partition
        def compute_J(iterator, centers_bc_local, m_local):
            from .ade_fcm import ADEFCM

            centers = centers_bc_local.value
            X_part = np.array(list(iterator))
            if len(X_part) == 0:
                return [(0.0, 0, 0.0)]
            n_local = X_part.shape[0]
            c = centers.shape[0]
            # Compute membership again (or we could pass U)
            dist = np.zeros((n_local, c))
            for j in range(c):
                diff = X_part - centers[j]
                dist[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
            dist = np.maximum(dist, 1e-10)
            inv = 2.0 / (m_local - 1) if m_local > 1 else 2.0 / 0.1
            U_part = np.zeros((n_local, c))
            for i in range(n_local):
                for j in range(c):
                    ratios = dist[i, j] / dist[i, :]
                    denom = np.sum(ratios ** inv)
                    U_part[i, j] = 1.0 / denom if denom > 1e-10 else 0.0

            J_local = 0.0
            for j in range(c):
                diff = X_part - centers[j]
                dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
                J_local += np.sum(U_part[:, j] ** m_local * dists_j)
            U_safe = np.maximum(U_part, 1e-10)
            entropy = -np.sum(U_part * np.log(U_safe))
            return [(J_local + 0.01 * entropy, n_local, 0.0)]

        J_parts = rdd.mapPartitions(
            lambda it: compute_J(it, centers_bc, m)
        )
        J_total = J_parts.map(lambda x: x[0]).sum()
        centers_bc.destroy()
        return J_total

    @staticmethod
    def _vector_to_rdd(spark, X):
        """Convert numpy array to RDD of rows."""
        X_list = [X[i] for i in range(X.shape[0])]
        rdd = spark.sparkContext.parallelize(X_list)
        return rdd

    def _adaptive_fuzzifier(self, t):
        """m(t) = m_min + (m_max - m_min) * exp(-alpha * t / T)"""
        if not isinstance(self.m, str):
            return float(self.m)
        m_min, m_max, alpha = 1.1, 2.5, 3.0
        ratio = t / max(self.max_iter - 1, 1)
        return m_min + (m_max - m_min) * np.exp(-alpha * ratio)

    def _dynamic_threshold(self, t):
        """epsilon(t) = eps_0 * exp(-beta * t / T)"""
        if not isinstance(self.epsilon, str):
            return float(self.epsilon)
        eps_0, beta = 1e-3, 5.0
        ratio = t / max(self.max_iter - 1, 1)
        eps_t = eps_0 * np.exp(-beta * ratio)
        return max(eps_t, 1e-8)

    def fit(self, X, y=None):
        """Fit Spark ADE-FCM to data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self : SparkADEFCM
        """
        n, d = X.shape
        logger.info(f"Spark ADE-FCM fitting: {n} samples, {d} features")

        self._init_spark()
        from .ade_fcm import ADEFCM

        # Auto cluster discovery
        if self.n_clusters == "auto":
            from .auto_cluster import AutomaticClusterDiscovery

            discover = AutomaticClusterDiscovery(random_state=self.random_state)
            k_range = range(2, min(15, int(np.sqrt(n)) + 2))
            if len(k_range) < 2:
                k_range = range(2, min(10, max(3, n // 2)))
            # Use a local ADEFCM for auto-discovery (faster)
            local_model = ADEFCM(
                max_iter=100,
                m=self.m,
                epsilon=self.epsilon,
                init_method=self.init_method,
                random_state=self.random_state,
                verbose=False,
            )
            self.n_clusters = discover.consensus_search(X, k_range, base_estimator=local_model)
            logger.info(f"Auto-discovered K = {self.n_clusters}")

        if self.n_clusters is None or self.n_clusters < 2:
            self.n_clusters = 2

        # Initialize centers locally
        local_model = ADEFCM(
            n_clusters=self.n_clusters,
            init_method=self.init_method,
            random_state=self.random_state,
            verbose=False,
        )
        self.centers_ = local_model._initialize_centers(X)

        # Parallelize data
        if self.num_partitions:
            rdd = self._vector_to_rdd(self.spark, X).repartition(self.num_partitions)
        else:
            rdd = self._vector_to_rdd(self.spark, X)

        # Initial membership (random)
        rng = np.random.RandomState(self.random_state)
        self.U_ = rng.rand(n, self.n_clusters)
        self.U_ = self.U_ / self.U_.sum(axis=1, keepdims=True)

        self.J_history_ = []
        patience_counter = 0
        iteration = 0

        for iteration in range(self.max_iter):
            m_t = self._adaptive_fuzzifier(iteration)
            eps_t = self._dynamic_threshold(iteration)

            # Distributed center update
            new_centers = self._distributed_center_update(rdd, self.centers_, m_t)

            # Compute membership locally (or distributedly)
            # For simplicity, compute full U locally after center update
            centers_bc = self.sc.broadcast(new_centers)
            U_parts = rdd.mapPartitions(
                lambda it: self._compute_membership_partition(it, centers_bc, m_t)
            )
            U_list = U_parts.collect()
            centers_bc.destroy()

            if U_list:
                U_new = np.vstack(U_list)
            else:
                U_new = self.U_.copy()

            # Confidence weighting (local)
            U_safe = np.maximum(U_new, 1e-10)
            mu = np.mean(U_safe, axis=1)
            sigma = np.std(U_safe, axis=1)
            ratio = np.where(mu > 1e-10, sigma / mu, 1e10)
            ratio = np.clip(ratio, 0, 1e10)
            conf = 1.0 - (2.0 / np.pi) * np.arctan(ratio)
            conf = np.maximum(conf, 1e-10)
            U_new = U_safe * conf[:, np.newaxis]
            row_sum = U_new.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, 1e-10)
            U_new = U_new / row_sum

            # Convergence
            change = np.linalg.norm(U_new - self.U_, "fro")
            self.U_ = U_new
            self.centers_ = new_centers

            # Distributed objective
            J = self._distributed_objective(rdd, self.centers_, self.U_, m_t)
            self.J_history_.append(J)

            # Early stopping
            if change < eps_t:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info(
                            f"Early stopping at iteration {iteration + 1}"
                        )
                    break
            else:
                patience_counter = 0

            if self.verbose and iteration % 20 == 0:
                logger.info(
                    f"Iter {iteration + 1:3d} | J={J:.4f} | m={m_t:.3f} | "
                    f"eps={eps_t:.2e} | change={change:.2e}"
                )

        self.n_iter_ = iteration + 1
        self.labels_ = np.argmax(self.U_, axis=1)

        # Outlier detection
        Um = self.U_ ** m_t
        scores = np.zeros(n)
        for j in range(self.n_clusters):
            diff = X - self.centers_[j]
            dists_j = np.sqrt(np.sum(diff ** 2, axis=1))
            scores += Um[:, j] * dists_j
        mu_s = np.mean(scores)
        sigma_s = np.std(scores)
        self.outlier_mask_ = scores > (mu_s + self.outlier_threshold * sigma_s)

        n_out = int(self.outlier_mask_.sum())
        logger.info(
            f"Spark ADE-FCM converged in {self.n_iter_}/{self.max_iter} iterations, "
            f"J*={self.J_history_[-1]:.4f}, outliers={n_out}/{n}"
        )

        return self

    def predict(self, X):
        """Predict cluster labels for new data using Spark.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        rdd = self._vector_to_rdd(self.spark, X)
        m_t = self._adaptive_fuzzifier(self.n_iter_)
        centers_bc = self.sc.broadcast(self.centers_)
        U_parts = rdd.mapPartitions(
            lambda it: self._compute_membership_partition(it, centers_bc, m_t)
        )
        U_list = U_parts.collect()
        centers_bc.destroy()
        if U_list:
            U = np.vstack(U_list)
            return np.argmax(U, axis=1)
        return np.array([])

    def stop(self):
        """Stop the Spark session."""
        if self.spark is not None:
            self.spark.stop()
            logger.info("Spark session stopped.")

    def __del__(self):
        self.stop()
