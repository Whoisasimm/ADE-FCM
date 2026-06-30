import time
import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist

from .spark_rdd_optimizer import SparkRDDOptimizer
from .spark_dataframe_optimizer import SparkDataFrameOptimizer


class ChunkedMembershipUpdate:
    def __init__(self, chunk_size=50000, m=2.0):
        self.chunk_size = chunk_size
        self.m = m

    def compute(self, X_chunk, centers):
        n_samples = X_chunk.shape[0]
        n_clusters = centers.shape[0]
        U_chunk = np.zeros((n_samples, n_clusters))

        for i in range(0, n_samples, self.chunk_size):
            end = min(i + self.chunk_size, n_samples)
            batch = X_chunk[i:end]
            dists = cdist(batch, centers, metric="euclidean")
            dists = np.clip(dists, 1e-12, None)
            inv_dists = dists ** (-2.0 / (self.m - 1 + 1e-12))
            U_chunk[i:end] = inv_dists / inv_dists.sum(axis=1, keepdims=True)

        return U_chunk


class ChunkedCenterUpdate:
    def __init__(self, m=2.0):
        self.m = m

    def compute(self, X_chunk, U_chunk):
        Um = U_chunk ** self.m
        numerator = Um.T @ X_chunk
        denominator = Um.sum(axis=0)
        return numerator, denominator

    def aggregate(self, partials):
        total_num = sum(p[0] for p in partials)
        total_den = sum(p[1] for p in partials)
        centers = total_num / (total_den[:, np.newaxis] + 1e-12)
        return centers


class ProgressTracker:
    def __init__(self, total_iterations, log_interval=1):
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.start_time = None
        self.current_iter = 0
        self.j_history = []

    def start(self):
        self.start_time = time.time()
        logger.info(f"Starting FCM: {self.total_iterations} max iterations")

    def update(self, iteration, J):
        self.current_iter = iteration + 1
        self.j_history.append(J)
        elapsed = time.time() - self.start_time
        if iteration % self.log_interval == 0:
            logger.info(
                f"Iter {iteration + 1:4d}/{self.total_iterations} | "
                f"J={J:.6f} | elapsed={elapsed:.2f}s"
            )

    def summary(self):
        elapsed = time.time() - self.start_time
        logger.info(
            f"FCM complete: {self.current_iter} iterations, "
            f"{elapsed:.2f}s elapsed, "
            f"final J={self.j_history[-1]:.6f}"
        )
        return {
            "iterations": self.current_iter,
            "elapsed_seconds": elapsed,
            "final_objective": self.j_history[-1],
            "j_history": self.j_history,
        }


class LargeScaleFCM:
    def __init__(
        self,
        spark_session,
        n_clusters,
        max_iter=100,
        m=2.0,
        epsilon=1e-5,
        chunk_size=50000,
        cache_storage_level="MEMORY_AND_DISK",
        use_rdd=True,
        use_dataframe=False,
        checkpoint_dir=None,
        verbose=True,
    ):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.chunk_size = chunk_size
        self.cache_storage_level = cache_storage_level
        self.use_rdd = use_rdd
        self.use_dataframe = use_dataframe
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        self.rdd_opt = SparkRDDOptimizer(spark_session)
        self.df_opt = SparkDataFrameOptimizer(spark_session)
        self.chunked_membership = ChunkedMembershipUpdate(chunk_size, m)
        self.chunked_centers = ChunkedCenterUpdate(m)
        self.tracker = None

        self._centers = None
        self._U = None
        self._J_history = []
        self._converged = False

    def _init_centers_kmeanspp(self, X, random_state=42):
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        centers = [X[rng.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            dists = np.array(
                [min(np.sum((x - c) ** 2) for c in centers) for x in X]
            )
            probs = dists / dists.sum()
            centers.append(X[rng.choice(n_samples, p=probs)])
        return np.array(centers)

    def fit_sequential(self, X):
        logger.info(
            f"LargeScaleFCM.fit_sequential: {X.shape[0]} samples, "
            f"{self.n_clusters} clusters, chunk_size={self.chunk_size}"
        )
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.tracker = ProgressTracker(self.max_iter)
        self.tracker.start()

        rng = np.random.RandomState(42)
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        centers = self._init_centers_kmeanspp(X)

        U_old = None
        for t in range(self.max_iter):
            U = self.chunked_membership.compute(X, centers)

            partials = []
            for i in range(0, n_samples, self.chunk_size):
                end = min(i + self.chunk_size, n_samples)
                num, den = self.chunked_centers.compute(X[i:end], U[i:end])
                partials.append((num, den))

            centers = self.chunked_centers.aggregate(partials)
            J = np.sum((U ** self.m) * cdist(X, centers) ** 2)
            self._J_history.append(J)
            self.tracker.update(t, J)

            if U_old is not None:
                change = np.linalg.norm(U - U_old)
                if change < self.epsilon:
                    self._converged = True
                    logger.info(f"Converged at iteration {t + 1}")
                    break

            U_old = U.copy()

        self._U = U
        self._centers = centers
        return centers, U, self._J_history

    def fit_rdd(self, X):
        logger.info(
            f"LargeScaleFCM.fit_rdd: {X.shape[0]} samples, "
            f"{self.n_clusters} clusters"
        )
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.tracker = ProgressTracker(self.max_iter)
        self.tracker.start()

        rdd = self.sc.parallelize(X, self.sc.defaultParallelism)
        rdd = self.rdd_opt.cache_strategy(rdd, self.cache_storage_level)
        if self.checkpoint_dir:
            rdd = self.rdd_opt.checkpoint(rdd, self.checkpoint_dir)

        rng = np.random.RandomState(42)
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)

        centers = self._init_centers_kmeanspp(X)

        U_old = None
        for t in range(self.max_iter):
            centers_bc = self.rdd_opt.broadcast_centers(self.sc, centers)

            def membership_func(iterator, centers):
                m_val = self.m
                results = []
                for row in iterator:
                    x = np.array(row, dtype=np.float64)
                    dists = np.sqrt(np.sum((centers - x) ** 2, axis=1))
                    dists = np.clip(dists, 1e-12, None)
                    inv_dists = dists ** (-2.0 / (m_val - 1 + 1e-12))
                    u = inv_dists / inv_dists.sum()
                    results.append(u.tolist())
                return iter(results)

            U_rows = self.rdd_opt.map_partitions_with_centers(
                rdd, centers_bc, membership_func
            )
            U = np.array(U_rows.collect())
            centers_bc.destroy()

            def seq_op(acc, val):
                x, u = val
                um = u ** self.m
                return (acc[0] + um[:, None] * x, acc[1] + um)

            def comb_op(a, b):
                return (a[0] + b[0], a[1] + b[1])

            init_val = (
                np.zeros((self.n_clusters, n_features), dtype=np.float64),
                np.zeros(self.n_clusters, dtype=np.float64),
            )

            labeled = rdd.zipWithIndex().map(
                lambda x: (
                    x[1] % self.n_clusters,
                    (np.array(x[0], dtype=np.float64), np.array(U[x[1]])),
                )
            )

            aggregated = labeled.aggregateByKey(
                init_val,
                lambda acc, val: (
                    acc[0] + (val[1] ** self.m)[:, None] * val[0],
                    acc[1] + (val[1] ** self.m),
                ),
                comb_op,
            ).collect()

            centers = np.zeros((self.n_clusters, n_features))
            for cluster_id, (num, den) in aggregated:
                centers[cluster_id] = num / (den[:, None] + 1e-12)

            J = np.sum((U ** self.m) * cdist(X, centers) ** 2)
            self._J_history.append(J)
            self.tracker.update(t, J)

            if U_old is not None:
                change = np.linalg.norm(U - U_old)
                if change < self.epsilon:
                    self._converged = True
                    logger.info(f"Converged at iteration {t + 1}")
                    break

            U_old = U.copy()

        rdd.unpersist()
        self._U = U
        self._centers = centers
        return centers, U, self._J_history

    def fit_dataframe(self, df, feature_columns=None):
        logger.info(
            f"LargeScaleFCM.fit_dataframe: DataFrame API, "
            f"{self.n_clusters} clusters"
        )
        from pyspark.ml.feature import VectorAssembler

        if feature_columns is None:
            from pyspark.sql.types import DoubleType, IntegerType
            feature_columns = [
                f.name
                for f in df.schema.fields
                if isinstance(f.dataType, (DoubleType, IntegerType))
                and f.name != "label"
            ]

        vec_col = "features_vec"
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol=vec_col
        )
        vec_df = assembler.transform(df)

        X = np.array(
            vec_df.select(vec_col)
            .rdd.map(lambda r: r[vec_col].toArray())
            .collect()
        )

        n_samples, n_features = X.shape
        self.tracker = ProgressTracker(self.max_iter)
        self.tracker.start()

        rng = np.random.RandomState(42)
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        centers = self._init_centers_kmeanspp(X)

        U_old = None
        for t in range(self.max_iter):
            centers_bc = self.sc.broadcast(centers)

            def predict_membership(features):
                import numpy as np
                x = np.array(features.toArray(), dtype=np.float64)
                c = centers_bc.value
                dists = np.sqrt(np.sum((c - x) ** 2, axis=1))
                dists = np.clip(dists, 1e-12, None)
                inv_dists = dists ** (-2.0 / (self.m - 1 + 1e-12))
                u = inv_dists / inv_dists.sum()
                return u.tolist()

            from pyspark.sql.types import ArrayType, DoubleType
            from pyspark.sql.functions import udf, col

            membership_udf = udf(predict_membership, ArrayType(DoubleType()))
            result_df = vec_df.select(
                membership_udf(col(vec_col)).alias("membership")
            )
            U_rows = result_df.collect()
            U = np.array([r["membership"] for r in U_rows])
            centers_bc.destroy()

            partials = []
            for i in range(0, n_samples, self.chunk_size):
                end = min(i + self.chunk_size, n_samples)
                Um = U[i:end] ** self.m
                num = Um.T @ X[i:end]
                den = Um.sum(axis=0)
                partials.append((num, den))

            centers = self.chunked_centers.aggregate(partials)

            J = np.sum((U ** self.m) * cdist(X, centers) ** 2)
            self._J_history.append(J)
            self.tracker.update(t, J)

            if U_old is not None:
                change = np.linalg.norm(U - U_old)
                if change < self.epsilon:
                    self._converged = True
                    logger.info(f"Converged at iteration {t + 1}")
                    break

            U_old = U.copy()

        self._U = U
        self._centers = centers
        return centers, U, self._J_history

    def fit_sql(self, df, temp_view="fcm_data"):
        logger.info(
            f"LargeScaleFCM.fit_sql: SQL API, "
            f"{self.n_clusters} clusters"
        )
        self.df_opt.adaptive_query_execution(True)
        df.createOrReplaceTempView(temp_view)

        from pyspark.sql.types import DoubleType, IntegerType
        feature_columns = [
            f.name
            for f in df.schema.fields
            if isinstance(f.dataType, (DoubleType, IntegerType))
            and f.name != "label"
        ]
        from pyspark.ml.feature import VectorAssembler

        vec_col = "features_vec"
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol=vec_col
        )
        vec_df = assembler.transform(df)

        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import ArrayType, DoubleType

        sql_view = f"{temp_view}_features"
        vec_df.createOrReplaceTempView(sql_view)

        X = np.array(
            vec_df.select(vec_col)
            .rdd.map(lambda r: r[vec_col].toArray())
            .collect()
        )
        n_samples, n_features = X.shape
        self.tracker = ProgressTracker(self.max_iter)
        self.tracker.start()

        rng = np.random.RandomState(42)
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        centers = self._init_centers_kmeanspp(X)

        U_old = None
        for t in range(self.max_iter):
            centers_bc = self.sc.broadcast(centers)

            def sql_membership(features):
                import numpy as np
                x = np.array(features.toArray(), dtype=np.float64)
                c = centers_bc.value
                dists = np.sqrt(np.sum((c - x) ** 2, axis=1))
                dists = np.clip(dists, 1e-12, None)
                inv_dists = dists ** (-2.0 / (self.m - 1 + 1e-12))
                u = inv_dists / inv_dists.sum()
                return u.tolist()

            m_udf = udf(sql_membership, ArrayType(DoubleType()))
            result_df = self.spark.sql(
                f"SELECT {vec_col} FROM {sql_view}"
            ).select(m_udf(col(vec_col)).alias("membership"))

            U_rows = result_df.collect()
            U = np.array([r["membership"] for r in U_rows])
            centers_bc.destroy()

            partials = []
            for i in range(0, n_samples, self.chunk_size):
                end = min(i + self.chunk_size, n_samples)
                Um = U[i:end] ** self.m
                num = Um.T @ X[i:end]
                den = Um.sum(axis=0)
                partials.append((num, den))

            centers = self.chunked_centers.aggregate(partials)

            J = np.sum((U ** self.m) * cdist(X, centers) ** 2)
            self._J_history.append(J)
            self.tracker.update(t, J)

            if U_old is not None:
                change = np.linalg.norm(U - U_old)
                if change < self.epsilon:
                    self._converged = True
                    logger.info(f"Converged at iteration {t + 1}")
                    break

            U_old = U.copy()

        self._U = U
        self._centers = centers
        return centers, U, self._J_history

    def get_metadata(self):
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "m": self.m,
            "epsilon": self.epsilon,
            "chunk_size": self.chunk_size,
            "converged": self._converged,
            "final_objective": (
                self._J_history[-1] if self._J_history else None
            ),
            "n_iterations": len(self._J_history),
        }
