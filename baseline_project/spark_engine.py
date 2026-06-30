import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from loguru import logger

class SparkFCMEngine:
    """Parallel FCM using Apache Spark."""
    
    def __init__(self, spark=None, n_clusters=5, max_iter=100, m=2.0, epsilon=1e-5):
        self.spark = spark or SparkSession.builder.appName("FCM").getOrCreate()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.centers = None
        self.U = None
        self.J_history = []
    
    def _dist_func(self, x, centers):
        """Compute distances from point to all centers."""
        return [np.sqrt(np.sum((x - c)**2)) for c in centers]
    
    def parallel_membership_update(self, rdd, centers_bc):
        """Parallel membership update using Spark map."""
        m = self.m
        n_clusters = self.n_clusters
        
        def update_partition(iterator):
            centers = centers_bc.value
            inv = 2.0 / (m - 1)
            for row in iterator:
                x = np.array(row[1:], dtype=float)
                dists = np.array([np.sqrt(np.sum((x - c)**2)) for c in centers])
                dists = np.maximum(dists, 1e-10)
                membership = np.zeros(n_clusters)
                for j in range(n_clusters):
                    denom = np.sum((dists[j] / dists) ** inv)
                    membership[j] = 1.0 / denom if denom > 0 else 0.0
                yield (row[0], *membership.tolist())
        
        return rdd.mapPartitions(update_partition)
    
    def parallel_center_update(self, rdd, n_features, n_clusters, m):
        """Parallel center update using Spark treeAggregate."""
        
        def seq_op(acc, row):
            x = np.array(row[1:1+n_features], dtype=float)
            u = np.array(row[1+n_features:], dtype=float)
            um = u ** m
            acc[0] += np.outer(x, um)
            acc[1] += um
            return acc
        
        def comb_op(a, b):
            return (a[0] + b[0], a[1] + b[1])
        
        zero = (np.zeros((n_features, n_clusters)), np.zeros(n_clusters))
        result = rdd.treeAggregate(zero, seq_op, comb_op, depth=2)
        numerator, denominator = result
        denominator = np.maximum(denominator, 1e-10)
        centers = (numerator / denominator).T
        return centers
    
    def fit(self, df, feature_columns):
        """Run FCM on Spark DataFrame."""
        logger.info(f"Starting Spark FCM with {self.n_clusters} clusters")
        
        # Get numpy array for initialization
        X = np.array(df.select(feature_columns).collect())
        n_features = len(feature_columns)
        
        # Initialize centers
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        self.centers = X[idx].copy()
        
        # Convert to RDD
        rdd = df.rdd.map(tuple).cache()
        n = rdd.count()
        
        for iteration in range(self.max_iter):
            # Broadcast centers
            centers_bc = self.spark.sparkContext.broadcast(self.centers.copy())
            
            # Parallel membership update
            membership_rdd = self.parallel_membership_update(rdd, centers_bc)
            
            # Collect U (for convergence check on driver)
            membership_data = membership_rdd.collect()
            U_new = np.array([row[1+n_features:] for row in membership_data])
            
            if self.U is not None:
                change = np.linalg.norm(U_new - self.U, 'fro')
                if change < self.epsilon:
                    logger.info(f"Converged at iteration {iteration+1}")
                    self.U = U_new
                    break
            
            self.U = U_new
            
            # Parallel center update
            self.centers = self.parallel_center_update(membership_rdd, n_features, self.n_clusters, self.m)
            
            # Compute objective
            J = 0.0
            for j in range(self.n_clusters):
                diff = X - self.centers[j]
                dists = np.sqrt(np.sum(diff**2, axis=1))
                J += np.sum(self.U[:, j]**self.m * dists)
            self.J_history.append(J)
            
            centers_bc.destroy()
        
        return self, self.centers, self.U, self.J_history
    
    def predict(self, df, feature_columns):
        """Assign hard labels."""
        X = np.array(df.select(feature_columns).collect())
        dists = np.zeros((X.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            diff = X - self.centers[j]
            dists[:, j] = np.sqrt(np.sum(diff**2, axis=1))
        return np.argmin(dists, axis=1)

class SparkFCLMEngine(SparkFCMEngine):
    """Parallel FCLM using Apache Spark (median-based)."""
    
    def parallel_center_update(self, rdd, n_features, n_clusters, m):
        """FCLM center update with median-based selection."""
        def seq_op(acc, row):
            x = np.array(row[1:1+n_features], dtype=float)
            u = np.array(row[1+n_features:], dtype=float)
            um = u ** m
            acc[0] += np.outer(x, um)
            acc[1] += um
            acc[2].append((x, u))
            return acc
        
        def comb_op(a, b):
            a[0] += b[0]
            a[1] += b[1]
            a[2].extend(b[2])
            return a
        
        zero = (np.zeros((n_features, n_clusters)), np.zeros(n_clusters), [])
        result = rdd.treeAggregate(zero, seq_op, comb_op, depth=2)
        numerator, denominator, points = result
        denominator = np.maximum(denominator, 1e-10)
        weighted_centers = (numerator / denominator).T
        
        # Median-based refinement
        all_points = np.array([p[0] for p in points])
        all_u = np.array([p[1] for p in points])
        centers = np.zeros_like(weighted_centers)
        
        for j in range(n_clusters):
            diff = all_points - weighted_centers[j]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            weighted_dists = dists * all_u[:, j]
            median_idx = np.argsort(weighted_dists)[len(weighted_dists) // 2]
            centers[j] = all_points[median_idx]
        
        return centers
