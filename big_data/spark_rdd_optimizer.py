import numpy as np
from loguru import logger
from pyspark.storagelevel import StorageLevel


class SparkRDDOptimizer:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.sc = spark_session.sparkContext

    def optimize_partitioning(self, rdd, target_partitions):
        current = rdd.getNumPartitions()
        if target_partitions < current:
            logger.info(
                f"Coalescing RDD from {current} to {target_partitions} partitions"
            )
            return rdd.coalesce(target_partitions, shuffle=False)
        elif target_partitions > current:
            logger.info(
                f"Repartitioning RDD from {current} to {target_partitions} partitions"
            )
            return rdd.repartition(target_partitions)
        return rdd

    def cache_strategy(self, rdd, storage_level="MEMORY_AND_DISK"):
        level_map = {
            "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
            "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
            "MEMORY_ONLY_SER": StorageLevel.MEMORY_ONLY_SER,
            "MEMORY_AND_DISK_SER": StorageLevel.MEMORY_AND_DISK_SER,
            "DISK_ONLY": StorageLevel.DISK_ONLY,
        }
        level = level_map.get(storage_level, StorageLevel.MEMORY_AND_DISK)
        logger.info(f"Persisting RDD with storage level: {storage_level}")
        return rdd.persist(level)

    def checkpoint(self, rdd, checkpoint_dir):
        self.sc.setCheckpointDir(checkpoint_dir)
        logger.info(f"Checkpointing RDD to {checkpoint_dir}")
        rdd.checkpoint()
        return rdd

    def broadcast_centers(self, sc, centers):
        logger.info(f"Broadcasting centers with shape {centers.shape}")
        return sc.broadcast(centers)

    def tree_aggregate(self, rdd, seq_op, comb_op, depth=2):
        logger.info(f"Performing treeAggregate with depth={depth}")
        return rdd.treeAggregate(
            zeroValue=None,
            seqOp=seq_op,
            combOp=comb_op,
            depth=depth,
        )

    def map_partitions_with_centers(self, rdd, centers_bc, func):
        def wrapper(iterator):
            centers = centers_bc.value
            return func(iterator, centers)

        return rdd.mapPartitions(wrapper)

    def fault_tolerant_compute(self, rdd, n_retries=3):
        for attempt in range(n_retries):
            try:
                result = rdd.collect()
                return result
            except Exception as e:
                logger.warning(
                    f"RDD compute attempt {attempt + 1}/{n_retries} failed: {e}"
                )
                if attempt == n_retries - 1:
                    logger.error("All retry attempts exhausted")
                    raise
        return None


class RDDMembershipComputer:
    def __init__(self, m, n_clusters):
        self.m = m
        self.n_clusters = n_clusters

    def compute(self, iterator, centers):
        results = []
        for row in iterator:
            x = np.array(row, dtype=np.float64)
            dists = np.sqrt(np.sum((centers - x) ** 2, axis=1))
            dists = np.clip(dists, 1e-12, None)
            inv_dists = dists ** (-2.0 / (self.m - 1 + 1e-12))
            u = inv_dists / inv_dists.sum()
            results.append(u.tolist())
        return iter(results)


class RDDAggregator:
    @staticmethod
    def center_aggregate(data_rdd, U, m, n_clusters, n_features):
        def seq_op(acc, val):
            x, u = val
            um = u ** m
            return (acc[0] + um[:, None] * x, acc[1] + um)

        def comb_op(a, b):
            return (a[0] + b[0], a[1] + b[1])

        init_val = (
            np.zeros((n_clusters, n_features), dtype=np.float64),
            np.zeros(n_clusters, dtype=np.float64),
        )

        labeled = data_rdd.zipWithIndex().map(
            lambda x: (
                x[1] % n_clusters,
                (np.array(x[0], dtype=np.float64), np.array(U[x[1]])),
            )
        )

        aggregated = labeled.aggregateByKey(
            init_val,
            lambda acc, val: (
                acc[0] + (val[1] ** m)[:, None] * val[0],
                acc[1] + (val[1] ** m),
            ),
            comb_op,
        ).collect()

        centers = np.zeros((n_clusters, n_features))
        for cluster_id, (num, den) in aggregated:
            centers[cluster_id] = num / (den[:, None] + 1e-12)
        return centers
