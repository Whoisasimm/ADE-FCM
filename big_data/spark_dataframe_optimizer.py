from loguru import logger
from pyspark.sql import DataFrame


class SparkDataFrameOptimizer:
    def __init__(self, spark_session):
        self.spark = spark_session

    def optimize_with_sql(self, df, temp_view_name, sql_query):
        df.createOrReplaceTempView(temp_view_name)
        logger.info(f"Executing SQL via temp view '{temp_view_name}': {sql_query}")
        result = self.spark.sql(sql_query)
        return result

    def adaptive_query_execution(self, enabled=True):
        self.spark.conf.set("spark.sql.adaptive.enabled", str(enabled).lower())
        if enabled:
            self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            self.spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
            self.spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
            logger.info("Adaptive Query Execution enabled")
        else:
            logger.info("Adaptive Query Execution disabled")
        return self.spark

    def broadcast_hash_join(self, df1, df2, join_key):
        from pyspark.sql.functions import broadcast

        logger.info(f"Performing broadcast hash join on key: {join_key}")
        joined = df1.join(broadcast(df2), join_key)
        return joined

    def partition_by_column(self, df, column, num_partitions):
        logger.info(
            f"Repartitioning DataFrame by column '{column}' into {num_partitions} partitions"
        )
        return df.repartition(num_partitions, column)

    def cache_dataframe(self, df):
        logger.info("Caching DataFrame in memory and disk")
        df.cache()
        df.count()
        logger.info(f"DataFrame cached: {df.rdd.getNumPartitions()} partitions")
        return df

    def optimize_shuffle(self, conf, shuffle_partitions=200):
        conf_name = "spark.sql.shuffle.partitions"
        if isinstance(conf, dict):
            conf[conf_name] = shuffle_partitions
        else:
            self.spark.conf.set(conf_name, shuffle_partitions)
        logger.info(f"Shuffle partitions set to {shuffle_partitions}")
        return conf

    def dynamic_allocation(self, enabled=True, min_executors=1, max_executors=10):
        self.spark.conf.set(
            "spark.dynamicAllocation.enabled", str(enabled).lower()
        )
        if enabled:
            self.spark.conf.set(
                "spark.dynamicAllocation.minExecutors", str(min_executors)
            )
            self.spark.conf.set(
                "spark.dynamicAllocation.maxExecutors", str(max_executors)
            )
            self.spark.conf.set(
                "spark.dynamicAllocation.initialExecutors", str(min_executors)
            )
            logger.info(
                f"Dynamic allocation enabled: min={min_executors}, max={max_executors}"
            )
        return self.spark


class DataFrameMembershipComputer:
    def __init__(self, spark, centers, m):
        self.spark = spark
        self.centers = centers
        self.m = m
        self.n_clusters = centers.shape[0]

    def compute_udf(self, features_col="features", output_col="membership"):
        centers = self.centers
        m_val = self.m

        def _membership_udf(features):
            import numpy as np

            x = np.array(features, dtype=np.float64)
            dists = np.sqrt(np.sum((centers - x) ** 2, axis=1))
            dists = np.clip(dists, 1e-12, None)
            inv_dists = dists ** (-2.0 / (m_val - 1 + 1e-12))
            u = inv_dists / inv_dists.sum()
            return u.tolist()

        from pyspark.sql.types import ArrayType, DoubleType
        from pyspark.sql.functions import udf

        return udf(_membership_udf, ArrayType(DoubleType()))


class DataFramePartitionManager:
    @staticmethod
    def estimate_optimal_partitions(data_size_bytes, target_size_mb=128):
        target_bytes = target_size_mb * 1024 * 1024
        partitions = max(1, data_size_bytes // target_bytes)
        return partitions

    @staticmethod
    def repartition_for_skew(df, column, skew_threshold=1.5):
        from pyspark.sql.functions import count

        counts = df.groupBy(column).agg(count("*").alias("cnt"))
        stats = counts.selectExpr(
            f"mean(cnt) as mean_cnt", f"stddev(cnt) as std_cnt"
        ).collect()[0]
        mean_cnt = stats["mean_cnt"]
        std_cnt = stats["std_cnt"]
        threshold = mean_cnt + skew_threshold * std_cnt

        skewed = counts.filter(f"cnt > {threshold}").select(column).collect()
        skewed_keys = [row[column] for row in skewed]
        if skewed_keys:
            logger.warning(
                f"Skew detected on {len(skewed_keys)} keys: {skewed_keys[:5]}"
            )
        return skewed_keys
