"""
ADE-FCM Big Data Enhancement Module
====================================
Provides large-scale distributed FCM optimizations using Apache Spark:
1. RDD-level optimizations (coalesce, broadcast, treeAggregate, checkpoint)
2. DataFrame/SQL-level optimizations (AQE, broadcast join, partitioning)
3. LargeScaleFCM - memory-efficient chunked FCM for data that exceeds RAM
4. Benchmark pipeline for sequential vs RDD vs DataFrame vs SQL comparison

Components:
    SparkRDDOptimizer       - RDD partitioning, caching, checkpoint, broadcast
    SparkDataFrameOptimizer - Spark SQL, AQE, broadcast hash join, partitioning
    LargeScaleFCM           - Chunked distributed FCM with progress tracking
"""

from .spark_rdd_optimizer import SparkRDDOptimizer
from .spark_dataframe_optimizer import SparkDataFrameOptimizer
from .large_scale_fcm import LargeScaleFCM

__all__ = [
    "SparkRDDOptimizer",
    "SparkDataFrameOptimizer",
    "LargeScaleFCM",
]
