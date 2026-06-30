"""
ADE-FCM Streaming Pipeline
===========================
End-to-end streaming pipeline integrating Kafka, Spark Structured Streaming,
and online incremental clustering with monitoring and checkpointing.

Usage:
    python -m ade_fcm.streaming.main \\
        --mode produce              \\
        --data-path data.csv         \\
        --bootstrap-servers localhost:9092

    python -m ade_fcm.streaming.main \\
        --mode stream                \\
        --bootstrap-servers localhost:9092 \\
        --n-clusters 5               \\
        --checkpoint-dir /tmp/ade-fcm-checkpoint
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger

from .kafka_producer import DataProducer
from .online_clustering import OnlineFCM


class StreamingPipeline:
    """End-to-end streaming pipeline orchestrator."""

    def __init__(
        self,
        n_clusters=5,
        fuzzifier=2.0,
        learning_rate=0.3,
        bootstrap_servers='localhost:9092',
        topic='clustering-data',
        checkpoint_dir='/tmp/ade-fcm-checkpoint',
        spark_master='local[*]',
        use_spark=False,
    ):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.learning_rate = learning_rate
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.checkpoint_dir = checkpoint_dir
        self.spark_master = spark_master
        self.use_spark = use_spark

        self.model = None
        self.producer = None
        self.consumer = None
        self.streaming_query = None
        self.metrics = {
            'batches_processed': 0,
            'total_points': 0,
            'start_time': None,
            'end_time': None,
            'centers_history': [],
        }

    def setup_model(self):
        """Initialize the online clustering model."""
        self.model = OnlineFCM(
            n_clusters=self.n_clusters,
            m=self.fuzzifier,
            learning_rate=self.learning_rate,
        )
        logger.info(f"Initialized OnlineFCM with k={self.n_clusters}, m={self.fuzzifier}")

    def setup_producer(self):
        """Initialize the Kafka producer."""
        self.producer = DataProducer(
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic,
        )
        logger.info(f"Initialized DataProducer for topic '{self.topic}'")

    def setup_spark_consumer(self):
        """Initialize Spark Structured Streaming consumer."""
        if not self.use_spark:
            return

        from .spark_streaming_consumer import ADEFCMStreaming

        import findspark
        findspark.init()

        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName("ADE-FCM-Streaming") \
            .master(self.spark_master) \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_dir) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        self.consumer = ADEFCMStreaming(
            spark=spark,
            n_clusters=self.n_clusters,
            model=self.model,
        )
        logger.info(f"Initialized Spark consumer with checkpoint at '{self.checkpoint_dir}'")

    def _process_batch(self, X, batch_size=100):
        """Process a single batch through the online model."""
        self.model.partial_fit(X)
        self.metrics['batches_processed'] += 1
        self.metrics['total_points'] += len(X)
        self.metrics['centers_history'].append(self.model.centers_.copy())
        return self.model.predict(X)

    def run_producer(self, data_path, batch_size=100, interval=0.1, stream_mode='batch'):
        """Run the producer to send data to Kafka."""
        self.setup_producer()

        data = np.loadtxt(data_path, delimiter=',') if os.path.exists(data_path) else None
        if data is None:
            logger.error(f"Data file not found: {data_path}")
            return

        logger.info(f"Loaded data with shape {data.shape} from {data_path}")

        if stream_mode == 'batch':
            self.producer.produce_batch(data, batch_size=batch_size, sleep=interval)
        else:
            self.producer.produce_stream(data, interval=interval)

        self.producer.close()

    def run_online(self, data_path, batch_size=100):
        """Run online clustering without Kafka (direct NumPy pipeline)."""
        self.setup_model()

        data = np.loadtxt(data_path, delimiter=',') if os.path.exists(data_path) else None
        if data is None:
            logger.error(f"Data file not found: {data_path}")
            return

        logger.info(f"Loaded data with shape {data.shape} from {data_path}")

        self.metrics['start_time'] = time.time()
        n = len(data)

        for i in range(0, n, batch_size):
            batch = data[i:i + batch_size]
            labels = self._process_batch(batch, batch_size=batch_size)
            logger.info(
                f"Batch {self.metrics['batches_processed']}: "
                f"points {i}-{i + len(batch)}, "
                f"clusters {np.unique(labels)}"
            )

        self.metrics['end_time'] = time.time()
        self._report_metrics()

    def run_spark_foreground(self, data_path=None):
        """Run the Spark streaming consumer (blocking)."""
        if not self.use_spark:
            logger.error("Spark mode not enabled; set use_spark=True")
            return

        self.setup_model()
        self.setup_spark_consumer()

        self.metrics['start_time'] = time.time()

        streaming_query = self.consumer.start_streaming(
            bootstrap_servers=self.bootstrap_servers,
            topic=self.topic,
        )

        try:
            streaming_query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
            streaming_query.stop()

        self.metrics['end_time'] = time.time()
        self._report_metrics()

    def _report_metrics(self):
        """Print performance metrics."""
        elapsed = self.metrics['end_time'] - self.metrics['start_time']
        logger.info("=" * 50)
        logger.info("Streaming Pipeline Metrics")
        logger.info("=" * 50)
        logger.info(f"  Total points processed : {self.metrics['total_points']}")
        logger.info(f"  Total batches          : {self.metrics['batches_processed']}")
        logger.info(f"  Elapsed time (s)       : {elapsed:.2f}")
        if self.metrics['total_points'] > 0:
            logger.info(f"  Throughput (pts/s)     : {self.metrics['total_points'] / elapsed:.1f}")
        if self.model and self.model.centers_ is not None:
            logger.info(f"  Final centers shape    : {self.model.centers_.shape}")
            logger.info(f"  Final centers          : {self.model.centers_}")
        logger.info("=" * 50)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ADE-FCM Streaming Pipeline")

    parser.add_argument('--mode', type=str, default='online',
                        choices=['produce', 'online', 'spark', 'stream'],
                        help='Pipeline mode: produce (kafka), online (direct), spark (structured streaming)')

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to input data file (CSV, no header)')

    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')

    parser.add_argument('--topic', type=str, default='clustering-data',
                        help='Kafka topic name')

    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of clusters')

    parser.add_argument('--fuzzifier', type=float, default=2.0,
                        help='Fuzzy exponent m')

    parser.add_argument('--learning-rate', type=float, default=0.3,
                        help='Online learning rate (decay for old centers)')

    parser.add_argument('--batch-size', type=int, default=100,
                        help='Mini-batch size for processing')

    parser.add_argument('--interval', type=float, default=0.1,
                        help='Sleep interval between batches (seconds)')

    parser.add_argument('--checkpoint-dir', type=str, default='/tmp/ade-fcm-checkpoint',
                        help='Spark checkpoint directory')

    parser.add_argument('--spark-master', type=str, default='local[*]',
                        help='Spark master URL')

    parser.add_argument('--use-spark', action='store_true', default=False,
                        help='Enable Spark Structured Streaming backend')

    parser.add_argument('--stream-mode', type=str, default='batch',
                        choices=['batch', 'single'],
                        help='Producer mode: batch or single-point streaming')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    return parser.parse_args()


def main():
    """Entry point for the streaming pipeline."""
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    pipeline = StreamingPipeline(
        n_clusters=args.n_clusters,
        fuzzifier=args.fuzzifier,
        learning_rate=args.learning_rate,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        checkpoint_dir=args.checkpoint_dir,
        spark_master=args.spark_master,
        use_spark=args.use_spark or args.mode == 'spark',
    )

    if args.mode == 'produce':
        if not args.data_path:
            logger.error("--data-path is required for produce mode")
            sys.exit(1)
        pipeline.run_producer(
            data_path=args.data_path,
            batch_size=args.batch_size,
            interval=args.interval,
            stream_mode=args.stream_mode,
        )
    elif args.mode == 'online':
        if not args.data_path:
            logger.error("--data-path is required for online mode")
            sys.exit(1)
        pipeline.run_online(
            data_path=args.data_path,
            batch_size=args.batch_size,
        )
    elif args.mode in ('spark', 'stream'):
        pipeline.run_spark_foreground(data_path=args.data_path)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
