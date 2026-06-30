import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, LongType
from loguru import logger


class ADEFCMStreaming:
    """Real-time ADE-FCM clustering with Spark Structured Streaming."""

    def __init__(self, spark=None, n_clusters=5, model=None):
        self.spark = spark or SparkSession.builder \
            .appName("ADE-FCM-Streaming") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/ade-fcm-checkpoint") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        self.n_clusters = n_clusters
        self.model = model
        self.batch_count = 0

    def create_kafka_stream(self, bootstrap_servers='localhost:9092', topic='clustering-data'):
        """Create a Kafka streaming source with parsed JSON schema."""
        schema = StructType([
            StructField("point", ArrayType(DoubleType())),
            StructField("timestamp", DoubleType()),
            StructField("index", LongType()),
        ])

        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", bootstrap_servers) \
            .option("subscribe", topic) \
            .load() \
            .select(from_json(col("value").cast("string"), schema).alias("data")) \
            .select("data.*")

        return df

    def online_update(self, micro_batch_df, batch_id):
        """Update cluster centers incrementally from a micro-batch."""
        self.batch_count += 1
        if micro_batch_df.count() == 0:
            return

        points = np.array([row.point for row in micro_batch_df.collect()])

        if self.model is None:
            logger.warning("No ADE-FCM model provided, skipping update")
            return

        alpha = 0.3
        m_value = getattr(self.model, 'm', 2.0)
        if m_value == 'adaptive':
            m_value = 2.0

        U = self.model._update_membership(points, self.model.centers_, m_value)
        new_centers = self.model._update_centers(points, U, m_value)
        self.model.centers_ = (1 - alpha) * self.model.centers_ + alpha * new_centers

        logger.info(f"Batch {self.batch_count}: Processed {len(points)} points, centers updated")

    def start_streaming(self, bootstrap_servers='localhost:9092', topic='clustering-data',
                        output_mode="update", trigger_interval=None):
        """Start the streaming query."""
        df = self.create_kafka_stream(bootstrap_servers, topic)

        query = df.writeStream \
            .foreachBatch(self.online_update) \
            .outputMode(output_mode) \
            .trigger(processingTime=trigger_interval) if trigger_interval else df.writeStream \
            .foreachBatch(self.online_update) \
            .outputMode(output_mode)

        if trigger_interval:
            from pyspark.sql.streaming import Trigger
            query = df.writeStream \
                .foreachBatch(self.online_update) \
                .outputMode(output_mode) \
                .trigger(processingTime=trigger_interval)

        return query.start()

    def start_console_stream(self, bootstrap_servers='localhost:9092', topic='clustering-data',
                             output_mode="append", trigger_interval="5 seconds"):
        """Start streaming with console output for debugging."""
        df = self.create_kafka_stream(bootstrap_servers, topic)

        query = df.writeStream \
            .outputMode(output_mode) \
            .format("console") \
            .option("truncate", "false") \
            .trigger(processingTime=trigger_interval) \
            .start()

        return query
