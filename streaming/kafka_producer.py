import json
import time
import numpy as np
from loguru import logger


class DataProducer:
    """Produces streaming data for real-time ADE-FCM clustering."""

    def __init__(self, bootstrap_servers='localhost:9092', topic='clustering-data'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None

    def _get_producer(self):
        if self.producer is None:
            from confluent_kafka import Producer
            self.producer = Producer({'bootstrap.servers': self.bootstrap_servers})
        return self.producer

    def produce_batch(self, X, batch_size=100, sleep=0.1, key_func=None):
        """Produce data points in batches to a Kafka topic."""
        producer = self._get_producer()
        n = len(X)
        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size]
            for j, point in enumerate(batch):
                data = {
                    'point': point.tolist() if hasattr(point, 'tolist') else list(point),
                    'timestamp': time.time(),
                    'index': i + j,
                }
                key = str(key_func(point)) if key_func else str(i + j)
                producer.produce(self.topic, key=key, value=json.dumps(data))
            producer.flush()
            logger.info(f"Produced batch {i // batch_size + 1}, points {i} to {i + len(batch)}")
            time.sleep(sleep)

    def produce_stream(self, X, interval=0.5):
        """Produce one data point at a time (true streaming)."""
        producer = self._get_producer()
        for i, point in enumerate(X):
            data = {
                'point': point.tolist() if hasattr(point, 'tolist') else list(point),
                'timestamp': time.time(),
                'index': i,
            }
            producer.produce(self.topic, key=str(i), value=json.dumps(data))
            if i % 10 == 0:
                producer.flush()
            time.sleep(interval)
        producer.flush()

    def close(self):
        if self.producer is not None:
            self.producer.flush()
            logger.info("Kafka producer closed")
