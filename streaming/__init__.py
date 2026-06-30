"""
ADE-FCM Streaming Module
========================
Real-time and online clustering support for ADE-FCM.

Components:
- DataProducer        - Kafka producer for streaming data
- ADEFCMStreaming     - Spark Structured Streaming consumer
- OnlineFCM           - Incremental online Fuzzy C-Means
- StreamingPipeline   - End-to-end streaming pipeline
"""

try:
    from .kafka_producer import DataProducer
except ImportError:
    DataProducer = None

try:
    from .spark_streaming_consumer import ADEFCMStreaming
except ImportError:
    ADEFCMStreaming = None

from .online_clustering import OnlineFCM

__all__ = [
    "DataProducer",
    "ADEFCMStreaming",
    "OnlineFCM",
]
