# Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 32 GB+ |
| CPU Cores | 4 | 8+ |
| Disk Space | 5 GB | 20 GB |
| GPU (optional) | CUDA 11.x | CUDA 12.x, 8 GB VRAM |
| Java (Spark) | JDK 11 | JDK 17 |
| Apache Spark | 3.3.0 | 3.3.4+ |

## pip Install

### Basic install (CPU only)

```bash
pip install ade-fcm
```

### Install from source

```bash
git clone https://github.com/your-org/ADE-FCM.git
cd ADE-FCM
pip install -e .
```

### Optional dependencies

```bash
# Spark support
pip install pyspark findspark

# GPU support (CUDA)
pip install cupy-cuda11x  # match your CUDA version
pip install cudf-cu11 cuml-cu11  # RAPIDS (optional)

# Streaming
pip install confluent-kafka

# Visualization
pip install matplotlib seaborn

# MLflow tracking
pip install mlflow

# All extras
pip install -e ".[all]"
```

### Requirements file

```bash
pip install -r baseline_project/requirements.txt
pip install pyspark cupy-cuda11x confluent-kafka mlflow matplotlib seaborn scipy scikit-learn pandas numpy loguru psutil tracemalloc
```

## Conda Environment Setup

```bash
conda create -n ade-fcm python=3.9 -y
conda activate ade-fcm

# Core
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn loguru psutil

# PySpark
conda install -c conda-forge pyspark findspark

# Kafka
pip install confluent-kafka

# GPU (CUDA 11.x)
conda install -c conda-forge cupy cudatoolkit=11.8
conda install -c rapidsai cuml=23.08

# MLflow
conda install -c conda-forge mlflow

# Verify
python -c "from novel_algorithm import ADEFCM; print('ADE-FCM ready')"
```

## Docker Setup

### Build Docker image

```bash
# Minimal image
docker build -t ade-fcm/spark:latest -f deployment/Dockerfile --target ade-fcm .

# With Jupyter
docker build -t ade-fcm/jupyter:latest -f deployment/Dockerfile --target jupyter .

# Training image
docker build -t ade-fcm/training:latest -f deployment/Dockerfile --target training .
```

### Run with Docker Compose (full stack)

```bash
docker compose -f deployment/docker-compose.yml up -d
```

This starts: ZooKeeper, Kafka, Spark Master + 3 Workers, Jupyter Lab, MLflow Tracking Server.

Services:
- Spark UI: http://localhost:8080
- Jupyter Lab: http://localhost:8888 (token: `ade-fcm`)
- MLflow UI: http://localhost:5000
- Kafka: localhost:9092

```bash
# Stop all
docker compose -f deployment/docker-compose.yml down

# Clean volumes
docker compose -f deployment/docker-compose.yml down -v
```

## Spark Cluster Setup

### Local Mode (single machine)

```python
from novel_algorithm import SparkADEFCM

model = SparkADEFCM(
    n_clusters=5,
    spark_master="local[*]",  # uses all local cores
    verbose=True
)
```

### Standalone Cluster

1. Start the Spark cluster:

```bash
# Start master
$SPARK_HOME/sbin/start-master.sh --host 0.0.0.0 --port 7077

# Start workers (repeat for each node)
$SPARK_HOME/sbin/start-worker.sh spark://master-node:7077
```

2. Connect ADE-FCM:

```python
model = SparkADEFCM(
    n_clusters=10,
    spark_master="spark://master-node:7077",
    checkpoint_dir="hdfs://namenode:8020/checkpoints"
)
```

### Databricks

1. Create a Databricks cluster (Runtime 12.x+, 8+ GB driver, 4+ workers).
2. Install the ADE-FCM library via PyPI or DBFS:
   - Notebook cell: `%pip install ade-fcm`
   - Or upload wheel to DBFS and: `%pip install /dbfs/path/to/ade_fcm.whl`
3. Use SparkADEFCM:

```python
spark.conf.set("spark.sql.adaptive.enabled", "true")

from novel_algorithm import SparkADEFCM
model = SparkADEFCM(
    n_clusters=5,
    spark_master="yarn",
    checkpoint_dir="/dbfs/checkpoints"
)
model.fit(X)
```

## GPU Setup

### CUDA Toolkit

```bash
# Verify CUDA
nvidia-smi

# Install CuPy matching your CUDA version
pip install cupy-cuda11x   # CUDA 11.x
pip install cupy-cuda12x   # CUDA 12.x

# Verify
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

### CuPy GPU FCM

```python
from gpu import GPUFCMManager

# GPU mode
manager = GPUFCMManager(use_gpu=True, n_clusters=5)
centers, U, J = manager.fit(X)
print(f"GPU speedup vs CPU: ~{manager.benchmark_cpu_vs_gpu(X)['speedup']:.1f}x")
```

### RAPIDS (cuML + cuDF)

```bash
# Via conda (recommended)
conda install -c rapidsai -c conda-forge cuml=23.08 cudf=23.08

# Verify
python -c "from cuml.cluster import KMeans; print('RAPIDS ready')"
```

```python
from gpu import RAPIDSFCM

model = RAPIDSFCM(n_clusters=5, max_iter=100)
centers, U, J = model.fit_fuzzy(X)
print(centers)
```

### Spark + GPU Hybrid

```python
from gpu import SparkGPUHybridEngine

engine = SparkGPUHybridEngine(n_clusters=5, max_iter=50, spark_mode="spark_gpu")
centers, U = engine.fit_spark_gpu(X)
```

## Kafka Setup

```bash
# Using Docker (recommended)
docker run -d --name zookeeper -p 2181:2181 confluentinc/cp-zookeeper:7.3.0
docker run -d --name kafka -p 9092:9092 -e KAFKA_ZOOKEEPER_CONNECT=localhost:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  confluentinc/cp-kafka:7.3.0

# Or via Docker Compose
docker compose -f deployment/docker-compose.yml up -d zookeeper kafka
```

### Verify Kafka

```bash
# Create a test topic
docker exec -it ade-fcm-kafka kafka-topics --bootstrap-server localhost:9092 --create --topic test --partitions 1 --replication-factor 1

# List topics
docker exec -it ade-fcm-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

```python
# Verify from Python
from streaming import DataProducer
producer = DataProducer(bootstrap_servers="localhost:9092", topic="clustering-data")
print("Kafka producer ready")
producer.close()
```

## Verification Steps

After installation, verify everything works:

```bash
# 1. Core ADE-FCM
python -c "
from novel_algorithm import ADEFCM
import numpy as np
X = np.random.randn(200, 5)
m = ADEFCM(n_clusters=3, max_iter=50, random_state=42)
m.fit(X)
print(f'OK: {m.n_clusters} clusters, {m.n_iter_} iterations, {m.outlier_mask_.sum()} outliers')
"

# 2. Spark (if configured)
python -c "
from novel_algorithm import SparkADEFCM
import numpy as np
X = np.random.randn(500, 4)
m = SparkADEFCM(n_clusters=3, max_iter=20, spark_master='local[2]', verbose=False)
m.fit(X)
print(f'Spark OK: {m.n_clusters} clusters, {m.n_iter_} iterations')
m.stop()
"

# 3. GPU (if available)
python -c "
from gpu import GPUFCMManager
import numpy as np
X = np.random.randn(1000, 5)
m = GPUFCMManager(use_gpu=True, n_clusters=3, max_iter=30)
centers, U, J = m.fit(X)
print(f'GPU OK: {centers.shape}')
"

# 4. Streaming
python -c "
from streaming import OnlineFCM
import numpy as np
m = OnlineFCM(n_clusters=3, m=2.0, learning_rate=0.3)
for _ in range(5):
    m.partial_fit(np.random.randn(50, 4))
print(f'Streaming OK: centers shape {m.centers_.shape}')
"

# 5. XAI
python -c "
from novel_algorithm import ADEFCM, explain_clusters
import numpy as np
X = np.random.randn(100, 4)
m = ADEFCM(n_clusters=3, random_state=42).fit(X)
exp = explain_clusters(X, m.labels_, m.centers_, feature_names=['a','b','c','d'])
print(f'XAI OK: {exp[\"n_clusters\"]} clusters, {exp[\"n_outliers\"]} outliers')
"
```
