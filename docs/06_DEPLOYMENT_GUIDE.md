# Deployment Guide

## Docker Compose Setup

### Architecture

The Docker Compose stack (`deployment/docker-compose.yml`) orchestrates:

```
                  ┌─────────────┐
                  │   Jupyter   │ :8888
                  └──────┬──────┘
                         │
┌──────────┐    ┌───────────────┐    ┌──────────┐
│ ZooKeeper│◄──►│    Kafka      │◄──►│ Producer │
│ :2181    │    │ :9092/:29092  │    └──────────┘
└──────────┘    └───────┬───────┘
                        │
              ┌─────────▼──────────┐
              │   Spark Master     │ :7077/:8080
              └─────────┬──────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │ Worker1 │    │ Worker2 │    │ Worker3 │
   │ :8081   │    │ :8082   │    │ :8083   │
   └─────────┘    └─────────┘    └─────────┘

   ┌──────────┐
   │  MLflow  │ :5000
   └──────────┘
```

### Quick Start

```bash
# 1. Create .env.spark configuration
cat > .env.spark <<EOF
SPARK_RPC_AUTHENTICATION_ENABLED=no
SPARK_RPC_ENCRYPTION_ENABLED=no
SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
SPARK_SSL_ENABLED=no
SPARK_WORKER_CORES=4
SPARK_WORKER_MEMORY=8g
EOF

# 2. Start all services
docker compose -f deployment/docker-compose.yml up -d

# 3. Monitor startup
docker compose -f deployment/docker-compose.yml logs -f

# 4. Access UIs
# Spark Master:  http://localhost:8080
# Jupyter Lab:   http://localhost:8888 (token: ade-fcm)
# MLflow UI:     http://localhost:5000
```

### Run a Training Job

```bash
# Via Spark submit (inside container)
docker exec ade-fcm-spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/ade-fcm/train.py \
  --input /data/input \
  --output /data/output \
  --n-clusters 10

# Via Jupyter notebook
# Open http://localhost:8888 and run:
# from novel_algorithm import SparkADEFCM
# model = SparkADEFCM(n_clusters=10, spark_master="spark://spark-master:7077")
# model.fit(X)
```

### Teardown

```bash
docker compose -f deployment/docker-compose.yml down
docker compose -f deployment/docker-compose.yml down -v  # + volumes
```

---

## Kubernetes Deployment

### Prerequisites

```bash
kubectl cluster-info
# Requires: Kubernetes 1.24+, kubectl configured
# Recommended: 3+ worker nodes, 8+ GB RAM per node
```

### Deploy

```bash
# Create namespace and deploy all resources
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Monitor rollout
kubectl get pods -n ade-fcm -w
kubectl get services -n ade-fcm
kubectl get hpa -n ade-fcm
```

### Access Services

```bash
# Spark Master UI
kubectl port-forward -n ade-fcm service/spark-master-ui 8080:8080

# Jupyter Lab
kubectl port-forward -n ade-fcm service/jupyter 8888:8888
# Open http://localhost:8888 (token: ade-fcm)

# MLflow
kubectl port-forward -n ade-fcm service/mlflow 5000:5000
```

### Scaling

```bash
# Manual scale
kubectl scale deployment/spark-worker -n ade-fcm --replicas=5

# The HPA auto-scales between 3-10 workers based on CPU (70%) and memory (80%)
kubectl get hpa -n ade-fcm spark-worker-hpa
```

### Teardown

```bash
kubectl delete namespace ade-fcm
# Or selectively:
kubectl delete -f deployment/kubernetes/
```

---

## CI/CD Pipeline

The CI/CD workflow (`deployment/ci_cd/.github/workflows/ci.yml`) has three stages:

```yaml
jobs:
  test:        # On every push/PR to main/develop
    - Lint (flake8, black, mypy)
    - Unit tests (pytest, coverage >= 80%)
    - Benchmarks (pytest-benchmark)

  build:       # On push to main (after tests pass)
    - Build Docker image
    - Push to ghcr.io
    - Generate SBOM

  deploy:      # On push to main or manual dispatch
    - Deploy to Kubernetes
    - Rollout verification
```

### Manual Trigger

```bash
gh workflow run CI/CD Pipeline -f deploy-to-k8s=true
```

---

## MLflow Tracking

### Setup

The MLflow tracking server runs at `http://localhost:5000` (Docker) or within the cluster.

### Python Integration

```python
from deployment.mlflow import ADEFCMMLflowConfig

# Initialize
mlflow_config = ADEFCMMLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment_name="ADE-FCM-Experiment",
)

# Start a run
with mlflow_config.start_run(run_name="iris-benchmark", tags={"dataset": "iris"}):
    # Log parameters
    mlflow_config.log_params({
        "n_clusters": 5,
        "max_iter": 300,
        "m": "adaptive",
        "init_method": "kmeans++",
    })

    # Run model
    model = ADEFCM(n_clusters=5).fit(X)

    # Log metrics
    mlflow_config.log_metrics({
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "n_iterations": model.n_iter_,
        "n_outliers": int(model.outlier_mask_.sum()),
    })

    # Log training curves
    mlflow_config.log_training_curve({
        "objective": model.J_history_,
        "convergence": model.convergence_history_,
    })

    # Log model
    mlflow_config.log_model(model, artifact_path="ade-fcm-model")

# Register best model
mlflow_config.register_best_model(
    metric_name="silhouette_score",
    model_name="ADE-FCM-Production",
    stage="Production",
)
```

### Compare Runs

```python
# Compare multiple runs
comparison = mlflow_config.compare_runs(
    run_ids=["run_id_1", "run_id_2", "run_id_3"],
    metrics=["silhouette_score", "davies_bouldin_index", "n_iterations"],
)
print(comparison)

# Get best run
best = ADEFCMMLflowConfig.get_best_run(
    "ADE-FCM-Experiment",
    metric_name="silhouette_score",
    maximize=True,
)
```

### Airflow Integration

The Airflow DAG (`deployment/airflow/dags/research_pipeline_dag.py`) runs weekly:

```
Start → Check Data → Ingest/Generate → Preprocess → SparkSubmit(ADE-FCM)
    → Evaluate → [Generate Report] → [Deploy Model] → Log to MLflow → Email → End
```

```bash
# List DAGs
airflow dags list

# Trigger manually
airflow dags trigger ade_fcm_research_pipeline --conf '{"dataset": "synthetic", "n_clusters": 10}'
```

---

## Monitoring with Prometheus + Grafana

### Prometheus Configuration

The Prometheus config (`deployment/monitoring/prometheus.yml`) scrapes:

| Job | Target | Interval |
|-----|--------|----------|
| Spark Master | spark-master:8080 | 10s |
| Spark Workers | worker-1:8081, worker-2:8081, worker-3:8081 | 10s |
| Spark Applications | auto-discovered (K8s SD) | 15s |
| Jupyter | jupyter:8888 | 30s |
| MLflow | mlflow:5000 | 15s |
| Kafka | kafka:9092 | 15s |
| Node Exporter | node-exporter:9100 | 30s |
| cAdvisor | cadvisor:8080 | 15s |

### Start Monitoring

```bash
# With Docker Compose (add to docker-compose.yml if not present)
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./deployment/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./deployment/monitoring/grafana_dashboard.json:/var/lib/grafana/dashboards/ade-fcm.json

volumes:
  prometheus-data:
  grafana-data:
```

### Grafana Dashboard

Import `deployment/monitoring/grafana_dashboard.json` into Grafana.

Pre-configured panels:
- **Spark Cluster Health**: Master status, worker count, active applications
- **Job Execution Time**: ADE-FCM training duration per run
- **Objective Convergence**: J_history with convergence threshold overlay
- **Throughput**: Points/second for streaming pipeline
- **GPU Utilization**: Memory, core usage (if nvidia-exporter configured)
- **Resource Usage**: CPU, memory, disk per node
- **Kafka Lag**: Consumer lag per partition
- **MLflow Run Count**: Experiments and model versions over time
