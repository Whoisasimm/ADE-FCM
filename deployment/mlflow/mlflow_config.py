import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import mlflow.spark
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

logger = logging.getLogger(__name__)


class ADEFCMMLflowConfig:
    """MLflow configuration and tracking utilities for ADE-FCM."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_location: Optional[str] = None,
        nested: bool = False,
    ):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "ADE-FCM"
        )
        self.artifact_location = artifact_location or os.getenv(
            "MLFLOW_DEFAULT_ARTIFACT_ROOT", "./mlflow-artifacts"
        )
        self.nested = nested
        self.client: Optional[MlflowClient] = None
        self.experiment_id: Optional[str] = None

        mlflow.set_tracking_uri(self.tracking_uri)
        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Create or retrieve the experiment."""
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = self.client.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            self.experiment_id = self.client.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_location,
            )
            logger.info("Created experiment '%s' (id=%s)", self.experiment_name, self.experiment_id)
        else:
            self.experiment_id = experiment.experiment_id
            logger.info("Using existing experiment '%s' (id=%s)", self.experiment_name, self.experiment_id)

        mlflow.set_experiment(self.experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        default_tags = {
            "project": "ADE-FCM",
            "mlflow.note.content": description or "",
        }
        if tags:
            default_tags.update(tags)

        return mlflow.start_run(
            run_name=run_name,
            tags=default_tags,
            nested=self.nested,
            description=description,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log training hyperparameters."""
        sanitized = {str(k): str(v) for k, v in params.items()}
        mlflow.log_params(sanitized)
        logger.debug("Logged params: %s", sanitized)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.debug("Logged metrics: %s", metrics)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        logger.info("Logged artifact: %s -> %s", local_path, artifact_path)

    def log_figure(self, figure, artifact_path: str) -> None:
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_path)
        logger.info("Logged figure: %s", artifact_path)

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        spark_model: bool = True,
        input_example: Optional[Any] = None,
        signature: Optional[ModelSignature] = None,
        pip_requirements: Optional[List[str]] = None,
    ) -> None:
        """Log a Spark ML or sklearn model."""
        if spark_model:
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                input_example=input_example,
                signature=signature,
                pip_requirements=pip_requirements,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                input_example=input_example,
                signature=signature,
                pip_requirements=pip_requirements,
            )
        logger.info("Logged model at '%s'", artifact_path)

    def log_dataframe(self, df: pd.DataFrame, artifact_path: str) -> None:
        """Log a pandas DataFrame as a CSV artifact."""
        tmp_path = Path(tempfile.gettempdir()) / f"mlflow_dataframe_{id(df)}.csv"
        df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(str(tmp_path), artifact_path=artifact_path)
        tmp_path.unlink()
        logger.info("Logged DataFrame: %s", artifact_path)

    def log_training_curve(self, history: Dict[str, List[float]]) -> None:
        """Log training curves from training history."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, (metric, values) in enumerate(history.items()):
            row, col = divmod(idx, 2)
            if row >= 2:
                break
            ax = axes[row, col]
            ax.plot(values, linewidth=2)
            ax.set_title(metric)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.log_figure(fig, "training_curves.png")
        plt.close(fig)

    @staticmethod
    def get_best_run(
        experiment_name: str,
        metric_name: str = "silhouette_score",
        maximize: bool = True,
        tracking_uri: Optional[str] = None,
    ) -> Optional[mlflow.entities.Run]:
        """Retrieve the best run from an experiment based on a metric."""
        client = MlflowClient(tracking_uri=tracking_uri or mlflow.get_tracking_uri())
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("Experiment '%s' not found", experiment_name)
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1,
        )
        if runs:
            logger.info(
                "Best run: %s (metric %s = %s)",
                runs[0].info.run_id,
                metric_name,
                runs[0].data.metrics.get(metric_name),
            )
            return runs[0]
        return None

    def register_best_model(
        self,
        metric_name: str = "silhouette_score",
        maximize: bool = True,
        model_name: str = "ADE-FCM-Best",
        stage: str = "Production",
    ) -> Optional[str]:
        """Find and register the best model to the Model Registry."""
        best_run = self.get_best_run(self.experiment_name, metric_name, maximize)
        if best_run is None:
            logger.warning("No runs found to register")
            return None

        model_uri = f"runs:/{best_run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info("Registered model '%s' version %s", model_name, result.version)

        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage,
        )
        logger.info("Transitioned model '%s' v%s to '%s'", model_name, result.version, stage)
        return result.version

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple runs and return a DataFrame."""
        if metrics is None:
            metrics = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score",
                       "runtime_seconds", "n_iterations"]

        rows = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {"run_id": run_id, "run_name": run.data.tags.get("mlflow.runName", "")}
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric, None)
            row["status"] = run.info.status
            rows.append(row)

        df = pd.DataFrame(rows).set_index("run_id")
        logger.info("Comparing %d runs across %d metrics", len(run_ids), len(metrics))
        return df

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        logger.info("Ended run with status '%s'", status)

    def cleanup_old_runs(self, max_runs: int = 100) -> int:
        """Delete old runs beyond the maximum allowed."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"],
        )
        if len(runs) <= max_runs:
            return 0

        deleted = 0
        for run in runs[max_runs:]:
            self.client.delete_run(run.info.run_id)
            deleted += 1

        logger.info("Cleaned up %d old runs", deleted)
        return deleted

    def create_experiment_from_config(self, config_path: Union[str, Path]) -> str:
        """Create and configure an experiment from a JSON config file."""
        with open(config_path) as f:
            config = json.load(f)

        experiment_name = config.get("experiment_name", self.experiment_name)
        tags = config.get("tags", {})

        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(
                name=experiment_name,
                artifact_location=config.get("artifact_location", self.artifact_location),
                tags=tags,
            )
        else:
            experiment_id = experiment.experiment_id
            if tags:
                for key, value in tags.items():
                    self.client.set_experiment_tag(experiment_id, key, value)

        return experiment_id
