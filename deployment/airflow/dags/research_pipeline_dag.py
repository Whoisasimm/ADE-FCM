from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

ENV = os.environ
DATA_DIR = Path(ENV.get("DATA_DIR", "/data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"

default_args = {
    "owner": "ade-fcm-team",
    "depends_on_past": False,
    "email": ["research@ade-fcm.org"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
    "sla": timedelta(hours=3),
}

with DAG(
    dag_id="ade_fcm_research_pipeline",
    default_args=default_args,
    description="End-to-end ADE-FCM research pipeline",
    schedule_interval="0 2 * * 6",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ade-fcm", "research", "training"],
    params={
        "dataset": "synthetic",
        "n_clusters": 10,
        "fuzzifier": 2.0,
        "max_iter": 100,
        "experiment_tag": "weekly-run",
        "generate_report": True,
        "deploy_model": False,
    },
    max_active_runs=1,
    concurrency=4,
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    check_data_availability = BashOperator(
        task_id="check_data_availability",
        bash_command=f"""
            echo "Checking data availability..."
            for f in "${{INPUT_DIR}}"/*.parquet "${{INPUT_DIR}}"/*.csv; do
                if [ -f "$f" ]; then
                    echo "Found: $f"
                fi
            done
            if [ "$(ls -A "${{INPUT_DIR}}"/ 2>/dev/null)" ]; then
                echo "DATA_AVAILABLE=true"
            else
                echo "DATA_AVAILABLE=false"
            fi
        """,
        dag=dag,
    )

    def decide_ingestion_strategy(**context) -> str:
        available = context["ti"].xcom_pull(task_ids="check_data_availability")
        if "DATA_AVAILABLE=true" in available:
            return "ingest_existing_data"
        return "generate_synthetic_data"

    ingest_or_generate = BranchPythonOperator(
        task_id="decide_ingestion_strategy",
        python_callable=decide_ingestion_strategy,
        dag=dag,
    )

    ingest_existing_data = BashOperator(
        task_id="ingest_existing_data",
        bash_command=f"""
            python /opt/ade-fcm/scripts/ingest_data.py \
                --input-dir {INPUT_DIR} \
                --format parquet \
                --output-dir {OUTPUT_DIR / "raw"}
        """,
        dag=dag,
    )

    generate_synthetic_data = BashOperator(
        task_id="generate_synthetic_data",
        bash_command=f"""
            python /opt/ade-fcm/scripts/generate_synthetic_data.py \
                --n-samples 100000 \
                --n-features 50 \
                --n-clusters 10 \
                --output-path {OUTPUT_DIR / "raw" / "synthetic.parquet"}
        """,
        dag=dag,
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"""
            python /opt/ade-fcm/scripts/preprocess.py \
                --input-dir {OUTPUT_DIR / "raw"} \
                --output-dir {OUTPUT_DIR / "processed"} \
                --normalize standard \
                --handle-missing mean \
                --encode-categorical onehot
        """,
        dag=dag,
    )

    run_ade_fcm = SparkSubmitOperator(
        task_id="run_ade_fcm",
        application="/opt/ade-fcm/train.py",
        conn_id="spark_default",
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.driver.memory": "4g",
            "spark.executor.memory": "8g",
            "spark.executor.cores": "4",
            "spark.executor.instances": "3",
            "spark.sql.adaptive.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        },
        application_args=[
            "--input", str(OUTPUT_DIR / "processed"),
            "--output", str(OUTPUT_DIR / "results"),
            "--model-dir", str(MODEL_DIR),
            "--n-clusters", "{{ params.n_clusters }}",
            "--fuzzifier", "{{ params.fuzzifier }}",
            "--max-iter", "{{ params.max_iter }}",
            "--experiment-tag", "{{ params.experiment_tag }}",
            "--tracking-uri", "{{ var.value.MLFLOW_TRACKING_URI }}",
        ],
        dag=dag,
    )

    evaluate_model = SparkSubmitOperator(
        task_id="evaluate_model",
        application="/opt/ade-fcm/evaluate.py",
        conn_id="spark_default",
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.driver.memory": "4g",
            "spark.executor.memory": "4g",
            "spark.executor.cores": "2",
            "spark.executor.instances": "2",
        },
        application_args=[
            "--model-dir", str(MODEL_DIR / "latest"),
            "--input", str(OUTPUT_DIR / "processed"),
            "--output", str(OUTPUT_DIR / "evaluation"),
            "--tracking-uri", "{{ var.value.MLFLOW_TRACKING_URI }}",
        ],
        dag=dag,
    )

    def should_generate_report(**context) -> str:
        if context["params"].get("generate_report", True):
            return "generate_report"
        return "skip_report"

    check_report_needed = BranchPythonOperator(
        task_id="check_report_needed",
        python_callable=should_generate_report,
        dag=dag,
    )

    generate_report = BashOperator(
        task_id="generate_report",
        bash_command=f"""
            python /opt/ade-fcm/scripts/generate_report.py \
                --results-dir {OUTPUT_DIR / "evaluation"} \
                --output-dir {REPORT_DIR} \
                --experiment-tag "{{{{ params.experiment_tag }}}}" \
                --include-figures \
                --format html \
                --format pdf
        """,
        dag=dag,
    )

    skip_report = DummyOperator(task_id="skip_report", dag=dag)

    def should_deploy_model(**context) -> str:
        if context["params"].get("deploy_model", False) and context["ti"].xcom_pull(
            task_ids="evaluate_model", key="model_performance"
        ):
            return "deploy_model"
        return "skip_deploy"

    check_deploy_needed = BranchPythonOperator(
        task_id="check_deploy_needed",
        python_callable=should_deploy_model,
        dag=dag,
    )

    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command=f"""
            python /opt/ade-fcm/scripts/deploy_model.py \
                --model-path {MODEL_DIR / "latest"} \
                --model-name ADE-FCM-{{{{ params.experiment_tag }}}} \
                --stage Production \
                --tracking-uri "{{{{ var.value.MLFLOW_TRACKING_URI }}}}"
        """,
        dag=dag,
    )

    skip_deploy = DummyOperator(task_id="skip_deploy", dag=dag)

    log_to_mlflow = BashOperator(
        task_id="log_to_mlflow",
        bash_command=f"""
            python /opt/ade-fcm/scripts/log_pipeline_metadata.py \
                --experiment-name ADE-FCM-Research \
                --dag-run-id "{{{{ run_id }}}}" \
                --params "{{{{ params | tojson }}}}" \
                --output-dir {OUTPUT_DIR / "evaluation"}
        """,
        dag=dag,
    )

    send_notification = EmailOperator(
        task_id="send_notification",
        to="research@ade-fcm.org",
        subject="ADE-FCM Research Pipeline Complete: {{ params.experiment_tag }}",
        html_content="""
            <h2>ADE-FCM Research Pipeline Complete</h2>
            <p>Experiment: {{ params.experiment_tag }}</p>
            <p>Dataset: {{ params.dataset }}</p>
            <p>Parameters: k={{ params.n_clusters }}, m={{ params.fuzzifier }}</p>
            <p>DAG Run ID: {{ run_id }}</p>
            <p>Reports: {{ REPORT_DIR }}</p>
        """,
        trigger_rule=TriggerRule.ALL_DONE,
        dag=dag,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE, dag=dag)

    start >> check_data_availability >> ingest_or_generate
    ingest_or_generate >> ingest_existing_data >> preprocess_data
    ingest_or_generate >> generate_synthetic_data >> preprocess_data
    preprocess_data >> run_ade_fcm >> evaluate_model >> check_report_needed
    check_report_needed >> generate_report >> check_deploy_needed
    check_report_needed >> skip_report >> check_deploy_needed
    check_deploy_needed >> deploy_model >> log_to_mlflow
    check_deploy_needed >> skip_deploy >> log_to_mlflow
    log_to_mlflow >> send_notification >> end
