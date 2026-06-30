from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.models import Variable, Param
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

ENV = os.environ
BENCHMARK_DIR = Path(ENV.get("BENCHMARK_DIR", "/data/benchmarks"))
RESULTS_DIR = BENCHMARK_DIR / "results"
REPORTS_DIR = BENCHMARK_DIR / "reports"

ALGORITHMS = ["ade-fcm", "fcm", "kmeans", "dbscan", "gaussian-mixture"]
DATASETS = ["synthetic", "adverse-effects-small", "adverse-effects-large", "clinical-trials"]
METRICS = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score",
           "runtime_seconds", "n_iterations", "convergence_rate"]

default_args = {
    "owner": "ade-fcm-team",
    "depends_on_past": False,
    "email": ["benchmarks@ade-fcm.org"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(hours=8),
}

with DAG(
    dag_id="ade_fcm_benchmark_pipeline",
    default_args=default_args,
    description="Automated benchmark suite for ADE-FCM vs baseline algorithms",
    schedule_interval="0 4 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ade-fcm", "benchmark", "comparison"],
    params={
        "algorithms": ALGORITHMS,
        "datasets": DATASETS,
        "n_trials": 5,
        "ab_testing": True,
        "statistical_significance": 0.05,
    },
    max_active_runs=1,
    doc_md="""
    # ADE-FCM Benchmark Pipeline
    Runs automated benchmarks comparing ADE-FCM against baseline clustering algorithms.
    Supports A/B testing with statistical significance validation.
    """,
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    setup_benchmark_env = BashOperator(
        task_id="setup_benchmark_env",
        bash_command=f"""
            mkdir -p {RESULTS_DIR} {REPORTS_DIR}
            echo "Benchmark directories initialized"
            echo "Algorithms: ${{{{ params.algorithms | join(', ') }}}}"
            echo "Datasets: ${{{{ params.datasets | join(', ') }}}}"
            echo "Trials: ${{{{ params.n_trials }}}}"
        """,
        dag=dag,
    )

    def generate_benchmark_matrix(**context) -> str:
        params = context["params"]
        algorithms = params.get("algorithms", ["ade-fcm"])
        datasets = params.get("datasets", ["synthetic"])
        n_trials = params.get("n_trials", 3)

        matrix = []
        for algo in algorithms:
            for dataset in datasets:
                for trial in range(1, n_trials + 1):
                    matrix.append({
                        "algorithm": algo,
                        "dataset": dataset,
                        "trial": trial,
                    })

        context["ti"].xcom_push(key="benchmark_matrix", value=json.dumps(matrix))
        context["ti"].xcom_push(key="n_runs", value=len(matrix))

        benchmark_config_path = BENCHMARK_DIR / "configs" / f"benchmark_{context['ds']}.json"
        benchmark_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_config_path, "w") as f:
            json.dump({"runs": matrix, "metrics": METRICS, "ab_testing": params.get("ab_testing", True)}, f)

        return f"Generated {len(matrix)} benchmark configurations"

    generate_matrix = PythonOperator(
        task_id="generate_benchmark_matrix",
        python_callable=generate_benchmark_matrix,
        dag=dag,
    )

    run_benchmarks = BashOperator(
        task_id="run_all_benchmarks",
        bash_command=f"""
            python /opt/ade-fcm/scripts/run_benchmarks.py \
                --config {BENCHMARK_DIR / "configs" / "benchmark_{{ ds }}.json"} \
                --results-dir {RESULTS_DIR} \
                --spark-master spark://spark-master:7077 \
                --parallel \
                --timeout 3600
        """,
        dag=dag,
    )

    compute_statistics = BashOperator(
        task_id="compute_statistics",
        bash_command=f"""
            python /opt/ade-fcm/scripts/benchmark_statistics.py \
                --results-dir {RESULTS_DIR} \
                --output {REPORTS_DIR / "statistics_{{ ds }}.json"} \
                --metrics {" ".join(METRICS)} \
                --alpha {{{{ params.statistical_significance }}}}
        """,
        dag=dag,
    )

    def perform_ab_testing(**context) -> str:
        if not context["params"].get("ab_testing", True):
            return "skip_ab_testing"

        stats_path = REPORTS_DIR / f"statistics_{context['ds']}.json"
        if stats_path.exists():
            return "run_ab_testing"
        return "skip_ab_testing"

    check_ab_testing = BranchPythonOperator(
        task_id="check_ab_testing",
        python_callable=perform_ab_testing,
        dag=dag,
    )

    run_ab_testing = BashOperator(
        task_id="run_ab_testing",
        bash_command=f"""
            python /opt/ade-fcm/scripts/ab_testing.py \
                --results-dir {RESULTS_DIR} \
                --output {REPORTS_DIR / "ab_testing_results_{{ ds }}.json"} \
                --control ade-fcm \
                --treatment {" ".join(a for a in ALGORITHMS if a != "ade-fcm")} \
                --metrics {" ".join(METRICS)} \
                --alpha {{{{ params.statistical_significance }}}} \
                --correction bonferroni
        """,
        dag=dag,
    )

    skip_ab_testing = DummyOperator(task_id="skip_ab_testing", dag=dag)

    generate_benchmark_report = BashOperator(
        task_id="generate_benchmark_report",
        bash_command=f"""
            python /opt/ade-fcm/scripts/generate_benchmark_report.py \
                --results-dir {RESULTS_DIR} \
                --statistics {REPORTS_DIR / "statistics_{{ ds }}.json"} \
                --ab-testing {REPORTS_DIR / "ab_testing_results_{{ ds }}.json"} \
                --output-dir {REPORTS_DIR} \
                --title "ADE-FCM Benchmark Report - {{ ds }}" \
                --format html \
                --format pdf \
                --include-figures
        """,
        dag=dag,
    )

    log_benchmark_to_mlflow = BashOperator(
        task_id="log_benchmark_to_mlflow",
        bash_command=f"""
            python /opt/ade-fcm/scripts/log_benchmark_mlflow.py \
                --results-dir {RESULTS_DIR} \
                --experiment-name ADE-FCM-Benchmarks \
                --tracking-uri "{{{{ var.value.MLFLOW_TRACKING_URI }}}}" \
                --dag-run-id "{{{{ run_id }}}}"
        """,
        dag=dag,
    )

    def compare_against_baseline(**context) -> None:
        import json
        from pathlib import Path

        ab_path = REPORTS_DIR / f"ab_testing_results_{context['ds']}.json"
        if not ab_path.exists():
            print("No A/B testing results found, skipping comparison")
            return

        with open(ab_path) as f:
            results = json.load(f)

        ade_fcm_wins = 0
        ade_fcm_losses = 0
        tie = 0

        for comparison in results.get("comparisons", []):
            if comparison.get("significant", False):
                if comparison.get("effect_size", 0) > 0:
                    ade_fcm_wins += 1
                else:
                    ade_fcm_losses += 1
            else:
                tie += 1

        summary = (
            f"ADE-FCM vs Baselines:\n"
            f"  Wins: {ade_fcm_wins}\n"
            f"  Losses: {ade_fcm_losses}\n"
            f"  Ties: {tie}\n"
            f"  Total comparisons: {ade_fcm_wins + ade_fcm_losses + tie}"
        )
        print(summary)

        context["ti"].xcom_push(key="comparison_summary", value=summary)

    compare_results = PythonOperator(
        task_id="compare_against_baseline",
        python_callable=compare_against_baseline,
        dag=dag,
    )

    def select_best_algorithm(**context) -> str:
        summary = context["ti"].xcom_pull(task_ids="compare_against_baseline", key="comparison_summary")
        if summary and "Wins:" in summary:
            return "archive_results"
        return "archive_results"

    select_best = BranchPythonOperator(
        task_id="select_best_algorithm",
        python_callable=select_best_algorithm,
        dag=dag,
    )

    archive_results = BashOperator(
        task_id="archive_results",
        bash_command=f"""
            archive_name="benchmark_{{ ds }}_{{ ts_nodash }}.tar.gz"
            tar -czf {BENCHMARK_DIR / "archives" / "$archive_name"} \
                -C {RESULTS_DIR} . \
                -C {REPORTS_DIR} .
            echo "Archived to $archive_name"
        """,
        dag=dag,
    )

    publish_dashboard_update = BashOperator(
        task_id="publish_dashboard_update",
        bash_command=f"""
            python /opt/ade-fcm/scripts/update_benchmark_dashboard.py \
                --results-dir {RESULTS_DIR} \
                --output {BENCHMARK_DIR / "dashboard" / "latest.json"} \
                --grafana-url "{{{{ var.value.GRAFANA_URL }}}}" \
                --dashboard-uid ade-fcm-benchmarks
        """,
        dag=dag,
    )

    send_benchmark_summary = BashOperator(
        task_id="send_benchmark_summary",
        bash_command=f"""
            python /opt/ade-fcm/scripts/send_benchmark_notification.py \
                --results-dir {RESULTS_DIR} \
                --reports-dir {REPORTS_DIR} \
                --channel benchmarks \
                --webhook-url "{{{{ var.value.SLACK_WEBHOOK_URL }}}}"
        """,
        dag=dag,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE, dag=dag)

    start >> setup_benchmark_env >> generate_matrix >> run_benchmarks >> compute_statistics
    compute_statistics >> check_ab_testing
    check_ab_testing >> run_ab_testing >> generate_benchmark_report
    check_ab_testing >> skip_ab_testing >> generate_benchmark_report
    generate_benchmark_report >> log_benchmark_to_mlflow >> compare_results >> select_best
    select_best >> archive_results >> publish_dashboard_update >> send_benchmark_summary >> end
