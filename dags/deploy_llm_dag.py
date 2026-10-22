"""Schedule a budgeted vLLM rollout.

Install this repo in the Airflow environment (`pip install -e .`) and point the
DagBag at this directory. The same steps run without Airflow:

    python -m mlops_cluster.cli deploy --model qwen2.5-7b-instruct --quant awq --replicas 2
"""

from __future__ import annotations


def run_deploy() -> dict:
    from mlops_cluster.pipeline import deploy_model

    return deploy_model("qwen2.5-7b-instruct", quant="awq", replicas=2, apply=False)


try:
    from datetime import datetime

    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow is optional for local planner runs and CI.
    DAG = None

if DAG is not None:
    with DAG(
        dag_id="deploy_llm_under_budget",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlops", "vllm", "budget"],
    ) as dag:
        PythonOperator(task_id="plan_register_render", python_callable=run_deploy)
