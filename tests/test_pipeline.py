import importlib.util
from pathlib import Path

import yaml

from mlops_cluster.cli import main
from mlops_cluster.pipeline import deploy_model


def test_deploy_writes_manifests_and_a_run(tmp_path):
    report = deploy_model(
        "qwen2.5-7b-instruct",
        quant="awq",
        replicas=2,
        out_dir=tmp_path,
    )
    assert report["accepted"]
    deployment = yaml.safe_load((tmp_path / "manifests" / "deployment.yaml").read_text(encoding="utf-8"))
    assert deployment["kind"] == "Deployment"
    assert list((tmp_path / "runs").glob("*.json"))


def test_cli_plan_exits_zero_for_an_accepted_model(capsys):
    code = main(["plan", "--model", "qwen2.5-7b-instruct", "--quant", "awq", "--replicas", "2"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "2g.10gb" in captured
    assert "accepted         yes" in captured


def test_cli_rejects_an_unknown_model():
    assert main(["plan", "--model", "not-a-model", "--quant", "awq"]) == 2


def test_module_entrypoint_prints_a_plan():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "mlops_cluster.cli", "plan", "--model", "tinyllama-1.1b-chat", "--quant", "awq"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "1g.5gb" in completed.stdout


def test_airflow_callable_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MLOPS_OUT_DIR", str(tmp_path))
    path = Path(__file__).resolve().parents[1] / "dags" / "deploy_llm_dag.py"
    spec = importlib.util.spec_from_file_location("deploy_llm_dag", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_deploy()
    assert report["accepted"] is True


def test_static_platform_manifests_parse():
    root = Path(__file__).resolve().parents[1] / "k8s"
    documents = []
    for path in sorted(root.rglob("*.yaml")):
        if path.name == "kustomization.yaml":
            continue
        documents.extend(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    kinds = {document["kind"] for document in documents if document}
    assert "ResourceQuota" in kinds
    assert "Deployment" in kinds
    quota = next(document for document in documents if document and document["kind"] == "ResourceQuota")
    assert quota["spec"]["hard"]["requests.nvidia.com/gpu"] == "2"
