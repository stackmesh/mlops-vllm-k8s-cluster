import yaml

from mlops_cluster.catalog import Budget, Cluster, NodePool, load_cluster, load_models
from mlops_cluster.planner import plan_deployment
from mlops_cluster.render import host_memory_gi, render_plan


def test_mig_deployment_requests_the_slice():
    models = load_models()
    plan = plan_deployment(models["qwen2.5-7b-instruct"], "awq", 2, load_cluster())
    rendered = render_plan(plan)
    deployment = yaml.safe_load(rendered["deployment.yaml"])
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/mig-2g.10gb"] == "1"
    assert deployment["spec"]["replicas"] == 2
    assert container["args"][container["args"].index("--model") + 1] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert "kustomization.yaml" in rendered
    assert "statefulset.yaml" not in rendered


def test_pipeline_parallel_renders_a_statefulset():
    models = load_models()
    cluster = Cluster(
        name="split",
        budget=Budget(2, 80, 2, 0.9, 1.15, 4),
        pools=(
            NodePool("node-a", "a100-40gb", 1, 1, True),
            NodePool("node-b", "a100-40gb", 1, 1, True),
        ),
        images={"vllm": "vllm/vllm-openai:v0.32.0"},
    )
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "awq", 1, cluster)
    rendered = render_plan(plan)
    stateful = yaml.safe_load(rendered["statefulset.yaml"])
    service = yaml.safe_load(rendered["service.yaml"])
    assert stateful["spec"]["replicas"] == 2
    assert stateful["spec"]["template"]["spec"]["affinity"]["podAntiAffinity"]
    assert service["spec"]["selector"]["statefulset.kubernetes.io/pod-name"].endswith("-0")
    assert host_memory_gi(plan) <= 64


def test_time_slice_config_matches_the_plan():
    models = load_models()
    cluster = Cluster(
        name="consumer",
        budget=Budget(2, 48, 8, 0.9, 1.15, 4),
        pools=(NodePool("node-a", "rtx-4090-24gb", 1, 2, False),),
        images={},
    )
    plan = plan_deployment(models["tinyllama-1.1b-chat"], "awq", 4, cluster)
    rendered = render_plan(plan)
    config = yaml.safe_load(rendered["time-slicing-config.yaml"])
    assert "replicas: 4" in config["data"]["config.yaml"]
    assert config["metadata"]["namespace"] == "kube-system"


def test_checked_in_examples_match_the_renderer():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    models = load_models()
    lab = load_cluster()
    cases = (
        ("qwen2.5-7b-instruct", "awq", 2, "qwen2.5-7b-awq"),
        ("llama-3.1-8b-instruct", "awq", 1, "llama-3.1-8b-awq"),
        ("llama-3.1-70b-instruct", "awq", 1, "llama-3.1-70b-awq"),
    )
    for model, quant, replicas, dirname in cases:
        plan = plan_deployment(models[model], quant, replicas, lab)
        rendered = render_plan(plan)
        folder = root / "k8s" / "examples" / dirname
        for name, body in rendered.items():
            assert (folder / name).read_text(encoding="utf-8") == body
        assert "&id" not in (folder / "deployment.yaml").read_text(encoding="utf-8")


def test_rejected_plan_is_not_rendered():
    models = load_models()
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "fp16", 1, load_cluster())
    with __import__("pytest").raises(ValueError):
        render_plan(plan)
