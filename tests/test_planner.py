import pytest

from mlops_cluster.catalog import Budget, Cluster, NodePool, load_cluster, load_models
from mlops_cluster.planner import candidates, plan_deployment


@pytest.fixture
def models():
    return load_models()


@pytest.fixture
def lab():
    return load_cluster()


def _cluster(name, budget, pools):
    return Cluster(name=name, budget=budget, pools=tuple(pools), images={})


def test_lab_budget_is_two_gpus(lab):
    assert lab.budget.max_physical_gpus == 2
    assert lab.budget.max_vram_gib == 80
    assert sum(pool.physical_gpus for pool in lab.pools) == 4


def test_qwen_awq_prefers_narrower_slice_with_more_sms(models, lab):
    plan = plan_deployment(models["qwen2.5-7b-instruct"], "awq", 2, lab)
    assert plan.accepted
    assert plan.strategy == "mig"
    assert plan.mig_profile == "2g.10gb"
    assert plan.physical_gpus == 1
    assert plan.reserved_vram_gib == pytest.approx(19.24)
    assert plan.pools == ("node-a",)
    assert plan.repo_id == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert plan.repo_matches_quant
    assert "--quantization" in plan.vllm_args
    profiles = {item.mig_profile for item in candidates(models["qwen2.5-7b-instruct"], "awq", 2, lab)}
    assert "1g.10gb" in profiles
    assert plan.mig_profile != "1g.10gb"


def test_qwen_fp16_takes_a_full_device(models, lab):
    plan = plan_deployment(models["qwen2.5-7b-instruct"], "fp16", 1, lab)
    assert plan.strategy == "dedicated"
    assert plan.mig_profile is None
    assert plan.physical_gpus == 1
    assert plan.reserved_vram_gib == pytest.approx(40)


def test_llama8_awq_single_replica_buys_more_sms(models, lab):
    plan = plan_deployment(models["llama-3.1-8b-instruct"], "awq", 1, lab)
    assert plan.mig_profile == "4g.20gb"
    assert plan.physical_gpus == 1
    assert plan.repo_matches_quant is False


def test_llama8_awq_two_replicas_pack_onto_one_gpu(models, lab):
    plan = plan_deployment(models["llama-3.1-8b-instruct"], "awq", 2, lab)
    assert plan.mig_profile == "3g.20gb"
    assert plan.physical_gpus == 1
    assert plan.reserved_vram_gib == pytest.approx(39.0)


def test_70b_awq_stays_on_one_node(models, lab):
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "awq", 1, lab)
    assert plan.accepted
    assert plan.strategy == "dedicated"
    assert plan.tensor_parallel == 2
    assert plan.pipeline_parallel == 1
    assert plan.cross_node is False
    assert plan.physical_gpus == 2
    assert plan.pools == ("node-a",)


def test_70b_fp16_is_rejected_and_points_at_awq(models, lab):
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "fp16", 1, lab)
    assert plan.accepted is False
    assert plan.hint is not None
    assert "awq" in plan.hint


def test_second_70b_replica_does_not_fit(models, lab):
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "awq", 2, lab)
    assert plan.accepted is False
    assert "replicas=1" in (plan.hint or "")


def test_70b_pipelines_across_single_gpu_nodes(models):
    cluster = _cluster(
        "split",
        Budget(2, 80, 2, 0.9, 1.15, 4),
        [
            NodePool("node-a", "a100-40gb", 1, 1, True),
            NodePool("node-b", "a100-40gb", 1, 1, True),
        ],
    )
    plan = plan_deployment(models["llama-3.1-70b-instruct"], "awq", 1, cluster)
    assert plan.accepted
    assert plan.tensor_parallel == 1
    assert plan.pipeline_parallel == 2
    assert plan.cross_node is True
    assert plan.pools == ("node-a", "node-b")


def test_time_slice_caps_utilization_on_a_4090(models):
    cluster = _cluster(
        "consumer",
        Budget(2, 48, 8, 0.9, 1.15, 4),
        [NodePool("node-a", "rtx-4090-24gb", 1, 2, False)],
    )
    plan = plan_deployment(models["tinyllama-1.1b-chat"], "awq", 4, cluster)
    assert plan.strategy == "time_slice"
    assert plan.physical_gpus == 1
    assert plan.time_slice_per_gpu == 4
    assert plan.gpu_memory_utilization == pytest.approx(0.225)
    assert plan.gpu_resource == "nvidia.com/gpu"


def test_tinyllama_uses_the_smallest_mig_slice(models, lab):
    plan = plan_deployment(models["tinyllama-1.1b-chat"], "awq", 2, lab)
    assert plan.mig_profile == "1g.5gb"
    assert plan.physical_gpus == 1


def test_fp16_8b_misses_a_24gb_card_and_awq_fits(models):
    cluster = _cluster(
        "one-4090",
        Budget(1, 24, 2, 0.9, 1.15, 4),
        [NodePool("node-a", "rtx-4090-24gb", 1, 1, False)],
    )
    rejected = plan_deployment(models["llama-3.1-8b-instruct"], "fp16", 1, cluster)
    assert rejected.accepted is False
    assert "awq" in (rejected.hint or "")
    accepted = plan_deployment(models["llama-3.1-8b-instruct"], "awq", 1, cluster)
    assert accepted.strategy == "dedicated"
    assert accepted.physical_gpus == 1


def test_replicas_must_be_positive(models, lab):
    with pytest.raises(ValueError):
        plan_deployment(models["qwen2.5-7b-instruct"], "awq", 0, lab)
