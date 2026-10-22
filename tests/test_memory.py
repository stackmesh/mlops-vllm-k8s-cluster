import pytest

from mlops_cluster.memory import estimate_memory


def test_qwen_awq_reserves_weights_and_kv():
    estimate = estimate_memory(
        params_b=7.61,
        n_layers=28,
        n_kv_heads=4,
        head_dim=128,
        max_model_len=4096,
        max_num_seqs=16,
        quant="awq",
        weight_overhead=1.15,
    )
    assert estimate.kv_gib == pytest.approx(3.5)
    assert estimate.total_gib == pytest.approx(7.575, abs=0.01)
    assert estimate.weight_with_overhead_gib < estimate.total_gib


def test_unknown_quant_is_rejected():
    with pytest.raises(ValueError):
        estimate_memory(
            params_b=1,
            n_layers=1,
            n_kv_heads=1,
            head_dim=1,
            max_model_len=8,
            max_num_seqs=1,
            quant="int3",
            weight_overhead=1.15,
        )
