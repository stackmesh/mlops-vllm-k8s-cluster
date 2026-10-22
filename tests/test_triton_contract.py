from pathlib import Path


def test_triton_config_batches_text_to_vectors():
    root = Path(__file__).resolve().parents[1] / "triton" / "model_repository" / "bge_embed"
    config = (root / "config.pbtxt").read_text(encoding="utf-8")
    assert 'name: "bge_embed"' in config
    assert "dynamic_batching" in config
    assert "TYPE_STRING" in config
    assert "384" in config
    model = (root / "1" / "model.py").read_text(encoding="utf-8")
    assert "load_torch_encoder" in model
    compile(model, "model.py", "exec")
