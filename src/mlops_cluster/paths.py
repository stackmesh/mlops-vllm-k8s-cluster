from __future__ import annotations

import os
from pathlib import Path


def config_dir() -> Path:
    """Directory that holds cluster.yaml and models.yaml."""
    override = os.environ.get("MLOPS_CONFIG_DIR")
    candidates = []
    if override:
        candidates.append(Path(override))
    packaged = Path(__file__).resolve().parent / "data"
    candidates.append(packaged)
    repo = Path(__file__).resolve().parents[2] / "configs"
    candidates.append(repo)
    candidates.append(Path.cwd() / "configs")
    for path in candidates:
        if (path / "cluster.yaml").is_file() and (path / "models.yaml").is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cluster.yaml not found. Looked in: {searched}")
