"""YAML config loader. `extends:` is a path relative to the YAML file.
Child values shallow-override parent values; lists are replaced."""
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = {**load_config(path.parent / parent), **cfg}
    return cfg
