"""
Resolve the path to a BrainOmni clone.
BrainOmni's source is not vendored in this repository. The path is read from
``brainomni_repo`` in ``configs/_base.yaml``; if the key is absent or null,
falls back to the bundled ``models/BrainOmni/BrainOmni_repo/``.
"""
import os
import sys
from pathlib import Path

import yaml


def _read_brainomni_repo_from_yaml():
    """Walk up from this file to find configs/_base.yaml and read brainomni_repo."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'configs' / '_base.yaml'
        if candidate.is_file():
            with open(candidate) as f:
                cfg = yaml.safe_load(f) or {}
            value = cfg.get('brainomni_repo')
            if value:
                # Resolve relative paths against the project root (parent of configs/).
                p = Path(os.path.expanduser(value))
                return p if p.is_absolute() else (parent / p).resolve()
            return None
    return None


def ensure_repo_on_path() -> str:
    repo_dir = _read_brainomni_repo_from_yaml()
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parent / 'BrainOmni_repo'

    repo_dir = str(repo_dir)
    marker = os.path.join(repo_dir, 'brainomni', 'model.py')
    if not os.path.isfile(marker):
        raise RuntimeError(
            f"BrainOmni repo path '{repo_dir}' does not look like a BrainOmni checkout "
            f"(missing {os.path.relpath(marker, repo_dir)}). "
            f"Set 'brainomni_repo' in configs/_base.yaml or clone "
            f"https://github.com/OpenTSLab/BrainOmni into models/BrainOmni/BrainOmni_repo."
        )

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir
