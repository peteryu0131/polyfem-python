"""Bookkeeping for the optional shape experiment runner; no FEM solves."""

import hashlib
import json
from pathlib import Path


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(config, source_files):
    """Refuse to mix runs after the meshes, settings or implementation change."""
    from polyfempy import polyfempy as backend

    config = config.as_dict()
    config.pop("output")
    files = [Path(__file__), *map(Path, source_files),
             Path(backend.__file__), *[Path(b["mesh"]) for b in config["geometry"]]]
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.resolve())] = digest.hexdigest()
    return {"configuration": config, "sha256": hashes}

