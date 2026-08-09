"""
Safe persistence utilities for models and experiment results.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib


def ensure_parent_directory(path: Path) -> None:
    """
    Create the parent directory of a path if it does not already exist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def save_json_atomic(data: dict[str, Any], path: Path) -> None:
    """
    Save JSON atomically.

    The result is first written to a temporary file and then moved to the
    destination. This prevents partially written metrics files if the process
    is interrupted.
    """

    ensure_parent_directory(path)

    temporary_file = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )

            file.flush()
            os.fsync(file.fileno())

            temporary_file = Path(file.name)

        os.replace(temporary_file, path)

    except Exception:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()

        raise


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON artifact.
    """

    if not path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


