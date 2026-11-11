"""Project-local configuration; no keys are embedded in source or run manifests."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = _read_config(config_path)
    config["config_path"] = str(config_path)
    config["project_root"] = str(PROJECT_ROOT)
    config["api"]["url"] = os.getenv("AERIALVP_CHAT_COMPLETIONS_URL", config["api"]["url"])
    config["mcp"]["python"] = _absolute(config["mcp"].get("python") or sys.executable)
    for settings in config["models"].values():
        for key in ("checkpoint", "sam_checkpoint", "source_path", "model_config", "clip_checkpoint", "clip_source_path"):
            if settings.get(key):
                settings[key] = _absolute(settings[key])
        settings["python"] = _absolute(settings.get("python") or config["mcp"]["python"])
    return config


def _read_config(path: Path, seen: frozenset = frozenset()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError("Circular config inheritance")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    parent = config.pop("extends", None)
    if parent:
        base = _read_config(path.parent / parent, seen | {path})
        config = _merge(base, config)
    return config


def _merge(base: dict, overrides: dict) -> dict:
    """Merge nested model settings without dropping unspecified options."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _absolute(value: str) -> str:
    path = Path(os.path.expandvars(value)).expanduser()
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())

