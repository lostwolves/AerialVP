import json
from pathlib import Path
import sys

from aerialvp.config import load_config, DEFAULT_CONFIG, PROJECT_ROOT


def test_default_configuration_uses_active_interpreter_and_relative_models(monkeypatch):
    monkeypatch.delenv("AERIALVP_CHAT_COMPLETIONS_URL", raising=False)
    config = load_config()
    assert Path(config["config_path"]) == DEFAULT_CONFIG
    assert DEFAULT_CONFIG.name == "default.json"
    assert config["mcp"]["python"] == str(Path(sys.executable))
    for model in config["models"].values():
        assert Path(model["checkpoint"]).is_relative_to(PROJECT_ROOT / "models")
        assert model["python"] == config["mcp"]["python"]


def test_nested_override_preserves_other_model_options(tmp_path):
    base = tmp_path / "base.json"
    overrides = tmp_path / "custom.json"
    base.write_text(json.dumps({
        "api": {"url": "https://example.invalid/v1/chat/completions"},
        "mcp": {"python": None},
        "models": {"example": {"checkpoint": "models/example", "device": "cuda:0", "max_new_tokens": 256}},
    }), encoding="utf-8")
    overrides.write_text(json.dumps({
        "extends": "base.json",
        "models": {"example": {"device": "cpu"}},
    }), encoding="utf-8")
    config = load_config(overrides)
    assert config["models"]["example"]["device"] == "cpu"
    assert config["models"]["example"]["max_new_tokens"] == 256
    assert Path(config["models"]["example"]["checkpoint"]).name == "example"
