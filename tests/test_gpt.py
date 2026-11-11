from types import SimpleNamespace

import pytest

from aerialvp.gpt import GPTClient, parse_object


def test_chat_completions_request_format(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setenv("TEST_AERIALVP_KEY", "local-test-key")
    def post(url, **kwargs):
        captured.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "OK"}}]})
    monkeypatch.setattr("aerialvp.gpt.requests.post", post)
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture-only")
    config = {"api_key_env": "TEST_AERIALVP_KEY", "url": "https://example.invalid/v1/chat/completions", "max_retries": 1}
    client = GPTClient(config, model="gpt-4.1")
    assert client.chat("question", "system", [str(image)]) == "OK"
    payload = captured[0][1]["json"]
    assert payload["model"] == "gpt-4.1"
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "system"}]
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured[0][1]["headers"]["Authorization"] == "Bearer local-test-key"


def test_parser_accepts_paper_dictionary_not_executable_code():
    assert parse_object("{'task_type': 'Visual Grounding'}") == {"task_type": "Visual Grounding"}
    with pytest.raises(ValueError):
        parse_object("__import__('os').system('echo no')")

