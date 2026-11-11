import json
import pytest

from aerialvp.engine import TaskEngine


class ScriptedClient:
    model = "test-fixture"
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
    def chat(self, question, system_prompt=None):
        self.requests.append(json.loads(question))
        return next(self.replies)


@pytest.mark.asyncio
async def test_invalid_instruction_is_fed_back_and_not_executed(tmp_path):
    client = ScriptedClient(['{"tool":"wrong"}', '{"tool":"correct"}'])
    engine = TaskEngine(client=client)
    engine.run_dir = tmp_path
    executed = []
    def validate(value):
        if value["tool"] != "correct":
            raise ValueError("Unknown tool; use correct")
    async def execute(value):
        executed.append(value["tool"])
        return {"ok": True}
    result = await engine._validated_decision("test", "system", {"original_prompt": "test task"}, validate, execute)
    assert result == {"ok": True}
    assert executed == ["correct"]
    assert len(engine.history) == 1
    assert client.requests[0]["validation_feedback"] == []
    assert "Unknown tool" in client.requests[1]["validation_feedback"][0]["error"]
    assert client.requests[1]["execution_history"] == []
    trace = json.loads((tmp_path / "engine_trace.json").read_text())
    assert [event["status"] for event in trace] == ["rejected", "accepted"]


@pytest.mark.asyncio
async def test_retry_exhaustion_never_accepts_invalid_instruction(tmp_path):
    client = ScriptedClient(['not JSON', '{}'])
    engine = TaskEngine(client=client)
    engine.run_dir = tmp_path
    engine.config["engine"]["max_validation_attempts"] = 2
    def validate(value):
        raise ValueError("missing required arguments")
    with pytest.raises(RuntimeError, match="no valid instruction"):
        await engine._validated_decision("test", "system", {}, validate)
    assert engine.history == []
    assert all(event["status"] == "rejected" for event in engine.trace)


@pytest.mark.asyncio
async def test_successful_output_becomes_next_stage_history(tmp_path):
    client = ScriptedClient(['{"step":1}', '{"step":2}'])
    engine = TaskEngine(client=client)
    engine.run_dir = tmp_path
    await engine._validated_decision("one", "system", {}, lambda x: None)
    await engine._validated_decision("two", "system", {}, lambda x: None)
    assert client.requests[1]["execution_history"][0]["output"] == {"step": 1}

