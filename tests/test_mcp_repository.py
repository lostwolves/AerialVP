"""Real MCP stdio protocol tests; fixture tools are explicitly not model tests."""
import asyncio
import os
import sys
import pytest
from mcp import StdioServerParameters
from aerialvp.config import DEFAULT_CONFIG
from aerialvp.repository import ToolRepository, ToolRepositoryError, ToolValidationError

FIXTURE_SERVER = '''
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
mcp = FastMCP("protocol-test-fixture", log_level="ERROR")
class Result(BaseModel):
    value: int
@mcp.tool()
def increment(value: int) -> Result:
    return Result(value=value + 1)
@mcp.tool()
def validation_failure() -> Result:
    raise ValueError("[AERIALVP_VALIDATION] malformed analysis dictionary")
@mcp.tool()
def service_failure() -> Result:
    raise RuntimeError("Provider unavailable")
if __name__ == "__main__":
    mcp.run(transport="stdio")
'''

@pytest.fixture
def protocol_server(tmp_path):
    script = tmp_path / 'fixture_server.py'
    script.write_text(FIXTURE_SERVER, encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    return StdioServerParameters(command=sys.executable, args=[str(script)], env=env)

def test_real_mcp_initialize_discover_call_and_error_types(protocol_server):
    async def exercise():
        async with ToolRepository(DEFAULT_CONFIG, servers=[protocol_server]) as repo:
            tools = await repo.list_tools()
            assert {t['name'] for t in tools} == {'increment', 'validation_failure', 'service_failure'}
            inc = next(t for t in tools if t['name'] == 'increment')
            assert 'value' in inc['inputSchema']['required']
            assert 'outputSchema' in inc
            assert await repo.call_tool('increment', {'value': 4}) == {'value': 5}
            with pytest.raises(ToolValidationError, match='input schema'):
                await repo.call_tool('increment', {'value': 'four'})
            with pytest.raises(ToolValidationError, match='Unknown registered'):
                await repo.call_tool('missing', {})
            with pytest.raises(ToolValidationError, match='malformed analysis'):
                await repo.call_tool('validation_failure', {})
            with pytest.raises(ToolRepositoryError, match='Provider unavailable'):
                await repo.call_tool('service_failure', {})
        with pytest.raises(ToolRepositoryError, match='Connect'):
            await repo.list_tools()
    asyncio.run(exercise())

def test_real_project_servers_expose_paper_tool_contracts():
    async def exercise():
        async with ToolRepository(DEFAULT_CONFIG) as repo:
            tools = {t['name']: t for t in await repo.list_tools()}
            assert len(tools) == 11
            for name in ['task_type_analysis', 'task_object_analysis', 'task_plan_analysis',
                         'spatial_position_description_yoloworld', 'semantic_description_DAM',
                         'spatial_relationship_description_SpaceOm']:
                assert name in tools and tools[name].get('outputSchema')
            assert set(tools['semantic_description_DAM']['inputSchema']['required']) == {'image_path', 'positions', 'guide_prompt'}
            assert set(tools['task_plan_analysis']['inputSchema']['required']) == {'original_prompt', 'task_type', 'task_objects'}
    asyncio.run(exercise())

def test_output_schema_rejection_is_engine_feedback():
    with pytest.raises(ToolValidationError, match='output schema'):
        ToolRepository._validate({'type': 'object', 'required': ['task_type']}, {}, 'analysis output')
