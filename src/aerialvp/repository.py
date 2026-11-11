"""Real MCP discovery, schema validation and invocation for AerialVP."""
from __future__ import annotations
from contextlib import AsyncExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any
from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import load_config


class ToolRepositoryError(RuntimeError):
    """Infrastructure or remote tool execution failure."""


class ToolValidationError(ValueError):
    """Tool contract failure suitable for Task Engine regeneration."""


class ToolRepository:
    def __init__(self, config_path: str | Path, *, servers: list[StdioServerParameters] | None = None):
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.config = load_config(self.config_path)
        self._server_parameters = servers
        self._stack = None
        self._tools: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, ClientSession] = {}

    def _default_servers(self):
        env = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        python = self.config.get('mcp', {}).get('python') or sys.executable
        return [StdioServerParameters(command=python,
            args=['-m', 'aerialvp.server', '--config', self.config_path, '--group', group],
            env=env, cwd=str(Path(__file__).resolve().parents[2])) for group in ('analysis', 'vision')]

    async def __aenter__(self):
        if self._stack is not None:
            raise ToolRepositoryError('ToolRepository is already connected')
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for parameters in self._server_parameters or self._default_servers():
                read, write = await self._stack.enter_async_context(stdio_client(parameters))
                timeout = timedelta(seconds=self.config.get('mcp', {}).get('timeout', 1800))
                session = await self._stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=timeout))
                await session.initialize()
                cursor = None
                while True:
                    response = await session.list_tools(cursor=cursor)
                    for tool in response.tools:
                        if tool.name in self._tools:
                            raise ToolRepositoryError(f'Duplicate MCP tool name: {tool.name}')
                        metadata = tool.model_dump(mode='json', exclude_none=True)
                        self._tools[tool.name] = metadata
                        self._sessions[tool.name] = session
                        Draft202012Validator.check_schema(metadata['inputSchema'])
                        if metadata.get('outputSchema'):
                            Draft202012Validator.check_schema(metadata['outputSchema'])
                    cursor = response.nextCursor
                    if not cursor:
                        break
            return self
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise

    async def __aexit__(self, exc_type, exc, tb):
        stack, self._stack = self._stack, None
        self._tools.clear()
        self._sessions.clear()
        if stack is not None:
            return await stack.__aexit__(exc_type, exc, tb)
        return None

    async def list_tools(self):
        if self._stack is None:
            raise ToolRepositoryError('Connect ToolRepository before discovering tools')
        return json.loads(json.dumps(list(self._tools.values())))

    @staticmethod
    def _validate(schema, value, label):
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
        if errors:
            details = '; '.join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:8])
            raise ToolValidationError(f'{label} schema validation failed: {details}')

    async def call_tool(self, name: str, arguments: dict):
        if self._stack is None:
            raise ToolRepositoryError('Connect ToolRepository before invoking a tool')
        if name not in self._tools:
            raise ToolValidationError(f'Unknown registered MCP tool: {name}')
        metadata = self._tools[name]
        self._validate(metadata['inputSchema'], arguments, f'{name} input')
        response = await self._sessions[name].call_tool(name, arguments=arguments)
        text = '\n'.join(c.text for c in response.content if getattr(c, 'type', None) == 'text')
        if response.isError:
            if '[AERIALVP_VALIDATION]' in text:
                raise ToolValidationError(f'MCP tool {name}: {text}')
            raise ToolRepositoryError(f'MCP tool {name} failed: {text or "Unspecified server error"}')
        output = getattr(response, 'structuredContent', None)
        if output is None:
            try:
                output = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ToolValidationError(f'MCP tool {name} returned no JSON result') from exc
        if not isinstance(output, dict):
            raise ToolValidationError(f'MCP tool {name} returned a non-object result')
        if metadata.get('outputSchema'):
            self._validate(metadata['outputSchema'], output, f'{name} output')
        return output

