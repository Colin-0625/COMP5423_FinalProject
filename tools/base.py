"""Tool base class and registry with OpenAI Function Calling schema export."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from agent.types import ToolResult


class Tool(ABC):
    name: str = ""
    description: str = ""
    # Compact human-readable parameter summary
    parameters: Dict[str, str] = {}
    # JSON Schema object for native OpenAI function calling
    json_schema: Dict[str, Any] = {}

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        ...

    def spec(self) -> str:
        params = ", ".join(f'"{k}": <{v}>' for k, v in self.parameters.items())
        return f"- {self.name}({{{params}}}): {self.description}"

    def openai_schema(self) -> Dict[str, Any]:
        if self.json_schema:
            params_schema = self.json_schema
        else:
            props = {k: {"type": "string", "description": v} for k, v in self.parameters.items()}
            params_schema = {"type": "object", "properties": props, "required": list(props.keys())}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params_schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        assert tool.name and tool.name not in self._tools, f"bad/duplicate tool {tool.name!r}"
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> str:
        return "\n".join(t.spec() for t in self._tools.values())

    def openai_tools(self) -> List[Dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(f"Error: unknown tool {name!r}. Available: {self.names()}", error=True)
        try:
            return tool.run(**(args or {}))
        except TypeError as e:
            return ToolResult(f"Error: bad arguments for {name}: {e}", error=True)
        except Exception as e:
            return ToolResult(f"Error: {name} failed: {type(e).__name__}: {e}", error=True)
