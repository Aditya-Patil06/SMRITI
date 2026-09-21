from abc import ABC, abstractmethod
from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel

class ProviderCapability(str, Enum):
    READ_CONTEXT = "READ_CONTEXT"
    SEARCH_MEMORY = "SEARCH_MEMORY"
    WRITE_MEMORY = "WRITE_MEMORY"
    IMPORT_HISTORY = "IMPORT_HISTORY"
    REAL_TIME_EVENTS = "REAL_TIME_EVENTS"
    DIRECT_HANDOFF = "DIRECT_HANDOFF"

class ProviderCapabilities(BaseModel):
    provider: str
    capabilities: Dict[str, bool]
    notes: Dict[str, str]

class BaseProviderAdapter(ABC):
    """Abstract base class for all AI provider integration adapters."""

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        pass

    @abstractmethod
    def export_context(self, context_package: Dict[str, Any]) -> str:
        """Format the portable context package according to provider-specific prompt conventions."""
        pass

class ChatGPTAdapter(BaseProviderAdapter):
    def get_provider_name(self) -> str:
        return "chatgpt"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="chatgpt",
            capabilities={
                ProviderCapability.READ_CONTEXT.value: True,
                ProviderCapability.SEARCH_MEMORY.value: True,
                ProviderCapability.WRITE_MEMORY.value: True,
                ProviderCapability.IMPORT_HISTORY.value: True,
                ProviderCapability.REAL_TIME_EVENTS.value: False,
                ProviderCapability.DIRECT_HANDOFF.value: False
            },
            notes={
                "IMPORT_HISTORY": "Supported via official OpenAI GDPR/Takeout conversations.json export bundle.",
                "DIRECT_HANDOFF": "Unsupported due to lack of public consumer session handoff API.",
                "REAL_TIME_EVENTS": "Unsupported; requires official webhook or MCP server."
            }
        )

    def export_context(self, context_package: Dict[str, Any]) -> str:
        return f"System Instruction for ChatGPT:\n{context_package.get('formatted_prompt', '')}"

class ClaudeAdapter(BaseProviderAdapter):
    def get_provider_name(self) -> str:
        return "claude"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="claude",
            capabilities={
                ProviderCapability.READ_CONTEXT.value: True,
                ProviderCapability.SEARCH_MEMORY.value: True,
                ProviderCapability.WRITE_MEMORY.value: True,
                ProviderCapability.IMPORT_HISTORY.value: True,
                ProviderCapability.REAL_TIME_EVENTS.value: False,
                ProviderCapability.DIRECT_HANDOFF.value: False
            },
            notes={
                "IMPORT_HISTORY": "Supported via official Anthropic data export JSON.",
                "DIRECT_HANDOFF": "Unsupported without private API session injection.",
                "REAL_TIME_EVENTS": "Unsupported."
            }
        )

    def export_context(self, context_package: Dict[str, Any]) -> str:
        return f"<project_context>\n{context_package.get('formatted_prompt', '')}\n</project_context>"

class GeminiAdapter(BaseProviderAdapter):
    def get_provider_name(self) -> str:
        return "gemini"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider="gemini",
            capabilities={
                ProviderCapability.READ_CONTEXT.value: True,
                ProviderCapability.SEARCH_MEMORY.value: True,
                ProviderCapability.WRITE_MEMORY.value: True,
                ProviderCapability.IMPORT_HISTORY.value: True,
                ProviderCapability.REAL_TIME_EVENTS.value: False,
                ProviderCapability.DIRECT_HANDOFF.value: False
            },
            notes={
                "IMPORT_HISTORY": "Supported via Google Takeout / Gemini JSON dumps.",
                "DIRECT_HANDOFF": "Unsupported for consumer web Gemini.",
                "REAL_TIME_EVENTS": "Unsupported."
            }
        )

    def export_context(self, context_package: Dict[str, Any]) -> str:
        return f"[Gemini Project System Context]\n{context_package.get('formatted_prompt', '')}"

class ProviderAdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, BaseProviderAdapter] = {
            "chatgpt": ChatGPTAdapter(),
            "claude": ClaudeAdapter(),
            "gemini": GeminiAdapter()
        }

    def get_adapter(self, provider: str) -> BaseProviderAdapter:
        return self._adapters.get(provider.lower(), self._adapters["chatgpt"])

    def list_all_capabilities(self) -> List[ProviderCapabilities]:
        return [adapter.get_capabilities() for adapter in self._adapters.values()]

provider_registry = ProviderAdapterRegistry()
