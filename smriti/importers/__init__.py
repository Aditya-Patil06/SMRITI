from typing import Dict, Any, List
from smriti.importers.base import BaseImporter, NormalizedConversation
from smriti.importers.chatgpt import ChatGPTImporter
from smriti.importers.claude import ClaudeImporter
from smriti.importers.gemini import GeminiImporter
from smriti.importers.generic import GenericImporter

class ImporterRegistry:
    def __init__(self):
        self._importers: Dict[str, BaseImporter] = {
            "chatgpt": ChatGPTImporter(),
            "claude": ClaudeImporter(),
            "gemini": GeminiImporter(),
            "generic": GenericImporter()
        }

    def get_importer(self, provider: str) -> BaseImporter:
        provider_clean = provider.lower().strip()
        if provider_clean in self._importers:
            return self._importers[provider_clean]
        return self._importers["generic"]

    def import_conversations(self, provider: str, raw_data: Any) -> List[NormalizedConversation]:
        importer = self.get_importer(provider)
        return importer.parse(raw_data)

registry = ImporterRegistry()
