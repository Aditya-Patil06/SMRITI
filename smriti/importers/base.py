from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class NormalizedMessage(BaseModel):
    role: str
    content: str
    sequence_index: float = 0.0
    external_id: Optional[str] = None
    created_at: Optional[datetime] = None
    raw_metadata: Dict[str, Any] = {}

class NormalizedConversation(BaseModel):
    title: str
    external_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: List[NormalizedMessage] = []
    raw_metadata: Dict[str, Any] = {}

class BaseImporter(ABC):
    """Abstract base class for all conversation importers."""
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the unique provider name (e.g. chatgpt, claude, gemini, generic)."""
        pass

    @abstractmethod
    def parse(self, raw_data: Any) -> List[NormalizedConversation]:
        """Parse raw input export format into canonical NormalizedConversation objects."""
        pass
