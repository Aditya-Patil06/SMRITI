import json
from datetime import datetime, timezone
from typing import List, Any
from dateutil import parser
from smriti.importers.base import BaseImporter, NormalizedConversation, NormalizedMessage

class GeminiImporter(BaseImporter):
    """Importer for Google Gemini exports (Google Takeout / JSON archives)."""

    def get_provider_name(self) -> str:
        return "gemini"

    def parse(self, raw_data: Any) -> List[NormalizedConversation]:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        elif not isinstance(raw_data, list):
            return []

        results: List[NormalizedConversation] = []
        for item in raw_data:
            title = item.get("title") or "Gemini Session"
            external_id = item.get("id") or item.get("sessionId")
            
            created_at = None
            if item.get("created_at") or item.get("createTime"):
                try:
                    created_at = parser.parse(item.get("created_at") or item.get("createTime"))
                except Exception:
                    pass

            raw_messages = item.get("messages") or item.get("turns") or []
            messages: List[NormalizedMessage] = []
            seq = 0.0

            for msg in raw_messages:
                author = msg.get("author") or msg.get("role") or "model"
                role = "user" if author in ["user", "human"] else "assistant"
                
                content = msg.get("content") or msg.get("text") or ""
                if isinstance(content, list):
                    content = " ".join([str(c) for c in content])

                if not str(content).strip():
                    continue

                msg_dt = None
                if msg.get("timestamp") or msg.get("createTime"):
                    try:
                        msg_dt = parser.parse(msg.get("timestamp") or msg.get("createTime"))
                    except Exception:
                        pass

                messages.append(NormalizedMessage(
                    role=role,
                    content=str(content).strip(),
                    sequence_index=seq,
                    external_id=msg.get("id"),
                    created_at=msg_dt,
                    raw_metadata={}
                ))
                seq += 1.0

            if messages:
                results.append(NormalizedConversation(
                    title=title,
                    external_id=external_id,
                    created_at=created_at,
                    updated_at=created_at,
                    messages=messages,
                    raw_metadata={}
                ))

        return results
