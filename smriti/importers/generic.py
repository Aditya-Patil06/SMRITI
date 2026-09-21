import json
from datetime import datetime, timezone
from typing import List, Any
from dateutil import parser
from smriti.importers.base import BaseImporter, NormalizedConversation, NormalizedMessage

class GenericImporter(BaseImporter):
    """Generic JSON importer supporting standard message turn lists."""

    def get_provider_name(self) -> str:
        return "generic"

    def parse(self, raw_data: Any) -> List[NormalizedConversation]:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        elif not isinstance(raw_data, list):
            return []

        results: List[NormalizedConversation] = []
        for conv in raw_data:
            title = conv.get("title") or "Generic Conversation"
            external_id = conv.get("id")

            created_at = None
            if conv.get("created_at"):
                try:
                    created_at = parser.parse(conv["created_at"])
                except Exception:
                    pass

            messages: List[NormalizedMessage] = []
            raw_msgs = conv.get("messages", [])
            seq = 0.0

            for msg in raw_msgs:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not str(content).strip():
                    continue

                msg_dt = None
                if msg.get("created_at") or msg.get("timestamp"):
                    try:
                        msg_dt = parser.parse(msg.get("created_at") or msg.get("timestamp"))
                    except Exception:
                        pass

                messages.append(NormalizedMessage(
                    role=role,
                    content=str(content).strip(),
                    sequence_index=seq,
                    external_id=msg.get("id"),
                    created_at=msg_dt,
                    raw_metadata=msg.get("metadata", {})
                ))
                seq += 1.0

            if messages:
                results.append(NormalizedConversation(
                    title=title,
                    external_id=external_id,
                    created_at=created_at,
                    updated_at=created_at,
                    messages=messages,
                    raw_metadata=conv.get("metadata", {})
                ))

        return results
