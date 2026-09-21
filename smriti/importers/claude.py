import json
from datetime import datetime, timezone
from typing import List, Any
from dateutil import parser
from smriti.importers.base import BaseImporter, NormalizedConversation, NormalizedMessage

class ClaudeImporter(BaseImporter):
    """Importer for Anthropic Claude export formats (conversations.json)."""

    def get_provider_name(self) -> str:
        return "claude"

    def parse(self, raw_data: Any) -> List[NormalizedConversation]:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        elif not isinstance(raw_data, list):
            return []

        results: List[NormalizedConversation] = []
        for conv in raw_data:
            title = conv.get("name") or conv.get("title") or "Claude Conversation"
            external_id = conv.get("uuid") or conv.get("id")

            created_at = None
            if conv.get("created_at"):
                try:
                    created_at = parser.parse(conv["created_at"])
                except Exception:
                    pass

            updated_at = None
            if conv.get("updated_at"):
                try:
                    updated_at = parser.parse(conv["updated_at"])
                except Exception:
                    pass

            messages: List[NormalizedMessage] = []
            chat_messages = conv.get("chat_messages") or conv.get("messages") or []
            
            seq = 0.0
            for msg in chat_messages:
                sender = msg.get("sender") or msg.get("role") or "human"
                role = "user" if sender in ["human", "user"] else "assistant"
                
                content = msg.get("text") or msg.get("content") or ""
                if isinstance(content, list):
                    # In newer formats, content can be blocks
                    text_blocks = []
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            text_blocks.append(block["text"])
                        elif isinstance(block, str):
                            text_blocks.append(block)
                    content = "\n".join(text_blocks)

                if not str(content).strip():
                    continue

                msg_dt = None
                if msg.get("created_at"):
                    try:
                        msg_dt = parser.parse(msg["created_at"])
                    except Exception:
                        pass

                messages.append(NormalizedMessage(
                    role=role,
                    content=str(content).strip(),
                    sequence_index=seq,
                    external_id=msg.get("uuid") or msg.get("id"),
                    created_at=msg_dt,
                    raw_metadata={}
                ))
                seq += 1.0

            if messages:
                results.append(NormalizedConversation(
                    title=title,
                    external_id=external_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    messages=messages,
                    raw_metadata={}
                ))

        return results
