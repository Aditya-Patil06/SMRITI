import json
from datetime import datetime, timezone
from typing import List, Any, Dict
from smriti.importers.base import BaseImporter, NormalizedConversation, NormalizedMessage

class ChatGPTImporter(BaseImporter):
    """Importer for official OpenAI ChatGPT export formats (conversations.json)."""

    def get_provider_name(self) -> str:
        return "chatgpt"

    def parse(self, raw_data: Any) -> List[NormalizedConversation]:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)
        
        if not isinstance(raw_data, list):
            # Sometimes a single conversation dictionary is passed
            if isinstance(raw_data, dict):
                raw_data = [raw_data]
            else:
                return []

        results: List[NormalizedConversation] = []
        for conv in raw_data:
            title = conv.get("title") or "ChatGPT Conversation"
            external_id = conv.get("id") or conv.get("conversation_id")
            
            create_time = None
            if conv.get("create_time"):
                try:
                    create_time = datetime.fromtimestamp(conv["create_time"], tz=timezone.utc)
                except Exception:
                    pass

            update_time = None
            if conv.get("update_time"):
                try:
                    update_time = datetime.fromtimestamp(conv["update_time"], tz=timezone.utc)
                except Exception:
                    pass

            # Handle ChatGPT mapping node tree
            messages: List[NormalizedMessage] = []
            mapping = conv.get("mapping", {})
            
            # Sort messages chronologically by traversing or collecting nodes
            sorted_nodes = []
            for node_id, node in mapping.items():
                msg = node.get("message")
                if msg and msg.get("content"):
                    sorted_nodes.append((node_id, node, msg.get("create_time") or 0))

            sorted_nodes.sort(key=lambda x: x[2])

            seq = 0.0
            for node_id, node, msg_time in sorted_nodes:
                msg = node.get("message", {})
                author = msg.get("author", {})
                role = author.get("role", "assistant")
                
                content_obj = msg.get("content", {})
                content_parts = content_obj.get("parts", [])
                
                # Combine parts
                text_content = ""
                if isinstance(content_parts, list):
                    text_content = "".join([str(p) for p in content_parts if p is not None])
                elif isinstance(content_parts, str):
                    text_content = content_parts
                elif content_obj.get("text"):
                    text_content = str(content_obj.get("text"))

                if not text_content.strip():
                    continue

                msg_dt = None
                if msg_time:
                    try:
                        msg_dt = datetime.fromtimestamp(msg_time, tz=timezone.utc)
                    except Exception:
                        pass

                messages.append(NormalizedMessage(
                    role=role,
                    content=text_content.strip(),
                    sequence_index=seq,
                    external_id=msg.get("id") or node_id,
                    created_at=msg_dt,
                    raw_metadata={"model": msg.get("metadata", {}).get("model_slug")}
                ))
                seq += 1.0

            if messages:
                results.append(NormalizedConversation(
                    title=title,
                    external_id=external_id,
                    created_at=create_time,
                    updated_at=update_time,
                    messages=messages,
                    raw_metadata={"model_slug": conv.get("default_model_slug")}
                ))

        return results
