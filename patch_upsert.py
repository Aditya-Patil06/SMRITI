import sys
import re

with open('smriti/vector_store.py', 'r', encoding='utf-8') as f:
    c = f.read()

upsert_body_new = '''    def upsert(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None, auto_save: bool = True):
        from smriti.models import EmbeddingMetadata
        from sqlalchemy.dialects.postgresql import insert
        db = self.db_session_factory()
        try:
            content_hash = self._hash(text)
            vec = self.embedding_provider.embed(text)
            
            stmt = insert(EmbeddingMetadata).values(
                memory_id=doc_id,
                model_name=self.model_name,
                model_version=self.model_version,
                embedding_dimension=self.dimension,
                content_hash=content_hash,
                embedding=vec
            ).on_conflict_do_nothing(
                index_elements=['memory_id', 'model_name', 'model_version', 'content_hash']
            )
            
            db.execute(stmt)
            
            # Clean up stale hashes atomically (if there's a new hash)
            db.query(EmbeddingMetadata).filter(
                EmbeddingMetadata.memory_id == doc_id,
                EmbeddingMetadata.model_name == self.model_name,
                EmbeddingMetadata.model_version == self.model_version,
                EmbeddingMetadata.content_hash != content_hash
            ).delete(synchronize_session=False)

            if auto_save:
                db.commit()
        finally:
            db.close()'''

c = re.sub(r'    def upsert\(self, doc_id: str, text: str.*?finally:\n            db\.close\(\)', upsert_body_new, c, flags=re.DOTALL)

with open('smriti/vector_store.py', 'w', encoding='utf-8') as f:
    f.write(c)
