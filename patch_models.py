import sys
with open('smriti/models.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'from sqlalchemy import (',
    '''from sqlalchemy import (
    create_engine, Column, String, Text, Float, DateTime, Boolean, ForeignKey, JSON, Integer, UniqueConstraint
)
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False

# dummy to replace old block
from sqlalchemy import ('''
)

new_model = '''
class EmbeddingMetadata(Base):
    __tablename__ = "embedding_metadata"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    memory_id = Column(String(64), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    
    if HAS_PGVECTOR:
        embedding = Column(Vector(), nullable=True)
        
    created_at = Column(DateTime, default=utcnow)
    
    __table_args__ = (
        UniqueConstraint('memory_id', 'model_name', 'model_version', 'content_hash', name='uix_embedding_idempotency'),
    )

'''
c = c.replace('# Engine and session initialization', new_model + '# Engine and session initialization')

with open('smriti/models.py', 'w', encoding='utf-8') as f:
    f.write(c)
