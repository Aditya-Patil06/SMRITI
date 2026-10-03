import sys
import re

with open('smriti/vector_store.py', 'r') as f:
    c = f.read()

c = c.replace('class BaseVectorStore:', '''
# Phase 2 Compatibility Adapters
from typing import Callable

class BaseVectorStore:''')

c = c.replace(
'''    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:''',
'''    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[VectorFilter] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:'''
)

in_memory_search_body = '''        for doc_id, doc_vec in self.vectors.items():
            meta = self.metadata.get(doc_id, {})
            
            if filter_fn and not filter_fn(meta):
                continue
                
            if filters:
                if filters.workspace_id and meta.get("workspace_id") != filters.workspace_id:
                    continue
                if filters.project_id and meta.get("project_id") != filters.project_id:
                    continue
                if not filters.include_superseded and meta.get("status") in ["superseded", "forgotten"]:
                    continue

            score = float(np.dot(q_vec, doc_vec))
            results.append((doc_id, score, meta))'''
            
c = re.sub(r'        for doc_id, doc_vec in self\.vectors\.items\(\):\n.*?(?=        results\.sort\(key=lambda)', in_memory_search_body + '\n\n', c, flags=re.DOTALL)

c += '''
# Expose Phase 2 compatibility aliases
EmbeddingService = MockEmbeddingProvider
VectorStore = InMemoryVectorStore
'''

with open('smriti/vector_store.py', 'w') as f:
    f.write(c)
