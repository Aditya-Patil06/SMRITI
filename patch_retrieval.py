import sys
with open('smriti/retrieval.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    'from smriti.vector_store import vector_store',
    'from smriti.vector_store import vector_store, VectorFilter'
)

c = c.replace(
    '''        vec_hits = vector_store.search(
            query=query_clean,
            top_k=limit * 2,
            filter_fn=lambda m: (not project_id or m.get("project_id") == project_id) and
                                (not workspace_id or m.get("workspace_id") == workspace_id) and
                                (include_superseded or m.get("status") not in ["superseded", "forgotten"])
        )''',
    '''        vec_hits = vector_store.search(
            query=query_clean,
            top_k=limit * 2,
            filters=VectorFilter(
                workspace_id=workspace_id,
                project_id=project_id,
                include_superseded=include_superseded
            )
        )'''
)

with open('smriti/retrieval.py', 'w', encoding='utf-8') as f:
    f.write(c)
