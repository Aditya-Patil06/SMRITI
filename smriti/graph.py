import networkx as nx
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from smriti.models import RelationshipEdge, Memory, Project, Conversation, Task
from smriti.schemas import GraphData, GraphNode, GraphEdge

class KnowledgeGraphService:
    """Manages the in-memory NetworkX graph enriched with persistent database edges."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def sync_from_db(self, db: Session, workspace_id: Optional[str] = None):
        """Rebuild or sync the in-memory NetworkX graph from database records."""
        self.graph.clear()

        # 1. Projects
        proj_q = db.query(Project)
        if workspace_id:
            proj_q = proj_q.filter(Project.workspace_id == workspace_id)
        for p in proj_q.all():
            self.graph.add_node(
                f"project:{p.id}",
                node_type="project",
                label=p.name,
                status=p.status,
                id=p.id,
                workspace_id=p.workspace_id
            )

        # 2. Memories
        mem_q = db.query(Memory).filter(Memory.status != "forgotten")
        if workspace_id:
            mem_q = mem_q.filter(Memory.workspace_id == workspace_id)
        for m in mem_q.all():
            self.graph.add_node(
                f"memory:{m.id}",
                node_type="memory",
                label=f"[{m.memory_type}] {m.statement[:40]}...",
                full_statement=m.statement,
                memory_type=m.memory_type,
                status=m.status,
                id=m.id,
                workspace_id=m.workspace_id,
                project_id=m.project_id
            )
            if m.project_id:
                self.graph.add_edge(
                    f"memory:{m.id}",
                    f"project:{m.project_id}",
                    relation="BELONGS_TO",
                    id=f"edge_m_p_{m.id}"
                )

        # 3. Tasks
        task_q = db.query(Task)
        for t in task_q.all():
            task_ws_id = t.project.workspace_id if t.project else None
            self.graph.add_node(
                f"task:{t.id}",
                node_type="task",
                label=t.title,
                status=t.status,
                id=t.id,
                project_id=t.project_id,
                workspace_id=task_ws_id
            )
            self.graph.add_edge(
                f"project:{t.project_id}",
                f"task:{t.id}",
                relation="CONTAINS",
                id=f"edge_p_t_{t.id}"
            )

        # 4. Explicit edges from DB
        edges = db.query(RelationshipEdge).all()
        for e in edges:
            u = f"{e.source_type}:{e.source_id}"
            v = f"{e.target_type}:{e.target_id}"
            self.graph.add_edge(u, v, relation=e.relation, id=e.id, properties=e.properties or {})

    def add_edge(self, db: Session, source_type: str, source_id: str, relation: str, target_type: str, target_id: str, properties: Dict[str, Any] = None) -> RelationshipEdge:
        edge_record = RelationshipEdge(
            source_type=source_type,
            source_id=source_id,
            relation=relation,
            target_type=target_type,
            target_id=target_id,
            properties=properties or {}
        )
        db.add(edge_record)
        db.commit()
        db.refresh(edge_record)

        u = f"{source_type}:{source_id}"
        v = f"{target_type}:{target_id}"
        self.graph.add_edge(u, v, relation=relation, id=edge_record.id, properties=properties or {})
        return edge_record

    def get_subgraph(self, center_node: Optional[str] = None, depth: int = 2) -> GraphData:
        """Return nodes and edges in serialized graph format for the UI."""
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        if center_node and self.graph.has_node(center_node):
            sub_nodes = set([center_node])
            frontier = set([center_node])
            for _ in range(depth):
                next_frontier = set()
                for n in frontier:
                    next_frontier.update(self.graph.neighbors(n))
                    next_frontier.update(self.graph.predecessors(n))
                sub_nodes.update(next_frontier)
                frontier = next_frontier
            view_graph = self.graph.subgraph(sub_nodes)
        else:
            view_graph = self.graph

        for n, data in view_graph.nodes(data=True):
            nodes.append(GraphNode(
                id=n,
                label=data.get("label", n),
                node_type=data.get("node_type", "entity"),
                properties={k: v for k, v in data.items() if k not in ["label", "node_type"]}
            ))

        for u, v, key, data in view_graph.edges(keys=True, data=True):
            edges.append(GraphEdge(
                id=str(data.get("id", f"{u}_{v}_{key}")),
                source=u,
                target=v,
                relation=data.get("relation", "RELATED_TO"),
                properties=data.get("properties", {})
            ))

        return GraphData(nodes=nodes, edges=edges)

    def find_connected_memories(
        self,
        node_id: str,
        max_hops: int = 2,
        workspace_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[str]:
        """Find memory IDs connected to a given node within workspace/project scope boundaries."""
        if not self.graph.has_node(node_id):
            return []
        
        visited = set([node_id])
        frontier = set([node_id])
        memory_ids = []

        for _ in range(max_hops):
            next_frontier = set()
            for n in frontier:
                neighbors = list(self.graph.neighbors(n)) + list(self.graph.predecessors(n))
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        nbr_data = self.graph.nodes[nbr]
                        # Scope validation during traversal
                        if workspace_id and nbr_data.get("node_type") == "task":
                            if nbr_data.get("workspace_id") != workspace_id:
                                continue
                        elif workspace_id and nbr_data.get("workspace_id") and nbr_data.get("workspace_id") != workspace_id:
                            continue
                        if project_id and nbr_data.get("project_id") and nbr_data.get("project_id") != project_id:
                            continue
                        next_frontier.add(nbr)
                        if nbr.startswith("memory:"):
                            # If it's a memory, ensure it strictly matches requested scope
                            if workspace_id and nbr_data.get("workspace_id") != workspace_id:
                                continue
                            if project_id and nbr_data.get("project_id") != project_id:
                                continue
                            memory_ids.append(nbr.replace("memory:", ""))
            frontier = next_frontier

        return memory_ids

graph_service = KnowledgeGraphService()
