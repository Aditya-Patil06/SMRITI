from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field

# --- User & Workspace ---
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None

class UserRead(UserBase):
    id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class WorkspaceBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_default: bool = False

class WorkspaceCreate(WorkspaceBase):
    user_id: Optional[str] = None

class WorkspaceRead(WorkspaceBase):
    id: str
    user_id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# --- Provider Account ---
class ProviderAccountBase(BaseModel):
    provider: str
    account_label: str
    auth_metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True

class ProviderAccountCreate(ProviderAccountBase):
    user_id: Optional[str] = None

class ProviderAccountRead(ProviderAccountBase):
    id: str
    user_id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# --- Message & Conversation ---
class MessageBase(BaseModel):
    role: str
    content: str
    sequence_index: float = 0.0
    external_id: Optional[str] = None
    created_at: Optional[datetime] = None
    raw_metadata: Optional[Dict[str, Any]] = None

class MessageRead(MessageBase):
    id: str
    conversation_id: str
    model_config = ConfigDict(from_attributes=True)

class ConversationBase(BaseModel):
    title: str
    external_id: Optional[str] = None
    raw_metadata: Optional[Dict[str, Any]] = None

class ConversationCreate(ConversationBase):
    provider_account_id: Optional[str] = None
    messages: List[MessageBase] = []

class ConversationRead(ConversationBase):
    id: str
    provider_account_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    is_deleted_source: bool
    messages: List[MessageRead] = []
    model_config = ConfigDict(from_attributes=True)

# --- Project & Tasks & Milestones ---
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "todo"
    priority: str = "medium"
    due_date: Optional[datetime] = None

class TaskCreate(TaskBase):
    project_id: str

class TaskRead(TaskBase):
    id: str
    project_id: str
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class MilestoneBase(BaseModel):
    title: str
    description: Optional[str] = None
    milestone_type: str  # project_created, architecture_decided, technology_selected, feature_completed, phase_completed
    evidence_memory_id: Optional[str] = None
    reached_at: Optional[datetime] = None

class MilestoneCreate(MilestoneBase):
    project_id: str

class MilestoneRead(MilestoneBase):
    id: str
    project_id: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    goal: Optional[str] = None
    status: str = "active"
    architecture_overview: Optional[str] = None
    tech_stack: List[str] = []
    constraints: List[str] = []

class ProjectCreate(ProjectBase):
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None

class ProjectRead(ProjectBase):
    id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime
    tasks: List[TaskRead] = []
    milestones: List[MilestoneRead] = []
    model_config = ConfigDict(from_attributes=True)

# --- Memory & Claims ---
class StructuredClaim(BaseModel):
    subject: str
    predicate: str
    object: str
    scope: Optional[Dict[str, Any]] = None  # e.g. {"env": "prod", "component": "backend"}
    temporal_context: Optional[str] = None

class MemoryBase(BaseModel):
    memory_type: str
    statement: str
    rationale: Optional[str] = None
    structured_claim: Optional[Union[StructuredClaim, Dict[str, Any]]] = None
    details: Optional[Dict[str, Any]] = None
    status: str = "active"
    confidence: float = 1.0
    extraction_method: str = "rule_heuristic"
    project_id: Optional[str] = None

class MemoryCreate(MemoryBase):
    workspace_id: Optional[str] = None
    source_message_id: Optional[str] = None
    source_conversation_id: Optional[str] = None

class MemoryUpdate(BaseModel):
    statement: Optional[str] = None
    rationale: Optional[str] = None
    structured_claim: Optional[Union[StructuredClaim, Dict[str, Any]]] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    change_reason: Optional[str] = None

class MemoryRead(MemoryBase):
    id: str
    workspace_id: str
    source_message_id: Optional[str] = None
    source_conversation_id: Optional[str] = None
    version: float
    superseded_by_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime
    model_config = ConfigDict(from_attributes=True)

# --- Explain Provenance ---
class ProvenanceExplanation(BaseModel):
    memory_id: str
    statement: str
    memory_type: str
    status: str
    confidence: float
    extraction_method: str
    created_at: datetime
    workspace_id: Optional[str] = None
    project_id: Optional[str] = None
    is_manual: bool = False
    source_unavailable: bool = False
    superseded_by_id: Optional[str] = None
    provider: Optional[str] = None
    provider_account: Optional[Dict[str, Any]] = None
    source_conversation: Optional[Dict[str, Any]] = None
    source_message: Optional[Dict[str, Any]] = None
    relevant_source_content: Optional[str] = None
    history_versions: List[Dict[str, Any]] = []

# --- Context Package ---
class PortableContextPackage(BaseModel):
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    query: str
    summary: str
    goal: Optional[str] = None
    architecture: Optional[str] = None
    tech_stack: List[str] = []
    decisions: List[Dict[str, Any]] = []
    constraints: List[str] = []
    tasks: List[Dict[str, Any]] = []
    relevant_memories: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    formatted_prompt: str

# --- Graph ---
class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any] = {}

class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = {}

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

# --- Search & Retrieval ---
class SearchQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    workspace_id: Optional[str] = None
    limit: int = 20
    include_superseded: bool = False

class SearchResultItem(BaseModel):
    id: str
    type: str  # memory, message, project
    title: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}
