from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    create_engine, Column, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from smriti.config import settings

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    username = Column(String(128), unique=True, nullable=False)
    email = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    
    workspaces = relationship("Workspace", back_populates="user", cascade="all, delete-orphan")
    provider_accounts = relationship("ProviderAccount", back_populates="user", cascade="all, delete-orphan")

class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    name = Column(String(128), nullable=False)  # Personal, Work, Research, etc.
    description = Column(Text, nullable=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="workspaces")
    projects = relationship("Project", back_populates="workspace", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="workspace", cascade="all, delete-orphan")

class ProviderAccount(Base):
    __tablename__ = "provider_accounts"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    provider = Column(String(64), nullable=False)  # chatgpt, claude, gemini, generic
    account_label = Column(String(128), nullable=False)  # Personal, College, Work
    auth_metadata = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="provider_accounts")
    conversations = relationship("Conversation", back_populates="provider_account")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    provider_account_id = Column(String(64), ForeignKey("provider_accounts.id"), nullable=True)
    external_id = Column(String(256), nullable=True)  # Provider's native conversation ID
    title = Column(String(512), nullable=False, default="Untitled Conversation")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    is_deleted_source = Column(Boolean, default=False)
    raw_metadata = Column(JSON, default=dict)

    provider_account = relationship("ProviderAccount", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.sequence_index")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(64), ForeignKey("conversations.id"), nullable=False)
    external_id = Column(String(256), nullable=True)
    role = Column(String(32), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    sequence_index = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)
    raw_metadata = Column(JSON, default=dict)

    conversation = relationship("Conversation", back_populates="messages")
    memories = relationship("Memory", back_populates="source_message")

class Project(Base):
    __tablename__ = "projects"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(64), ForeignKey("workspaces.id"), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    goal = Column(Text, nullable=True)
    status = Column(String(64), default="active")  # active, completed, paused, archived
    architecture_overview = Column(Text, nullable=True)
    tech_stack = Column(JSON, default=list)  # list of strings
    constraints = Column(JSON, default=list)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    last_confirmed_at = Column(DateTime, default=utcnow)

    workspace = relationship("Workspace", back_populates="projects")
    memories = relationship("Memory", back_populates="project")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = "tasks"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    project_id = Column(String(64), ForeignKey("projects.id"), nullable=False)
    title = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(64), default="todo")  # todo, in_progress, completed, blocked
    priority = Column(String(32), default="medium")  # low, medium, high, critical
    due_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="tasks")

class Memory(Base):
    __tablename__ = "memories"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    workspace_id = Column(String(64), ForeignKey("workspaces.id"), nullable=False)
    project_id = Column(String(64), ForeignKey("projects.id"), nullable=True)
    source_message_id = Column(String(64), ForeignKey("messages.id"), nullable=True)
    source_conversation_id = Column(String(64), nullable=True)
    
    memory_type = Column(String(64), nullable=False)  # decision, task, problem, solution, technology, concept, constraint, fact, status_change
    statement = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    details = Column(JSON, default=dict)
    
    # Status: active, superseded, conflicting, deprecated, source_unavailable, forgotten, review_required
    status = Column(String(64), default="active")
    confidence = Column(Float, default=1.0)
    extraction_method = Column(String(64), default="rule_heuristic")  # rule_heuristic, llm, user_explicit
    
    version = Column(Float, default=1.0)
    superseded_by_id = Column(String(64), nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    last_confirmed_at = Column(DateTime, default=utcnow)

    workspace = relationship("Workspace", back_populates="memories")
    project = relationship("Project", back_populates="memories")
    source_message = relationship("Message", back_populates="memories")
    versions = relationship("MemoryVersion", back_populates="memory", cascade="all, delete-orphan")

class MemoryVersion(Base):
    __tablename__ = "memory_versions"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    memory_id = Column(String(64), ForeignKey("memories.id"), nullable=False)
    version_number = Column(Float, nullable=False)
    statement = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    details = Column(JSON, default=dict)
    status = Column(String(64), nullable=False)
    change_reason = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    memory = relationship("Memory", back_populates="versions")

class RelationshipEdge(Base):
    __tablename__ = "relationship_edges"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    source_type = Column(String(64), nullable=False)  # memory, project, conversation, technology, etc.
    source_id = Column(String(64), nullable=False)
    relation = Column(String(64), nullable=False)    # CONTAINS, DISCUSSES, MENTIONS, PRODUCED, RELATED_TO, HAS_SOLUTION, DEPENDS_ON, CONTINUES, DERIVED_FROM, BELONGS_TO, SUPERSEDES, CONFLICTS_WITH, CONFIRMED_BY
    target_type = Column(String(64), nullable=False)
    target_id = Column(String(64), nullable=False)
    properties = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String(64), primary_key=True, default=generate_uuid)
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False)  # create, update, delete, supersede, resolve_conflict, confirm
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)

# Engine and session initialization
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    # Seed default user and workspace if empty
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if not user:
            user = User(id="default-user", username="smriti_user", email="user@smriti.local")
            db.add(user)
            db.commit()
            db.refresh(user)

        workspace = db.query(Workspace).filter_by(user_id=user.id).first()
        if not workspace:
            workspace = Workspace(
                id="default-workspace",
                user_id=user.id,
                name="Personal Workspace",
                description="Default local personal memory space",
                is_default=True
            )
            db.add(workspace)
            db.commit()
    finally:
        db.close()
