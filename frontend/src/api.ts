export interface DashboardStats {
  total_memories: number;
  active_projects: number;
  total_conversations: number;
  pending_reviews: number;
  stale_memories: number;
  provider_accounts: number;
  recent_memories: Array<{
    id: string;
    statement: string;
    type: string;
    status: string;
    confidence: number;
    created_at: string;
  }>;
}

export interface Milestone {
  id: string;
  project_id: string;
  title: string;
  description?: string;
  milestone_type: string;
  evidence_memory_id?: string;
  reached_at?: string;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  goal?: string;
  status: string;
  architecture_overview?: string;
  tech_stack: string[];
  constraints: string[];
  created_at: string;
  updated_at: string;
  tasks: Task[];
  milestones?: Milestone[];
}

export interface ConflictItem {
  id: string;
  type: string;
  subject?: string;
  predicate?: string;
  memory_a: {
    id: string;
    statement: string;
    object?: string;
    scope?: Record<string, any>;
    status: string;
    confidence: number;
    created_at?: string;
  };
  memory_b?: {
    id: string;
    statement: string;
    object?: string;
    scope?: Record<string, any>;
    status: string;
    confidence: number;
    created_at?: string;
  } | null;
  reason: string;
}

export interface Task {
  id: string;
  project_id: string;
  title: string;
  description?: string;
  status: string;
  priority: string;
}

export interface Memory {
  id: string;
  workspace_id: string;
  project_id?: string;
  memory_type: string;
  statement: string;
  rationale?: string;
  status: string;
  confidence: number;
  extraction_method: string;
  version: number;
  superseded_by_id?: string;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  is_deleted_source: boolean;
  messages: Array<{
    id: string;
    role: string;
    content: string;
    sequence_index: number;
    created_at: string;
  }>;
}

export interface ProvenanceExplanation {
  memory_id: string;
  statement: string;
  memory_type: string;
  status: string;
  confidence: number;
  extraction_method: string;
  created_at: string;
  source_conversation?: {
    id: string;
    title: string;
    provider?: string;
    created_at: string;
  };
  source_message?: {
    id: string;
    role: string;
    created_at: string;
  };
  relevant_source_content?: string;
  history_versions: Array<{
    version: number;
    statement: string;
    status: string;
    change_reason?: string;
    created_at: string;
  }>;
}

export interface ContextPackage {
  project_name?: string;
  query: string;
  summary: string;
  goal?: string;
  architecture?: string;
  tech_stack: string[];
  decisions: Array<{ statement: string; rationale?: string }>;
  constraints: string[];
  tasks: Array<{ title: string; status: string; priority: string }>;
  relevant_memories: Array<{ statement: string; type: string }>;
  sources: Array<{ message_id: string; conversation_title: string; snippet: string }>;
  formatted_prompt: string;
}

export interface GraphData {
  nodes: Array<{ id: string; label: string; node_type: string }>;
  edges: Array<{ id: string; source: string; target: string; relation: string }>;
}

export interface ProviderCapability {
  provider: string;
  capabilities: Record<string, boolean>;
  notes: Record<string, string>;
}

const API_BASE = "http://localhost:8000/api/v1";

export const api = {
  async getDashboard(): Promise<DashboardStats> {
    const res = await fetch(`${API_BASE}/dashboard/stats`);
    if (!res.ok) throw new Error("Failed to fetch dashboard stats");
    return res.json();
  },

  async getProjects(): Promise<Project[]> {
    const res = await fetch(`${API_BASE}/projects`);
    return res.json();
  },

  async createProject(data: Partial<Project>): Promise<Project> {
    const res = await fetch(`${API_BASE}/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async getMemories(projectId?: string, status?: string): Promise<Memory[]> {
    let url = `${API_BASE}/memories`;
    const params = new URLSearchParams();
    if (projectId) params.append("project_id", projectId);
    if (status) params.append("status", status);
    if (params.toString()) url += `?${params.toString()}`;
    const res = await fetch(url);
    return res.json();
  },

  async explainMemory(id: string): Promise<ProvenanceExplanation> {
    const res = await fetch(`${API_BASE}/memories/${id}/explain`);
    return res.json();
  },

  async resolveMemory(id: string, action: string, reason?: string): Promise<Memory> {
    const url = `${API_BASE}/memories/${id}/resolve?resolution_action=${action}${reason ? `&reason=${encodeURIComponent(reason)}` : ""}`;
    const res = await fetch(url, { method: "POST" });
    return res.json();
  },

  async getConversations(): Promise<Conversation[]> {
    const res = await fetch(`${API_BASE}/conversations`);
    return res.json();
  },

  async importConversations(provider: string, data: any, projectId?: string): Promise<any> {
    let url = `${API_BASE}/imports/conversations?provider=${provider}`;
    if (projectId) url += `&project_id=${projectId}`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.json();
  },

  async getContextPackage(query: string, projectId?: string): Promise<ContextPackage> {
    let url = `${API_BASE}/context?query=${encodeURIComponent(query)}`;
    if (projectId) url += `&project_id=${projectId}`;
    const res = await fetch(url);
    return res.json();
  },

  async getKnowledgeGraph(): Promise<GraphData> {
    const res = await fetch(`${API_BASE}/graph`);
    return res.json();
  },

  async getCapabilities(): Promise<ProviderCapability[]> {
    const res = await fetch(`${API_BASE}/providers/capabilities`);
    return res.json();
  },

  async exportCanonical(): Promise<any> {
    const res = await fetch(`${API_BASE}/exports/canonical`);
    return res.json();
  },

  async importCanonical(payload: any): Promise<any> {
    const res = await fetch(`${API_BASE}/imports/canonical`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  },

  async getConflicts(workspaceId?: string, projectId?: string): Promise<ConflictItem[]> {
    let url = `${API_BASE}/conflicts`;
    const params = new URLSearchParams();
    if (workspaceId) params.append("workspace_id", workspaceId);
    if (projectId) params.append("project_id", projectId);
    if (params.toString()) url += `?${params.toString()}`;
    const res = await fetch(url);
    return res.json();
  },

  async resolveConflictFlow(
    memoryId: string,
    action: string,
    pairedMemoryId?: string,
    supersededById?: string,
    reason?: string
  ): Promise<any> {
    const params = new URLSearchParams();
    params.append("memory_id", memoryId);
    params.append("resolution_action", action);
    if (pairedMemoryId) params.append("paired_memory_id", pairedMemoryId);
    if (supersededById) params.append("superseded_by_id", supersededById);
    if (reason) params.append("reason", reason);
    const res = await fetch(`${API_BASE}/conflicts/resolve?${params.toString()}`, { method: "POST" });
    return res.json();
  },

  async synthesizeProject(projectId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/projects/${projectId}/synthesize`, { method: "POST" });
    return res.json();
  }
};
