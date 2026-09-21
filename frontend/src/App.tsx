import { useState, useEffect } from 'react';
import { 
  Brain, LayoutDashboard, FolderGit2, History, Network, ArrowLeftRight, 
  Download, Copy, Check, Sparkles, Database
} from 'lucide-react';
import { 
  api, 
  type DashboardStats, 
  type Project, 
  type Memory, 
  type Conversation, 
  type ProvenanceExplanation, 
  type ContextPackage, 
  type GraphData, 
  type ProviderCapability 
} from './api';

export function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'projects' | 'memories' | 'conversations' | 'graph' | 'context' | 'imports' | 'providers'>('dashboard');
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [capabilities, setCapabilities] = useState<ProviderCapability[]>([]);

  // Modal / Detailed view states
  const [selectedMemory, setSelectedMemory] = useState<ProvenanceExplanation | null>(null);
  const [contextQuery, setContextQuery] = useState<string>('What did we decide about the system architecture?');
  const [contextPackage, setContextPackage] = useState<ContextPackage | null>(null);
  const [copiedPrompt, setCopiedPrompt] = useState<boolean>(false);
  const [importJson, setImportJson] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<string>('chatgpt');
  const [importStatus, setImportStatus] = useState<string>('');

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      const [s, p, m, c, caps] = await Promise.all([
        api.getDashboard().catch(() => null),
        api.getProjects().catch(() => []),
        api.getMemories().catch(() => []),
        api.getConversations().catch(() => []),
        api.getCapabilities().catch(() => [])
      ]);
      setStats(s);
      setProjects(p);
      setMemories(m);
      setConversations(c);
      setCapabilities(caps);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleExplain(memId: string) {
    try {
      const exp = await api.explainMemory(memId);
      setSelectedMemory(exp);
    } catch (e) {
      alert("Error retrieving provenance explanation");
    }
  }

  async function handleResolve(memId: string, action: string) {
    try {
      await api.resolveMemory(memId, action, `User resolved as ${action}`);
      alert(`Memory updated to ${action}`);
      setSelectedMemory(null);
      loadData();
    } catch (e) {
      alert("Error resolving conflict");
    }
  }

  async function handleGenerateContext() {
    if (!contextQuery.trim()) return;
    try {
      const pkg = await api.getContextPackage(contextQuery);
      setContextPackage(pkg);
    } catch (e) {
      alert("Failed to compile context package");
    }
  }

  async function handleImport() {
    if (!importJson.trim()) return;
    try {
      setImportStatus("Importing and extracting structured memory...");
      const parsed = JSON.parse(importJson);
      const res = await api.importConversations(selectedProvider, parsed);
      setImportStatus(`Success! Imported ${res.imported_conversations} conversations and extracted ${res.extracted_memories} memories.`);
      loadData();
    } catch (e: any) {
      setImportStatus(`Import error: ${e.message}`);
    }
  }

  async function handleCanonicalExport() {
    try {
      const bundle = await api.exportCanonical();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `smriti-export-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
    } catch (e) {
      alert("Export failed");
    }
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0a0d14', color: '#e2e8f0', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Sidebar */}
      <aside style={{ width: '260px', background: '#0f172a', borderRight: '1px solid #1e293b', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '24px 20px', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Brain style={{ color: '#38bdf8', width: '28px', height: '28px' }} />
          <div>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: '700', letterSpacing: '0.05em' }}>SMRITI</h1>
            <span style={{ fontSize: '11px', color: '#64748b' }}>Personal AI Memory Core</span>
          </div>
        </div>

        <nav style={{ padding: '16px 12px', flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {[
            { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
            { id: 'projects', label: 'Projects & State', icon: FolderGit2 },
            { id: 'memories', label: 'Memory Bank', icon: Database },
            { id: 'context', label: 'Context Engine', icon: Sparkles },
            { id: 'conversations', label: 'Conversations', icon: History },
            { id: 'graph', label: 'Knowledge Graph', icon: Network },
            { id: 'providers', label: 'AI Adapters', icon: ArrowLeftRight },
            { id: 'imports', label: 'Import / Export', icon: Download },
          ].map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id as any);
                  if (item.id === 'graph') {
                    api.getKnowledgeGraph().then(setGraphData).catch(console.error);
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: 'none',
                  background: active ? '#1e293b' : 'transparent',
                  color: active ? '#38bdf8' : '#94a3b8',
                  fontSize: '14px',
                  fontWeight: active ? '600' : '400',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease'
                }}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div style={{ padding: '16px 20px', borderTop: '1px solid #1e293b', fontSize: '12px', color: '#64748b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ color: '#10b981', fontWeight: 'bold' }}>●</span>
            <span>Local-First & Private</span>
          </div>
          <span>Status: Verified Core v0.1.0</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '32px 40px', overflowY: 'auto' }}>
        {/* Top Header */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '700' }}>
              {activeTab === 'dashboard' && 'Control Center Overview'}
              {activeTab === 'projects' && 'Project Memory & State'}
              {activeTab === 'memories' && 'Persistent Canonical Memories'}
              {activeTab === 'context' && 'Portable Context Generator'}
              {activeTab === 'conversations' && 'Normalized Conversation Archives'}
              {activeTab === 'graph' && 'Relational Knowledge Graph'}
              {activeTab === 'providers' && 'AI Provider Capability Matrix'}
              {activeTab === 'imports' && 'Import & Zero-Loss Portability'}
            </h2>
            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
              Universal personal AI memory layer independent of proprietary platforms.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              onClick={handleCanonicalExport}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                background: '#1e293b',
                color: '#e2e8f0',
                border: '1px solid #334155',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '500'
              }}
            >
              <Download size={16} />
              Export Canonical Memory
            </button>
          </div>
        </header>

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && stats && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px' }}>
              {[
                { label: 'Total Memories', value: stats.total_memories, sub: 'Decisions, tasks, tech' },
                { label: 'Active Projects', value: stats.active_projects, sub: 'Structured states' },
                { label: 'Stored Dialogues', value: stats.total_conversations, sub: 'Normalized sessions' },
                { label: 'Pending Reviews', value: stats.pending_reviews, sub: 'Requires verification' },
              ].map((c, i) => (
                <div key={i} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                  <div style={{ fontSize: '13px', color: '#64748b', fontWeight: '500' }}>{c.label}</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: '#f8fafc', margin: '8px 0 4px 0' }}>{c.value}</div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>{c.sub}</div>
                </div>
              ))}
            </div>

            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px' }}>
              <h3 style={{ margin: '0 0 16px 0', fontSize: '16px' }}>Recently Recorded Project Memories</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {stats.recent_memories.length === 0 ? (
                  <p style={{ color: '#64748b', fontSize: '14px' }}>No memories registered yet. Import conversation exports to begin.</p>
                ) : (
                  stats.recent_memories.map((m) => (
                    <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: '#1e293b', borderRadius: '8px' }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                          <span style={{ fontSize: '11px', background: '#38bdf822', color: '#38bdf8', padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase', fontWeight: '600' }}>
                            {m.type}
                          </span>
                          <span style={{ fontSize: '11px', color: '#94a3b8' }}>Confidence: {(m.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ fontSize: '14px', color: '#f1f5f9' }}>{m.statement}</div>
                      </div>
                      <button
                        onClick={() => handleExplain(m.id)}
                        style={{ background: '#334155', border: 'none', color: '#e2e8f0', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px' }}
                      >
                        Explain
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* Memories Bank Tab */}
        {activeTab === 'memories' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '8px' }}>
              <input
                type="text"
                placeholder="Search memories or filter by topic..."
                style={{ flex: 1, padding: '10px 16px', background: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', color: '#e2e8f0' }}
                onChange={async (e) => {
                  const q = e.target.value;
                  if (q.trim().length > 2) {
                    const hits = await api.getMemories();
                    setMemories(hits.filter(h => h.statement.toLowerCase().includes(q.toLowerCase())));
                  } else {
                    api.getMemories().then(setMemories);
                  }
                }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {memories.map((m) => (
                <div key={m.id} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
                        <span style={{ fontSize: '11px', background: '#38bdf822', color: '#38bdf8', padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase', fontWeight: '600' }}>
                          {m.memory_type}
                        </span>
                        <span style={{ fontSize: '12px', color: m.status === 'active' ? '#10b981' : '#f59e0b' }}>
                          ● {m.status} (v{m.version.toFixed(1)})
                        </span>
                        <span style={{ fontSize: '12px', color: '#64748b' }}>Method: {m.extraction_method}</span>
                      </div>
                      <div style={{ fontSize: '15px', fontWeight: '500', color: '#f8fafc', marginBottom: m.rationale ? '4px' : '0' }}>
                        {m.statement}
                      </div>
                      {m.rationale && (
                        <div style={{ fontSize: '13px', color: '#94a3b8' }}>
                          <strong>Rationale:</strong> {m.rationale}
                        </div>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        onClick={() => handleExplain(m.id)}
                        style={{ padding: '6px 14px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', borderRadius: '6px', fontSize: '13px', cursor: 'pointer' }}
                      >
                        Explain Provenance
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Context Engine Tab */}
        {activeTab === 'context' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: '600', marginBottom: '8px' }}>
                Query / Active Focus
              </label>
              <div style={{ display: 'flex', gap: '12px' }}>
                <input
                  type="text"
                  value={contextQuery}
                  onChange={(e) => setContextQuery(e.target.value)}
                  style={{ flex: 1, padding: '10px 16px', background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#f8fafc', fontSize: '14px' }}
                />
                <button
                  onClick={handleGenerateContext}
                  style={{ padding: '10px 24px', background: '#0284c7', border: 'none', borderRadius: '8px', color: '#fff', fontWeight: '600', cursor: 'pointer' }}
                >
                  Compile Context
                </button>
              </div>
            </div>

            {contextPackage && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                  <h3 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Extracted Context Summary</h3>
                  <p style={{ fontSize: '13px', color: '#94a3b8' }}>{contextPackage.summary}</p>
                  
                  <h4 style={{ margin: '16px 0 8px 0', fontSize: '14px' }}>Active Decisions ({contextPackage.decisions.length})</h4>
                  {contextPackage.decisions.map((d, i) => (
                    <div key={i} style={{ fontSize: '13px', padding: '6px 0', borderBottom: '1px solid #1e293b' }}>
                      • {d.statement}
                    </div>
                  ))}

                  <h4 style={{ margin: '16px 0 8px 0', fontSize: '14px' }}>Active Tasks ({contextPackage.tasks.length})</h4>
                  {contextPackage.tasks.map((t, i) => (
                    <div key={i} style={{ fontSize: '13px', padding: '6px 0', borderBottom: '1px solid #1e293b' }}>
                      [{t.status}] {t.title}
                    </div>
                  ))}
                </div>

                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <h3 style={{ margin: 0, fontSize: '16px' }}>Portable AI System Prompt</h3>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(contextPackage.formatted_prompt);
                        setCopiedPrompt(true);
                        setTimeout(() => setCopiedPrompt(false), 2000);
                      }}
                      style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', background: '#1e293b', border: '1px solid #334155', borderRadius: '6px', color: '#e2e8f0', cursor: 'pointer', fontSize: '12px' }}
                    >
                      {copiedPrompt ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                      {copiedPrompt ? "Copied!" : "Copy Context"}
                    </button>
                  </div>
                  <pre style={{ margin: 0, padding: '12px', background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '12px', color: '#38bdf8', overflowX: 'auto', whiteSpace: 'pre-wrap', maxHeight: '400px' }}>
                    {contextPackage.formatted_prompt}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Conversations Archive Tab */}
        {activeTab === 'conversations' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {conversations.length === 0 ? (
              <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px', color: '#64748b' }}>
                No conversations imported yet.
              </div>
            ) : (
              conversations.map((c) => (
                <div key={c.id} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <h3 style={{ margin: 0, fontSize: '16px', color: '#f8fafc' }}>{c.title}</h3>
                    <span style={{ fontSize: '12px', color: '#64748b' }}>{c.messages?.length || 0} messages</span>
                  </div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>Created: {new Date(c.created_at).toLocaleString()}</div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Projects Tab */}
        {activeTab === 'projects' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '20px' }}>
              {projects.length === 0 ? (
                <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px', color: '#64748b' }}>
                  No active projects found.
                </div>
              ) : (
                projects.map((p) => (
                  <div key={p.id} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <h3 style={{ margin: 0, fontSize: '17px', color: '#f8fafc' }}>{p.name}</h3>
                      <span style={{ fontSize: '11px', background: '#10b98122', color: '#10b981', padding: '2px 8px', borderRadius: '4px', fontWeight: '600' }}>
                        {p.status}
                      </span>
                    </div>
                    <p style={{ fontSize: '13px', color: '#94a3b8', margin: '0 0 12px 0' }}>{p.goal || p.description || "No goal specified"}</p>
                    {p.tech_stack && p.tech_stack.length > 0 && (
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                        {p.tech_stack.map((t, i) => (
                          <span key={i} style={{ fontSize: '11px', background: '#1e293b', color: '#cbd5e1', padding: '2px 6px', borderRadius: '4px' }}>
                            {t}
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ fontSize: '12px', color: '#64748b' }}>
                      Tasks: {p.tasks?.length || 0} active
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Knowledge Graph Tab */}
        {activeTab === 'graph' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px' }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '16px' }}>Network Graph Visualizer</h3>
            <p style={{ fontSize: '13px', color: '#64748b', margin: '0 0 16px 0' }}>
              Connected topological view linking projects, decisions, tasks, and technologies.
            </p>
            {graphData && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div style={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', maxHeight: '450px', overflowY: 'auto' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#38bdf8' }}>Graph Entities ({graphData.nodes.length})</h4>
                  {graphData.nodes.map((n) => (
                    <div key={n.id} style={{ fontSize: '12px', padding: '6px 8px', borderBottom: '1px solid #0f172a', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#94a3b8', fontWeight: '600' }}>[{n.node_type}]</span>
                      <span style={{ color: '#f1f5f9' }}>{n.label}</span>
                    </div>
                  ))}
                </div>

                <div style={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', maxHeight: '450px', overflowY: 'auto' }}>
                  <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#10b981' }}>Graph Edges ({graphData.edges.length})</h4>
                  {graphData.edges.map((e) => (
                    <div key={e.id} style={{ fontSize: '12px', padding: '6px 8px', borderBottom: '1px solid #0f172a', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#f1f5f9' }}>{e.source.split(':')[1]?.slice(0, 8)}</span>
                      <span style={{ color: '#f59e0b', fontWeight: 'bold' }}>── {e.relation} ──&gt;</span>
                      <span style={{ color: '#f1f5f9' }}>{e.target.split(':')[1]?.slice(0, 8)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* AI Providers Tab */}
        {activeTab === 'providers' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
            {capabilities.map((c) => (
              <div key={c.provider} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '18px', textTransform: 'capitalize' }}>{c.provider}</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '16px' }}>
                  {Object.entries(c.capabilities).map(([cap, supported]) => (
                    <div key={cap} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
                      <span style={{ color: '#94a3b8' }}>{cap}</span>
                      <span style={{ color: supported ? '#10b981' : '#64748b', fontWeight: '600' }}>
                        {supported ? 'SUPPORTED' : 'UNSUPPORTED'}
                      </span>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', borderTop: '1px solid #1e293b', paddingTop: '12px' }}>
                  {Object.entries(c.notes).map(([k, note]) => (
                    <div key={k} style={{ marginBottom: '4px' }}>
                      <strong>{k}:</strong> {note}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Imports & Portability Tab */}
        {activeTab === 'imports' && (
          <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '10px', padding: '24px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Ingest Conversation Archive</h3>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                style={{ padding: '8px 12px', background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', borderRadius: '6px' }}
              >
                <option value="chatgpt">ChatGPT (conversations.json)</option>
                <option value="claude">Anthropic Claude (conversations.json)</option>
                <option value="gemini">Google Gemini (JSON Takeout)</option>
                <option value="generic">Generic Standard JSON</option>
              </select>
              <button
                onClick={handleImport}
                style={{ padding: '8px 20px', background: '#0284c7', border: 'none', color: '#fff', borderRadius: '6px', fontWeight: '600', cursor: 'pointer' }}
              >
                Ingest & Extract
              </button>
            </div>
            <textarea
              rows={8}
              value={importJson}
              onChange={(e) => setImportJson(e.target.value)}
              placeholder="Paste raw JSON export archive contents here..."
              style={{ width: '100%', padding: '12px', background: '#020617', border: '1px solid #1e293b', borderRadius: '8px', color: '#f8fafc', fontFamily: 'monospace', fontSize: '13px' }}
            />
            {importStatus && (
              <div style={{ marginTop: '12px', fontSize: '13px', color: importStatus.startsWith('Error') ? '#ef4444' : '#10b981' }}>
                {importStatus}
              </div>
            )}
          </div>
        )}

        {/* Explain / Provenance Modal */}
        {selectedMemory && (
          <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
            <div style={{ width: '640px', background: '#0f172a', border: '1px solid #334155', borderRadius: '12px', padding: '24px', maxHeight: '80vh', overflowY: 'auto' }}>
              <h3 style={{ margin: '0 0 12px 0', fontSize: '18px' }}>Explain This Memory (Provenance)</h3>
              <div style={{ fontSize: '15px', fontWeight: '600', color: '#38bdf8', marginBottom: '8px' }}>
                {selectedMemory.statement}
              </div>
              <div style={{ fontSize: '13px', color: '#94a3b8', marginBottom: '16px' }}>
                Type: {selectedMemory.memory_type} | Status: {selectedMemory.status} | Confidence: {(selectedMemory.confidence * 100).toFixed(0)}%
              </div>

              <div style={{ background: '#020617', padding: '12px', borderRadius: '8px', border: '1px solid #1e293b', marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '4px' }}>Original Source Message Content</div>
                <div style={{ fontSize: '13px', color: '#f1f5f9', whiteSpace: 'pre-wrap' }}>
                  {selectedMemory.relevant_source_content || "Direct user assertion / import"}
                </div>
              </div>

              {selectedMemory.source_conversation && (
                <div style={{ fontSize: '12px', color: '#64748b', marginBottom: '16px' }}>
                  Source Conversation: "{selectedMemory.source_conversation.title}" ({selectedMemory.source_conversation.provider})
                </div>
              )}

              <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', borderTop: '1px solid #1e293b', paddingTop: '16px' }}>
                <button
                  onClick={() => handleResolve(selectedMemory.memory_id, 'active')}
                  style={{ padding: '6px 14px', background: '#10b981', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
                >
                  Confirm Active
                </button>
                <button
                  onClick={() => handleResolve(selectedMemory.memory_id, 'supersede')}
                  style={{ padding: '6px 14px', background: '#f59e0b', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
                >
                  Supersede
                </button>
                <button
                  onClick={() => handleResolve(selectedMemory.memory_id, 'deprecate')}
                  style={{ padding: '6px 14px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
                >
                  Deprecate
                </button>
                <button
                  onClick={() => setSelectedMemory(null)}
                  style={{ padding: '6px 14px', background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
export default App;
