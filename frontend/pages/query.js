import React, { useState, useEffect, useRef } from "react";
import useSWR from "swr";
import NavBar from "../components/NavBar";
import api from "../lib/api";
// Use window-based require for react-force-graph to avoid SSR and hydration issues
let ForceGraph2D = () => <div className="small-muted" style={{padding: 20}}>Loading visualization...</div>;
if (typeof window !== "undefined") {
  ForceGraph2D = require("react-force-graph-2d").default;
}
const fetcher = (url) => api.get(url).then((res) => res.data);

export default function QueryDashboard() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  
  const { data: history, mutate: mutateHistory } = useSWR("/query/history", fetcher);

  useEffect(() => {
    const saved = localStorage.getItem("last_query_result");
    if (saved) {
      try {
        setResult(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load saved result", e);
      }
    }
  }, []);

  const handleQuery = async (e, customQuery = null) => {
    e?.preventDefault();
    const targetQuery = customQuery || query;
    if (!targetQuery) return;
    
    setLoading(true);
    setResult(null);
    setActiveStep(0);
    setQuery(targetQuery);

    try {
      // Phase simulation for dynamic feel
      await new Promise(r => setTimeout(r, 600));
      setActiveStep(1);
      
      const res = await api.post("/query/", { query: targetQuery });
      
      await new Promise(r => setTimeout(r, 600));
      setActiveStep(2);
      
      setResult(res.data);
      localStorage.setItem("last_query_result", JSON.stringify(res.data));
      mutateHistory();
      setActiveStep(3);
    } catch (err) {
      console.error("Query failed:", err);
      setActiveStep(-1);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    if (confirm("Clear all query history and learned policy?")) {
      await api.post("/agentic/reset");
      setResult(null);
      localStorage.removeItem("last_query_result");
      mutateHistory();
    }
  };

  return (
    <div className="app-shell">
      <NavBar />
      <div className="dashboard-container">
        {/* Left Sidebar: Query & History */}
        <div className="dash-section sidebar">
          <div className="section-header">
            <h3>Query Interface</h3>
            <button className="text-btn" onClick={clearHistory}>Clear</button>
          </div>
          
          <form onSubmit={handleQuery} className="input-area">
            <textarea
              placeholder="Enter your research query..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleQuery(e);
                }
              }}
            />
            <button className="btn-submit" type="submit" disabled={loading}>
              {loading ? "PROCESSING..." : "EXECUTE"}
            </button>
          </form>

          <div className="history-area">
            <div className="small-muted" style={{ marginBottom: 10 }}>RECENT QUERIES</div>
            <div className="history-list">
              {(history || []).map((h, i) => (
                <button key={i} className="history-item" onClick={(e) => handleQuery(e, h)}>
                  {h}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Middle Section: Process & Answer */}
        <div className="dash-section main-content">
          <div className="section-header">
            <h3>Agentic Response</h3>
            {result && <span className="route-tag">{result.route?.toUpperCase()} RAG</span>}
          </div>

          <div className="response-scroll">
            {loading ? (
              <div className="status-container">
                <div className="loader"></div>
                <div className="status-text">
                  {activeStep === 0 && "Analyzing query features..."}
                  {activeStep === 1 && "Routing to optimal retrieval store..."}
                  {activeStep === 2 && "Synthesizing evidence via LLM..."}
                </div>
              </div>
            ) : result ? (
              <div className="answer-wrapper">
                <div className="answer-text">
                  {result.answer}
                </div>
                <div className="answer-meta">
                  <div className="meta-item">
                    <span className="meta-label">Confidence</span>
                    <span className="meta-value">{(result.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Reward</span>
                    <span className="meta-value" style={{ color: "var(--accent)" }}>+{(result.reward_details?.reward || 0).toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="placeholder-msg">
                System idle. Awaiting input.
              </div>
            )}
          </div>
        </div>

        {/* Right Section: Retrieved Evidence (Dynamic) */}
        <div className="dash-section evidence-panel">
          <div className="section-header">
            <h3>Retrieved Evidence</h3>
          </div>
          
          <div className="evidence-scroll">
            {result ? (
              <div className="evidence-content">
                
                {/* Always show Graph if Graph or Hybrid */}
                {(result.route === 'graph' || result.route === 'hybrid') && (
                  <div className="evidence-block">
                    <div className="block-title">Knowledge Subgraph</div>
                    <div className="graph-mini-viz">
                      <ForceGraph2D
                        graphData={result.thought_process?.subgraph || { nodes: [], links: [] }}
                        height={250}
                        width={350}
                        nodeAutoColorBy="group"
                        linkColor={() => "rgba(255,255,255,0.15)"}
                        backgroundColor="rgba(0,0,0,0)"
                        nodeCanvasObject={(node, ctx, globalScale) => {
                          const label = node.text || node.id;
                          const fontSize = 12 / globalScale;
                          ctx.font = `${fontSize}px Inter, sans-serif`;
                          const textWidth = ctx.measureText(label).width;
                          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); 

                          ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
                          ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

                          ctx.textAlign = 'center';
                          ctx.textBaseline = 'middle';
                          ctx.fillStyle = node.color || '#38bdf8';
                          ctx.fillText(label, node.x, node.y);
                        }}
                      />
                    </div>
                    {result.thought_process?.relationships?.length > 0 && (
                      <div className="rel-list">
                        {result.thought_process.relationships.slice(0, 5).map((r, i) => (
                          <div key={i} className="rel-item">
                            {r.source} → <b>{r.rel_type}</b> → {r.target}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Always show Vector Chunks if Vector or Hybrid */}
                {(result.route === 'vectorial' || result.route === 'hybrid') && (
                  <div className="evidence-block">
                    <div className="block-title">Vector Snippets</div>
                    <div className="chunk-list">
                      {result.sources.filter(s => s.text).map((s, i) => (
                        <div key={i} className="snippet-item">
                          <div className="snippet-header">PAGE {s.page} | SIMILARITY {(s.score * 100).toFixed(0)}%</div>
                          <div className="snippet-text">{s.text}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            ) : (
              <div className="placeholder-msg">Contextual data will appear here.</div>
            )}
          </div>
        </div>
      </div>

      <style jsx>{`
        .dashboard-container {
          display: flex;
          height: calc(100vh - 80px);
          width: 100%;
          overflow: hidden;
          background: #020617;
        }
        .dash-section {
          display: flex;
          flex-direction: column;
          border-right: 1px solid rgba(255,255,255,0.05);
          height: 100%;
        }
        .sidebar { flex: 0 0 300px; padding: 20px; }
        .main-content { flex: 1; padding: 20px; background: rgba(255,255,255,0.01); }
        .evidence-panel { flex: 0 0 400px; padding: 20px; border-right: none; }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }
        .section-header h3 { font-size: 14px; margin: 0; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; }
        
        .input-area { display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }
        textarea {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 8px;
          color: white;
          padding: 12px;
          min-height: 120px;
          resize: none;
          outline: none;
          font-family: inherit;
        }
        textarea:focus { border-color: var(--accent); }
        .btn-submit {
          background: var(--accent);
          color: white;
          border: none;
          padding: 10px;
          border-radius: 6px;
          font-weight: 800;
          cursor: pointer;
        }

        .history-list { display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
        .history-item {
          background: transparent;
          border: 1px solid rgba(255,255,255,0.05);
          color: #94a3b8;
          padding: 10px;
          border-radius: 6px;
          text-align: left;
          font-size: 12px;
          cursor: pointer;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .history-item:hover { border-color: var(--accent); color: white; }

        .response-scroll, .evidence-scroll { flex: 1; overflow-y: auto; padding-right: 5px; }
        .placeholder-msg { height: 100%; display: flex; align-items: center; justify-content: center; opacity: 0.3; font-style: italic; }

        .route-tag { font-size: 10px; background: var(--accent); color: white; padding: 2px 8px; border-radius: 12px; font-weight: 800; }
        .answer-wrapper { background: rgba(15, 23, 42, 0.4); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05); }
        .answer-text { font-size: 1.1rem; line-height: 1.7; margin-bottom: 20px; color: #f1f5f9; }
        .answer-meta { display: flex; gap: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); }
        .meta-item { display: flex; flex-direction: column; }
        .meta-label { font-size: 10px; opacity: 0.5; text-transform: uppercase; }
        .meta-value { font-weight: 800; font-size: 16px; }

        .evidence-block { margin-bottom: 30px; }
        .block-title { font-size: 12px; font-weight: 800; color: var(--accent); margin-bottom: 15px; border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 5px; }
        .graph-mini-viz { background: rgba(0,0,0,0.2); border-radius: 8px; margin-bottom: 10px; }
        .rel-item { font-size: 11px; padding: 4px 8px; background: rgba(255,255,255,0.02); margin-bottom: 4px; border-radius: 4px; }
        
        .snippet-item { background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 3px solid #334155; }
        .snippet-header { font-size: 9px; font-weight: 800; opacity: 0.5; margin-bottom: 5px; }
        .snippet-text { font-size: 12px; line-height: 1.5; opacity: 0.8; }

        .status-container { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px; }
        .loader { width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.1); border-top-color: var(--accent); border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-text { font-size: 14px; opacity: 0.7; letter-spacing: 0.5px; }

        .text-btn { background: none; border: none; color: #fb7185; font-size: 11px; cursor: pointer; text-decoration: underline; opacity: 0.7; }
        .text-btn:hover { opacity: 1; }
      `}</style>
    </div>
  );
}
