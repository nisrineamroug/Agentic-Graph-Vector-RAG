import React, { useState, useEffect, useMemo } from "react";
import useSWR from "swr";
import NavBar from "../components/NavBar";
import api from "../lib/api";
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
// Use window-based require for react-force-graph to avoid SSR and hydration issues
let ForceGraph2D = () => <div className="small-muted" style={{padding: 20}}>Loading visualization...</div>;
if (typeof window !== "undefined") {
  ForceGraph2D = require("react-force-graph-2d").default;
}
const fetcher = (url) => api.get(url).then((res) => res.data);

export default function AgenticDashboard() {
  const { data: policy } = useSWR("/agentic/policy", fetcher);
  const { data: history } = useSWR("/agentic/reward-history", fetcher, { refreshInterval: 3000 });
  
  const [result, setResult] = useState(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    const saved = localStorage.getItem("last_query_result");
    if (saved) {
      try {
        setResult(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load saved result", e);
      }
    }
    
    // Listen for storage changes (updates when query is run on other page)
    const handleStorage = () => {
      const saved = localStorage.getItem("last_query_result");
      if (saved) setResult(JSON.parse(saved));
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const chartData = useMemo(() => {
    if (!history || history.length === 0) return [];
    return history.map((h, i) => ({
      iteration: i + 1,
      reward: (h.reward || 0) * 100
    }));
  }, [history]);

  if (!isClient) return null;

  return (
    <div className="app-shell">
      <NavBar />
      <div className="agentic-dashboard">
        <div className="dashboard-header">
          <h2>Agentic Graph RAG Control Center</h2>
        </div>

        <div className="dashboard-grid">
          
          {/* TOP ROW: Policy & Reward */}
          <div className="grid-row top">
            <div className="dash-card policy-viz">
              <div className="card-header">Policy Visualization</div>
              <div className="policy-content">
                <div className="q-learning-diagram">
                   <div className="state-nodes">
                     {policy?.slice(0, 3).map((p, i) => (
                       <div key={i} className={`state-node ${result?.thought_process?.state === p.state ? 'active' : ''}`}>
                         {p.state}
                       </div>
                     ))}
                   </div>
                   <div className="flow-arrow">→</div>
                   <div className="action-nodes">
                      <div className={`action-node ${result?.route === 'graph' ? 'highlight' : ''}`}>Graph Action</div>
                      <div className={`action-node ${result?.route === 'vectorial' ? 'highlight' : ''}`}>Vector Action</div>
                   </div>
                </div>
                <div className="policy-label">Q-Learning Policy Transition</div>
              </div>
            </div>

            <div className="dash-card reward-monitor">
              <div className="card-header">Reward Monitor</div>
              <div className="reward-chart-wrapper">
                <div className="chart-header">Reward Over Time</div>
                <ResponsiveContainer width="100%" height="80%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="iteration" hide />
                    <YAxis domain={[0, 100]} stroke="#475569" fontSize={10} />
                    <Area type="monotone" dataKey="reward" stroke="var(--accent)" fill="rgba(56, 189, 248, 0.2)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
                <div className="chart-footer">Iterations</div>
              </div>
            </div>
          </div>

          {/* MIDDLE ROW: Decision Path */}
          <div className="grid-row full">
            <div className="dash-card decision-path">
              <div className="card-header">Decision Path</div>
              <div className="path-flow">
                <div className="path-step">User Query</div>
                <div className="path-arrow">→</div>
                <div className="path-step">Semantic Analysis</div>
                <div className="path-arrow">→</div>
                <div className="path-step">Systematic Analysis</div>
                <div className="path-arrow">→</div>
                <div className="path-step active">{result?.route?.toUpperCase() || 'IDLE'} RAG</div>
              </div>
            </div>
          </div>

          {/* BOTTOM ROW: Execution, Feedback, Result */}
          <div className="grid-row bottom">
            <div className="dash-card graph-execution">
              <div className="card-header">Graph Query Execution</div>
              <div className="execution-content">
                <div className="neo4j-results">
                   <div className="small-muted">Neo4j Query Results</div>
                   <div className="query-code">{result?.thought_process?.cypher_queries?.[0]?.cypher || 'No query executed'}</div>
                </div>
                <div className="extracted-subgraph">
                   <div className="small-muted">Extracted Subgraph</div>
                   <div className="mini-graph">
                      <ForceGraph2D
                        graphData={result?.thought_process?.subgraph || { nodes: [], links: [] }}
                        height={120}
                        width={200}
                        nodeAutoColorBy="group"
                        enableNodeDrag={false}
                        enableZoom={false}
                        backgroundColor="rgba(0,0,0,0)"
                        nodeCanvasObject={(node, ctx, globalScale) => {
                          const label = node.text || node.id;
                          const fontSize = 10 / globalScale;
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
                </div>
              </div>
            </div>

            <div className="dash-card feedback-loop">
              <div className="card-header">Hybrid Feedback Loop</div>
              <div className="feedback-content">
                <div className="loop-item">
                  <div className="loop-label">Q-Table Update</div>
                  <div className="loop-val">+{result?.reward_details?.reward?.toFixed(2) || '0.00'}</div>
                </div>
                <div className="loop-animation">
                   <div className="sync-icon">🔄</div>
                   <div className="brain-icon">🧠</div>
                </div>
                <div className="loop-footer">Learning from Rewards</div>
              </div>
            </div>

            <div className="dash-card final-answer">
              <div className="card-header">Final Answer</div>
              <div className="answer-content">
                <div className="small-muted">Generated Response</div>
                <div className="truncated-answer">{result?.answer?.substring(0, 150)}...</div>
                <div className="confidence-score">
                  Confidence Score: <span className="highlight">{(result?.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>

      <style jsx>{`
        .agentic-dashboard {
          padding: 20px;
          height: calc(100vh - 80px);
          overflow-y: auto;
          background: #020617;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .header-tabs { display: flex; gap: 10px; }
        .tab { 
          background: rgba(255,255,255,0.03); 
          border: 1px solid rgba(255,255,255,0.1); 
          color: white; 
          padding: 6px 15px; 
          border-radius: 6px; 
          font-size: 12px;
          cursor: pointer;
        }
        .tab.active { background: var(--accent); border-color: var(--accent); font-weight: 700; }

        .dashboard-grid { display: flex; flex-direction: column; gap: 20px; }
        .grid-row { display: flex; gap: 20px; }
        .grid-row.top { height: 300px; }
        .grid-row.full { height: 100px; }
        .grid-row.bottom { height: 320px; }

        .dash-card {
          flex: 1;
          background: rgba(15, 23, 42, 0.6);
          border: 1px solid rgba(255,255,255,0.05);
          border-radius: 12px;
          padding: 15px;
          display: flex;
          flex-direction: column;
        }
        .card-header {
          font-size: 13px;
          font-weight: 800;
          color: #38bdf8;
          margin-bottom: 15px;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        /* Policy Viz */
        .q-learning-diagram { display: flex; align-items: center; justify-content: center; gap: 20px; margin-top: 20px; }
        .state-nodes { display: flex; flex-direction: column; gap: 10px; }
        .state-node { 
          background: #1e293b; 
          border: 1px solid #334155; 
          padding: 8px 12px; 
          border-radius: 20px; 
          font-size: 11px; 
          opacity: 0.5;
        }
        .state-node.active { border-color: var(--accent); background: rgba(56, 189, 248, 0.1); opacity: 1; box-shadow: 0 0 10px rgba(56, 189, 248, 0.2); }
        .action-node { 
          background: #1e293b; 
          border: 1px solid #334155; 
          padding: 10px 15px; 
          border-radius: 6px; 
          font-size: 12px; 
          margin-bottom: 10px; 
          opacity: 0.7;
        }
        .action-node.highlight { background: rgba(251, 191, 36, 0.1); border-color: #fbbf24; color: #fbbf24; opacity: 1; }
        .policy-label { text-align: center; font-size: 10px; margin-top: 15px; opacity: 0.4; }

        /* Reward Monitor */
        .reward-chart-wrapper { flex: 1; display: flex; flex-direction: column; }
        .chart-header { text-align: center; font-size: 11px; margin-bottom: 5px; opacity: 0.7; }
        .chart-footer { text-align: center; font-size: 10px; margin-top: 5px; opacity: 0.4; }

        /* Decision Path */
        .path-flow { display: flex; align-items: center; justify-content: center; gap: 15px; height: 100%; }
        .path-step { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); padding: 8px 15px; border-radius: 6px; font-size: 12px; }
        .path-step.active { background: var(--accent); color: white; border-color: var(--accent); font-weight: 800; }
        .path-arrow { opacity: 0.3; }

        /* Bottom Row */
        .execution-content { display: flex; gap: 15px; }
        .neo4j-results { flex: 1.5; }
        .query-code { 
          background: rgba(0,0,0,0.3); 
          padding: 8px; 
          border-radius: 4px; 
          font-family: monospace; 
          font-size: 10px; 
          margin-top: 10px; 
          color: #94a3b8;
          height: 120px;
          overflow-y: auto;
        }
        .extracted-subgraph { flex: 1; }
        .mini-graph { background: rgba(0,0,0,0.2); border-radius: 8px; margin-top: 10px; height: 120px; display: flex; align-items: center; justify-content: center; }

        .feedback-content { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .loop-animation { display: flex; gap: 20px; font-size: 30px; margin: 20px 0; }
        .loop-footer { font-size: 11px; opacity: 0.5; }

        .answer-content { display: flex; flex-direction: column; gap: 10px; height: 100%; justify-content: center; }
        .truncated-answer { font-size: 13px; opacity: 0.8; line-height: 1.5; background: rgba(255,255,255,0.02); padding: 10px; border-radius: 6px; }
        .confidence-score { font-size: 14px; font-weight: 700; }
        .highlight { color: var(--accent); }

        .small-muted { font-size: 10px; font-weight: 700; opacity: 0.5; }
      `}</style>
    </div>
  );
}
