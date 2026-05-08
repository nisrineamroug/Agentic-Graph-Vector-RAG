import React from "react";
import useSWR from "swr";
import NavBar from "../components/NavBar";
import api from "../lib/api";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const fetcher = (url) => api.get(url).then((res) => res.data);

export default function VectorialPage() {
  const { data: stats } = useSWR("/vectorial/stats", fetcher);
  const { data: evals } = useSWR("/vectorial/eval-metrics", fetcher);

  return (
    <div className="app-shell">
      <NavBar />
      <div className="content">
        <div className="page" style={{ padding: "24px", gap: "24px", display: "flex", flexDirection: "row", overflowY: "auto" }}>
          
          {/* Left: Stats & Samples */}
          <div className="section" style={{ flex: 1, minWidth: "300px" }}>
            <h3>Vectorial RAG Config</h3>
            <div className="small-muted" style={{ marginBottom: 20 }}>
              Indexing and chunking parameters.
            </div>

            <div className="section-scroll-area">
              {stats ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
                  <div className="chunk" style={{ borderLeft: "4px solid #4ade80" }}>
                    <div className="small-muted">Index Status</div>
                    <div style={{ fontSize: "16px", fontWeight: 700, color: stats.loaded ? "#4ade80" : "#fb7185" }}>
                      {stats.loaded ? "LOADED & ACTIVE" : "NOT LOADED"}
                    </div>
                  </div>

                  <div className="chunk">
                    <div className="small-muted">Chunking Strategy</div>
                    <div style={{ fontSize: "14px", fontWeight: 700, color: "var(--accent)" }}>{stats.chunking_method || "N/A"}</div>
                    <div style={{ display: "flex", gap: 15, marginTop: 8 }}>
                      <div>
                        <div className="small-muted" style={{ fontSize: "10px" }}>SIZE</div>
                        <div style={{ fontSize: "12px" }}>{stats.chunk_params?.size || 1500} chars</div>
                      </div>
                      <div>
                        <div className="small-muted" style={{ fontSize: "10px" }}>OVERLAP</div>
                        <div style={{ fontSize: "12px" }}>{stats.chunk_params?.overlap || 400} chars</div>
                      </div>
                    </div>
                  </div>

                  <h3 style={{ marginTop: 10 }}>Chunk Samples</h3>
                  <div className="chunks-list">
                    {(stats.samples || []).map((s, i) => (
                      <div key={i} className="chunk" style={{ fontSize: "12px", background: "rgba(255,255,255,0.02)" }}>
                        <div style={{ color: "var(--accent)", marginBottom: 5, fontWeight: 700 }}>{s.page}</div>
                        <div style={{ fontStyle: "italic", opacity: 0.8 }}>"{s.text}"</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="small-muted">Loading configuration...</div>
              )}
            </div>
          </div>

          {/* Middle: Visualization (PCA) */}
          <div className="section" style={{ flex: 1.2, minWidth: "400px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3>Embeddings Cloud (PCA)</h3>
              {stats && (
                <span className="small-muted" style={{ fontSize: "10px", background: "rgba(56, 189, 248, 0.1)", padding: "4px 8px", borderRadius: "12px", color: "var(--accent)", fontWeight: 700 }}>
                  {stats.model?.split('/').pop()}
                </span>
              )}
            </div>
            <div className="small-muted" style={{ marginBottom: 15 }}>
              Visual cluster distribution of all chunks.
            </div>
            <div className="section-scroll-area">
               <div style={{ 
                width: "100%", 
                background: "#050b16", 
                borderRadius: "16px", 
                border: "1px solid var(--border)",
                overflow: "hidden",
                boxShadow: "0 8px 32px rgba(0,0,0,0.4)"
              }}>
                <img 
                  src={`${api.defaults.baseURL}/vectorial/pca_image?t=${Date.now()}`} 
                  alt="PCA Nuage" 
                  style={{ width: "100%", display: "block" }}
                />
              </div>
            </div>
          </div>

          {/* Right: Evaluation & Metrics */}
          <div className="section" style={{ flex: 1.5, minWidth: "450px" }}>
            <h3>Evaluation Benchmarks</h3>
            <div className="small-muted" style={{ marginBottom: 20 }}>
              Performance of different chunking & retrieval methods.
            </div>

            <div className="section-scroll-area">
              {evals && evals.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  
                  {/* Leaderboard Table */}
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "14px", marginBottom: 12 }}>Retrieval Leaderboard</div>
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                        <thead>
                          <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                            <th style={{ padding: "8px" }}>Config</th>
                            <th style={{ padding: "8px" }}>Recall@5</th>
                            <th style={{ padding: "8px" }}>MRR@5</th>
                            <th style={{ padding: "8px" }}>Winner</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evals.map((e, idx) => (
                            <tr key={idx} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: e.winner === "YES" ? "rgba(56, 189, 248, 0.05)" : "transparent" }}>
                              <td style={{ padding: "8px", fontWeight: 600 }}>{e.config}</td>
                              <td style={{ padding: "8px" }}>{(e.recall_at_k * 100).toFixed(1)}%</td>
                              <td style={{ padding: "8px" }}>{e.mrr_at_k.toFixed(3)}</td>
                              <td style={{ padding: "8px", color: e.winner === "YES" ? "#4ade80" : "inherit" }}>{e.winner}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Chart: Hit Rates */}
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "14px", marginBottom: 12 }}>Chunk Hit Rate by Config</div>
                    <div style={{ width: "100%", height: "200px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={evals}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                          <XAxis dataKey="config" fontSize={10} stroke="#475569" axisLine={false} tickLine={false} />
                          <YAxis fontSize={10} stroke="#475569" axisLine={false} tickLine={false} domain={[0, 1]} />
                          <Tooltip 
                            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '11px' }}
                          />
                          <Bar dataKey="chunk_hit_rate" radius={[4, 4, 0, 0]}>
                            {evals.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.winner === "YES" ? "var(--accent)" : "rgba(56, 189, 248, 0.3)"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Summary Card */}
                  <div className="chunk" style={{ borderLeft: "4px solid var(--accent)", background: "rgba(56, 189, 248, 0.05)" }}>
                    <div style={{ fontWeight: 700, fontSize: "14px", marginBottom: 5 }}>Optimization Insight</div>
                    <div style={{ fontSize: "12px", lineHeight: "1.5" }}>
                      {evals[0]?.summary || "The evaluation shows that a chunk size of 1000 with 200 overlap provides the best balance between recall and precision."}
                    </div>
                  </div>

                </div>
              ) : (
                <div className="small-muted" style={{ textAlign: "center", padding: 40 }}>
                  No evaluation data found. Run evaluate_retrieval.py to generate metrics.
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
