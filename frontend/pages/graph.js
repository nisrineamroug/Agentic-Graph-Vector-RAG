import React, { useEffect, useMemo, useState, useRef, useCallback } from "react";
import useSWR from "swr";
import NavBar from "../components/NavBar";
import api from "../lib/api";

// Old-school dynamic import to ensure ref forwarding works better than next/dynamic
let ForceGraph2D = () => <div style={{ color: "#94a3b8", padding: 20 }}>Loading visualization engine...</div>;
if (typeof window !== "undefined") {
  ForceGraph2D = require("react-force-graph-2d").default;
}

const fetcher = (url) => api.get(url).then((res) => res.data);

export default function GraphPage() {
  const fgRef = useRef();
  const [selectedCommunity, setSelectedCommunity] = useState(null);
  const [communityGraph, setCommunityGraph] = useState({ nodes: [], links: [] });
  const [loadingCommunity, setLoadingCommunity] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [minWeight, setMinWeight] = useState(0.1);
  const [relType, setRelType] = useState("ALL");
  const [hoveredNode, setHoveredNode] = useState(null);
  const [isClient, setIsClient] = useState(false);

  // SWR for primary data
  const { data: vizData, mutate: mutateViz } = useSWR("/graph/visualization", fetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 60000,
  });

  const { data: commData, mutate: mutateComm } = useSWR("/graph/communities", fetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 60000,
  });

  const { data: metrics } = useSWR(
    selectedCommunity !== null ? `/graph/metrics?community_id=${selectedCommunity}` : "/graph/metrics",
    fetcher,
    { revalidateOnFocus: false }
  );

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleFit = useCallback(() => {
    if (fgRef.current && typeof fgRef.current.zoomToFit === "function") {
      fgRef.current.zoomToFit(600, 80);
    }
  }, []);

  useEffect(() => {
    if (vizData && isClient) {
      const timer = setTimeout(handleFit, 2000);
      return () => clearTimeout(timer);
    }
  }, [vizData, isClient, handleFit]);

  const graphData = vizData || { nodes: [], links: [] };
  const sortedCommunities = useMemo(() => {
    const raw = commData?.communities || [];
    return [...raw].sort((a, b) => a.community_id - b.community_id);
  }, [commData]);

  const activeGraph = selectedCommunity !== null ? communityGraph : graphData;

  const relationTypes = useMemo(() => {
    const types = new Set((activeGraph.links || []).map((l) => l.value || "UNKNOWN"));
    return ["ALL", ...Array.from(types)];
  }, [activeGraph]);

  const filteredData = useMemo(() => {
    const links = (activeGraph.links || []).filter((l) => {
      const w = Number(l.weight ?? 1);
      const typeOk = relType === "ALL" ? true : l.value === relType;
      return w >= minWeight && typeOk;
    });

    const nodeIds = new Set();
    links.forEach((l) => {
      nodeIds.add(typeof l.source === 'object' ? l.source.id : l.source);
      nodeIds.add(typeof l.target === 'object' ? l.target.id : l.target);
    });

    const nodes = (activeGraph.nodes || []).filter(
      (n) => nodeIds.has(n.id) || !links.length
    );

    return { nodes, links };
  }, [activeGraph, relType, minWeight]);

  async function openCommunity(communityId) {
    setSelectedCommunity(communityId);
    setLoadingCommunity(true);
    try {
      const res = await api.get(
        `/graph/communities/${communityId}?min_weight=${minWeight}&limit=1200`
      );
      setCommunityGraph({
        nodes: res.data.nodes || [],
        links: res.data.links || [],
      });
      setTimeout(handleFit, 1000);
    } catch (err) {
      console.error("Failed to load community subgraph:", err);
      setCommunityGraph({ nodes: [], links: [] });
    } finally {
      setLoadingCommunity(false);
    }
  }

  function resetCommunityView() {
    setSelectedCommunity(null);
    setCommunityGraph({ nodes: [], links: [] });
    setTimeout(handleFit, 1000);
  }

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      const [newViz, newComm] = await Promise.all([
        api.get("/graph/visualization?refresh=true").then(res => res.data),
        api.get("/graph/communities?refresh=true").then(res => res.data)
      ]);
      mutateViz(newViz, false);
      mutateComm(newComm, false);
      resetCommunityView();
    } catch (err) {
      console.error("Refresh failed:", err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const getCommunityColor = (communityId) => {
    if (communityId == null) return "#94a3b8";
    const colors = [
      "#38bdf8", "#fb7185", "#34d399", "#fbbf24", "#a78bfa", 
      "#f472b6", "#818cf8", "#4ade80", "#fb923c", "#2dd4bf"
    ];
    return colors[communityId % colors.length];
  };

  return (
    <div className="app-shell">
      <NavBar />
      <div className="content">
        <div className="page">
          {/* Main Graph Section */}
          <div className="section" style={{ flex: 2, position: "relative", padding: 0 }}>
            {/* Control Panel (Top Left) */}
            <div style={{ 
              position: "absolute", 
              top: 20, 
              left: 20, 
              zIndex: 100,
              background: "rgba(11, 31, 58, 0.9)", 
              padding: "15px", 
              borderRadius: "12px", 
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(255,255,255,0.2)",
              width: "220px",
              pointerEvents: "auto"
            }}>
              <h3 style={{ margin: "0 0 5px 0", fontSize: "1.1rem" }}>Graph Explorer</h3>
              <div className="small-muted" style={{ marginBottom: 15 }}>
                {selectedCommunity !== null ? `Community ${selectedCommunity}` : "Knowledge Map"}
              </div>
              
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div>
                  <label className="small-muted" style={{ display: "block", marginBottom: 5 }}>Min weight: {minWeight}</label>
                  <input
                    type="range"
                    min="0"
                    max="5"
                    step="0.1"
                    value={minWeight}
                    onChange={(e) => setMinWeight(Number(e.target.value))}
                    style={{ width: "100%", cursor: "pointer" }}
                  />
                </div>

                <div>
                  <label className="small-muted" style={{ display: "block", marginBottom: 5 }}>Relation</label>
                  <select
                    value={relType}
                    onChange={(e) => setRelType(e.target.value)}
                    className="query-input"
                    style={{ width: "100%", padding: "6px", fontSize: "12px", background: "#050b16" }}
                  >
                    {relationTypes.map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>

                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn" style={{ flex: 1, padding: "8px", fontSize: "12px" }} onClick={handleFit}>Fit View</button>
                  <button 
                    className="btn" 
                    style={{ flex: 1, padding: "8px", fontSize: "12px", opacity: isRefreshing ? 0.6 : 1 }} 
                    onClick={handleRefresh}
                    disabled={isRefreshing}
                  >
                    {isRefreshing ? "..." : "Reload"}
                  </button>
                </div>
              </div>
            </div>

            {/* Stats Overlay (Top Right) */}
            <div style={{
              position: "absolute",
              top: 20,
              right: 20,
              zIndex: 100,
              background: "rgba(11, 31, 58, 0.7)",
              padding: "10px 20px",
              borderRadius: "30px",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(255,255,255,0.1)",
              display: "flex",
              gap: 20,
              fontSize: "12px",
              color: "#fff"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#38bdf8" }}></div>
                <span style={{ opacity: 0.6 }}>Nodes:</span> 
                <b>{metrics?.summary?.nodes || filteredData.nodes.length}</b>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#fb7185" }}></div>
                <span style={{ opacity: 0.6 }}>Relations:</span> 
                <b>{metrics?.summary?.links || filteredData.links.length}</b>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#34d399" }}></div>
                <span style={{ opacity: 0.6 }}>Clusters:</span> 
                <b>{commData?.communities?.length || 0}</b>
              </div>
            </div>

            {isClient && (
              <ForceGraph2D
                ref={fgRef}
                graphData={filteredData}
                backgroundColor="#050b16"
                nodeLabel={(node) => `
                  <div style="background: #0b1f3a; padding: 10px; border-radius: 8px; border: 1px solid #38bdf8; color: white;">
                    <strong>${node.text || node.id}</strong><br/>
                    <span style="opacity: 0.8; font-size: 12px;">Comm: ${node.community_id ?? 'N/A'}</span>
                  </div>
                `}
                nodeColor={(node) => getCommunityColor(node.community_id)}
                nodeRelSize={6}
                linkWidth={(link) => Math.sqrt(link.weight || 1) * 0.6}
                linkColor={() => "rgba(255, 255, 255, 0.15)"}
                linkDirectionalArrowLength={3}
                linkDirectionalArrowRelPos={1}
                onNodeHover={setHoveredNode}
                 onNodeClick={(node) => {
                  if (node.community_id != null) openCommunity(node.community_id);
                }}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const label = node.text || node.id;
                  if (globalScale > 1.8 || (hoveredNode && hoveredNode.id === node.id)) {
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Inter, sans-serif`;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.fillStyle = "#ffffff";
                    ctx.fillText(label, node.x, node.y + 7);
                  }
                }}
                nodeCanvasObjectMode={() => 'after'}
                linkCanvasObject={(link, ctx, globalScale) => {
                  const isHovered = hoveredNode && (
                    (typeof link.source === 'object' ? link.source.id === hoveredNode.id : link.source === hoveredNode.id) ||
                    (typeof link.target === 'object' ? link.target.id === hoveredNode.id : link.target === hoveredNode.id)
                  );
                  
                  if (globalScale < 1.5 && !isHovered) return;
                  
                  const start = link.source;
                  const end = link.target;
                  if (typeof start !== 'object' || typeof end !== 'object') return;

                  const label = link.value || "";
                  const fontSize = 11 / globalScale;
                  ctx.font = `${isHovered ? 'bold' : 'normal'} ${fontSize}px Inter, sans-serif`;
                  const textWidth = ctx.measureText(label).width;
                  
                  const x = start.x + (end.x - start.x) / 2;
                  const y = start.y + (end.y - start.y) / 2;

                  // Background for label
                  ctx.fillStyle = isHovered ? "rgba(11, 31, 58, 0.95)" : "rgba(5, 11, 22, 0.85)";
                  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
                  ctx.shadowBlur = 4 / globalScale;
                  ctx.fillRect(x - textWidth/2 - 2/globalScale, y - fontSize/2 - 2/globalScale, textWidth + 4/globalScale, fontSize + 4/globalScale);
                  ctx.shadowBlur = 0;

                  // Text
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillStyle = isHovered ? "#38bdf8" : "rgba(255, 255, 255, 0.7)";
                  ctx.fillText(label, x, y);
                }}
                linkCanvasObjectMode={() => 'after'}
                onEngineStop={() => handleFit()}
              />
            )}
          </div>

          {/* Communities Sidebar */}
          <div className="section" style={{ flex: 0.8 }}>
            <h3>Communities</h3>
            <div className="small-muted" style={{ marginBottom: 15 }}>Thematic clusters</div>
            
            <div className="section-scroll-area">
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {sortedCommunities.map((c) => (
                  <button
                    key={c.community_id}
                    className="history-item"
                    style={{ 
                      textAlign: "left", 
                      border: selectedCommunity === c.community_id ? `2px solid ${getCommunityColor(c.community_id)}` : "1px solid #1f2a3a",
                      position: "relative",
                      padding: "12px",
                      margin: 0
                    }}
                    onClick={() => openCommunity(c.community_id)}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                      <div style={{ color: "#e6eef8", fontWeight: 700, fontSize: "14px" }}>Comm {c.community_id}</div>
                      <div style={{ 
                        width: "10px", 
                        height: "10px", 
                        borderRadius: "50%", 
                        background: getCommunityColor(c.community_id)
                      }} />
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--accent)", marginBottom: 4 }}>
                      {c.top_terms?.join(", ") || "No summary available"}
                    </div>
                    <div className="small-muted" style={{ fontSize: "11px" }}>{c.size} nodes</div>
                  </button>
                ))}
              </div>
            </div>
            
            {selectedCommunity !== null && (
              <button className="btn" style={{ marginTop: 12, width: "100%" }} onClick={resetCommunityView}>Show All Map</button>
            )}

            <div style={{ marginTop: 20 }}>
              <h3>Active Node</h3>
              <div className="chunk" style={{ minHeight: "60px", background: hoveredNode ? "rgba(56, 189, 248, 0.05)" : "transparent" }}>
                {hoveredNode ? (
                  <div style={{ wordBreak: "break-all" }}>
                    <div style={{ color: getCommunityColor(hoveredNode.community_id), fontWeight: 700 }}>
                      {hoveredNode.text || hoveredNode.id}
                    </div>
                    <div className="small-muted" style={{ fontSize: "11px" }}>
                      Community {hoveredNode.community_id ?? 'N/A'}
                    </div>
                  </div>
                ) : (
                  <div className="small-muted" style={{ fontSize: "12px" }}>Hover a node</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
