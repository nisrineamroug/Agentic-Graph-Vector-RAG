import React from "react";
import NavBar from "../components/NavBar";

export default function Home() {
  return (
    <div className="app-shell">
      <NavBar />
      <div className="content">
        <div
          className="page"
          style={{ alignItems: "center", justifyContent: "center" }}
        >
          <div style={{ textAlign: "center" }}>
            <h1 style={{ color: "#fff" }}>PFA RAG Demo</h1>
            <p className="small-muted">
              Use the buttons above to navigate: Vectorial, Graph, Agentic or
              Query.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
