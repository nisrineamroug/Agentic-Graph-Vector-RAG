import React from "react";
import { useRouter } from "next/router";

export default function NavBar() {
  const router = useRouter();
  const items = [
    { key: "vectorial", label: "Vectorial RAG", href: "/vectorial" },
    { key: "graph", label: "Graph RAG", href: "/graph" },
    { key: "agentic", label: "Agentic RAG", href: "/agentic" },
    { key: "query", label: "Query", href: "/query" },
  ];

  return (
    <div className="topbar">
      {items.map((it) => (
        <button
          key={it.key}
          className={`nav-button ${router.pathname === it.href ? "active" : ""}`}
          onClick={() => router.push(it.href)}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
