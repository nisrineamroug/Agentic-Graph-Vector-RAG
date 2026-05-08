# -*- coding: utf-8 -*-
"""
Service for building a graph store from chunks and syncing it to Neo4j when available.
"""

import json
import os
import re
import unicodedata
from collections import defaultdict, deque
from itertools import combinations
from typing import List, Dict, Any, Optional

import numpy as np
from neo4j import GraphDatabase
from transformers import pipeline

from backend.config import get_settings
from backend.services.llm import LLMService

settings = get_settings()


class GraphService:
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.graph_store_path = settings.GRAPH_STORE_PATH
        self.ner_model_name = settings.GRAPH_NER_MODEL
        self.llm = LLMService()
    _driver = None  # Class-level singleton driver
    _cache = {}     # Class-level cache to persist across requests

    def connect(self):
        """Establishes connection to Neo4j AuraDB / Local using a shared driver."""
        if not self.uri or not self.user or not self.password:
            raise ValueError(
                "Neo4j configuration is missing. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in the environment."
            )
        
        if GraphService._driver is None:
            try:
                # Use a shared driver with connection pooling
                GraphService._driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50
                )
                # Verify only once on creation
                GraphService._driver.verify_connectivity()
                print("Global Neo4j Driver initialized.")
            except Exception as e:
                print(f"Failed to initialize Neo4j driver: {e}")
                raise

        self.driver = GraphService._driver

    def close(self):
        # We don't close the shared driver per request
        pass

    @staticmethod
    def _document_node_id(source_name: str) -> str:
        return f"document::{source_name}"

    @staticmethod
    def _page_node_id(source_name: str, page_label: str) -> str:
        safe_page = re.sub(r"\s+", "_", page_label.strip()) if page_label else "unknown_page"
        return f"page::{source_name}::{safe_page}"

    @staticmethod
    def _chunk_node_id(chunk_index: int) -> str:
        return f"chunk_{chunk_index}"

    @staticmethod
    def _term_node_id(term: str) -> str:
        return f"term::{term}"

    def _upsert_node(self, nodes: Dict[str, Dict[str, Any]], node_id: str, group: str, text: str = "", **attrs):
        node = nodes.get(node_id)
        if node is None:
            node = {"id": node_id, "group": group, "text": text or ""}
            node.update({k: v for k, v in attrs.items() if v is not None})
            nodes[node_id] = node
            return node

        if text and not node.get("text"):
            node["text"] = text

        for key, value in attrs.items():
            if value is not None and key not in node:
                node[key] = value

        return node

    def _upsert_link(
        self,
        links: Dict[tuple, Dict[str, Any]],
        source_id: str,
        target_id: str,
        value: str,
        weight: float = 1.0,
        **attrs,
    ):
        key = (source_id, target_id, value)
        link = links.get(key)
        if link is None:
            link = {"source": source_id, "target": target_id, "value": value, "weight": float(weight)}
            link.update({k: v for k, v in attrs.items() if v is not None})
            links[key] = link
            return link

        link["weight"] = float(link.get("weight", 0.0)) + float(weight)
        for key_name, value_name in attrs.items():
            if value_name is not None and key_name not in link:
                link[key_name] = value_name
        return link

    def _build_local_graph_store(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        top_k_similar: int = 3,
        similarity_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        links: Dict[tuple, Dict[str, Any]] = {}
        term_cooccurrence = defaultdict(float)
        prev_chunk_by_source: Dict[str, str] = {}

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            source_name = metadata.get("source", "unknown.pdf")
            page_label = metadata.get("page_label", "") or "Unknown page"
            chunk_uid = self._chunk_node_id(i)
            page_uid = self._page_node_id(source_name, page_label)
            document_uid = self._document_node_id(source_name)
            text = chunk.get("text", "")[:1000]

            self._upsert_node(nodes, document_uid, "Document", text=source_name, source=source_name)
            self._upsert_node(nodes, page_uid, "Page", text=page_label, source=source_name, page_label=page_label)
            self._upsert_link(links, document_uid, page_uid, "HAS_PAGE", weight=1.0)

            self._upsert_node(
                nodes,
                chunk_uid,
                "Chunk",
                text=text,
                source=source_name,
                page_label=page_label,
                chunk_index=i,
            )
            self._upsert_link(links, page_uid, chunk_uid, "HAS_CHUNK", weight=1.0)

            previous_chunk_uid = prev_chunk_by_source.get(source_name)
            if previous_chunk_uid:
                self._upsert_link(links, previous_chunk_uid, chunk_uid, "NEXT_CHUNK", weight=1.0)
            prev_chunk_by_source[source_name] = chunk_uid

            terms = self._extract_terms(chunk)
            for rank, term in enumerate(terms):
                term_uid = self._term_node_id(term)
                self._upsert_node(nodes, term_uid, "Term", text=term, name=term, display=term, source=source_name)
                self._upsert_link(links, chunk_uid, term_uid, "MENTIONS", weight=1.0, rank=rank, source=source_name)

            for term_a, term_b in combinations(terms, 2):
                if term_a == term_b:
                    continue
                left_term, right_term = sorted([term_a, term_b])
                term_cooccurrence[(left_term, right_term)] += 1.0

        for (left_term, right_term), weight in term_cooccurrence.items():
            self._upsert_link(
                links,
                self._term_node_id(left_term),
                self._term_node_id(right_term),
                "CO_OCCURS_WITH",
                weight=weight,
            )

        if embeddings is not None and len(embeddings) == len(chunks):
            vectors = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = vectors / norms
            similarities = normalized @ normalized.T

            for i in range(len(chunks)):
                scores = similarities[i].copy()
                scores[i] = -1.0
                neighbor_ids = np.argsort(scores)[::-1][:top_k_similar]
                for neighbor_index in neighbor_ids:
                    score = float(scores[neighbor_index])
                    if score < similarity_threshold:
                        continue
                    self._upsert_link(
                        links,
                        self._chunk_node_id(i),
                        self._chunk_node_id(neighbor_index),
                        "SIMILAR_TO",
                        weight=score,
                        score=score,
                    )

        node_list = list(nodes.values())
        link_list = list(links.values())
        summary = {
            "document_count": len({node["source"] for node in node_list if node["group"] == "Document"}),
            "page_count": sum(1 for node in node_list if node["group"] == "Page"),
            "chunk_count": sum(1 for node in node_list if node["group"] == "Chunk"),
            "term_count": sum(1 for node in node_list if node["group"] == "Term"),
            "node_count": len(node_list),
            "link_count": len(link_list),
            "similarity_link_count": sum(1 for link in link_list if link["value"] == "SIMILAR_TO"),
            "co_occurrence_link_count": sum(1 for link in link_list if link["value"] == "CO_OCCURS_WITH"),
        }

        return {"nodes": node_list, "links": link_list, "summary": summary}

    def save_local_graph_store(self, graph_store: Dict[str, Any], output_path: Optional[str] = None) -> str:
        output_path = output_path or self.graph_store_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_store, f, ensure_ascii=False, indent=2)

        self.local_graph_store = graph_store
        return output_path

    def load_local_graph_store(self, input_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        input_path = input_path or self.graph_store_path
        if self.local_graph_store is not None:
            return self.local_graph_store

        if not os.path.exists(input_path):
            return None

        with open(input_path, "r", encoding="utf-8") as f:
            self.local_graph_store = json.load(f)

        return self.local_graph_store

    def _local_term_metrics(self, graph_store: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = graph_store.get("nodes", [])
        links = graph_store.get("links", [])
        node_by_id = {node["id"]: node for node in nodes}
        degree_counts = defaultdict(int)

        for link in links:
            source = link.get("source")
            target = link.get("target")
            if node_by_id.get(source, {}).get("group") == "Term":
                degree_counts[source] += 1
            if node_by_id.get(target, {}).get("group") == "Term":
                degree_counts[target] += 1

        ranked_terms = []
        for term_id, degree in sorted(degree_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
            node = node_by_id.get(term_id, {})
            ranked_terms.append({"term": node.get("display") or node.get("text") or term_id, "degree": degree})

        return ranked_terms

    def _local_communities(self, graph_store: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = graph_store.get("nodes", [])
        links = graph_store.get("links", [])
        node_by_id = {node["id"]: node for node in nodes if node.get("group") == "Term"}
        adjacency = defaultdict(set)

        for link in links:
            if link.get("value") != "CO_OCCURS_WITH":
                continue
            source = link.get("source")
            target = link.get("target")
            if source in node_by_id and target in node_by_id:
                adjacency[source].add(target)
                adjacency[target].add(source)

        communities = []
        visited = set()
        community_id = 0

        for term_id in sorted(node_by_id):
            if term_id in visited:
                continue

            queue = deque([term_id])
            visited.add(term_id)
            component = []

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            for component_term in component:
                term_node = node_by_id.get(component_term, {})
                communities.append(
                    {
                        "term": term_node.get("display") or term_node.get("text") or component_term,
                        "community_id": community_id,
                    }
                )
                if len(communities) >= 100:
                    return communities

            community_id += 1

        return communities

    def clear_database(self):
        """Wipes the database - USE WITH CAUTION."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ Database cleared.")

    def _get_ner_pipeline(self):
        if self.ner_pipeline is False:
            return None

        if self.ner_pipeline is not None:
            return self.ner_pipeline

        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
            )
            print(f"Loaded HF NER model: {self.ner_model_name}")
        except Exception as exc:
            print(f"Could not load HF NER model ({self.ner_model_name}): {exc}")
            self.ner_pipeline = False

        return self.ner_pipeline if self.ner_pipeline not in (None, False) else None

    @staticmethod
    def _normalize_term(term: str) -> str:
        term = term.strip().lower()
        term = unicodedata.normalize("NFKD", term)
        term = "".join(ch for ch in term if not unicodedata.combining(ch))
        term = re.sub(r"[^\w\s-]", " ", term)
        term = re.sub(r"\s+", " ", term).strip()
        return term

    @staticmethod
    def _is_valid_term(term: str) -> bool:
        """Keeps only jargon-like terms and removes numeric/noisy tokens."""
        if not term:
            return False
        if len(term) < 3:
            return False
        if not term[0].isalpha():
            return False
        if term.isdigit():
            return False
        if re.fullmatch(r"[\d_\-\s]+", term):
            return False
        if not re.search(r"[a-zA-Z]", term):
            return False
        alpha_count = sum(ch.isalpha() for ch in term)
        digit_count = sum(ch.isdigit() for ch in term)
        compact_len = len(term.replace(" ", ""))
        if alpha_count < 3:
            return False
        if compact_len > 0 and (digit_count / compact_len) > 0.25:
            return False
        tokens = term.split()
        if len(tokens) >= 3 and all(len(tok) == 1 for tok in tokens):
            return False
        unique_alpha = {ch for ch in term if ch.isalpha()}
        if alpha_count >= 5 and len(unique_alpha) <= 2:
            return False
        if re.fullmatch(r"chunk[_\-]?\d+", term):
            return False
        return True

    def _extract_terms(self, chunk: Dict[str, Any]) -> List[str]:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        raw_terms = metadata.get("keywords", [])
        terms = []

        ner_pipeline = self._get_ner_pipeline()
        if ner_pipeline is not None and text.strip():
            try:
                entities = ner_pipeline(text[:1500])
                for entity in entities:
                    value = entity.get("word") or entity.get("entity_group") or ""
                    normalized = self._normalize_term(str(value))
                    if self._is_valid_term(normalized) and normalized not in terms:
                        terms.append(normalized)
            except Exception as exc:
                print(f"HF NER failed on chunk; falling back to keywords ({exc})")

        for term in raw_terms:
            normalized = self._normalize_term(str(term))
            if self._is_valid_term(normalized) and normalized not in terms:
                terms.append(normalized)

        return terms[:12]

    def _extract_semantic_relations(self, text: str, terms: List[str]) -> List[Dict[str, Any]]:
        """Uses LLM to find semantic relationships between extracted terms within a chunk."""
        if not self.llm.api_key or len(terms) < 2:
            return []
        
        prompt = (
            f"Given this text: '{text[:800]}...'\n"
            f"And these terms: {', '.join(terms)}\n"
            "Identify semantic relationships between these terms. "
            "Use ONLY these relationship types: CONTROLS, INFLUENCES, PART_OF, INCREASES, CAUSES, DRIVES, RELATES_TO.\n"
            "Return a JSON list of objects: [{\"source\": \"term1\", \"target\": \"term2\", \"type\": \"TYPE\", \"reason\": \"brief explanation\"}]\n"
            "If no clear relationship exists, return []. Return ONLY JSON."
        )
        
        try:
            # Reuse LLM generate_answer but with custom prompt
            raw = self.llm.generate_answer("Extract relationships", text[:1000], "GraphExtraction")
            # Clean JSON if LLM added markdown wrappers
            clean_json = re.sub(r'```json|```', '', raw).strip()
            relations = json.loads(clean_json)
            return [r for r in relations if isinstance(r, dict) and r.get('type') in ['CONTROLS', 'INFLUENCES', 'PART_OF', 'INCREASES', 'CAUSES', 'DRIVES', 'RELATES_TO']]
        except Exception:
            return []

    def _create_similarity_edges(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        top_k_similar: int = 3,
        similarity_threshold: float = 0.65,
    ):
        if embeddings is None or len(embeddings) != len(chunks):
            return

        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        similarities = normalized @ normalized.T

        with self.driver.session() as session:
            for i in range(len(chunks)):
                scores = similarities[i].copy()
                scores[i] = -1.0
                neighbor_ids = np.argsort(scores)[::-1][:top_k_similar]

                for neighbor_index in neighbor_ids:
                    score = float(scores[neighbor_index])
                    if score < similarity_threshold:
                        continue

                    session.run(
                        """
                        MATCH (a:Chunk {uid: $source_uid})
                        MATCH (b:Chunk {uid: $target_uid})
                        MERGE (a)-[r:SIMILAR_TO]->(b)
                        SET r.score = $score
                        """,
                        source_uid=f"chunk_{i}",
                        target_uid=f"chunk_{neighbor_index}",
                        score=score,
                    )

    def create_graph_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
        top_k_similar: int = 3,
        similarity_threshold: float = 0.65,
        sync_to_neo4j: bool = True,
        clear_existing: bool = True,
        export_local_graph: bool = False,
    ):
        """
        Builds a graph store from text chunks and optionally syncs it to Neo4j.

        Nodes:
        - Document: thesis file source
        - Page: page-level grouping
        - Chunk: chunk-level text snippets
        - Term: extracted jargon/keywords from each chunk

        Relationships:
        - (Document)-[:HAS_PAGE]->(Page)
        - (Page)-[:HAS_CHUNK]->(Chunk)
        - (Chunk)-[:MENTIONS]->(Term)
        - (Term)-[:CO_OCCURS_WITH]->(Term)
        - (Chunk)-[:NEXT_CHUNK]->(Chunk)
        - (Chunk)-[:SIMILAR_TO]->(Chunk) from vector embeddings
        """
        graph_store = None

        if export_local_graph or not sync_to_neo4j:
            graph_store = self._build_local_graph_store(
                chunks,
                embeddings=embeddings,
                top_k_similar=top_k_similar,
                similarity_threshold=similarity_threshold,
            )
            if export_local_graph:
                saved_path = self.save_local_graph_store(graph_store)
                print(f"Local graph store exported to {saved_path}")

        if not sync_to_neo4j:
            return graph_store or {"nodes": [], "links": [], "summary": {}}

        self.connect()

        if clear_existing:
            self.clear_database()

        with self.driver.session() as session:
            print(f"Building jargon-first graph from {len(chunks)} chunks...")

            for i, chunk in enumerate(chunks):
                metadata = chunk.get("metadata", {})
                source_name = metadata.get("source", "unknown.pdf")
                page_label = metadata.get("page_label", "")
                chunk_uid = f"chunk_{i}"
                page_uid = self._page_node_id(source_name, page_label or "Unknown page")
                document_uid = self._document_node_id(source_name)
                text = chunk.get("text", "")[:1000]
                terms = self._extract_terms(chunk)

                session.run(
                    """
                    MERGE (d:Document {name: $source_name})
                    SET d.kind = 'document'
                    MERGE (p:Page {uid: $page_uid})
                    SET p.label = $page_label,
                        p.source = $source_name,
                        p.kind = 'page'
                    MERGE (c:Chunk {uid: $chunk_uid})
                    SET c.text = $text,
                        c.source = $source_name,
                        c.chunk_index = $chunk_index,
                        c.page_label = $page_label,
                        c.kind = 'chunk'
                    MERGE (d)-[:HAS_PAGE]->(p)
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    source_name=source_name,
                    page_uid=page_uid,
                    chunk_uid=chunk_uid,
                    text=text,
                    chunk_index=i,
                    page_label=page_label,
                )

                if i > 0:
                    prev_chunk_uid = f"chunk_{i - 1}"
                    session.run(
                        """
                        MATCH (a:Chunk {uid: $prev_chunk_uid})
                        MATCH (b:Chunk {uid: $chunk_uid})
                        MERGE (a)-[r:NEXT_CHUNK]->(b)
                        SET r.weight = 1
                        """,
                        prev_chunk_uid=prev_chunk_uid,
                        chunk_uid=chunk_uid,
                    )

                for rank, term in enumerate(terms):
                    session.run(
                        """
                        MATCH (c:Chunk {uid: $chunk_uid})
                        MERGE (t:Term {name: $term})
                        SET t.display = $term,
                            t.kind = 'term'
                        MERGE (c)-[m:MENTIONS]->(t)
                        SET m.rank = $rank,
                            m.source = $source_name
                        """,
                        chunk_uid=chunk_uid,
                        term=term,
                        rank=rank,
                        source_name=source_name,
                    )

                # New: Semantic relationships via LLM
                semantic_rels = self._extract_semantic_relations(text, terms)
                for rel in semantic_rels:
                    session.run(
                        f"""
                        MATCH (a:Term {{name: $source}})
                        MATCH (b:Term {{name: $target}})
                        MERGE (a)-[r:{rel['type']}]->(b)
                        SET r.weight = coalesce(r.weight, 0) + 1,
                            r.reason = $reason
                        """,
                        source=rel['source'],
                        target=rel['target'],
                        reason=rel.get('reason', '')
                    )

                for term_a, term_b in combinations(terms, 2):
                    if term_a == term_b:
                        continue

                    left_term, right_term = sorted([term_a, term_b])
                    session.run(
                        """
                        MERGE (a:Term {name: $left_term})
                        MERGE (b:Term {name: $right_term})
                        MERGE (a)-[r:CO_OCCURS_WITH]->(b)
                        SET r.weight = coalesce(r.weight, 0) + 1
                        """,
                        left_term=left_term,
                        right_term=right_term,
                    )

            self._create_similarity_edges(
                chunks,
                embeddings=embeddings,
                top_k_similar=top_k_similar,
                similarity_threshold=similarity_threshold,
            )

            print("Jargon-first graph structure built.")
            return graph_store or {"status": "neo4j_synced"}

    def run_louvain_communities(self, force_refresh: bool = False):
        """
        Executes Louvain community detection (requires GDS plugin),
        writes communityId on Term nodes, and returns community sizes.
        """
        if not force_refresh:
            # Check if communities already exist in Neo4j (basic check)
            self.connect()
            with self.driver.session() as session:
                has_comm = session.run("MATCH (t:Term) WHERE t.communityId IS NOT NULL RETURN count(t) > 0 AS has_comm").single()["has_comm"]
                if has_comm:
                    communities_res = session.run(
                        """
                        MATCH (t:Term)
                        WHERE t.communityId IS NOT NULL
                        WITH t.communityId AS community_id, count(*) AS size, collect(coalesce(t.display, t.name)) AS all_terms
                        RETURN community_id, size, all_terms[0..5] AS top_terms
                        ORDER BY size DESC
                        LIMIT 100
                        """
                    )
                    return {
                        "status": "cached",
                        "communities": [dict(record) for record in communities_res],
                    }
        else:
            # Clear cache on force refresh
            self.__class__._cache.clear()

        self.connect()

        with self.driver.session() as session:
            try:
                # Drop stale in-memory projection if present.
                exists_res = session.run(
                    """
                    CALL gds.graph.exists($graph_name)
                    YIELD exists
                    RETURN exists
                    """,
                    graph_name="termGraph",
                ).single()
                if exists_res and exists_res.get("exists"):
                    session.run(
                        """
                        CALL gds.graph.drop($graph_name)
                        YIELD graphName
                        RETURN graphName
                        """,
                        graph_name="termGraph",
                    ).consume()

                session.run(
                    """
                    CALL gds.graph.project(
                      $graph_name,
                      'Term',
                      {
                        CO_OCCURS_WITH: {
                          type: 'CO_OCCURS_WITH',
                          orientation: 'UNDIRECTED',
                          properties: 'weight'
                        }
                      }
                    )
                    """,
                    graph_name="termGraph",
                ).consume()

                louvain_stats = session.run(
                    """
                    CALL gds.louvain.write(
                      $graph_name,
                      {
                        relationshipWeightProperty: 'weight',
                        writeProperty: 'communityId'
                      }
                    )
                    YIELD communityCount, modularity, ranLevels
                    RETURN communityCount, modularity, ranLevels
                    """,
                    graph_name="termGraph",
                ).single()

                communities = session.run(
                    """
                    MATCH (t:Term)
                    WHERE t.communityId IS NOT NULL
                    WITH t.communityId AS community_id, count(*) AS size, collect(coalesce(t.display, t.name)) AS all_terms
                    RETURN community_id, size, all_terms[0..5] AS top_terms
                    ORDER BY size DESC
                    LIMIT 100
                    """
                )

                session.run(
                    """
                    CALL gds.graph.drop($graph_name)
                    YIELD graphName
                    RETURN graphName
                    """,
                    graph_name="termGraph",
                ).consume()

                return {
                    "status": "ok",
                    "stats": dict(louvain_stats) if louvain_stats else {},
                    "communities": [dict(record) for record in communities],
                }
            except Exception as e:
                print(f"GDS Louvain failed or is unavailable ({e}). Falling back to python-louvain.")
                return self._compute_communities_python_louvain()

    def _compute_communities_python_louvain(self):
        """
        Fallback community detection using python-louvain (NetworkX).
        Works on any Neo4j instance, including Aura Free.
        """
        import networkx as nx
        from community import community_louvain
        
        self.connect()

        with self.driver.session() as session:
            # Fetch all term-to-term relationships
            result = session.run("""
                MATCH (a:Term)-[r:CO_OCCURS_WITH]-(b:Term)
                RETURN a.name AS source, b.name AS target, coalesce(r.weight, 1.0) AS weight
            """)
            
            G = nx.Graph()
            for record in result:
                G.add_edge(record["source"], record["target"], weight=record["weight"])
            
            if G.number_of_nodes() == 0:
                return {"status": "empty", "communities": []}
                
            # Run Louvain
            partition = community_louvain.best_partition(G, weight='weight')
            
            # Write back to Neo4j
            batch = [{"name": name, "community_id": int(cid)} for name, cid in partition.items()]
            for i in range(0, len(batch), 500):
                sub_batch = batch[i:i+500]
                session.run("""
                    UNWIND $rows AS row
                    MATCH (t:Term {name: row.name})
                    SET t.communityId = row.community_id
                """, rows=sub_batch)
            
            communities_res = session.run(
                """
                MATCH (t:Term)
                WHERE t.communityId IS NOT NULL
                WITH t.communityId AS community_id, count(*) AS size, collect(coalesce(t.display, t.name)) AS all_terms
                RETURN community_id, size, all_terms[0..5] AS top_terms
                ORDER BY size DESC
                LIMIT 100
                """
            )
            
            return {
                "status": "ok",
                "method": "python-louvain-fallback",
                "communities": [dict(record) for record in communities_res]
            }

    def get_community_subgraph(self, community_id: int, min_weight: float = 1.0, limit: int = 500):
        """
        Returns a single Louvain community as an isolated subgraph.
        Requires Term.communityId to already be written (run_louvain_communities first).
        """
        self.connect()

        with self.driver.session() as session:
            nodes_res = session.run(
                """
                MATCH (a:Term)
                WHERE a.communityId = $community_id
                RETURN a.name as id,
                       'Term' as group,
                       coalesce(a.display, a.name) as text,
                       a.communityId as community_id
                LIMIT $limit
                """,
                community_id=community_id,
                limit=limit,
            )

            links_res = session.run(
                """
                MATCH (a:Term)-[r]->(b:Term)
                WHERE a.communityId = $community_id
                  AND b.communityId = $community_id
                  AND coalesce(r.weight, r.confidence, 1) >= $min_weight
                RETURN a.name as source,
                       b.name as target,
                       type(r) as value,
                       coalesce(r.weight, r.confidence, 1) as weight
                LIMIT $limit
                """,
                community_id=community_id,
                min_weight=min_weight,
                limit=limit,
            )

            nodes = [dict(n) for n in nodes_res]
            links = [dict(l) for l in links_res]
            return {
                "community_id": community_id,
                "nodes": nodes,
                "links": links,
                "node_count": len(nodes),
                "link_count": len(links),
            }

    def get_graph_metrics(self, community_id: Optional[int] = None):
        """Computes basic graph metrics and summary stats, optionally for a specific community."""
        self.connect()

        try:
            with self.driver.session() as session:
                # 1. Summary Stats (Counts)
                # FOR DEBUGGING: Count EVERYTHING in the database first
                all_nodes_res = session.run("MATCH (n) RETURN count(n) as count").single()
                total_db_nodes = all_nodes_res["count"] if all_nodes_res else 0
                
                all_links_res = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()
                total_db_links = all_links_res["count"] if all_links_res else 0

                if community_id is not None:
                    # Filtered view
                    nodes_res = session.run(
                        "MATCH (n) WHERE toString(n.communityId) = toString($cid) OR toString(n.community_id) = toString($cid) RETURN count(n) as count",
                        cid=community_id
                    ).single()
                    nodes_count = nodes_res["count"] if nodes_res else 0
                    
                    links_res = session.run(
                        "MATCH (n)-[r]->(m) WHERE (toString(n.communityId) = toString($cid)) AND (toString(m.communityId) = toString($cid)) RETURN count(DISTINCT r) as count",
                        cid=community_id
                    ).single()
                    links_count = links_res["count"] if links_res else 0
                else:
                    # Global view: Use the "Count Everything" values
                    nodes_count = total_db_nodes
                    links_count = total_db_links

                summary = {"nodes": nodes_count, "links": links_count}

                # 2. Top Terms (Degree-based ranking)
                if community_id is not None:
                    terms_query = """
                        MATCH (t) 
                        WHERE (toString(t.communityId) = toString($cid) OR toString(t.community_id) = toString($cid))
                          AND t.name IS NOT NULL
                        RETURN t.name as term, COUNT { (t)--() } as degree 
                        ORDER BY degree DESC LIMIT 10
                    """
                else:
                    terms_query = "MATCH (t) WHERE t.name IS NOT NULL RETURN t.name as term, COUNT { (t)--() } as degree ORDER BY degree DESC LIMIT 10"
                
                terms_res = session.run(terms_query, cid=community_id)
                top_terms = [dict(record) for record in terms_res]
                
                return {
                    "summary": summary,
                    "top_terms": top_terms
                }
        except Exception as e:
            err_msg = str(e)
            print(f"❌ Graph metrics error: {err_msg}")
            return {
                "summary": {"nodes": 0, "links": 0, "error": err_msg}, 
                "top_terms": []
            }

    def get_visualization_data(self, force_refresh: bool = False):
        """Returns a term-centric graph for clearer jargon visualization.
        Includes semantic relationships (CONTROLS, INFLUENCES, etc.) and co-occurrence.
        Excludes chunk/page nodes - knowledge base contains only terms and their relationships.
        """
        # Check cache
        if not force_refresh and "viz_data" in self._cache:
            return self._cache["viz_data"]

        self.connect()

        try:
            with self.driver.session() as session:
                # Fetch Term nodes AND any nodes connected to them (e.g. Chunks)
                nodes_res = session.run(
                    """
                    MATCH (n)
                    WHERE (n:Term OR n:Chunk)
                    RETURN
                      coalesce(n.name, n.uid, n.id) as id,
                      labels(n)[0] as group,
                      coalesce(n.display, n.text, n.name) as text,
                      n.communityId as community_id
                    LIMIT 1000
                    """
                )
                
                # Fetch ALL relationships connected to Terms, prioritizing varied types
                links_res = session.run(
                    """
                    MATCH (s)-[r]->(t)
                    WHERE (s:Term OR s:Chunk) AND (t:Term OR t:Chunk)
                      AND s <> t
                    WITH s, r, t, type(r) as rel_type
                    ORDER BY coalesce(r.weight, r.confidence, 1) DESC
                    WITH rel_type, collect({
                      source: coalesce(s.name, s.uid, s.id),
                      target: coalesce(t.name, t.uid, t.id),
                      value: rel_type,
                      weight: coalesce(r.weight, r.confidence, 1)
                    }) as rels_of_type
                    // Take top 1000 of each type to ensure variety, then flatten
                    UNWIND rels_of_type[0..1000] as rel
                    RETURN rel.source as source, rel.target as target, rel.value as value, rel.weight as weight
                    LIMIT 5000
                    """
                )

                nodes = [dict(n) for n in nodes_res]
                links = [dict(l) for l in links_res]

                # Ensure nodes that are targets/sources of links are included even if not in nodes_res (due to LIMIT)
                linked_node_ids = set()
                for link in links:
                    linked_node_ids.add(link["source"])
                    linked_node_ids.add(link["target"])
                
                existing_node_ids = {node["id"] for node in nodes}
                missing_node_ids = linked_node_ids - existing_node_ids
                
                if missing_node_ids:
                    missing_res = session.run("""
                        MATCH (t:Term)
                        WHERE t.name IN $names
                        RETURN
                          t.name as id,
                          'Term' as group,
                          coalesce(t.display, t.name) as text,
                          t.communityId as community_id
                    """, names=list(missing_node_ids))
                    nodes.extend([dict(n) for n in missing_res])

                if not nodes:
                    raise Exception("No data in Neo4j")

                res = {
                    "nodes": nodes, 
                    "links": links,
                    "metadata": {
                        "queries": [
                            "MATCH (n) WHERE (n:Term OR n:Chunk) RETURN ... LIMIT 1000",
                            "MATCH (s)-[r]->(t) WHERE (s:Term OR s:Chunk) AND (t:Term OR t:Chunk) ... LIMIT 5000"
                        ]
                    }
                }
                self._cache["viz_data"] = res
                return res
        except Exception as e:
            return {"nodes": [], "links": [], "status": "unavailable", "detail": str(e)}

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from a user query (FR + EN stop words removed)."""
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'en',
            'est', 'sont', 'a', 'au', 'aux', 'ce', 'ces', 'que', 'qui', 'quoi',
            'ne', 'pas', 'plus', 'aussi', 'comme', 'mais', 'donc', 'car', 'si',
            'dans', 'sur', 'par', 'pour', 'avec', 'sans', 'entre', 'vers',
            'sous', 'cette', 'ses', 'son', 'sa', 'mon', 'ma', 'mes',
            'il', 'elle', 'ils', 'elles', 'nous', 'vous', 'on', 'se',
            'quel', 'quelle', 'quels', 'quelles', 'comment', 'fait', 'faire',
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'need', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'an',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'about', 'this', 'that', 'these',
        }
        words = re.sub(r'[?!.,;:\'"()\[\]{}]', ' ', query.lower()).split()
        return [w for w in words if len(w) > 2 and w not in stop_words]

    def graph_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query-driven graph retrieval: finds relevant terms, their relationships,
        and connected chunks for LLM context. Tracks Cypher queries executed.
        """
        self.connect()
        query_terms = self._extract_search_terms(query)
        cypher_log = []
        subgraph = {"nodes": [], "links": []}
        empty = {"context": "", "matched_terms": [], "relationships": [],
                 "chunks_retrieved": 0, "cypher_queries": [], "subgraph": subgraph, "sources": []}

        if not query_terms:
            return empty

        try:
            with self.driver.session() as session:
                # ── Step 1: Find matching Term nodes ──
                q1 = ("MATCH (t:Term) "
                      "WHERE ANY(term IN $terms WHERE toLower(t.name) CONTAINS term) "
                      "RETURN t.name AS term, t.communityId AS community_id "
                      "LIMIT 30")
                cypher_log.append({"step": 1, "description": "Find matching terms", "cypher": q1, "params": {"terms": query_terms}})
                matched_terms = [dict(r) for r in session.run(q1, terms=query_terms)]

                if not matched_terms:
                    # Fallback: search chunk text directly
                    q_fb = ("MATCH (c:Chunk) "
                            "WHERE ANY(term IN $terms WHERE toLower(c.text) CONTAINS term) "
                            "RETURN c.text AS text, c.source AS source, c.page_label AS page_label, c.uid AS uid "
                            "LIMIT $top_k")
                    cypher_log.append({"step": "1b", "description": "Fallback: search chunk text", "cypher": q_fb})
                    chunks = [dict(r) for r in session.run(q_fb, terms=query_terms, top_k=top_k)]
                    ctx = "\n\n".join([f"[Chunk] {c['text']}" for c in chunks if c.get("text")])
                    
                    fb_subgraph = {
                        "nodes": [{"id": c.get("uid", f"chunk_{i}"), "text": f"Chunk {c.get('page_label', '')}", "group": "Chunk"} for i, c in enumerate(chunks)],
                        "links": []
                    }
                    
                    return {**empty, "context": ctx, "chunks_retrieved": len(chunks),
                            "cypher_queries": cypher_log, "sources": chunks, "subgraph": fb_subgraph}

                term_names = [t["term"] for t in matched_terms]

                # ── Step 2: Get relationships between matched terms ──
                q2 = ("MATCH (a:Term)-[r]->(b:Term) "
                      "WHERE a.name IN $terms AND b.name IN $terms "
                      "RETURN a.name AS source, b.name AS target, type(r) AS rel_type, "
                      "coalesce(r.weight, 1) AS weight, r.reason AS reason "
                      "LIMIT 50")
                cypher_log.append({"step": 2, "description": "Get term relationships", "cypher": q2})
                relationships = [dict(r) for r in session.run(q2, terms=term_names)]

                subgraph["nodes"] = [
                    {
                        "id": t["term"], 
                        "text": t["term"], 
                        "group": "Term", 
                        "community_id": t.get("community_id")
                    } for t in matched_terms
                ]
                subgraph["links"] = [{"source": r["source"], "target": r["target"], "value": r["rel_type"]} for r in relationships]

                # ── Step 3: Retrieve chunks connected to matched terms ──
                q3 = ("MATCH (c:Chunk)-[:MENTIONS]->(t:Term) "
                      "WHERE t.name IN $terms "
                      "WITH c, collect(DISTINCT t.name) AS related_terms "
                      "RETURN c.text AS text, c.source AS source, c.page_label AS page_label, related_terms "
                      "ORDER BY size(related_terms) DESC "
                      "LIMIT $top_k")
                cypher_log.append({"step": 3, "description": "Retrieve chunks via MENTIONS edges", "cypher": q3})
                chunks = [dict(r) for r in session.run(q3, terms=term_names, top_k=top_k)]

                # ── Build rich context for LLM ──
                parts = []
                if relationships:
                    rel_lines = []
                    for r in relationships[:15]:
                        reason = f" ({r['reason']})" if r.get("reason") else ""
                        rel_lines.append(f"  {r['source']} --[{r['rel_type']}]--> {r['target']}{reason}")
                    parts.append("Knowledge graph relationships:\n" + "\n".join(rel_lines))

                for i, c in enumerate(chunks):
                    terms_str = ", ".join(c.get("related_terms", []))
                    parts.append(f"[Graph Evidence {i+1} | Terms: {terms_str}]\n{c['text']}")

                return {
                    "context": "\n\n".join(parts),
                    "matched_terms": term_names,
                    "relationships": relationships,
                    "chunks_retrieved": len(chunks),
                    "cypher_queries": cypher_log,
                    "subgraph": subgraph,
                    "sources": chunks,
                }
        except Exception as e:
            print(f"Graph search error: {e}")
            return {**empty, "cypher_queries": cypher_log, "error": str(e)}
