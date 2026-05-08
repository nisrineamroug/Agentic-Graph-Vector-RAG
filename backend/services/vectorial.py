# -*- coding: utf-8 -*-
"""
Service for generating embeddings and managing the FAISS vector index.
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from backend.config import get_settings

settings = get_settings()

class VectorService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.index_path = settings.FAISS_INDEX_PATH
        self.model = None
        self.index = None
        self.bm25 = None  # Keyword search
        self.chunks = []
        self.embeddings_cached = None

    def _ensure_model_loaded(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def create_index(self, chunks: List[Dict[str, Any]]):
        """Builds both FAISS index and BM25 index."""
        self._ensure_model_loaded()
        self.chunks = chunks
        
        texts = [c["text"] for c in chunks]
        
        # 1. FAISS (Vector)
        print(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings_cached = self.model.encode(texts, show_progress_bar=True)
        dimension = self.embeddings_cached.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings_cached).astype('float32'))
        
        # 2. BM25 (Keyword)
        from rank_bm25 import BM25Okapi
        # Combine text and keywords for better keyword matching
        tokenized_corpus = []
        for c in chunks:
            text_content = c["text"].lower()
            keywords_content = " ".join(c["metadata"].get("keywords", [])).lower()
            tokenized_corpus.append((text_content + " " + keywords_content).split())
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Hybrid Indexing complete (FAISS + BM25).")

    def hybrid_search(self, query: str, top_k: int = 5, vector_weight: float = None):
        """Combines Vector similarities and BM25 keyword scores."""
        self._ensure_model_loaded()

        if vector_weight is None:
            vector_weight = float(settings.HYBRID_VECTOR_WEIGHT)
        
        # 1. Vector Search
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), len(self.chunks))
        
        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Combine scores (Simple weighted sum)
        # Normalize: FAISS is distance (small is better), BM25 is score (large is better)
        combined_results = []
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            # Normalize vector distance to a score (0 to 1 range, inverted)
            v_score = 1 / (1 + distances[0][i]) 
            # Normalize BM25 score
            k_score = bm25_scores[idx] / max_bm25
            
            final_score = (v_score * vector_weight) + (k_score * (1 - vector_weight))
            
            res = self.chunks[idx].copy()
            res["score"] = float(final_score)
            res["method"] = "hybrid"
            combined_results.append(res)
            
        # Sort by final score descending
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]

    def get_pca_projection(self):
        """Reduces high-dim embeddings to 2D for the 'PCA nuage de points' button."""
        if self.embeddings_cached is None:
            # If not in memory, we'd need to re-generate or load them
            # For simplicity, we assume they are available after creation
            return []
            
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(self.embeddings_cached)
        
        projection = []
        for i, coords in enumerate(pcs):
            projection.append({
                "x": float(coords[0]),
                "y": float(coords[1]),
                "label": self.chunks[i]["metadata"]["page_label"]
            })
        return projection

    def save_index(self):
        """Persists FAISS, BM25, and text chunks to disk."""
        if self.index:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            import json, pickle
            # Save chunks
            with open(f"{self.index_path}_chunks.json", "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
            # Save BM25
            with open(f"{self.index_path}_bm25.pkl", "wb") as f:
                pickle.dump(self.bm25, f)
            
            # Save raw embeddings (needed for PCA)
            np.save(f"{self.index_path}_embeddings.npy", self.embeddings_cached)
                
            print(f"All Vectorial RAG data saved to {os.path.dirname(self.index_path)}")

    def load_index(self):
        """Loads FAISS, BM25, chunks, and embeddings from disk."""
        try:
            import json, pickle
            self.index = faiss.read_index(f"{self.index_path}.index")
            
            with open(f"{self.index_path}_chunks.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            
            with open(f"{self.index_path}_bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
                
            self.embeddings_cached = np.load(f"{self.index_path}_embeddings.npy")
            
            print("Vector index, BM25, and PCA data loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
