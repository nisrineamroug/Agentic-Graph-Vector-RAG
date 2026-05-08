# -*- coding: utf-8 -*-
"""
Agentic RAG Service using a Q-Learning approach to route queries.
"""

import numpy as np
import random
import pickle
import os
from typing import Dict, Any
from backend.config import get_settings

settings = get_settings()

class QLearningRouter:
    def __init__(self, actions: list, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.actions = actions  # [0: Vector, 1: Graph, 2: Hybrid]
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_initial = epsilon
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9995  # Decay per query
        self.q_table = {}       # State -> Action-Value mapping
        self.q_table_path = "data/q_table.pkl"
        self.query_count = 0    # Track queries for annealing
        self.load_q_table()

    def _get_state_details(self, query: str) -> Dict[str, Any]:
        """
        Classifies a query into a state based on semantic, structural, and complexity features.
        This state drives the Q-learning action selection.
        """
        q = query.lower()
        words = q.split()
        length_bucket = "short" if len(words) < 6 else "long"

        # Structural/relational keywords → favor Graph RAG
        structural_keywords = [
            'résumé', 'synthèse', 'liste', 'tous', 'relation', 'lien', 'global',
            'comparer', 'différence', 'structure', 'hiérarchie', 'composants',
            'relationship', 'compare', 'overview', 'connected', 'related',
            'types', 'categories', 'classification', 'between', 'link',
            'limites', 'limitent', 'encadrent', 'impact', 'influencent', 
            'interactions', 'dépend', 'relient', 'composé'
        ]
        # Semantic/contextual keywords → favor Vectorial RAG
        semantic_keywords = [
            'expliquer', 'définir', 'décrire', 'signifie', 'pourquoi',
            'explain', 'define', 'describe', 'meaning', 'detail',
            'elaborate', 'context', 'specific', 'example', 'precisely',
            'quel', 'quelle', 'quels', 'quelles', 'valeur', 'estimé', 
            'concept', 'est-ce'
        ]

        found_structural = [kw for kw in structural_keywords if kw in q]
        found_semantic = [kw for kw in semantic_keywords if kw in q]

        if len(found_structural) > len(found_semantic):
            query_type = "structural"
        elif len(found_semantic) > 0:
            query_type = "semantic"
        else:
            query_type = "factual"

        state = f"{length_bucket}_{query_type}"
        return {
            "state": state,
            "features": {
                "length": len(words),
                "bucket": length_bucket,
                "query_type": query_type,
                "is_structural": len(found_structural) > 0,
                "is_semantic": len(found_semantic) > 0,
                "structural_keywords": found_structural,
                "semantic_keywords": found_semantic,
            }
        }

    def _get_state(self, query: str) -> str:
        return self._get_state_details(query)["state"]

    def choose_action(self, query: str) -> int:
        """Epsilon-greedy action selection with annealing."""
        state = self._get_state(query)
        
        # Initialize state in Q-table if new
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        # Anneal epsilon: decay exploration over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.query_count += 1

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.actions)))  # Explore
        else:
            return int(np.argmax(self.q_table[state]))      # Exploit

    def update_q_table(self, query: str, action: int, reward: float):
        """Standard Q-learning update rule."""
        state = self._get_state(query)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        # In a real one-step router, we don't have a 'next state' 
        # so it's a simplified Q-update
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.alpha * (reward - old_value)
        self.save_q_table()

    def save_q_table(self):
        os.makedirs("data", exist_ok=True)
        with open(self.q_table_path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                self.q_table = pickle.load(f)

class AgenticService:
    def __init__(self):
        self.router = QLearningRouter(actions=["VECTOR", "GRAPH", "HYBRID"])

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Decides which RAG route to take.
        Returns the chosen route name.
        """
        action_idx = self.router.choose_action(query)
        route_map = {0: "vectorial", 1: "graph", 2: "hybrid"}
        state_details = self.router._get_state_details(query)
        state = state_details["state"]
        values = self.router.q_table.get(state, np.zeros(len(self.router.actions)))
        
        if np.allclose(values, 0):
            confidence = 1.0 / len(self.router.actions)
        else:
            shifted = values - np.max(values)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            confidence = float(probabilities[int(action_idx)])
            
        return {
            "query": query,
            "route": route_map[int(action_idx)],
            "action_index": int(action_idx),
            "confidence": float(confidence),
            "state": str(state),
            "analysis": state_details["features"],
            "q_values": [float(v) for v in values]
        }

    def get_policy_data(self):
        """Returns the Q-Table for frontend visualization."""
        policy = []
        for state, values in self.router.q_table.items():
            policy.append({
                "state": state,
                "vector_q": float(values[0]),
                "graph_q": float(values[1]),
                "hybrid_q": float(values[2]),
                "best_action": ["Vector", "Graph", "Hybrid"][int(np.argmax(values))]
            })
        return policy

    def auto_update_from_reward(self, query: str, action_index: int, reward: float) -> Dict[str, Any]:
        """
        Automatically update Q-table based on retrieval quality reward.
        This is called implicitly after each query (no manual feedback needed).
        
        Args:
            query: The user query
            action_index: Which route was chosen (0=vector, 1=graph, 2=hybrid)
            reward: Auto-computed quality score (0-1)
        
        Returns: Updated Q-values and epsilon status
        """
        self.router.update_q_table(query, action_index, reward)
        state = self.router._get_state(query)
        
        return {
            "updated": True,
            "state": state,
            "action": action_index,
            "reward": float(reward),
            "new_q_values": [float(v) for v in self.router.q_table[state]],
            "epsilon": float(self.router.epsilon),
            "query_count": self.router.query_count,
        }
