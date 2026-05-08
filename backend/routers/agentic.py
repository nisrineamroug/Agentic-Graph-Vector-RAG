# -*- coding: utf-8 -*-
"""
Router for Agentic Router operations (Routing, Q-Learning visualization).
"""

import os
import json
from fastapi import APIRouter
from backend.services.agentic import AgenticService
from backend.services.auto_reward import AutoRewardScorer

router = APIRouter(prefix="/agentic", tags=["Agentic"])
agent_service = AgenticService()
reward_scorer = AutoRewardScorer()

@router.get("/route")
async def get_route(query: str):
    """Classifies a query and suggests a route."""
    return agent_service.route_query(query)

@router.get("/policy")
async def get_policy():
    """Returns the Q-Table for visualization."""
    return agent_service.get_policy_data()

@router.get("/stats")
async def get_route_stats():
    """Returns performance statistics for each route based on implicit rewards."""
    return reward_scorer.get_route_stats()

@router.post("/feedback")
async def submit_feedback(query: str, action_index: int, reward: float):
    """Trains the Q-Learning agent with user feedback (optional, for explicit rewards)."""
    agent_service.router.update_q_table(query, action_index, reward)
    return {"status": "trained", "reward": reward}

@router.post("/reset")
async def reset_agent():
    """Clears history and resets Q-table."""
    # Reset Q-table
    agent_service.router.q_table = {}
    if os.path.exists(agent_service.router.q_table_path):
        os.remove(agent_service.router.q_table_path)
    
    # Clear logs
    if os.path.exists(reward_scorer.log_path):
        os.remove(reward_scorer.log_path)
    
    return {"status": "reset_successful"}

@router.get("/reward-history")
async def get_reward_history():
    """Parses query logs to return reward trend over time."""
    history = []
    if not os.path.exists(reward_scorer.log_path):
        return history
        
    try:
        with open(reward_scorer.log_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if "reward" in data:
                    history.append({
                        "timestamp": data.get("timestamp", ""),
                        "reward": float(data["reward"]),
                        "route": data.get("route", "unknown")
                    })
    except Exception as e:
        print(f"Error reading reward history: {e}")
        
    return history
