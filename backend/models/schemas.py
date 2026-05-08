from pydantic import BaseModel
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    route_override: Optional[str] = None

class GraphNode(BaseModel):
    id: str
    group: str
    text: Optional[str] = ""

class GraphLink(BaseModel):
    source: str
    target: str
    value: Optional[str] = "NEXT"

class GraphData(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]

class PCAData(BaseModel):
    x: float
    y: float
    label: str

class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    confidence: float
    sources: List[Dict]
    thought_process: Optional[Dict] = None
    reward_details: Optional[Dict] = None
    update_details: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
