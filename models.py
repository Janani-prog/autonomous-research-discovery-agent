from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class Phase(str, Enum):
    PLAN = "plan"
    SEARCH = "search"
    SCORE = "score"
    ANALYZE = "analyze"
    INQUIRE = "inquire"
    TERMINATE = "terminate"


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    year: int
    score: float = 0.0
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: int = 0

    # --- Ranking features (learned / heuristic) ---
    semantic_score: float = 0.0
    recency_score: float = 0.0
    citation_score: float = 0.0
    concept_coverage: float = 0.0

    # --- Graph-related ---
    citation_edges: List[Tuple[str, str]] = Field(default_factory=list)
    
    class Config:
        validate_assignment = True


class Subgoal(BaseModel):
    name: str
    description: str
    completed: bool = False
    refinements: int = 0
    refined_queries: Optional[List[str]] = None
    papers: Dict[str, Paper] = Field(default_factory=dict)
    gaps: List[str] = Field(default_factory=list)
    concept_map: Dict[str, List[str]] = Field(default_factory=dict)

class ResearchState(BaseModel):
    objective: str
    phase: Phase = Phase.PLAN
    subgoals: Dict[str, Subgoal] = Field(default_factory=dict)
    confidence: float = 0.3
    user_intent: Optional[str] = None

