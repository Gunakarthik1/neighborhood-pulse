"""
Pydantic schemas for the NeighborhoodPulse API.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    location: str = Field(..., min_length=2, max_length=200, description="City, neighborhood, or address")
    address: Optional[str] = Field(None, max_length=300, description="Optional specific address")


class ResearchResponse(BaseModel):
    report_id: str
    location: str
    status: str
    message: str


class AgentStatusEvent(BaseModel):
    event_type: str
    agent: Optional[str] = None
    status: Optional[str] = None
    latency_ms: Optional[float] = None
    citation_count: Optional[int] = None
    error: Optional[str] = None
    state: Optional[str] = None
    timestamp: str


class CitationSchema(BaseModel):
    source_name: str
    url: str
    retrieved_at: str
    data_field: str
    raw_value: str


class AgentResultSchema(BaseModel):
    agent_name: str
    status: str
    data: dict[str, Any]
    citations: list[CitationSchema]
    latency_ms: float
    error: Optional[str] = None


class ValidationResultSchema(BaseModel):
    accuracy_rate: float
    accuracy_rate_pct: float
    total_claims: int
    total_verified: int
    total_unverified: int
    citation_coverage_by_section: dict[str, float]
    citation_count_by_agent: dict[str, int]
    validated_claims: list[dict]
    unverified_claims: list[dict]


class ReportSummary(BaseModel):
    report_id: str
    location: str
    status: str
    risk_score: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
    citation_accuracy_pct: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
