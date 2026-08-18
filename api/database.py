"""
SQLite + SQLAlchemy persistence layer.
Tables: research_reports, agent_results, citations
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./neighborhood_pulse.db")
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(String(36), primary_key=True)
    location = Column(String(300), nullable=False, index=True)
    address = Column(String(400), nullable=True)
    status = Column(String(50), nullable=False, default="PENDING")
    graph_state = Column(String(50), nullable=True)
    risk_score = Column(Float, nullable=True)
    citation_accuracy_pct = Column(Float, nullable=True)
    report_json = Column(Text, nullable=True)           # Full synthesized report
    validation_json = Column(Text, nullable=True)       # Validation result JSON
    graph_json = Column(Text, nullable=True)            # AgentGraph state JSON
    error_message = Column(Text, nullable=True)
    created_at = Column(String(30), nullable=False)
    completed_at = Column(String(30), nullable=True)
    total_latency_ms = Column(Float, nullable=True)


class AgentResultRecord(Base):
    __tablename__ = "agent_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(36), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    data_json = Column(Text, nullable=True)
    citations_json = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(String(30), nullable=False)


class CitationRecord(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(36), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    source_name = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False)
    retrieved_at = Column(String(30), nullable=False)
    data_field = Column(String(200), nullable=False)
    raw_value = Column(Text, nullable=False)
    created_at = Column(String(30), nullable=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Report operations ─────────────────────────────────────────────────────────

def create_report(db: Session, report_id: str, location: str, address: Optional[str] = None) -> ResearchReport:
    record = ResearchReport(
        id=report_id,
        location=location,
        address=address,
        status="PENDING",
        created_at=_now(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def update_report_status(db: Session, report_id: str, status: str, graph_state: Optional[str] = None) -> None:
    db.query(ResearchReport).filter(ResearchReport.id == report_id).update(
        {"status": status, "graph_state": graph_state}
    )
    db.commit()


def complete_report(
    db: Session,
    report_id: str,
    report_dict: dict,
    validation_dict: dict,
    graph_dict: dict,
    total_latency_ms: float,
) -> None:
    risk_score = report_dict.get("risk_score")
    citation_accuracy = validation_dict.get("accuracy_rate_pct") if validation_dict else None

    db.query(ResearchReport).filter(ResearchReport.id == report_id).update(
        {
            "status": "COMPLETE",
            "graph_state": graph_dict.get("graph_state"),
            "risk_score": risk_score,
            "citation_accuracy_pct": citation_accuracy,
            "report_json": json.dumps(report_dict),
            "validation_json": json.dumps(validation_dict) if validation_dict else None,
            "graph_json": json.dumps(graph_dict),
            "completed_at": _now(),
            "total_latency_ms": total_latency_ms,
        }
    )
    db.commit()


def fail_report(db: Session, report_id: str, error: str) -> None:
    db.query(ResearchReport).filter(ResearchReport.id == report_id).update(
        {"status": "FAILED", "error_message": error, "completed_at": _now()}
    )
    db.commit()


def get_report(db: Session, report_id: str) -> Optional[ResearchReport]:
    return db.query(ResearchReport).filter(ResearchReport.id == report_id).first()


def list_reports(db: Session, limit: int = 50) -> list[ResearchReport]:
    return (
        db.query(ResearchReport)
        .order_by(ResearchReport.created_at.desc())
        .limit(limit)
        .all()
    )


# ── Agent result operations ───────────────────────────────────────────────────

def save_agent_results(db: Session, report_id: str, agent_results: list) -> None:
    now = _now()
    for result in agent_results:
        if hasattr(result, "to_dict"):
            rd = result.to_dict()
        else:
            rd = result

        record = AgentResultRecord(
            report_id=report_id,
            agent_name=rd.get("agent_name", ""),
            status=rd.get("status", ""),
            data_json=json.dumps(rd.get("data", {})),
            citations_json=json.dumps(rd.get("citations", [])),
            latency_ms=rd.get("latency_ms"),
            error=rd.get("error"),
            created_at=now,
        )
        db.add(record)

        # Also store citations individually for queryability
        for cit in rd.get("citations", []):
            cit_record = CitationRecord(
                report_id=report_id,
                agent_name=rd.get("agent_name", ""),
                source_name=cit.get("source_name", ""),
                url=cit.get("url", ""),
                retrieved_at=cit.get("retrieved_at", now),
                data_field=cit.get("data_field", ""),
                raw_value=cit.get("raw_value", ""),
                created_at=now,
            )
            db.add(cit_record)

    db.commit()


def get_agent_results(db: Session, report_id: str) -> list[AgentResultRecord]:
    return (
        db.query(AgentResultRecord)
        .filter(AgentResultRecord.report_id == report_id)
        .order_by(AgentResultRecord.id)
        .all()
    )


def get_citations(db: Session, report_id: str) -> list[CitationRecord]:
    return (
        db.query(CitationRecord)
        .filter(CitationRecord.report_id == report_id)
        .order_by(CitationRecord.id)
        .all()
    )
