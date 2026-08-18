"""
NeighborhoodPulse FastAPI application.
Async/event-driven with SSE streaming for real-time agent progress.
"""

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from agents.base_agent import AgentResult
from agents.orchestrator import AgentOrchestrator, OrchestratorResult
from api.database import (
    SessionLocal,
    complete_report,
    create_report,
    fail_report,
    get_agent_results,
    get_citations,
    get_db,
    get_report,
    init_db,
    list_reports,
    save_agent_results,
    update_report_status,
)
from api.models import (
    HealthResponse,
    ReportSummary,
    ResearchRequest,
    ResearchResponse,
    ValidationResultSchema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeighborhoodPulse API",
    description="Distributed multi-agent neighborhood data platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory event queues for SSE streaming: report_id → list of queues
_sse_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)


@app.on_event("startup")
async def on_startup():
    init_db()
    logger.info("NeighborhoodPulse API started.")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _push_event(report_id: str, event_type: str, payload: dict) -> None:
    """Push an SSE event to all active listeners for this report."""
    payload["event"] = event_type
    payload.setdefault("timestamp", _now_iso())
    message = json.dumps(payload)
    for queue in list(_sse_queues.get(report_id, [])):
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            pass


def _push_done(report_id: str) -> None:
    """Signal SSE stream completion."""
    _push_event(report_id, "done", {"report_id": report_id})
    # Also push sentinel None to close queues
    for queue in list(_sse_queues.get(report_id, [])):
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


async def _run_pipeline(report_id: str, location: str, address: Optional[str]) -> None:
    """Background task: run the full agent orchestration pipeline."""
    db = SessionLocal()
    try:
        update_report_status(db, report_id, "RUNNING", "PARALLEL_FETCH")
        _push_event(report_id, "status", {"report_id": report_id, "status": "RUNNING"})

        orchestrator = AgentOrchestrator()

        def on_agent_event(event_type: str, payload: dict) -> None:
            _push_event(report_id, event_type, payload)

        orchestrator.on_event(on_agent_event)

        result: OrchestratorResult = await orchestrator.run(location)

        # Persist agent results and citations
        agent_result_dicts = []
        for ar in result.agent_results:
            agent_result_dicts.append(ar)
        if result.synthesis_result:
            agent_result_dicts.append(result.synthesis_result)

        save_agent_results(db, report_id, [r.to_dict() for r in agent_result_dicts])

        # Persist completed report
        validation_dict = result.validation_result.to_dict() if result.validation_result else {}
        complete_report(
            db=db,
            report_id=report_id,
            report_dict=result.report or {},
            validation_dict=validation_dict,
            graph_dict=result.graph.to_dict(),
            total_latency_ms=result.graph.total_latency_ms,
        )

        _push_event(report_id, "report_ready", {
            "report_id": report_id,
            "risk_score": result.report.get("risk_score") if result.report else None,
            "citation_accuracy_pct": round(result.validation_result.accuracy_rate * 100, 1) if result.validation_result else None,
        })

    except Exception as exc:
        logger.exception(f"Pipeline error for report {report_id}: {exc}")
        fail_report(db, report_id, str(exc))
        _push_event(report_id, "error", {"report_id": report_id, "error": str(exc)})

    finally:
        _push_done(report_id)
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=_now_iso(),
    )


@app.post("/api/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
):
    """Start an async research pipeline for a location. Returns report_id immediately."""
    report_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        create_report(db, report_id, request.location, request.address)
    finally:
        db.close()

    background_tasks.add_task(_run_pipeline, report_id, request.location, request.address)

    return ResearchResponse(
        report_id=report_id,
        location=request.location,
        status="PENDING",
        message="Research pipeline started. Connect to /api/research/{report_id}/stream for live updates.",
    )


@app.get("/api/research/{report_id}/stream")
async def stream_report_events(report_id: str):
    """
    SSE endpoint for real-time agent pipeline events.
    Streams events until pipeline completes or client disconnects.
    """
    db = SessionLocal()
    report = get_report(db, report_id)
    db.close()

    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    # If already complete, emit a synthetic completion event and close
    if report.status in ("COMPLETE", "FAILED"):
        async def immediate_stream():
            payload = {
                "event": "report_ready" if report.status == "COMPLETE" else "error",
                "report_id": report_id,
                "status": report.status,
                "timestamp": _now_iso(),
            }
            if report.status == "COMPLETE":
                payload["risk_score"] = report.risk_score
                payload["citation_accuracy_pct"] = report.citation_accuracy_pct
            yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'report_id': report_id})}\n\n"

        return StreamingResponse(immediate_stream(), media_type="text/event-stream")

    # Register a queue for this client
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_queues[report_id].append(queue)

    async def event_generator() -> AsyncIterator[str]:
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                if message is None:
                    # Pipeline complete — close stream
                    break

                yield f"data: {message}\n\n"

                # Stop after done event
                try:
                    parsed = json.loads(message)
                    if parsed.get("event") in ("done", "error"):
                        break
                except (json.JSONDecodeError, KeyError):
                    pass
        finally:
            if queue in _sse_queues.get(report_id, []):
                _sse_queues[report_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/research/{report_id}/report")
async def get_full_report(report_id: str):
    """Return the full completed report JSON."""
    db = SessionLocal()
    try:
        report = get_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        if report.status == "PENDING" or report.status == "RUNNING":
            raise HTTPException(status_code=202, detail="Report is still being generated")
        if report.status == "FAILED":
            raise HTTPException(status_code=500, detail=report.error_message or "Pipeline failed")
        if not report.report_json:
            raise HTTPException(status_code=404, detail="Report data not found")
        return json.loads(report.report_json)
    finally:
        db.close()


@app.get("/api/research/{report_id}/citations")
async def get_report_citations(report_id: str):
    """Return all citations for a report, queryable separately."""
    db = SessionLocal()
    try:
        report = get_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        citations = get_citations(db, report_id)
        return {
            "report_id": report_id,
            "location": report.location,
            "citation_count": len(citations),
            "citations": [
                {
                    "id": c.id,
                    "agent_name": c.agent_name,
                    "source_name": c.source_name,
                    "url": c.url,
                    "retrieved_at": c.retrieved_at,
                    "data_field": c.data_field,
                    "raw_value": c.raw_value,
                }
                for c in citations
            ],
        }
    finally:
        db.close()


@app.get("/api/research/{report_id}/validation")
async def get_report_validation(report_id: str):
    """Return citation validation results for a report."""
    db = SessionLocal()
    try:
        report = get_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        if not report.validation_json:
            raise HTTPException(status_code=404, detail="Validation data not available")
        return json.loads(report.validation_json)
    finally:
        db.close()


@app.get("/api/reports")
async def list_all_reports(limit: int = 20):
    """List all generated reports, newest first."""
    db = SessionLocal()
    try:
        reports = list_reports(db, limit=limit)
        return {
            "total": len(reports),
            "reports": [
                {
                    "report_id": r.id,
                    "location": r.location,
                    "status": r.status,
                    "risk_score": r.risk_score,
                    "citation_accuracy_pct": r.citation_accuracy_pct,
                    "created_at": r.created_at,
                    "completed_at": r.completed_at,
                    "total_latency_ms": r.total_latency_ms,
                }
                for r in reports
            ],
        }
    finally:
        db.close()


@app.get("/api/research/{report_id}/status")
async def get_report_status(report_id: str):
    """Return the current status of a research job."""
    db = SessionLocal()
    try:
        report = get_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        return {
            "report_id": report_id,
            "location": report.location,
            "status": report.status,
            "graph_state": report.graph_state,
            "risk_score": report.risk_score,
            "citation_accuracy_pct": report.citation_accuracy_pct,
            "created_at": report.created_at,
            "completed_at": report.completed_at,
            "total_latency_ms": report.total_latency_ms,
            "error": report.error_message,
        }
    finally:
        db.close()
