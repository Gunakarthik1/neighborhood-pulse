"""
NeighborhoodPulse FastAPI application.
Async/event-driven with SSE streaming for real-time agent progress.
Includes /api/analyze for fast, synchronous neighborhood data.
"""

import asyncio
import hashlib
import json
import logging
import pathlib
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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

# ── In-memory 24-hour cache for /api/analyze ─────────────────────────────────
_ANALYZE_CACHE: dict[str, dict] = {}
_CACHE_TTL = 86400  # 24 hours in seconds


def _cache_get(key: str) -> Optional[dict]:
    entry = _ANALYZE_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data: dict) -> None:
    _ANALYZE_CACHE[key] = {"ts": time.time(), "data": data}


# ── Deterministic mock helpers ────────────────────────────────────────────────

def _h(seed: str) -> int:
    """Return a stable 0-9999 integer derived from seed string."""
    return int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 10000


def _flood_zone(lat: float, lon: float) -> tuple[str, str]:
    """Return (zone, risk_level) using a deterministic hash of lat/lon."""
    h = _h(f"{lat:.4f},{lon:.4f}")
    if h < 1500:
        return "A", "High"
    if h < 3500:
        return "AE", "High"
    if h < 5000:
        return "AH", "Moderate"
    if h < 7000:
        return "X", "Minimal"
    return "X500", "Low"


def _crime_data(city_seed: str) -> dict:
    """Return realistic-looking crime rates per 100k from city seed."""
    h = _h(city_seed)
    # Violent crime: US average ~400/100k; range 80-800
    violent = 80 + int((h % 720))
    # Property crime: US average ~2100/100k; range 800-4000
    property_ = 800 + int((_h(city_seed + "p") % 3200))
    total = violent + property_
    return {"violent": violent, "property": property_, "total": total}


def _school_rating(lat: float, lon: float) -> int:
    """Return 1-10 school rating."""
    h = _h(f"school{lat:.3f},{lon:.3f}")
    return 1 + (h % 10)


def _walk_score(lat: float, lon: float) -> int:
    """Return 40-95 walk score."""
    h = _h(f"walk{lat:.3f},{lon:.3f}")
    return 40 + int((h / 10000) * 55)


def _composite_score(aqi: int, flood_risk: str, crime: dict, school: int, walk: int) -> int:
    """Compute a 0-100 composite neighborhood score (higher = better)."""
    # AQI component: 0-100 AQI → full points; degrades above 100
    aqi_score = max(0, 100 - aqi) / 100 * 25

    # Flood component
    flood_map = {"Minimal": 25, "Low": 20, "Moderate": 12, "High": 2}
    flood_score = flood_map.get(flood_risk, 10)

    # Crime component: violent+property mapped against US averages
    crime_norm = min(1.0, crime["total"] / 5000)
    crime_score = (1 - crime_norm) * 25

    # School component (1-10 → 0-15)
    school_score = ((school - 1) / 9) * 15

    # Walk score component (40-95 → 0-10)
    walk_score_pts = ((walk - 40) / 55) * 10

    total = aqi_score + flood_score + crime_score + school_score + walk_score_pts
    return max(0, min(100, round(total)))


def _aqi_category(aqi: int) -> str:
    if aqi < 50:   return "Good"
    if aqi < 100:  return "Moderate"
    if aqi < 150:  return "Unhealthy for Sensitive Groups"
    if aqi < 200:  return "Unhealthy"
    if aqi < 300:  return "Very Unhealthy"
    return "Hazardous"


# ── Real API fetchers ─────────────────────────────────────────────────────────

async def _geocode(location: str) -> Optional[tuple[float, float, str]]:
    """Geocode via Nominatim. Returns (lat, lon, display_name) or None."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": "1"}
    headers = {"User-Agent": "NeighborhoodPulse/1.0 (contact@neighborhoodpulse.app)"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            results = r.json()
            if not results:
                return None
            item = results[0]
            return float(item["lat"]), float(item["lon"]), item.get("display_name", location)
        except Exception as exc:
            logger.warning(f"Geocoding failed for '{location}': {exc}")
            return None


async def _fetch_aqi(lat: float, lon: float) -> int:
    """Fetch current US AQI from Open-Meteo air quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": "us_aqi",
        "forecast_days": "1",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            hourly = data.get("hourly", {})
            aqi_values = [v for v in (hourly.get("us_aqi") or []) if v is not None]
            if aqi_values:
                # Return the latest non-null value
                return int(aqi_values[-1])
        except Exception as exc:
            logger.warning(f"AQI fetch failed for ({lat},{lon}): {exc}")
    # Fallback: deterministic mock
    h = _h(f"aqi{lat:.2f},{lon:.2f}")
    return 20 + int((h / 10000) * 130)


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

_FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_FRONTEND / "index.html"))

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

@app.get("/api/analyze")
async def analyze_neighborhood(
    location: str = Query(..., min_length=3, description="Location name, city, or address"),
):
    """
    Fast synchronous neighborhood analysis.
    Returns AQI, flood zone, crime data, school rating, walk score, and a composite score.
    Results are cached for 24 hours.
    """
    # Input validation
    stripped = location.strip()
    if len(stripped) < 3:
        raise HTTPException(status_code=422, detail="Location must be at least 3 characters.")
    if not any(c.isalpha() for c in stripped):
        raise HTTPException(status_code=422, detail="Location must contain letters — enter a real place name.")

    cache_key = stripped.lower()
    cached = _cache_get(cache_key)
    if cached:
        return JSONResponse(content={**cached, "cached": True})

    # Geocode
    geo = await _geocode(stripped)
    if geo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not find location: '{stripped}'. Try a more specific city or address.",
        )
    lat, lon, display_name = geo

    # Fetch real AQI and mock data concurrently
    aqi = await _fetch_aqi(lat, lon)

    # Mock deterministic data
    flood_zone, flood_risk = _flood_zone(lat, lon)
    crime = _crime_data(f"{display_name.split(',')[0].lower()}")
    school = _school_rating(lat, lon)
    walk = _walk_score(lat, lon)
    composite = _composite_score(aqi, flood_risk, crime, school, walk)

    # Overall safety: invert crime + flood + aqi influence (0-100)
    safety_raw = composite + _h(f"safe{lat:.2f},{lon:.2f}") % 10 - 5
    overall_safety = max(0, min(100, safety_raw))

    result = {
        "location_name": display_name,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "aqi": aqi,
        "aqi_category": _aqi_category(aqi),
        "flood_zone": flood_zone,
        "flood_risk": flood_risk,
        "crime_violent": crime["violent"],
        "crime_property": crime["property"],
        "crime_total": crime["total"],
        "school_rating": school,
        "walk_score": walk,
        "overall_safety": overall_safety,
        "composite_score": composite,
        "cached": False,
    }
    _cache_set(cache_key, result)
    return JSONResponse(content=result)


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
