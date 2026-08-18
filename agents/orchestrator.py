"""
Agent Orchestrator — LangGraph-style state machine that coordinates
FEMA, Zoning, and Climate agents concurrently, then runs Synthesis and
Citation Validation in sequence.

State machine:
  START → PARALLEL_FETCH → SYNTHESIS → VALIDATION → COMPLETE / PARTIAL_COMPLETE
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncIterator, Callable, Optional

from .base_agent import AgentResult, AgentState, Citation
from .citation_validator import CitationValidator, ValidationResult
from .climate_agent import ClimateAgent
from .fema_agent import FEMAAgent
from .synthesis_agent import SynthesisAgent
from .zoning_agent import ZoningAgent

logger = logging.getLogger(__name__)


class GraphState(str, Enum):
    START = "START"
    PARALLEL_FETCH = "PARALLEL_FETCH"
    SYNTHESIS = "SYNTHESIS"
    VALIDATION = "VALIDATION"
    COMPLETE = "COMPLETE"
    PARTIAL_COMPLETE = "PARTIAL_COMPLETE"
    ERROR = "ERROR"


@dataclass
class AgentGraph:
    """Tracks the current graph state, per-agent status, and timing."""
    graph_state: GraphState = GraphState.START
    agent_statuses: dict = field(default_factory=dict)
    agent_latencies: dict = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "graph_state": self.graph_state.value,
            "agent_statuses": self.agent_statuses,
            "agent_latencies": self.agent_latencies,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_latency_ms": self.total_latency_ms,
            "errors": self.errors,
        }


@dataclass
class OrchestratorResult:
    graph: AgentGraph
    agent_results: list[AgentResult]
    synthesis_result: Optional[AgentResult]
    validation_result: Optional[ValidationResult]
    report: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "graph": self.graph.to_dict(),
            "agent_results": [r.to_dict() for r in self.agent_results],
            "synthesis_result": self.synthesis_result.to_dict() if self.synthesis_result else None,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "report": self.report,
        }


EventCallback = Callable[[str, dict], None]


class AgentOrchestrator:
    """
    LangGraph-style multi-agent orchestrator.

    Runs FEMA, Zoning, and Climate agents concurrently via asyncio.gather,
    then feeds their results into SynthesisAgent and CitationValidator.

    Emits structured events at each state transition for SSE streaming.
    """

    def __init__(self):
        self.fema_agent = FEMAAgent()
        self.zoning_agent = ZoningAgent()
        self.climate_agent = ClimateAgent()
        self.synthesis_agent = SynthesisAgent()
        self.citation_validator = CitationValidator()
        self._event_callbacks: list[EventCallback] = []

    def on_event(self, callback: EventCallback) -> None:
        """Register a callback for real-time orchestration events."""
        self._event_callbacks.append(callback)

    def _emit(self, event_type: str, payload: dict) -> None:
        payload["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for cb in self._event_callbacks:
            try:
                cb(event_type, payload)
            except Exception as exc:
                logger.warning(f"Event callback error: {exc}")

    async def run(self, location: str) -> OrchestratorResult:
        """
        Execute the full agent pipeline for a location.
        Returns OrchestratorResult with report, validation, and graph state.
        """
        graph = AgentGraph()
        graph.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pipeline_start = time.monotonic()

        # ── State: START ──────────────────────────────────────────────────────
        graph.graph_state = GraphState.START
        self._emit("state_change", {"state": graph.graph_state.value, "location": location})
        logger.info(f"[Orchestrator] Starting pipeline for '{location}'")

        # Initialize agent statuses
        for agent_name in ["FEMA Data", "Zoning Records", "Climate Risk", "Report Synthesis"]:
            graph.agent_statuses[agent_name] = AgentState.IDLE.value
        self._emit("agents_initialized", {"agents": list(graph.agent_statuses.keys())})

        # ── State: PARALLEL_FETCH ─────────────────────────────────────────────
        graph.graph_state = GraphState.PARALLEL_FETCH
        self._emit("state_change", {"state": graph.graph_state.value})

        for name in ["FEMA Data", "Zoning Records", "Climate Risk"]:
            graph.agent_statuses[name] = AgentState.RUNNING.value
        self._emit("agents_running", {"agents": ["FEMA Data", "Zoning Records", "Climate Risk"]})

        # Wrap each agent to emit events on completion
        async def run_agent_with_events(agent, location: str) -> AgentResult:
            result = await agent.execute(location)
            graph.agent_statuses[agent.name] = result.status.value
            graph.agent_latencies[agent.name] = round(result.latency_ms, 1)
            self._emit("agent_complete", {
                "agent": agent.name,
                "status": result.status.value,
                "latency_ms": result.latency_ms,
                "citation_count": len(result.citations),
                "error": result.error,
            })
            if result.error:
                graph.errors.append(f"{agent.name}: {result.error}")
            return result

        fema_result, zoning_result, climate_result = await asyncio.gather(
            run_agent_with_events(self.fema_agent, location),
            run_agent_with_events(self.zoning_agent, location),
            run_agent_with_events(self.climate_agent, location),
        )

        agent_results = [fema_result, zoning_result, climate_result]

        # Determine transition: if 2+ agents FAILED, go to PARTIAL_COMPLETE
        failed_count = sum(
            1 for r in agent_results if r.status == AgentState.FAILED
        )
        degraded_count = sum(
            1 for r in agent_results if r.status in (AgentState.DEGRADED, AgentState.FAILED)
        )

        if failed_count >= 2:
            logger.warning(f"[Orchestrator] {failed_count} agents failed — entering PARTIAL_COMPLETE")
            graph.graph_state = GraphState.PARTIAL_COMPLETE
            self._emit("state_change", {
                "state": graph.graph_state.value,
                "reason": f"{failed_count} data agents failed",
            })

            # Still attempt synthesis with whatever we have
            graph.agent_statuses["Report Synthesis"] = AgentState.RUNNING.value
            self._emit("agent_started", {"agent": "Report Synthesis"})
            synthesis_result = await self.synthesis_agent.execute(
                location, context={"agent_results": agent_results}
            )
            graph.agent_statuses["Report Synthesis"] = synthesis_result.status.value
            graph.agent_latencies["Report Synthesis"] = round(synthesis_result.latency_ms, 1)
            self._emit("agent_complete", {
                "agent": "Report Synthesis",
                "status": synthesis_result.status.value,
                "latency_ms": synthesis_result.latency_ms,
            })
            graph.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            graph.total_latency_ms = round((time.monotonic() - pipeline_start) * 1000, 1)
            self._emit("pipeline_complete", {
                "state": graph.graph_state.value,
                "total_latency_ms": graph.total_latency_ms,
            })
            return OrchestratorResult(
                graph=graph,
                agent_results=agent_results,
                synthesis_result=synthesis_result,
                validation_result=None,
                report=synthesis_result.data if synthesis_result else None,
            )

        # ── State: SYNTHESIS ──────────────────────────────────────────────────
        graph.graph_state = GraphState.SYNTHESIS
        self._emit("state_change", {"state": graph.graph_state.value})

        graph.agent_statuses["Report Synthesis"] = AgentState.RUNNING.value
        self._emit("agent_started", {"agent": "Report Synthesis"})

        synthesis_result = await self.synthesis_agent.execute(
            location, context={"agent_results": agent_results}
        )
        graph.agent_statuses["Report Synthesis"] = synthesis_result.status.value
        graph.agent_latencies["Report Synthesis"] = round(synthesis_result.latency_ms, 1)
        self._emit("agent_complete", {
            "agent": "Report Synthesis",
            "status": synthesis_result.status.value,
            "latency_ms": synthesis_result.latency_ms,
            "citation_count": len(synthesis_result.citations),
        })

        # ── State: VALIDATION ─────────────────────────────────────────────────
        graph.graph_state = GraphState.VALIDATION
        self._emit("state_change", {"state": graph.graph_state.value})

        validation_result: Optional[ValidationResult] = None
        try:
            report_dict = synthesis_result.data
            validation_result = self.citation_validator.validate(report_dict, agent_results)
            self._emit("validation_complete", {
                "accuracy_rate": validation_result.accuracy_rate,
                "accuracy_rate_pct": round(validation_result.accuracy_rate * 100, 1),
                "total_claims": validation_result.total_claims,
                "total_verified": validation_result.total_verified,
            })
            logger.info(
                f"[Orchestrator] Validation: {validation_result.accuracy_rate*100:.1f}% accuracy "
                f"({validation_result.total_verified}/{validation_result.total_claims} claims)"
            )
        except Exception as exc:
            logger.error(f"[Orchestrator] Citation validation failed: {exc}")
            graph.errors.append(f"Validation: {exc}")

        # ── State: COMPLETE ───────────────────────────────────────────────────
        graph.graph_state = GraphState.COMPLETE if degraded_count == 0 else GraphState.PARTIAL_COMPLETE
        graph.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        graph.total_latency_ms = round((time.monotonic() - pipeline_start) * 1000, 1)

        self._emit("state_change", {"state": graph.graph_state.value})
        self._emit("pipeline_complete", {
            "state": graph.graph_state.value,
            "total_latency_ms": graph.total_latency_ms,
            "risk_score": synthesis_result.data.get("risk_score") if synthesis_result.data else None,
            "citation_accuracy_pct": round(validation_result.accuracy_rate * 100, 1) if validation_result else None,
        })

        logger.info(
            f"[Orchestrator] Pipeline complete for '{location}' in {graph.total_latency_ms:.0f}ms — "
            f"state={graph.graph_state.value}"
        )

        return OrchestratorResult(
            graph=graph,
            agent_results=agent_results,
            synthesis_result=synthesis_result,
            validation_result=validation_result,
            report=synthesis_result.data,
        )
