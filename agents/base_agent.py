"""
Base agent class with retry, timeout, and graceful degradation.
All agents inherit from BaseAgent.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


@dataclass
class Citation:
    source_name: str
    url: str
    retrieved_at: str
    data_field: str
    raw_value: str

    def to_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "data_field": self.data_field,
            "raw_value": self.raw_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Citation":
        return cls(**d)


@dataclass
class AgentResult:
    agent_name: str
    status: AgentState
    data: dict = field(default_factory=dict)
    citations: list = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "status": self.status.value if isinstance(self.status, AgentState) else self.status,
            "data": self.data,
            "citations": [c.to_dict() if isinstance(c, Citation) else c for c in self.citations],
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentResult":
        citations = [Citation.from_dict(c) if isinstance(c, dict) else c for c in d.get("citations", [])]
        status_val = d.get("status", "FAILED")
        try:
            status = AgentState(status_val)
        except ValueError:
            status = AgentState.FAILED
        return cls(
            agent_name=d["agent_name"],
            status=status,
            data=d.get("data", {}),
            citations=citations,
            latency_ms=d.get("latency_ms", 0.0),
            error=d.get("error"),
        )


class BaseAgent(ABC):
    """
    Abstract base class for all NeighborhoodPulse agents.
    Provides exponential backoff retry (3 attempts), 10s timeout,
    and graceful degradation — returns partial results instead of crashing.
    """

    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 10.0
    BASE_BACKOFF_SECONDS = 0.5

    def __init__(self, name: str):
        self.name = name
        self.state = AgentState.IDLE
        self._logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def _fetch(self, location: str, context: dict) -> AgentResult:
        """
        Core fetch logic implemented by each subclass.
        Should raise exceptions on failure; BaseAgent handles retry/degradation.
        """
        ...

    @abstractmethod
    async def _fetch_partial(self, location: str, context: dict) -> AgentResult:
        """
        Returns minimal/partial data when full fetch fails.
        Must not raise; used for graceful degradation.
        """
        ...

    async def execute(self, location: str, context: dict | None = None) -> AgentResult:
        """
        Execute the agent with retry logic and graceful degradation.
        Returns AgentResult regardless of success/failure.
        """
        context = context or {}
        self.state = AgentState.RUNNING
        start_ms = time.monotonic() * 1000

        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._logger.debug(f"[{self.name}] Attempt {attempt}/{self.MAX_RETRIES} for '{location}'")
                result = await asyncio.wait_for(
                    self._fetch(location, context),
                    timeout=self.TIMEOUT_SECONDS,
                )
                result.latency_ms = time.monotonic() * 1000 - start_ms
                self.state = result.status
                self._logger.info(
                    f"[{self.name}] Completed in {result.latency_ms:.1f}ms — status={result.status.value}"
                )
                return result

            except asyncio.TimeoutError as exc:
                last_error = exc
                self._logger.warning(f"[{self.name}] Attempt {attempt} timed out after {self.TIMEOUT_SECONDS}s")
            except Exception as exc:
                last_error = exc
                self._logger.warning(f"[{self.name}] Attempt {attempt} failed: {exc}")

            if attempt < self.MAX_RETRIES:
                backoff = self.BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                self._logger.debug(f"[{self.name}] Backing off {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

        # All retries exhausted — attempt graceful degradation
        self._logger.warning(f"[{self.name}] All retries exhausted; returning degraded partial result")
        try:
            partial = await asyncio.wait_for(
                self._fetch_partial(location, context),
                timeout=self.TIMEOUT_SECONDS,
            )
            partial.latency_ms = time.monotonic() * 1000 - start_ms
            partial.status = AgentState.DEGRADED
            partial.error = str(last_error) if last_error else "Max retries exceeded"
            self.state = AgentState.DEGRADED
            return partial
        except Exception as exc:
            self._logger.error(f"[{self.name}] Partial fetch also failed: {exc}")
            self.state = AgentState.FAILED
            return AgentResult(
                agent_name=self.name,
                status=AgentState.FAILED,
                data={},
                citations=[],
                latency_ms=time.monotonic() * 1000 - start_ms,
                error=f"Agent failed completely: {last_error}",
            )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
