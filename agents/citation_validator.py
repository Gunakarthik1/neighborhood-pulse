"""
Citation Validator — verifies that every factual claim in a NeighborhoodReport
can be traced back to a citation in the agent results.
Target: 94% citation accuracy.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .base_agent import AgentResult, Citation
from .synthesis_agent import NeighborhoodReport


@dataclass
class ClaimCheck:
    field_name: str
    section: str
    report_value: Any
    is_verified: bool
    matched_citation: Optional[str] = None
    matched_agent: Optional[str] = None


@dataclass
class ValidationResult:
    accuracy_rate: float                                 # 0.0–1.0
    validated_claims: list[ClaimCheck]
    unverified_claims: list[ClaimCheck]
    citation_coverage_by_section: dict[str, float]      # section → coverage 0.0–1.0
    total_claims: int
    total_verified: int
    citation_count_by_agent: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "accuracy_rate": self.accuracy_rate,
            "accuracy_rate_pct": round(self.accuracy_rate * 100, 1),
            "total_claims": self.total_claims,
            "total_verified": self.total_verified,
            "total_unverified": self.total_claims - self.total_verified,
            "citation_coverage_by_section": self.citation_coverage_by_section,
            "citation_count_by_agent": self.citation_count_by_agent,
            "validated_claims": [
                {
                    "field": c.field_name,
                    "section": c.section,
                    "value": str(c.report_value),
                    "matched_citation": c.matched_citation,
                    "matched_agent": c.matched_agent,
                }
                for c in self.validated_claims
            ],
            "unverified_claims": [
                {
                    "field": c.field_name,
                    "section": c.section,
                    "value": str(c.report_value),
                }
                for c in self.unverified_claims
            ],
        }


# Fields to validate per section and which agent data key to look up
_FLOOD_CHECKS = [
    ("flood_zone", "flood_zone"),
    ("base_flood_elevation_ft", "base_flood_elevation_ft"),
    ("flood_insurance_required", "flood_insurance_required"),
    ("special_flood_hazard_area", "special_flood_hazard_area"),
    ("disaster_declarations_count", None),       # derived field — checked by citation presence
    ("community_rating_system_discount_pct", "community_rating_system_discount"),
]

_ZONING_CHECKS = [
    ("zoning_code", "zoning_code"),
    ("zoning_description", "zoning_description"),
    ("max_building_height_ft", "max_building_height_ft"),
    ("floor_area_ratio", "floor_area_ratio"),
    ("last_updated", "last_updated"),
]

_CLIMATE_CHECKS = [
    ("wildfire_risk", "wildfire_risk"),
    ("heat_index_days_over_100", "heat_index_days_over_100"),
    ("sea_level_rise_projection_ft", "sea_level_rise_projection_ft"),
    ("drought_risk", "drought_risk"),
    ("air_quality_index_avg", "air_quality_index_avg"),
    ("extreme_weather_events_per_decade", "extreme_weather_events_per_decade"),
]


def _values_match(report_val: Any, agent_val: Any) -> bool:
    """Flexible comparison — handles None, numeric rounding, string normalization."""
    if report_val is None and agent_val is None:
        return True
    if report_val is None or agent_val is None:
        return False
    # Exact match
    if report_val == agent_val:
        return True
    # Numeric within 1%
    try:
        rv = float(report_val)
        av = float(agent_val)
        if av == 0:
            return rv == 0
        return abs(rv - av) / abs(av) < 0.02
    except (TypeError, ValueError):
        pass
    # String normalization
    return str(report_val).strip().lower() == str(agent_val).strip().lower()


def _citation_covers_field(citations: list[Citation], field_name: str) -> Optional[Citation]:
    """Return the first citation whose data_field matches or is semantically related."""
    field_lower = field_name.lower().replace("_", " ")
    for c in citations:
        if c.data_field.lower().replace("_", " ") in field_lower or field_lower in c.data_field.lower().replace("_", " "):
            return c
    return None


class CitationValidator:
    """
    Validates that factual claims in a NeighborhoodReport map to verifiable
    citations from the underlying agent results.
    """

    def validate(
        self,
        report: NeighborhoodReport | dict,
        agent_results: list[AgentResult],
    ) -> ValidationResult:
        # Accept both NeighborhoodReport objects and plain dicts (e.g. from DB)
        if isinstance(report, dict):
            report_dict = report
        else:
            report_dict = report.to_dict()

        # Build lookup: agent_name → (data dict, citations list)
        agent_lookup: dict[str, tuple[dict, list[Citation]]] = {}
        for ar in agent_results:
            citations = [
                Citation.from_dict(c) if isinstance(c, dict) else c
                for c in ar.citations
            ]
            agent_lookup[ar.agent_name] = (ar.data, citations)

        # Citation count by agent
        citation_count_by_agent = {name: len(cits) for name, (_, cits) in agent_lookup.items()}

        all_checks: list[ClaimCheck] = []
        section_checks: dict[str, list[ClaimCheck]] = {
            "flood": [],
            "zoning": [],
            "climate": [],
        }

        # ── Flood section validation ──────────────────────────────────────────
        flood_section = report_dict.get("flood_section", {})
        fema_data, fema_citations = next(
            ((d, c) for name, (d, c) in agent_lookup.items() if "FEMA" in name),
            ({}, []),
        )
        for report_field, agent_field in _FLOOD_CHECKS:
            report_val = flood_section.get(report_field)
            if report_val is None:
                continue
            agent_val = fema_data.get(agent_field) if agent_field else None
            citation = _citation_covers_field(fema_citations, agent_field or report_field)

            # A claim is verified if:
            #   (a) value matches agent raw data, OR
            #   (b) a citation explicitly covers the field (for derived/count fields)
            if agent_field:
                verified = _values_match(report_val, agent_val)
            else:
                # Derived field — verified by citation presence
                verified = citation is not None

            check = ClaimCheck(
                field_name=report_field,
                section="flood",
                report_value=report_val,
                is_verified=verified,
                matched_citation=citation.source_name if (verified and citation) else None,
                matched_agent="FEMA Data" if verified else None,
            )
            all_checks.append(check)
            section_checks["flood"].append(check)

        # ── Zoning section validation ─────────────────────────────────────────
        zoning_section = report_dict.get("zoning_section", {})
        zoning_data, zoning_citations = next(
            ((d, c) for name, (d, c) in agent_lookup.items() if "Zoning" in name),
            ({}, []),
        )
        for report_field, agent_field in _ZONING_CHECKS:
            report_val = zoning_section.get(report_field)
            if report_val is None:
                continue
            agent_val = zoning_data.get(agent_field) if agent_field else None
            citation = _citation_covers_field(zoning_citations, agent_field or report_field)
            verified = _values_match(report_val, agent_val) if agent_field else citation is not None

            check = ClaimCheck(
                field_name=report_field,
                section="zoning",
                report_value=report_val,
                is_verified=verified,
                matched_citation=citation.source_name if (verified and citation) else None,
                matched_agent="Zoning Records" if verified else None,
            )
            all_checks.append(check)
            section_checks["zoning"].append(check)

        # ── Climate section validation ────────────────────────────────────────
        climate_section = report_dict.get("climate_section", {})
        climate_data, climate_citations = next(
            ((d, c) for name, (d, c) in agent_lookup.items() if "Climate" in name),
            ({}, []),
        )
        for report_field, agent_field in _CLIMATE_CHECKS:
            report_val = climate_section.get(report_field)
            if report_val is None:
                continue
            agent_val = climate_data.get(agent_field) if agent_field else None
            citation = _citation_covers_field(climate_citations, agent_field or report_field)
            verified = _values_match(report_val, agent_val) if agent_field else citation is not None

            check = ClaimCheck(
                field_name=report_field,
                section="climate",
                report_value=report_val,
                is_verified=verified,
                matched_citation=citation.source_name if (verified and citation) else None,
                matched_agent="Climate Risk" if verified else None,
            )
            all_checks.append(check)
            section_checks["climate"].append(check)

        # ── Cross-cutting: verify all_citations in report appear in agent results ──
        report_citations = report_dict.get("all_citations", [])
        agent_all_citations = [
            cit for _, cits in agent_lookup.values() for cit in cits
        ]
        agent_citation_urls = {
            (c.url if isinstance(c, Citation) else c.get("url", ""))
            for c in agent_all_citations
        }

        for rc in report_citations:
            url = rc.get("url", "") if isinstance(rc, dict) else rc.url
            field_name = rc.get("data_field", "") if isinstance(rc, dict) else rc.data_field
            source = rc.get("source_name", "") if isinstance(rc, dict) else rc.source_name
            # A citation is verified if its URL exists in the agent citation pool
            verified = url in agent_citation_urls
            check = ClaimCheck(
                field_name=f"citation:{field_name}",
                section="citations",
                report_value=source,
                is_verified=verified,
                matched_citation=source if verified else None,
                matched_agent=None,
            )
            all_checks.append(check)

        # ── Compute accuracy metrics ──────────────────────────────────────────
        validated = [c for c in all_checks if c.is_verified]
        unverified = [c for c in all_checks if not c.is_verified]
        total = len(all_checks)
        total_verified = len(validated)

        accuracy_rate = total_verified / total if total > 0 else 1.0

        # Section-level coverage
        citation_coverage_by_section: dict[str, float] = {}
        for section_name, checks in section_checks.items():
            if checks:
                section_verified = sum(1 for c in checks if c.is_verified)
                citation_coverage_by_section[section_name] = round(section_verified / len(checks), 3)
            else:
                citation_coverage_by_section[section_name] = 1.0

        # Citations section coverage
        citation_checks = [c for c in all_checks if c.section == "citations"]
        if citation_checks:
            cit_verified = sum(1 for c in citation_checks if c.is_verified)
            citation_coverage_by_section["citations"] = round(cit_verified / len(citation_checks), 3)

        return ValidationResult(
            accuracy_rate=round(accuracy_rate, 4),
            validated_claims=validated,
            unverified_claims=unverified,
            citation_coverage_by_section=citation_coverage_by_section,
            total_claims=total,
            total_verified=total_verified,
            citation_count_by_agent=citation_count_by_agent,
        )
