"""
Synthesis Agent — aggregates data from all source agents into a structured
NeighborhoodReport with inline citations and computed risk scores.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional

from .base_agent import AgentResult, AgentState, BaseAgent, Citation


@dataclass
class NeighborhoodReport:
    location: str
    generated_at: str
    executive_summary: str
    risk_score: float                         # 0-100
    risk_breakdown: dict                      # factor → {score, weight, label}
    flood_section: dict
    zoning_section: dict
    climate_section: dict
    recommendations: list[str]
    data_gaps: list[str]
    all_citations: list[Citation]
    agent_statuses: dict                      # agent_name → status

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "generated_at": self.generated_at,
            "executive_summary": self.executive_summary,
            "risk_score": self.risk_score,
            "risk_breakdown": self.risk_breakdown,
            "flood_section": self.flood_section,
            "zoning_section": self.zoning_section,
            "climate_section": self.climate_section,
            "recommendations": self.recommendations,
            "data_gaps": self.data_gaps,
            "all_citations": [c.to_dict() if isinstance(c, Citation) else c for c in self.all_citations],
            "agent_statuses": self.agent_statuses,
        }


def _safe_get(data: dict, key: str, default=None):
    val = data.get(key, default)
    return default if val is None else val


def _compute_risk_score(fema_data: dict, climate_data: dict) -> tuple[float, dict]:
    """
    Weighted risk score computation.
    Returns (overall_score_0_100, breakdown_dict)
    """
    breakdown = {}

    # ── Flood Risk (30%)
    zone = _safe_get(fema_data, "flood_zone", "X")
    sfha = _safe_get(fema_data, "special_flood_hazard_area", False)
    num_disasters = len(_safe_get(fema_data, "disaster_declarations", []))
    flood_base = 0
    if zone in ("V", "VE"):
        flood_base = 95
    elif zone in ("A", "AE"):
        flood_base = 80
    elif zone in ("X500", "AO", "AH"):
        flood_base = 45
    else:
        flood_base = 10
    disaster_bonus = min(25, num_disasters * 3)
    flood_score = min(100, flood_base + disaster_bonus)
    breakdown["flood_risk"] = {"score": flood_score, "weight": 0.30, "label": "Flood & Disaster Risk"}

    # ── Wildfire Risk (20%)
    wildfire_map = {"low": 10, "medium": 40, "high": 70, "very_high": 95}
    wildfire_str = _safe_get(climate_data, "wildfire_risk", "low")
    wildfire_score = wildfire_map.get(wildfire_str, 10)
    breakdown["wildfire_risk"] = {"score": wildfire_score, "weight": 0.20, "label": "Wildfire Risk"}

    # ── Heat Risk (15%)
    heat_days = _safe_get(climate_data, "heat_index_days_over_100", 0)
    heat_score = min(100, (heat_days / 80) * 100) if heat_days else 5
    breakdown["heat_risk"] = {"score": round(heat_score, 1), "weight": 0.15, "label": "Extreme Heat Risk"}

    # ── Sea Level Rise (15%)
    slr = _safe_get(climate_data, "sea_level_rise_projection_ft", 0.0)
    slr_score = min(100, slr * 70) if slr else 0
    breakdown["sea_level_risk"] = {"score": round(slr_score, 1), "weight": 0.15, "label": "Sea Level Rise Risk"}

    # ── Drought Risk (10%)
    drought_map = {"low": 5, "moderate": 30, "severe": 65, "extreme": 90}
    drought_str = _safe_get(climate_data, "drought_risk", "low")
    drought_score = drought_map.get(drought_str, 5)
    breakdown["drought_risk"] = {"score": drought_score, "weight": 0.10, "label": "Drought Risk"}

    # ── Air Quality (10%)
    aqi = _safe_get(climate_data, "air_quality_index_avg", 50)
    aqi_score = min(100, max(0, (aqi - 25) * 1.5)) if aqi else 30
    breakdown["air_quality_risk"] = {"score": round(aqi_score, 1), "weight": 0.10, "label": "Air Quality Risk"}

    # Weighted sum
    total = sum(v["score"] * v["weight"] for v in breakdown.values())
    return round(total, 1), breakdown


def _build_flood_section(fema_data: dict, fema_citations: list[Citation], citation_offset: int) -> dict:
    zone = _safe_get(fema_data, "flood_zone", "Unknown")
    bfe = _safe_get(fema_data, "base_flood_elevation_ft")
    sfha = _safe_get(fema_data, "special_flood_hazard_area", False)
    insurance_req = _safe_get(fema_data, "flood_insurance_required", False)
    declarations = _safe_get(fema_data, "disaster_declarations", [])
    risk_level = _safe_get(fema_data, "risk_level", "moderate")
    crs_discount = _safe_get(fema_data, "community_rating_system_discount", 0)
    firm_date = _safe_get(fema_data, "firm_panel_effective_date", "N/A")

    sfha_text = "within" if sfha else "outside of"
    insurance_text = "is required" if insurance_req else "is not federally required, though may be advisable"
    bfe_text = f" The base flood elevation is {bfe} ft." if bfe is not None else ""

    decl_text = ""
    if declarations:
        recent = declarations[:3]
        types = list({d["disaster_type"] for d in recent})
        years = [str(d["year"]) for d in recent]
        decl_text = (
            f" FEMA has recorded {len(declarations)} major disaster declaration(s) in the past decade for this area, "
            f"including {', '.join(types[:2])} events in {', '.join(years[:3])}."
        )
    else:
        decl_text = " No major FEMA disaster declarations were recorded for this area in the past decade."

    crs_text = f" The community participates in the FEMA Community Rating System (CRS), qualifying homeowners for up to {crs_discount}% flood insurance premium discounts." if crs_discount else ""

    ci = citation_offset
    prose = (
        f"This property falls {sfha_text} a FEMA-designated Special Flood Hazard Area (SFHA), "
        f"classified as Flood Zone {zone} [{ci+1}].{bfe_text} "
        f"Federal flood insurance {insurance_text} for mortgage holders in this zone [{ci+2}]."
        f"{decl_text} [{ci+3}]"
        f"{crs_text}"
    )

    return {
        "title": "Flood & Disaster Risk",
        "flood_zone": zone,
        "base_flood_elevation_ft": bfe,
        "special_flood_hazard_area": sfha,
        "flood_insurance_required": insurance_req,
        "disaster_declarations_count": len(declarations),
        "disaster_declarations": declarations,
        "community_rating_system_discount_pct": crs_discount,
        "firm_panel_effective_date": firm_date,
        "prose": prose,
        "citation_indices": list(range(ci + 1, ci + len(fema_citations) + 1)),
    }


def _build_zoning_section(zoning_data: dict, zoning_citations: list[Citation], citation_offset: int) -> dict:
    code = _safe_get(zoning_data, "zoning_code", "N/A")
    desc = _safe_get(zoning_data, "zoning_description", "Unknown")
    uses = _safe_get(zoning_data, "permitted_uses", [])
    max_height = _safe_get(zoning_data, "max_building_height_ft")
    far = _safe_get(zoning_data, "floor_area_ratio")
    setbacks = _safe_get(zoning_data, "setback_requirements", {})
    overlays = _safe_get(zoning_data, "overlay_zones", [])
    last_updated = _safe_get(zoning_data, "last_updated", "N/A")
    jurisdiction = _safe_get(zoning_data, "jurisdiction", "Local Planning Department")
    parking = _safe_get(zoning_data, "parking_requirement", "varies")

    ci = citation_offset
    overlay_text = f" Additionally, the parcel falls within {', '.join(overlays)} overlay zone(s), adding further regulatory considerations." if overlays else ""
    setback_text = ""
    if setbacks:
        setback_text = (
            f" Required setbacks are {setbacks.get('front_ft', 0)} ft front, "
            f"{setbacks.get('rear_ft', 0)} ft rear, and {setbacks.get('side_ft', 0)} ft side."
        )

    uses_preview = ", ".join(uses[:4]) if uses else "not available"
    height_text = f" Maximum building height is {max_height} ft." if max_height else ""
    far_text = f" with a floor area ratio (FAR) of {far}" if far else ""

    prose = (
        f"The property is zoned {code} — {desc} under the {jurisdiction} ordinance [{ci+1}]{far_text}. "
        f"Permitted uses include: {uses_preview}.{height_text}"
        f"{setback_text}{overlay_text} "
        f"Parking requirements are set at {parking} [{ci+2}]. "
        f"Zoning classifications were last updated {last_updated} [{ci+3}]."
    )

    return {
        "title": "Zoning & Land Use",
        "zoning_code": code,
        "zoning_description": desc,
        "permitted_uses": uses,
        "max_building_height_ft": max_height,
        "floor_area_ratio": far,
        "setback_requirements": setbacks,
        "overlay_zones": overlays,
        "parking_requirement": parking,
        "last_updated": last_updated,
        "jurisdiction": jurisdiction,
        "prose": prose,
        "citation_indices": list(range(ci + 1, ci + len(zoning_citations) + 1)),
    }


def _build_climate_section(climate_data: dict, climate_citations: list[Citation], citation_offset: int) -> dict:
    wildfire = _safe_get(climate_data, "wildfire_risk", "low")
    heat_days = _safe_get(climate_data, "heat_index_days_over_100", 0)
    slr = _safe_get(climate_data, "sea_level_rise_projection_ft", 0.0)
    drought = _safe_get(climate_data, "drought_risk", "low")
    extreme_events = _safe_get(climate_data, "extreme_weather_events_per_decade", 0)
    aqi = _safe_get(climate_data, "air_quality_index_avg", 50)
    temp_increase = _safe_get(climate_data, "projected_temperature_increase_f_by_2050")
    region = _safe_get(climate_data, "region_profile", "general")
    carbon_trend = _safe_get(climate_data, "carbon_emission_trend", "stable")
    green_space = _safe_get(climate_data, "green_space_coverage_pct")

    ci = citation_offset
    wildfire_readable = wildfire.replace("_", " ").title()
    drought_readable = drought.title()

    heat_text = f"{heat_days} days per year" if heat_days else "minimal occurrence"
    slr_text = f"{slr} ft by 2050" if slr else "not a significant risk for inland locations"
    temp_text = f" Projected mean temperature increase of {temp_increase}°F by 2050." if temp_increase else ""

    aqi_category = (
        "Good" if aqi <= 50 else
        "Moderate" if aqi <= 100 else
        "Unhealthy for Sensitive Groups" if aqi <= 150 else "Unhealthy"
    )

    prose = (
        f"This location carries a {wildfire_readable} wildfire risk rating per USFS risk modeling [{ci+1}]. "
        f"Heat stress exposure averages {heat_text} with temperatures exceeding 100°F [{ci+2}]. "
        f"Sea level rise projections indicate {slr_text} [{ci+3}]. "
        f"Drought conditions are classified as {drought_readable} for this region [{ci+4}]. "
        f"Annual average air quality index of {aqi} ({aqi_category}) per EPA monitoring [{ci+4}]."
        f"{temp_text} "
        f"The area experiences approximately {extreme_events} major weather events per decade."
    )

    return {
        "title": "Climate Risk",
        "wildfire_risk": wildfire,
        "heat_index_days_over_100": heat_days,
        "sea_level_rise_projection_ft": slr,
        "drought_risk": drought,
        "extreme_weather_events_per_decade": extreme_events,
        "air_quality_index_avg": aqi,
        "aqi_category": aqi_category,
        "projected_temperature_increase_f_by_2050": temp_increase,
        "carbon_emission_trend": carbon_trend,
        "green_space_coverage_pct": green_space,
        "region_profile": region,
        "prose": prose,
        "citation_indices": list(range(ci + 1, ci + len(climate_citations) + 1)),
    }


def _generate_recommendations(
    risk_score: float,
    fema_data: dict,
    zoning_data: dict,
    climate_data: dict,
    data_gaps: list[str],
) -> list[str]:
    recs = []

    # Flood-related
    sfha = _safe_get(fema_data, "special_flood_hazard_area", False)
    insurance_req = _safe_get(fema_data, "flood_insurance_required", False)
    crs_discount = _safe_get(fema_data, "community_rating_system_discount", 0)
    zone = _safe_get(fema_data, "flood_zone", "X")
    declarations = _safe_get(fema_data, "disaster_declarations", [])

    if sfha or zone in ("A", "AE", "V", "VE"):
        recs.append(
            "Obtain an Elevation Certificate from a licensed surveyor before closing — this document "
            "is required for accurate NFIP flood insurance quotes and may significantly reduce premiums."
        )
    if insurance_req:
        recs.append(
            "Flood insurance is federally required for this property. " + (
                f"Consider the CRS discount ({crs_discount}% reduction) when shopping NFIP policies."
                if crs_discount else
                "Compare NFIP and private flood insurance carriers for the best rate."
            )
        )
    if len(declarations) >= 4:
        recs.append(
            "Given the area's history of repeated disaster declarations, request seller disclosure "
            "of prior flood or storm damage, and budget for resilience upgrades such as elevated utilities and sump pumps."
        )

    # Climate-related
    wildfire = _safe_get(climate_data, "wildfire_risk", "low")
    heat_days = _safe_get(climate_data, "heat_index_days_over_100", 0)
    slr = _safe_get(climate_data, "sea_level_rise_projection_ft", 0.0)
    drought = _safe_get(climate_data, "drought_risk", "low")
    aqi = _safe_get(climate_data, "air_quality_index_avg", 50)

    if wildfire in ("high", "very_high"):
        recs.append(
            "Consult California FAIR Plan or equivalent state insurer of last resort for wildfire coverage, "
            "and verify defensible space compliance (100-ft clearance) with local fire authority."
        )
    if heat_days and heat_days > 40:
        recs.append(
            f"With {heat_days} days per year exceeding 100°F, prioritize HVAC capacity review, "
            "attic insulation upgrades, and confirm utility grid reliability for summer load demands."
        )
    if slr and slr > 0.6:
        recs.append(
            f"Sea level rise projections of {slr} ft by 2050 pose long-term asset risk. "
            "Consult a coastal engineer and review the property's 30-year mortgage horizon against NOAA projections."
        )
    if drought in ("severe", "extreme"):
        recs.append(
            "Verify municipal water supply reliability and review HOA/local restrictions on irrigation. "
            "Consider drought-tolerant landscaping and rainwater harvesting where permitted."
        )

    # Zoning-related
    overlays = _safe_get(zoning_data, "overlay_zones", [])
    if "Historic Preservation Overlay" in overlays:
        recs.append(
            "Historic Preservation Overlay designation restricts exterior modifications and may require "
            "design review board approval for renovations — budget additional time and cost for permitting."
        )

    # Data gaps
    if data_gaps:
        recs.append(
            f"Data for {', '.join(data_gaps)} was incomplete during this analysis. "
            "Contact the relevant agencies directly for a comprehensive risk assessment before making investment decisions."
        )

    # High overall risk
    if risk_score >= 65:
        recs.append(
            "Given an elevated overall risk score, engage a licensed home inspector with environmental "
            "risk specialization, and review all natural hazard disclosure statements in the purchase agreement."
        )

    # Cap at 6 recommendations
    return recs[:6]


def _generate_executive_summary(
    location: str,
    risk_score: float,
    fema_data: dict,
    zoning_data: dict,
    climate_data: dict,
    data_gaps: list[str],
) -> str:
    risk_label = (
        "low" if risk_score < 30 else
        "moderate" if risk_score < 55 else
        "elevated" if risk_score < 75 else
        "high"
    )

    zone = _safe_get(fema_data, "flood_zone", "X")
    zoning_code = _safe_get(zoning_data, "zoning_code", "N/A")
    zoning_desc = _safe_get(zoning_data, "zoning_description", "")
    wildfire = _safe_get(climate_data, "wildfire_risk", "low").replace("_", " ")
    drought = _safe_get(climate_data, "drought_risk", "low")
    declarations = _safe_get(fema_data, "disaster_declarations", [])

    gap_note = (
        f" Note: data from {', '.join(data_gaps)} was partially unavailable and may affect completeness."
        if data_gaps else ""
    )

    summary = (
        f"{location} carries an overall neighborhood risk score of {risk_score}/100 ({risk_label} risk). "
        f"The area is designated FEMA Flood Zone {zone} with {len(declarations)} disaster declaration(s) in the past decade, "
        f"and carries a {wildfire} wildfire risk alongside {drought} drought conditions. "
        f"Zoning is classified {zoning_code} ({zoning_desc}), reflecting the area's land-use character.{gap_note}"
    )
    return summary


class SynthesisAgent(BaseAgent):
    """
    Aggregates outputs from FEMA, Zoning, and Climate agents into a
    structured NeighborhoodReport with computed risk scores, prose sections,
    inline citations, and actionable recommendations.
    """

    def __init__(self):
        super().__init__("Report Synthesis")

    async def _fetch(self, location: str, context: dict) -> AgentResult:
        await asyncio.sleep(random.uniform(0.8, 2.0))

        agent_results: list[AgentResult] = context.get("agent_results", [])
        retrieved_at = self._now_iso()

        fema_result = next((r for r in agent_results if "FEMA" in r.agent_name), None)
        zoning_result = next((r for r in agent_results if "Zoning" in r.agent_name), None)
        climate_result = next((r for r in agent_results if "Climate" in r.agent_name), None)

        fema_data = fema_result.data if fema_result else {}
        zoning_data = zoning_result.data if zoning_result else {}
        climate_data = climate_result.data if climate_result else {}

        # Identify data gaps (degraded or missing agents)
        data_gaps = []
        agent_statuses = {}
        for result in agent_results:
            agent_statuses[result.agent_name] = result.status.value if isinstance(result.status, AgentState) else result.status
            if result.status in (AgentState.DEGRADED, AgentState.FAILED):
                data_gaps.append(result.agent_name)

        # Aggregate all citations with global indices
        all_citations: list[Citation] = []
        fema_citations = fema_result.citations if fema_result else []
        zoning_citations = zoning_result.citations if zoning_result else []
        climate_citations = climate_result.citations if climate_result else []

        all_citations.extend(fema_citations)
        zoning_offset = len(all_citations)
        all_citations.extend(zoning_citations)
        climate_offset = len(all_citations)
        all_citations.extend(climate_citations)

        # Compute risk score
        risk_score, risk_breakdown = _compute_risk_score(fema_data, climate_data)

        # Build report sections
        flood_section = _build_flood_section(fema_data, fema_citations, 0)
        zoning_section = _build_zoning_section(zoning_data, zoning_citations, zoning_offset)
        climate_section = _build_climate_section(climate_data, climate_citations, climate_offset)

        # Generate narrative content
        executive_summary = _generate_executive_summary(
            location, risk_score, fema_data, zoning_data, climate_data, data_gaps
        )
        recommendations = _generate_recommendations(
            risk_score, fema_data, zoning_data, climate_data, data_gaps
        )

        report = NeighborhoodReport(
            location=location,
            generated_at=retrieved_at,
            executive_summary=executive_summary,
            risk_score=risk_score,
            risk_breakdown=risk_breakdown,
            flood_section=flood_section,
            zoning_section=zoning_section,
            climate_section=climate_section,
            recommendations=recommendations,
            data_gaps=data_gaps,
            all_citations=all_citations,
            agent_statuses=agent_statuses,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentState.SUCCESS,
            data=report.to_dict(),
            citations=all_citations,
        )

    async def _fetch_partial(self, location: str, context: dict) -> AgentResult:
        """Minimal stub report when synthesis itself fails (extremely rare)."""
        await asyncio.sleep(0.05)
        retrieved_at = self._now_iso()
        stub = {
            "location": location,
            "generated_at": retrieved_at,
            "executive_summary": f"Report for {location} could not be fully generated due to a synthesis error.",
            "risk_score": 50.0,
            "risk_breakdown": {},
            "flood_section": {"prose": "Data unavailable.", "title": "Flood & Disaster Risk"},
            "zoning_section": {"prose": "Data unavailable.", "title": "Zoning & Land Use"},
            "climate_section": {"prose": "Data unavailable.", "title": "Climate Risk"},
            "recommendations": ["Please retry the research request for a complete report."],
            "data_gaps": ["All agents"],
            "all_citations": [],
            "agent_statuses": {},
        }
        return AgentResult(
            agent_name=self.name,
            status=AgentState.DEGRADED,
            data=stub,
            citations=[],
        )
