"""
FEMA Agent — retrieves flood zone and disaster declaration data.
Simulates FEMA API with location-aware realistic data.
15% chance of DEGRADED partial response to test graceful degradation.
"""

import asyncio
import hashlib
import random
from typing import Any

from .base_agent import AgentResult, AgentState, BaseAgent, Citation


# Coastal cities → high flood risk; mountain cities → low; midwest → moderate
_COASTAL_KEYWORDS = {
    "miami", "houston", "new orleans", "tampa", "jacksonville", "virginia beach",
    "charleston", "savannah", "galveston", "corpus christi", "mobile", "pensacola",
    "fort lauderdale", "west palm beach", "boca raton", "naples", "cape coral",
    "myrtle beach", "wilmington", "norfolk", "baltimore", "boston", "new york",
    "brooklyn", "jersey city", "hoboken", "newark", "providence", "newport",
    "portland", "seattle", "tacoma", "san diego", "los angeles", "long beach",
    "san francisco", "oakland", "sacramento",
}

_MOUNTAIN_KEYWORDS = {
    "denver", "salt lake", "boise", "helena", "cheyenne", "albuquerque",
    "tucson", "flagstaff", "boulder", "fort collins", "colorado springs",
    "reno", "provo", "missoula", "billings",
}

_FLOOD_ZONES = {
    "high": ["AE", "A", "VE", "V"],
    "moderate": ["AE", "X500", "AO"],
    "low": ["X", "X", "X", "B"],
}

_DISASTER_TYPES = [
    "Hurricane", "Severe Storm", "Flooding", "Tornado", "Winter Storm",
    "Earthquake", "Wildfire", "Drought", "Landslide", "Coastal Erosion",
]


def _location_seed(location: str) -> int:
    """Deterministic seed from location string so same city always returns same data."""
    return int(hashlib.md5(location.lower().strip().encode()).hexdigest(), 16) % (2**31)


def _classify_risk(location: str) -> str:
    loc_lower = location.lower()
    for kw in _COASTAL_KEYWORDS:
        if kw in loc_lower:
            return "high"
    for kw in _MOUNTAIN_KEYWORDS:
        if kw in loc_lower:
            return "low"
    return "moderate"


def _generate_flood_data(location: str, rng: random.Random) -> dict:
    risk_level = _classify_risk(location)
    flood_zone = rng.choice(_FLOOD_ZONES[risk_level])

    if risk_level == "high":
        bfe = round(rng.uniform(6.0, 18.0), 1)
        sfha = flood_zone in ("A", "AE", "V", "VE")
        insurance_required = True
        num_declarations = rng.randint(4, 10)
    elif risk_level == "moderate":
        bfe = round(rng.uniform(1.0, 8.0), 1)
        sfha = flood_zone in ("AE", "AO")
        insurance_required = sfha
        num_declarations = rng.randint(1, 5)
    else:
        bfe = round(rng.uniform(0.0, 2.0), 1)
        sfha = False
        insurance_required = False
        num_declarations = rng.randint(0, 2)

    current_year = 2026
    declarations = []
    years_used: set = set()
    for _ in range(num_declarations):
        year = rng.randint(current_year - 10, current_year - 1)
        while year in years_used:
            year = rng.randint(current_year - 10, current_year - 1)
        years_used.add(year)
        dtype = rng.choice(_DISASTER_TYPES)
        if risk_level == "high" and rng.random() < 0.6:
            dtype = rng.choice(["Hurricane", "Flooding", "Severe Storm"])
        declarations.append({
            "year": year,
            "disaster_type": dtype,
            "declaration_number": f"DR-{rng.randint(4000, 4999)}",
            "incident_period": f"{year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        })
    declarations.sort(key=lambda d: d["year"], reverse=True)

    return {
        "flood_zone": flood_zone,
        "base_flood_elevation_ft": bfe,
        "flood_insurance_required": insurance_required,
        "special_flood_hazard_area": sfha,
        "disaster_declarations": declarations,
        "risk_level": risk_level,
        "community_rating_system_discount": rng.choice([0, 5, 10, 15, 20]) if risk_level == "high" else 0,
        "firm_panel_effective_date": f"{rng.randint(2010, 2023)}-{rng.randint(1,12):02d}-01",
    }


def _make_citations(location: str, data: dict, retrieved_at: str) -> list[Citation]:
    citations = [
        Citation(
            source_name="FEMA National Flood Insurance Program (NFIP)",
            url="https://www.fema.gov/flood-insurance",
            retrieved_at=retrieved_at,
            data_field="flood_zone",
            raw_value=str(data.get("flood_zone", "")),
        ),
        Citation(
            source_name="FEMA Flood Map Service Center",
            url=f"https://msc.fema.gov/portal/search?AddressQuery={location.replace(' ', '+')}",
            retrieved_at=retrieved_at,
            data_field="base_flood_elevation_ft",
            raw_value=str(data.get("base_flood_elevation_ft", "")),
        ),
        Citation(
            source_name="FEMA Disaster Declarations Summary",
            url="https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2",
            retrieved_at=retrieved_at,
            data_field="disaster_declarations",
            raw_value=f"{len(data.get('disaster_declarations', []))} declarations in last 10 years",
        ),
    ]
    return citations


class FEMAAgent(BaseAgent):
    """
    Retrieves FEMA flood zone and disaster declaration data.
    Simulates realistic location-specific data with a 15% DEGRADED rate.
    """

    DEGRADED_PROBABILITY = 0.15

    def __init__(self):
        super().__init__("FEMA Data")

    async def _fetch(self, location: str, context: dict) -> AgentResult:
        # Simulate network latency
        await asyncio.sleep(random.uniform(0.3, 1.2))

        rng = random.Random(_location_seed(location))

        # 15% chance of simulated partial outage
        if rng.random() < self.DEGRADED_PROBABILITY:
            raise ConnectionError("FEMA API endpoint temporarily unavailable (simulated outage)")

        retrieved_at = self._now_iso()
        data = _generate_flood_data(location, rng)
        citations = _make_citations(location, data, retrieved_at)

        return AgentResult(
            agent_name=self.name,
            status=AgentState.SUCCESS,
            data=data,
            citations=citations,
        )

    async def _fetch_partial(self, location: str, context: dict) -> AgentResult:
        """Returns minimal flood data when the full fetch fails."""
        await asyncio.sleep(0.05)
        rng = random.Random(_location_seed(location) + 1)
        risk_level = _classify_risk(location)
        flood_zone = rng.choice(_FLOOD_ZONES[risk_level])
        retrieved_at = self._now_iso()

        partial_data = {
            "flood_zone": flood_zone,
            "base_flood_elevation_ft": None,
            "flood_insurance_required": risk_level == "high",
            "special_flood_hazard_area": risk_level == "high",
            "disaster_declarations": [],
            "risk_level": risk_level,
            "community_rating_system_discount": None,
            "firm_panel_effective_date": None,
        }

        citations = [
            Citation(
                source_name="FEMA National Flood Insurance Program (NFIP) — Partial Data",
                url="https://www.fema.gov/flood-insurance",
                retrieved_at=retrieved_at,
                data_field="flood_zone",
                raw_value=f"{flood_zone} (estimated from risk profile)",
            )
        ]

        return AgentResult(
            agent_name=self.name,
            status=AgentState.DEGRADED,
            data=partial_data,
            citations=citations,
        )
