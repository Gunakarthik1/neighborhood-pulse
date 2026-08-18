"""
Zoning Agent — retrieves municipal zoning and land-use data.
Simulates municipal APIs with urban/suburban/rural heuristics.
"""

import asyncio
import hashlib
import random

from .base_agent import AgentResult, AgentState, BaseAgent, Citation


_URBAN_KEYWORDS = {
    "new york", "brooklyn", "manhattan", "chicago", "los angeles", "san francisco",
    "boston", "philadelphia", "washington", "seattle", "portland", "denver",
    "atlanta", "miami", "dallas", "houston", "phoenix", "san diego", "austin",
    "minneapolis", "detroit", "baltimore", "pittsburgh", "cleveland", "st. louis",
    "new orleans", "tampa", "orlando", "charlotte", "nashville", "memphis",
    "kansas city", "indianapolis", "columbus", "milwaukee", "jacksonville",
    "las vegas", "sacramento", "salt lake city", "richmond", "raleigh",
}

_RURAL_KEYWORDS = {
    "rural", "county", "township", "unincorporated", "village", "hamlet",
    "springs", "junction", "crossing", "mills", "falls",
}

_URBAN_ZONES = [
    ("MX-5", "Mixed-Use Urban Core", ["multi-family residential", "retail", "office", "restaurant", "hotel", "entertainment"], 120, 4.0),
    ("C-3", "General Commercial District", ["retail", "office", "restaurant", "personal services", "auto-oriented commercial"], 65, 2.5),
    ("R-10", "High-Density Residential", ["multi-family residential", "assisted living", "daycare"], 75, 3.0),
    ("MX-2", "Neighborhood Mixed-Use", ["multi-family residential", "ground-floor retail", "office", "restaurant"], 55, 2.0),
]

_SUBURBAN_ZONES = [
    ("R-2", "Medium-Density Residential", ["single-family residential", "duplex", "home occupation", "daycare"], 35, 0.5),
    ("R-1", "Single-Family Residential", ["single-family residential", "accessory dwelling unit", "home occupation"], 30, 0.4),
    ("C-1", "Neighborhood Commercial", ["retail", "personal services", "restaurant (limited)", "office"], 40, 1.0),
    ("O-M", "Office/Medical", ["office", "medical clinic", "pharmacy", "laboratory"], 45, 1.5),
]

_RURAL_ZONES = [
    ("A-1", "Agricultural", ["farming", "livestock", "single-family residential (limited)", "ranch"], 35, 0.1),
    ("R-R", "Rural Residential", ["single-family residential", "agriculture", "equestrian", "vacation rental"], 28, 0.2),
    ("RC", "Resource Conservation", ["conservation", "open space", "hiking", "limited agriculture"], 20, 0.05),
]

_SETBACK_TEMPLATES = {
    "urban": {"front_ft": 0, "rear_ft": 10, "side_ft": 0},
    "suburban": {"front_ft": 20, "rear_ft": 25, "side_ft": 5},
    "rural": {"front_ft": 40, "rear_ft": 50, "side_ft": 15},
}


def _location_seed(location: str) -> int:
    return int(hashlib.md5(location.lower().strip().encode()).hexdigest(), 16) % (2**31)


def _classify_urban_type(location: str) -> str:
    loc_lower = location.lower()
    for kw in _RURAL_KEYWORDS:
        if kw in loc_lower:
            return "rural"
    for kw in _URBAN_KEYWORDS:
        if kw in loc_lower:
            return "urban"
    # Heuristic: if location is short single word, often suburban
    words = [w for w in loc_lower.split() if w not in ("city", "town", "of", "the")]
    if len(words) == 1 and len(words[0]) < 8:
        return "suburban"
    return "suburban"


def _generate_zoning_data(location: str, rng: random.Random) -> dict:
    urban_type = _classify_urban_type(location)

    if urban_type == "urban":
        zone_code, zone_desc, permitted_uses, max_height, far = rng.choice(_URBAN_ZONES)
        max_height = rng.randint(max_height - 15, max_height + 30)
    elif urban_type == "rural":
        zone_code, zone_desc, permitted_uses, max_height, far = rng.choice(_RURAL_ZONES)
        max_height = rng.randint(max_height - 5, max_height + 10)
    else:
        zone_code, zone_desc, permitted_uses, max_height, far = rng.choice(_SUBURBAN_ZONES)
        max_height = rng.randint(max_height - 5, max_height + 15)

    setbacks = _SETBACK_TEMPLATES[urban_type].copy()
    setbacks["front_ft"] = max(0, setbacks["front_ft"] + rng.randint(-5, 5))
    setbacks["rear_ft"] = max(5, setbacks["rear_ft"] + rng.randint(-5, 10))
    setbacks["side_ft"] = max(0, setbacks["side_ft"] + rng.randint(-3, 5))

    update_year = rng.randint(2018, 2025)
    update_month = rng.randint(1, 12)

    overlay_zones = []
    if rng.random() < 0.3:
        overlay_zones.append("Historic Preservation Overlay")
    if rng.random() < 0.25:
        overlay_zones.append("Floodplain Development Overlay")
    if rng.random() < 0.2:
        overlay_zones.append("Transit-Oriented Development Overlay")

    return {
        "zoning_code": zone_code,
        "zoning_description": zone_desc,
        "urban_type": urban_type,
        "permitted_uses": permitted_uses,
        "max_building_height_ft": max_height,
        "floor_area_ratio": round(far + rng.uniform(-0.1, 0.3), 2),
        "setback_requirements": setbacks,
        "overlay_zones": overlay_zones,
        "minimum_lot_size_sqft": rng.randint(2000, 40000) if urban_type != "urban" else None,
        "parking_requirement": f"{round(rng.uniform(0.5, 2.0), 1)} spaces per unit" if urban_type != "rural" else "No minimum",
        "last_updated": f"{update_year}-{update_month:02d}-01",
        "jurisdiction": f"{location.split(',')[0].strip()} Planning Department",
    }


def _make_citations(location: str, data: dict, retrieved_at: str) -> list[Citation]:
    city_name = location.split(",")[0].strip().replace(" ", "-").lower()
    jurisdiction = data.get("jurisdiction", f"{location} Planning Department")

    citations = [
        Citation(
            source_name=f"{jurisdiction} — Official Zoning Map",
            url=f"https://www.{city_name.replace('-','')}.gov/zoning/map",
            retrieved_at=retrieved_at,
            data_field="zoning_code",
            raw_value=f"{data['zoning_code']} — {data['zoning_description']}",
        ),
        Citation(
            source_name=f"{jurisdiction} — Municipal Code, Title 23 (Zoning Ordinance)",
            url=f"https://www.{city_name.replace('-','')}.gov/municipal-code/title-23",
            retrieved_at=retrieved_at,
            data_field="floor_area_ratio",
            raw_value=f"FAR {data['floor_area_ratio']}, Max Height {data['max_building_height_ft']} ft",
        ),
        Citation(
            source_name="American Planning Association — Zoning Practice Database",
            url="https://www.planning.org/publications/zoningpractice/",
            retrieved_at=retrieved_at,
            data_field="permitted_uses",
            raw_value=", ".join(data["permitted_uses"][:3]),
        ),
    ]
    return citations


class ZoningAgent(BaseAgent):
    """
    Retrieves municipal zoning and land-use data.
    Uses urban/suburban/rural heuristics to generate realistic data.
    """

    def __init__(self):
        super().__init__("Zoning Records")

    async def _fetch(self, location: str, context: dict) -> AgentResult:
        await asyncio.sleep(random.uniform(0.4, 1.5))

        rng = random.Random(_location_seed(location))

        retrieved_at = self._now_iso()
        data = _generate_zoning_data(location, rng)
        citations = _make_citations(location, data, retrieved_at)

        return AgentResult(
            agent_name=self.name,
            status=AgentState.SUCCESS,
            data=data,
            citations=citations,
        )

    async def _fetch_partial(self, location: str, context: dict) -> AgentResult:
        await asyncio.sleep(0.05)
        rng = random.Random(_location_seed(location) + 7)
        urban_type = _classify_urban_type(location)
        retrieved_at = self._now_iso()

        if urban_type == "urban":
            zone_code, zone_desc = "MX", "Mixed-Use (estimated)"
        elif urban_type == "rural":
            zone_code, zone_desc = "A-1", "Agricultural (estimated)"
        else:
            zone_code, zone_desc = "R-1", "Residential (estimated)"

        partial_data = {
            "zoning_code": zone_code,
            "zoning_description": zone_desc,
            "urban_type": urban_type,
            "permitted_uses": [],
            "max_building_height_ft": None,
            "floor_area_ratio": None,
            "setback_requirements": None,
            "overlay_zones": [],
            "minimum_lot_size_sqft": None,
            "parking_requirement": None,
            "last_updated": None,
            "jurisdiction": f"{location.split(',')[0].strip()} Planning Department",
        }

        citations = [
            Citation(
                source_name=f"{location.split(',')[0].strip()} Planning Department — Partial Data",
                url=f"https://www.{location.split(',')[0].strip().lower().replace(' ','')}.gov/planning",
                retrieved_at=retrieved_at,
                data_field="zoning_code",
                raw_value=f"{zone_code} (estimated from location profile)",
            )
        ]

        return AgentResult(
            agent_name=self.name,
            status=AgentState.DEGRADED,
            data=partial_data,
            citations=citations,
        )
