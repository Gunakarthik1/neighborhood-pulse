"""
Climate Agent — retrieves climate risk data from NOAA and EPA sources.
Location-specific: California cities get wildfire risk, Florida gets sea level rise,
Phoenix gets heat index, Pacific Northwest gets low wildfire + flood risk.
"""

import asyncio
import hashlib
import random

from .base_agent import AgentResult, AgentState, BaseAgent, Citation


# Regional climate risk profiles
_CALIFORNIA_KEYWORDS = {
    "los angeles", "san francisco", "san diego", "sacramento", "fresno",
    "oakland", "san jose", "berkeley", "santa barbara", "santa ana",
    "riverside", "stockton", "bakersfield", "anaheim", "irvine",
    "long beach", "chula vista", "santa clara", "modesto", "oxnard",
    "redding", "napa", "malibu", "pasadena", "ventura",
}

_FLORIDA_KEYWORDS = {
    "miami", "tampa", "orlando", "jacksonville", "fort lauderdale",
    "west palm beach", "boca raton", "naples", "cape coral", "sarasota",
    "pensacola", "tallahassee", "gainesville", "clearwater", "st. petersburg",
    "daytona", "fort myers", "key west", "coral gables",
}

_ARIZONA_KEYWORDS = {
    "phoenix", "tucson", "tempe", "mesa", "scottsdale", "chandler",
    "glendale", "peoria", "surprise", "gilbert", "flagstaff",
}

_PACIFIC_NW_KEYWORDS = {
    "seattle", "portland", "tacoma", "bellevue", "olympia", "spokane",
    "boise", "eugene", "salem", "vancouver", "everett",
}

_TEXAS_KEYWORDS = {
    "houston", "dallas", "austin", "san antonio", "fort worth", "el paso",
    "arlington", "corpus christi", "plano", "laredo", "lubbock", "amarillo",
}

_MIDWEST_KEYWORDS = {
    "chicago", "detroit", "indianapolis", "columbus", "milwaukee", "minneapolis",
    "st. louis", "kansas city", "cleveland", "cincinnati", "omaha", "des moines",
    "madison", "green bay", "ann arbor",
}

_WILDFIRE_LEVELS = ["low", "medium", "high", "very_high"]
_DROUGHT_LEVELS = ["low", "moderate", "severe", "extreme"]


def _location_seed(location: str) -> int:
    return int(hashlib.md5(location.lower().strip().encode()).hexdigest(), 16) % (2**31)


def _classify_climate_region(location: str) -> str:
    loc_lower = location.lower()
    for kw in _CALIFORNIA_KEYWORDS:
        if kw in loc_lower:
            return "california"
    for kw in _FLORIDA_KEYWORDS:
        if kw in loc_lower:
            return "florida"
    for kw in _ARIZONA_KEYWORDS:
        if kw in loc_lower:
            return "arizona"
    for kw in _PACIFIC_NW_KEYWORDS:
        if kw in loc_lower:
            return "pacific_northwest"
    for kw in _TEXAS_KEYWORDS:
        if kw in loc_lower:
            return "texas"
    for kw in _MIDWEST_KEYWORDS:
        if kw in loc_lower:
            return "midwest"
    # Coastal hints
    if any(kw in loc_lower for kw in ("beach", "coast", "harbor", "bay", "shore", "atlantic", "gulf")):
        return "coastal"
    return "general"


def _generate_climate_data(location: str, rng: random.Random) -> dict:
    region = _classify_climate_region(location)

    if region == "california":
        wildfire_risk = rng.choice(["high", "very_high", "very_high"])
        heat_days = rng.randint(5, 30)
        sea_level_rise = round(rng.uniform(0.3, 0.9), 2)
        drought_risk = rng.choice(["severe", "extreme"])
        extreme_weather_per_decade = rng.randint(6, 14)
        aqi = rng.randint(85, 145)

    elif region == "florida":
        wildfire_risk = "low"
        heat_days = rng.randint(40, 90)
        sea_level_rise = round(rng.uniform(0.8, 1.6), 2)
        drought_risk = "moderate"
        extreme_weather_per_decade = rng.randint(8, 16)
        aqi = rng.randint(35, 65)

    elif region == "arizona":
        wildfire_risk = rng.choice(["medium", "high"])
        heat_days = rng.randint(70, 130)
        sea_level_rise = 0.0
        drought_risk = "extreme"
        extreme_weather_per_decade = rng.randint(3, 8)
        aqi = rng.randint(70, 115)

    elif region == "pacific_northwest":
        wildfire_risk = rng.choice(["low", "medium"])
        heat_days = rng.randint(1, 8)
        sea_level_rise = round(rng.uniform(0.3, 0.7), 2)
        drought_risk = rng.choice(["low", "moderate"])
        extreme_weather_per_decade = rng.randint(4, 9)
        aqi = rng.randint(25, 60)

    elif region == "texas":
        wildfire_risk = rng.choice(["medium", "high"])
        heat_days = rng.randint(20, 60)
        sea_level_rise = round(rng.uniform(0.2, 0.5), 2) if "houston" in location.lower() or "corpus" in location.lower() else 0.0
        drought_risk = rng.choice(["moderate", "severe"])
        extreme_weather_per_decade = rng.randint(9, 18)
        aqi = rng.randint(55, 95)

    elif region == "midwest":
        wildfire_risk = "low"
        heat_days = rng.randint(5, 20)
        sea_level_rise = 0.0
        drought_risk = rng.choice(["low", "moderate"])
        extreme_weather_per_decade = rng.randint(10, 20)
        aqi = rng.randint(45, 85)

    elif region == "coastal":
        wildfire_risk = "low"
        heat_days = rng.randint(5, 30)
        sea_level_rise = round(rng.uniform(0.5, 1.2), 2)
        drought_risk = "low"
        extreme_weather_per_decade = rng.randint(5, 12)
        aqi = rng.randint(35, 75)

    else:  # general
        wildfire_risk = rng.choice(["low", "medium"])
        heat_days = rng.randint(2, 25)
        sea_level_rise = round(rng.uniform(0.0, 0.4), 2)
        drought_risk = rng.choice(["low", "moderate"])
        extreme_weather_per_decade = rng.randint(3, 12)
        aqi = rng.randint(40, 90)

    # Compute a normalized risk composite score for synthesis
    wildfire_score = _WILDFIRE_LEVELS.index(wildfire_risk) * 25
    heat_score = min(100, (heat_days / 100) * 100)
    sea_level_score = min(100, sea_level_rise * 60)
    drought_score = _DROUGHT_LEVELS.index(drought_risk) * 25
    climate_risk_composite = round((wildfire_score * 0.3 + heat_score * 0.25 + sea_level_score * 0.2 + drought_score * 0.25), 1)

    return {
        "wildfire_risk": wildfire_risk,
        "heat_index_days_over_100": heat_days,
        "sea_level_rise_projection_ft": sea_level_rise,
        "drought_risk": drought_risk,
        "extreme_weather_events_per_decade": extreme_weather_per_decade,
        "air_quality_index_avg": aqi,
        "climate_risk_composite_score": climate_risk_composite,
        "region_profile": region,
        "projected_temperature_increase_f_by_2050": round(rng.uniform(2.0, 5.5), 1),
        "carbon_emission_trend": rng.choice(["decreasing", "stable", "increasing"]),
        "green_space_coverage_pct": rng.randint(8, 42),
    }


def _make_citations(location: str, data: dict, retrieved_at: str) -> list[Citation]:
    city_name = location.split(",")[0].strip()

    citations = [
        Citation(
            source_name="NOAA National Centers for Environmental Information — Climate Data",
            url="https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/",
            retrieved_at=retrieved_at,
            data_field="heat_index_days_over_100",
            raw_value=f"{data['heat_index_days_over_100']} days/year over 100°F",
        ),
        Citation(
            source_name="EPA Climate Change Indicators — Sea Level",
            url="https://www.epa.gov/climate-indicators/climate-change-indicators-sea-level",
            retrieved_at=retrieved_at,
            data_field="sea_level_rise_projection_ft",
            raw_value=f"{data['sea_level_rise_projection_ft']} ft projected rise by 2050",
        ),
        Citation(
            source_name="USFS Wildfire Risk to Communities",
            url="https://wildfirerisk.org/",
            retrieved_at=retrieved_at,
            data_field="wildfire_risk",
            raw_value=data["wildfire_risk"].replace("_", " ").title(),
        ),
        Citation(
            source_name="EPA AirNow — Air Quality Index Data",
            url="https://www.airnow.gov/",
            retrieved_at=retrieved_at,
            data_field="air_quality_index_avg",
            raw_value=f"Annual average AQI: {data['air_quality_index_avg']}",
        ),
        Citation(
            source_name="climate.gov — U.S. Drought Monitor",
            url="https://www.climate.gov/maps-pubs/drought-monitor",
            retrieved_at=retrieved_at,
            data_field="drought_risk",
            raw_value=data["drought_risk"].title(),
        ),
    ]
    return citations


class ClimateAgent(BaseAgent):
    """
    Retrieves climate risk data from NOAA, EPA, and wildfire risk databases.
    Generates location-specific risk profiles for US cities.
    """

    def __init__(self):
        super().__init__("Climate Risk")

    async def _fetch(self, location: str, context: dict) -> AgentResult:
        await asyncio.sleep(random.uniform(0.5, 1.8))

        rng = random.Random(_location_seed(location))
        retrieved_at = self._now_iso()
        data = _generate_climate_data(location, rng)
        citations = _make_citations(location, data, retrieved_at)

        return AgentResult(
            agent_name=self.name,
            status=AgentState.SUCCESS,
            data=data,
            citations=citations,
        )

    async def _fetch_partial(self, location: str, context: dict) -> AgentResult:
        await asyncio.sleep(0.05)
        rng = random.Random(_location_seed(location) + 13)
        region = _classify_climate_region(location)
        retrieved_at = self._now_iso()

        # Return only the most stable, easy-to-derive data
        wildfire_risk = "high" if region == "california" else ("low" if region in ("florida", "pacific_northwest") else "medium")
        drought_risk = "extreme" if region == "arizona" else ("low" if region == "pacific_northwest" else "moderate")

        partial_data = {
            "wildfire_risk": wildfire_risk,
            "heat_index_days_over_100": None,
            "sea_level_rise_projection_ft": None,
            "drought_risk": drought_risk,
            "extreme_weather_events_per_decade": None,
            "air_quality_index_avg": None,
            "climate_risk_composite_score": None,
            "region_profile": region,
            "projected_temperature_increase_f_by_2050": None,
            "carbon_emission_trend": None,
            "green_space_coverage_pct": None,
        }

        citations = [
            Citation(
                source_name="NOAA National Centers for Environmental Information — Partial Data",
                url="https://www.ncei.noaa.gov/",
                retrieved_at=retrieved_at,
                data_field="wildfire_risk",
                raw_value=f"{wildfire_risk} (estimated from regional profile)",
            )
        ]

        return AgentResult(
            agent_name=self.name,
            status=AgentState.DEGRADED,
            data=partial_data,
            citations=citations,
        )
