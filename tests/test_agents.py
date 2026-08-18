"""
Tests for individual agents — schema correctness, graceful degradation,
and citation structure.
"""

import asyncio
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import AgentResult, AgentState, Citation
from agents.fema_agent import FEMAAgent, _classify_risk
from agents.zoning_agent import ZoningAgent, _classify_urban_type
from agents.climate_agent import ClimateAgent, _classify_climate_region
from agents.synthesis_agent import SynthesisAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fema_agent():
    return FEMAAgent()


@pytest.fixture
def zoning_agent():
    return ZoningAgent()


@pytest.fixture
def climate_agent():
    return ClimateAgent()


@pytest.fixture
def synthesis_agent():
    return SynthesisAgent()


# ── FEMA Agent Tests ──────────────────────────────────────────────────────────

class TestFEMAAgent:

    @pytest.mark.asyncio
    async def test_returns_agent_result(self, fema_agent):
        result = await fema_agent.execute("Austin, TX")
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_result_schema_fields(self, fema_agent):
        result = await fema_agent.execute("Denver, CO")
        assert result.agent_name == "FEMA Data"
        assert result.status in (AgentState.SUCCESS, AgentState.DEGRADED)
        assert isinstance(result.data, dict)
        assert isinstance(result.citations, list)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_flood_data_keys_present(self, fema_agent):
        result = await fema_agent.execute("Seattle, WA")
        required_keys = [
            "flood_zone", "flood_insurance_required",
            "special_flood_hazard_area", "disaster_declarations", "risk_level",
        ]
        for key in required_keys:
            assert key in result.data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_coastal_city_gets_higher_risk(self, fema_agent):
        miami_result = await fema_agent.execute("Miami, FL")
        denver_result = await fema_agent.execute("Denver, CO")
        # Both may be SUCCESS or DEGRADED; check risk_level when data is present
        if miami_result.status == AgentState.SUCCESS and denver_result.status == AgentState.SUCCESS:
            assert miami_result.data.get("risk_level") == "high"
            assert denver_result.data.get("risk_level") == "low"

    @pytest.mark.asyncio
    async def test_citations_have_required_fields(self, fema_agent):
        result = await fema_agent.execute("Portland, OR")
        for citation in result.citations:
            if isinstance(citation, Citation):
                assert citation.source_name
                assert citation.url.startswith("http")
                assert citation.retrieved_at
                assert citation.data_field
                assert citation.raw_value is not None
            else:
                assert citation["source_name"]
                assert citation["url"].startswith("http")

    @pytest.mark.asyncio
    async def test_degraded_returns_partial_data(self, fema_agent):
        """Force degraded by monkeypatching _fetch to fail."""
        async def always_fail(location, context):
            raise ConnectionError("Forced failure for test")

        fema_agent._fetch = always_fail
        result = await fema_agent.execute("Miami, FL")
        assert result.status in (AgentState.DEGRADED, AgentState.FAILED)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_degraded_still_returns_agent_result(self, fema_agent):
        async def always_fail(location, context):
            raise RuntimeError("Simulated timeout")

        fema_agent._fetch = always_fail
        result = await fema_agent.execute("Houston, TX")
        # Must return AgentResult, not raise
        assert isinstance(result, AgentResult)
        assert result.agent_name == "FEMA Data"

    def test_coastal_risk_classification(self):
        assert _classify_risk("Miami, FL") == "high"
        assert _classify_risk("Houston, TX") == "high"
        assert _classify_risk("Denver, CO") == "low"
        assert _classify_risk("Indianapolis, IN") == "moderate"

    @pytest.mark.asyncio
    async def test_disaster_declarations_are_list(self, fema_agent):
        result = await fema_agent.execute("New Orleans, LA")
        if result.status == AgentState.SUCCESS:
            assert isinstance(result.data["disaster_declarations"], list)
            for decl in result.data["disaster_declarations"]:
                assert "year" in decl
                assert "disaster_type" in decl
                assert "declaration_number" in decl


# ── Zoning Agent Tests ────────────────────────────────────────────────────────

class TestZoningAgent:

    @pytest.mark.asyncio
    async def test_returns_agent_result(self, zoning_agent):
        result = await zoning_agent.execute("Chicago, IL")
        assert isinstance(result, AgentResult)
        assert result.agent_name == "Zoning Records"

    @pytest.mark.asyncio
    async def test_schema_keys_present(self, zoning_agent):
        result = await zoning_agent.execute("Austin, TX")
        required = [
            "zoning_code", "zoning_description", "permitted_uses",
            "max_building_height_ft", "floor_area_ratio", "setback_requirements",
            "last_updated",
        ]
        for key in required:
            assert key in result.data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_urban_city_gets_higher_far(self, zoning_agent):
        nyc_result = await zoning_agent.execute("New York, NY")
        rural_result = await zoning_agent.execute("Rural Junction, WY")
        if (nyc_result.status == AgentState.SUCCESS
                and rural_result.status == AgentState.SUCCESS
                and nyc_result.data.get("floor_area_ratio")
                and rural_result.data.get("floor_area_ratio")):
            assert nyc_result.data["floor_area_ratio"] > rural_result.data["floor_area_ratio"]

    @pytest.mark.asyncio
    async def test_permitted_uses_is_list(self, zoning_agent):
        result = await zoning_agent.execute("Phoenix, AZ")
        if result.status == AgentState.SUCCESS:
            assert isinstance(result.data["permitted_uses"], list)

    @pytest.mark.asyncio
    async def test_setback_requirements_has_front_rear_side(self, zoning_agent):
        result = await zoning_agent.execute("Boston, MA")
        if result.status == AgentState.SUCCESS and result.data.get("setback_requirements"):
            setbacks = result.data["setback_requirements"]
            assert "front_ft" in setbacks
            assert "rear_ft" in setbacks
            assert "side_ft" in setbacks

    @pytest.mark.asyncio
    async def test_degraded_returns_partial(self, zoning_agent):
        async def always_fail(location, context):
            raise ConnectionError("Forced failure")

        zoning_agent._fetch = always_fail
        result = await zoning_agent.execute("Seattle, WA")
        assert isinstance(result, AgentResult)
        assert result.status in (AgentState.DEGRADED, AgentState.FAILED)

    def test_urban_classification(self):
        assert _classify_urban_type("New York, NY") == "urban"
        assert _classify_urban_type("Chicago, IL") == "urban"
        assert _classify_urban_type("Rural Springs, WY") == "rural"

    @pytest.mark.asyncio
    async def test_citations_cite_planning_sources(self, zoning_agent):
        result = await zoning_agent.execute("Denver, CO")
        # At least one citation should reference a planning or zoning source
        if result.citations:
            cit = result.citations[0]
            source = cit.source_name if isinstance(cit, Citation) else cit["source_name"]
            assert any(kw in source.lower() for kw in ["planning", "zoning", "municipal", "ordinance", "american"])


# ── Climate Agent Tests ───────────────────────────────────────────────────────

class TestClimateAgent:

    @pytest.mark.asyncio
    async def test_returns_agent_result(self, climate_agent):
        result = await climate_agent.execute("Phoenix, AZ")
        assert isinstance(result, AgentResult)
        assert result.agent_name == "Climate Risk"

    @pytest.mark.asyncio
    async def test_schema_keys_present(self, climate_agent):
        result = await climate_agent.execute("Miami, FL")
        required = [
            "wildfire_risk", "heat_index_days_over_100", "sea_level_rise_projection_ft",
            "drought_risk", "extreme_weather_events_per_decade", "air_quality_index_avg",
        ]
        for key in required:
            assert key in result.data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_california_wildfire_risk_elevated(self, climate_agent):
        result = await climate_agent.execute("Los Angeles, CA")
        if result.status == AgentState.SUCCESS:
            assert result.data["wildfire_risk"] in ("high", "very_high")

    @pytest.mark.asyncio
    async def test_florida_sea_level_rise(self, climate_agent):
        result = await climate_agent.execute("Miami, FL")
        if result.status == AgentState.SUCCESS:
            slr = result.data.get("sea_level_rise_projection_ft", 0)
            assert slr >= 0.5, f"Expected high sea level rise for Miami, got {slr}"

    @pytest.mark.asyncio
    async def test_phoenix_heat_days_high(self, climate_agent):
        result = await climate_agent.execute("Phoenix, AZ")
        if result.status == AgentState.SUCCESS:
            heat_days = result.data.get("heat_index_days_over_100", 0)
            assert heat_days >= 50, f"Expected high heat days for Phoenix, got {heat_days}"

    @pytest.mark.asyncio
    async def test_pacific_nw_lower_wildfire(self, climate_agent):
        result = await climate_agent.execute("Seattle, WA")
        if result.status == AgentState.SUCCESS:
            assert result.data["wildfire_risk"] in ("low", "medium")

    @pytest.mark.asyncio
    async def test_wildfire_risk_valid_values(self, climate_agent):
        result = await climate_agent.execute("Sacramento, CA")
        valid = ("low", "medium", "high", "very_high")
        if result.status == AgentState.SUCCESS:
            assert result.data["wildfire_risk"] in valid

    @pytest.mark.asyncio
    async def test_citations_reference_noaa_epa(self, climate_agent):
        result = await climate_agent.execute("Houston, TX")
        if result.citations:
            sources = [
                (c.source_name if isinstance(c, Citation) else c["source_name"]).lower()
                for c in result.citations
            ]
            assert any("noaa" in s or "epa" in s or "climate" in s for s in sources)

    @pytest.mark.asyncio
    async def test_aqi_is_positive_integer(self, climate_agent):
        result = await climate_agent.execute("Denver, CO")
        if result.status == AgentState.SUCCESS:
            aqi = result.data.get("air_quality_index_avg")
            assert isinstance(aqi, int)
            assert aqi > 0

    def test_region_classification(self):
        assert _classify_climate_region("Los Angeles, CA") == "california"
        assert _classify_climate_region("Miami, FL") == "florida"
        assert _classify_climate_region("Phoenix, AZ") == "arizona"
        assert _classify_climate_region("Seattle, WA") == "pacific_northwest"


# ── Synthesis Agent Tests ─────────────────────────────────────────────────────

class TestSynthesisAgent:

    async def _get_source_results(self):
        fema = await FEMAAgent().execute("Austin, TX")
        zoning = await ZoningAgent().execute("Austin, TX")
        climate = await ClimateAgent().execute("Austin, TX")
        return [fema, zoning, climate]

    @pytest.mark.asyncio
    async def test_returns_agent_result(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        assert isinstance(result, AgentResult)
        assert result.agent_name == "Report Synthesis"

    @pytest.mark.asyncio
    async def test_report_dict_has_required_keys(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        assert result.status == AgentState.SUCCESS
        required = [
            "location", "generated_at", "executive_summary", "risk_score",
            "risk_breakdown", "flood_section", "zoning_section", "climate_section",
            "recommendations", "data_gaps", "all_citations",
        ]
        for key in required:
            assert key in result.data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_risk_score_in_valid_range(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        if result.status == AgentState.SUCCESS:
            score = result.data["risk_score"]
            assert 0 <= score <= 100, f"Risk score {score} out of range"

    @pytest.mark.asyncio
    async def test_recommendations_is_list(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        if result.status == AgentState.SUCCESS:
            recs = result.data["recommendations"]
            assert isinstance(recs, list)
            assert len(recs) >= 1

    @pytest.mark.asyncio
    async def test_executive_summary_is_nonempty_string(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Miami, FL", context={"agent_results": source_results}
        )
        if result.status == AgentState.SUCCESS:
            summary = result.data["executive_summary"]
            assert isinstance(summary, str)
            assert len(summary) > 50

    @pytest.mark.asyncio
    async def test_data_gaps_reflects_degraded_agents(self, synthesis_agent):
        fema = await FEMAAgent().execute("Austin, TX")
        # Force zoning to degraded
        async def always_fail(location, context):
            raise RuntimeError("Forced fail")
        zoning = ZoningAgent()
        zoning._fetch = always_fail
        zoning_result = await zoning.execute("Austin, TX")
        climate = await ClimateAgent().execute("Austin, TX")

        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": [fema, zoning_result, climate]}
        )
        if result.status in (AgentState.SUCCESS, AgentState.DEGRADED):
            data_gaps = result.data.get("data_gaps", [])
            # If zoning degraded, its name should appear in data_gaps
            if zoning_result.status in (AgentState.DEGRADED, AgentState.FAILED):
                assert "Zoning Records" in data_gaps

    @pytest.mark.asyncio
    async def test_all_citations_aggregated(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        if result.status == AgentState.SUCCESS:
            cits = result.data["all_citations"]
            assert isinstance(cits, list)
            assert len(cits) >= 3  # At minimum one per source agent

    @pytest.mark.asyncio
    async def test_risk_breakdown_weights_sum_to_one(self, synthesis_agent):
        source_results = await self._get_source_results()
        result = await synthesis_agent.execute(
            "Austin, TX", context={"agent_results": source_results}
        )
        if result.status == AgentState.SUCCESS:
            breakdown = result.data["risk_breakdown"]
            total_weight = sum(v["weight"] for v in breakdown.values())
            assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected ~1.0"
