# NeighborhoodPulse

Concurrent multi-agent platform that gathers FEMA flood data, zoning codes, and climate risk in parallel — synthesizes weighted composite risk scores (0–100) with cited sources and actionable recommendations.

**Live:** [neighborhood-pulse.onrender.com](https://neighborhood-pulse.onrender.com)

---

## What it does

Enter any US city, neighborhood, or address. NeighborhoodPulse spins up 4 specialized AI agents concurrently, each pulling from a different data source, and synthesizes their findings into a single risk report with a 0–100 composite score.

| Agent | Data Source | What it finds |
|-------|------------|---------------|
| **FEMA Agent** | NFIP flood zone data | Flood zone designation, historical claims, disaster declarations |
| **Zoning Agent** | Municipal zoning records | Land use type, density, permitted uses, overlay districts |
| **Climate Agent** | NOAA / climate APIs | Heat risk, drought index, extreme weather frequency, AQI |
| **Synthesis Agent** | All agent outputs | Weighted composite score, risk breakdown, citations, recommendations |

---

## Architecture

```
POST /api/analyze
        │
        ▼
  AgentOrchestrator
  ┌─────┬──────┬────────┐
  │FEMA │Zoning│Climate │   ← run concurrently via asyncio.gather
  └──┬──┴───┬──┴────┬───┘
     └──────┴───────┘
              │
       SynthesisAgent
              │
     Risk Score 0–100
     + Breakdown by category
     + Citation chips
     + Recommendations
```

Agents run in parallel using `asyncio.gather`. Results are cached in SQLite and streamed back via SSE for real-time progress updates. Each result includes source citations so every data point is verifiable.

---

## Risk Score Breakdown

The composite score weights four dimensions:

| Dimension | Weight | Source |
|-----------|--------|--------|
| Flood Risk | 35% | FEMA NFIP |
| Climate Risk | 30% | NOAA / climate data |
| Zoning Risk | 20% | Municipal records |
| Infrastructure | 15% | Synthesized |

Score ranges: **0–25** Low · **26–50** Moderate · **51–75** High · **76–100** Critical

---

## Tech Stack

- **Backend:** Python · FastAPI · asyncio · SQLAlchemy · SQLite
- **Agents:** Custom agent framework with `base_agent.py` + `AgentOrchestrator`
- **External APIs:** FEMA NFIP · NOAA · httpx for concurrent fetching
- **Frontend:** Vanilla JS · CSS Grid · SVG score ring animation
- **Infra:** Docker · Render

---

## Running locally

```bash
git clone https://github.com/Gunakarthik1/neighborhood-pulse
cd neighborhood-pulse
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`

---

## API

```
POST /api/analyze
Body: { "location": "Austin, TX" }

Response:
{
  "composite_score": 34,
  "risk_level": "Moderate",
  "breakdown": {
    "flood": { "score": 28, "zone": "Zone X", "source": "FEMA NFIP" },
    "climate": { "score": 41, "heat_days": 89, "source": "NOAA" },
    "zoning": { "score": 22, "type": "Mixed Use", "source": "Municipal" }
  },
  "citations": [...],
  "recommendations": [...]
}

GET /api/health
GET /api/reports          ← list past analyses
GET /api/reports/{id}     ← retrieve cached report
```
