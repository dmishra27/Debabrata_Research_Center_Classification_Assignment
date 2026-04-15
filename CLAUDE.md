# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Unsupervised ML pipeline that classifies UK research centres into **Premium**, **Standard**, or **Basic** quality tiers using K-Means clustering (k=3). The notebook trains and serializes the model; the FastAPI app loads that artifact and serves predictions.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Generate model_bundle.pkl (required before starting the API)
jupyter notebook                # open EDA_and_Model.ipynb and run all cells

# Start API (local)
python -m uvicorn app:app --reload

# Start API (Docker)
docker-compose up --build
```

API runs on `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

There is no automated test suite. Smoke-test the API with:
```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"internalFacilitiesCount":9,"hospitals_10km":3,"pharmacies_10km":2,"facilityDiversity_10km":0.82,"facilityDensity_10km":0.45}'
```

## Architecture

**Training → Serving pipeline:**

1. `EDA_and_Model.ipynb` (primary notebook, required filename per assignment) — performs EDA, trains StandardScaler + KMeans, and saves `model_bundle.pkl` via joblib. Also writes `research_centers_clustered.csv`.
2. `app.py` — FastAPI service. Loads `model_bundle.pkl` at import time (fails with `RuntimeError` if missing). Exposes `GET /` (health) and `POST /predict` (inference).
3. `model_bundle.pkl` — dict containing `scaler`, `kmeans`, `tier_map` (cluster int → tier label), and `model_features` (ordered list).

`Debabrata_Mishra_Research_Center_Quality_Classification.ipynb` is an identical duplicate of `EDA_and_Model.ipynb`. Keep both in sync if notebook content changes.

## Critical Constraints

- **Feature contract**: inference in `app.py` assembles the feature vector in a fixed order matching training: `internalFacilitiesCount`, `hospitals_10km`, `pharmacies_10km`, `facilityDiversity_10km`, `facilityDensity_10km`. Changing this order breaks predictions silently.
- **Generated artifacts**: `model_bundle.pkl` and `research_centers_clustered.csv` are outputs of the notebook — never edit manually, regenerate from the notebook when needed.
- **model_bundle.pkl must exist** before `app.py` can start. A missing or stale bundle will cause runtime errors or stale predictions.
- Docker and local dev both bind port 8000 — check for conflicts if the service appears broken.
- Interactive Plotly charts in the notebooks require JavaScript; GitHub's notebook preview does not render them. Use `nbviewer` (link in README) or run locally.
