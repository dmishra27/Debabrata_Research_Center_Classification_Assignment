"""
Research Centre Quality Classification — FastAPI Endpoint
POST /predict  →  {"predictedCategory": "Premium" | "Standard" | "Basic"}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os

# ── Load model bundle ────────────────────────────────────────────────────────
BUNDLE_PATH = os.getenv("MODEL_PATH", "model_bundle.pkl")

try:
    bundle         = joblib.load(BUNDLE_PATH)
    scaler         = bundle["scaler"]
    kmeans         = bundle["kmeans"]
    tier_map       = bundle["tier_map"]
    MODEL_FEATURES = bundle["model_features"]
except FileNotFoundError:
    raise RuntimeError(
        f"Model bundle not found at '{BUNDLE_PATH}'. "
        "Run EDA_and_Model.ipynb first to generate model_bundle.pkl."
    )

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Research Centre Quality Classifier",
    description=(
        "Classifies a UK research centre into **Premium**, **Standard**, or **Basic** "
        "using a K-Means clustering model trained on internal facility count and nearby "
        "healthcare access features."
    ),
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class CentreFeatures(BaseModel):
    """Input features for a single research centre."""
    internalFacilitiesCount: int   = Field(..., ge=0, le=50,  example=9,
                                           description="Number of internal facilities")
    hospitals_10km:          int   = Field(..., ge=0, le=20,  example=3,
                                           description="Hospitals within 10 km")
    pharmacies_10km:         int   = Field(..., ge=0, le=20,  example=2,
                                           description="Pharmacies within 10 km")
    facilityDiversity_10km:  float = Field(..., ge=0.0, le=1.0, example=0.82,
                                           description="Facility diversity index (0–1)")
    facilityDensity_10km:    float = Field(..., ge=0.0,          example=0.45,
                                           description="Approximate facility density")


class PredictionResponse(BaseModel):
    """Minimal response matching assignment specification."""
    predictedCategory: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    """Returns API status and the feature list expected by /predict."""
    return {
        "status":   "ok",
        "model":    "K-Means (k=3)",
        "features": MODEL_FEATURES,
        "tiers":    list(tier_map.values()),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a research centre",
    responses={
        200: {
            "description": "Quality tier prediction",
            "content": {
                "application/json": {
                    "example": {"predictedCategory": "Premium"}
                }
            },
        }
    },
)
def predict(centre: CentreFeatures) -> PredictionResponse:
    """
    Accept a research centre's five quality features and return the predicted
    quality tier: **Premium**, **Standard**, or **Basic**.

    Steps performed internally:
    1. Assemble feature vector in training order.
    2. Apply the saved StandardScaler.
    3. Run KMeans.predict() on the scaled vector.
    4. Map the cluster integer to a human-readable tier label.
    """
    try:
        feature_vector = np.array([[
            centre.internalFacilitiesCount,
            centre.hospitals_10km,
            centre.pharmacies_10km,
            centre.facilityDiversity_10km,
            centre.facilityDensity_10km,
        ]])                                        # shape (1, 5)

        X_scaled   = scaler.transform(feature_vector)
        cluster_id = int(kmeans.predict(X_scaled)[0])
        tier       = tier_map[cluster_id]

        return PredictionResponse(predictedCategory=tier)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
