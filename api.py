"""
api.py — FastAPI backend for AI Salary Predictor.
Start: uvicorn api:app --host 127.0.0.1 --port 8000
"""

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "salary_model.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ─── LOCATION BOOST (NEW — FIXES YOUR CORE PROBLEM) ──────────────────────────
LOCATION_BOOST = {
    "Bangalore": 1.15,
    "Delhi": 1.10,
    "Mumbai": 1.12,
    "Hyderabad": 1.05,
    "Pune": 0.95,
    "Chennai": 0.93,
    "Gurgaon": 1.11,
    "Noida": 1.08,
    "Remote": 0.97,
}

# ─── Pydantic schemas ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    experience_years: float = Field(..., ge=0, le=60)
    job_role: str
    skills: str
    location: str


class PredictResponse(BaseModel):
    predicted_salary: float
    justification: str


# ─── Model helpers ──────────────────────────────────────────────────────────

def _load_bundle() -> Dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run: python train.py"
        )
    return joblib.load(MODEL_PATH)


_ROLE_MAP: Dict[str, str] = {
    "intern": "Junior",
    "junior": "Junior",
    "mid-level": "Mid",
    "mid": "Mid",
    "senior": "Senior",
    "lead": "Lead",
    "manager": "Manager",
    "director": "Director",
}


def _resolve_position(job_role: str, allowed: List[str]) -> str:
    mapped = _ROLE_MAP.get(job_role.strip().lower(), job_role.strip())
    return mapped if mapped in allowed else allowed[0]


# ❌ REMOVED FALLBACK LOGIC → STRICT MATCH
def _resolve_location(location: str, allowed: List[str]) -> str:
    loc = location.strip()
    if loc not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid location '{loc}'. Allowed: {allowed}"
        )
    return loc


def _parse_skills(skills_csv: str, allowed: List[str]) -> List[str]:
    allowed_set = set(allowed)
    seen = set()
    out: List[str] = []

    for part in skills_csv.split(","):
        s = part.strip()
        if s in allowed_set and s not in seen:
            out.append(s)
            seen.add(s)

    return out


def _build_features(
    bundle: Dict[str, Any],
    experience_years: float,
    position: str,
    location: str,
    skills: List[str],
) -> np.ndarray:

    x_years = bundle["scaler"].transform(
        pd.DataFrame({"years_experience": [experience_years]})
    )

    x_cat = bundle["ohe"].transform(
        pd.DataFrame({"position": [position], "location": [location]})
    )

    x_skills = bundle["mlb"].transform([skills])

    return np.hstack([x_years, x_cat, x_skills])


# ─── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="AI Salary Predictor API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    try:
        bundle = _load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    allowed_positions = bundle["ohe"].categories_[0].tolist()
    allowed_locations = bundle["ohe"].categories_[1].tolist()
    allowed_skills = bundle["mlb"].classes_.tolist()

    # ─── DEBUG (IMPORTANT) ───────────────────────────────────────────────────
    print("INPUT LOCATION:", req.location)

    position = _resolve_position(req.job_role, allowed_positions)
    location = _resolve_location(req.location, allowed_locations)
    skills = _parse_skills(req.skills, allowed_skills)

    print("USED LOCATION:", location)

    x = _build_features(
        bundle=bundle,
        experience_years=float(req.experience_years),
        position=position,
        location=location,
        skills=skills,
    )

    try:
        raw_pred = float(bundle["model"].predict(x)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # ─── APPLY LOCATION BOOST (CRITICAL FIX) ─────────────────────────────────
    base_salary = float(np.clip(raw_pred, 300_000, 12_500_000))
    boost = LOCATION_BOOST.get(location, 1.0)
    salary = base_salary * boost

    # ─── Justification ───────────────────────────────────────────────────────
    lpa = salary / 100_000

    skill_note = (
        f"{len(skills)} skill(s): {', '.join(skills[:4])}"
        if skills else "no recognized skills"
    )

    justification = (
        f"₹{salary:,.0f}/yr ({lpa:.2f} LPA) for a {position} in {location} "
        f"with {req.experience_years:.1f} yrs experience. "
        f"{skill_note}. "
        f"{location} market adjustment applied (~{int((boost-1)*100)}%). "
        "Model trained on India tech salary patterns."
    )

    return PredictResponse(
        predicted_salary=salary,
        justification=justification
    )
