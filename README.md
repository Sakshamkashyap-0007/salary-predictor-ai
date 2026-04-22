# SalaryAI — India Tech Salary Predictor

Production-grade AI salary predictor for the Indian tech market.
FastAPI backend · scikit-learn RandomForest · Vanilla HTML/CSS/JS frontend.

---

## Stack

| Layer    | Tech                                    |
|----------|-----------------------------------------|
| ML       | scikit-learn RandomForestRegressor      |
| Backend  | FastAPI + Uvicorn                       |
| Frontend | HTML5 · CSS3 · Vanilla JS               |
| Data     | Synthetic India-market dataset (800 rows)|
| Runner   | `run.py` — single-command launcher      |

---

## Project Structure

```
salary-predictor/
├── run.py              ← Single-command launcher (start here)
├── api.py              ← FastAPI backend
├── train.py            ← Model training
├── generate_data.py    ← Synthetic dataset generator
├── requirements.txt
├── data/
│   └── salary_data.csv
├── models/
│   └── salary_model.pkl
└── frontend/
    ├── index.html
    ├── styles.css
    └── script.js
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run everything with ONE command
```bash
python run.py
```

This will:
- Generate dataset if missing
- Train model if missing
- Start FastAPI on `http://127.0.0.1:8000`
- Serve frontend on `http://127.0.0.1:5500`
- Open your browser automatically

### 3. Manual steps (if preferred)
```bash
python generate_data.py        # Generate dataset
python train.py                # Train & save model
python run.py   # Start API
# Open frontend/index.html in browser
```

---

## API

### POST `/predict`
```json
{
  "experience_years": 5.5,
  "job_role": "Senior",
  "skills": "Python, Machine Learning, AWS",
  "location": "Bangalore"
}
```

Response:
```json
{
  "predicted_salary": 2100000.0,
  "justification": "Predicted ₹21,00,000/yr (21.00 LPA) ..."
}
```

### GET `/health`
Returns `{"status": "ok"}`.

---

## Features

- **Location-aware predictions** — Bangalore, Delhi, Gurgaon, Noida, Mumbai, Hyderabad, Chennai, Pune, Remote all produce different salaries
- **Experience granularity** — Enter years + months
- **Skill tagging** — Multi-skill chip input with 20+ suggestions
- **Annual + monthly** output
- **Live API status** indicator
- **Single-command startup**

---

## Notes

- Trained on **synthetic** data calibrated to the Indian market. For learning/demo only.
- Salary range: ₹3L – ₹1.25Cr/yr depending on profile.
- Model: RandomForestRegressor, R² ≈ 0.88.