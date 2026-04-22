"""
generate_data.py — PRO VERSION (Improved signal strength + realism)
Run: python generate_data.py
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_rows: int = 5000   # 🔥 increased dataset size
    out_csv: str = os.path.join("data", "salary_data.csv")


POSITIONS: List[str] = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]

LOCATIONS: List[str] = [
    "Bangalore", "Delhi", "Mumbai", "Hyderabad", "Pune",
    "Chennai", "Gurgaon", "Noida", "Remote",
]

SKILLS: List[str] = [
    # Programming
    "Python","JavaScript","TypeScript","Java","C++","C","Go","Rust","Kotlin","Swift","Dart",

    # Frontend
    "HTML","CSS","React","Next.js","Vue.js","Angular","Tailwind CSS","Bootstrap","MLOps",
"NLP",
"Computer Vision",
"Spark",

    # Backend
    "Node.js","Express.js","FastAPI","Django","Spring Boot","REST API","GraphQL",

    # Databases
    "SQL","MySQL","PostgreSQL","MongoDB","Redis","Firebase","DynamoDB","SQLite",

    # DevOps
    "Docker","Kubernetes","CI/CD","GitHub Actions","AWS","Azure","GCP","Terraform","Nginx",

    # Data / ML
    "Machine Learning","Deep Learning","Data Science","Data Analysis","Statistics",
    "Pandas","NumPy","Scikit-learn","TensorFlow","PyTorch",

    # AI / GenAI
    "LLM","Generative AI","Prompt Engineering","LangChain","LangGraph","RAG",
    "Vector Databases","FAISS","OpenAI API","Groq API",

    # Data Engineering
    "ETL","Data Pipelines","Apache Spark","Hadoop","Kafka",

    # Analytics
    "Power BI","Tableau","Excel","Dashboarding",

    # Mobile
    "React Native","Flutter","Android Development","iOS Development",

    # Core
    "System Design","Microservices","API Design","Authentication","Testing","Agile"
]

BASE_BY_POSITION = {
    "Junior":   550_000,
    "Mid":    1_050_000,
    "Senior": 1_900_000,
    "Lead":   2_800_000,
    "Manager":3_300_000,
    "Director":5_000_000,
}

# 🔥 STRONGER LOCATION IMPACT (KEY FIX)
LOCATION_MULTIPLIER = {
    "Bangalore": 1.20,
    "Delhi":     1.15,
    "Gurgaon":   1.16,
    "Noida":     1.12,
    "Mumbai":    1.18,
    "Hyderabad": 1.05,
    "Chennai":   0.95,
    "Pune":      0.90,
    "Remote":    0.97,
}

SKILL_PREMIUM = {
    "Python": 40000,
    "Machine Learning": 90000,
    "Deep Learning": 90000,
    "SQL": 35000,
    "Data Engineering": 70000,
    "AWS": 70000,
    "Azure": 60000,
    "GCP": 60000,
    "Docker": 45000,
    "Kubernetes": 55000,
    "React": 35000,
    "Java": 40000,
    "Node.js": 30000,
    "System Design": 110000,
    "NLP": 60000,
    "Computer Vision": 60000,
    "Spark": 55000,
    "MLOps": 85000,
    "Tableau": 25000,
}

SKILL_PREMIUM.update({
    "LangChain": 80000,
    "LangGraph": 85000,
    "RAG": 90000,
    "LLM": 95000,
    "Generative AI": 90000,
    "Prompt Engineering": 70000,
    "Next.js": 40000,
    "React": 35000,
    "Node.js": 30000,
    "MongoDB": 30000,
    "PostgreSQL": 35000,
    "Firebase": 25000,
    "GraphQL": 40000,
    "CI/CD": 50000,
    "GitHub Actions": 40000,
    "Terraform": 60000,
    "Nginx": 30000,
    "Power BI": 30000,
    "Excel": 20000,
    "Kafka": 60000,
    "Hadoop": 55000
})

LOCATION_PROBS = [0.22, 0.16, 0.16, 0.12, 0.10, 0.07, 0.06, 0.06, 0.05]


def _sample_position(rng: np.random.Generator, years: int) -> str:
    if years <= 1:   p = [0.78, 0.18, 0.03, 0.01, 0.00, 0.00]
    elif years <= 4: p = [0.28, 0.52, 0.16, 0.03, 0.01, 0.00]
    elif years <= 7: p = [0.08, 0.33, 0.43, 0.12, 0.03, 0.01]
    elif years <= 11:p = [0.02, 0.16, 0.42, 0.26, 0.10, 0.04]
    elif years <= 15:p = [0.01, 0.09, 0.30, 0.30, 0.20, 0.10]
    else:            p = [0.00, 0.05, 0.20, 0.25, 0.25, 0.25]
    return rng.choice(POSITIONS, p=p)


def _sample_skills(rng: np.random.Generator, position: str) -> List[str]:
    lo_hi = {"Junior":(2,5),"Mid":(3,7),"Senior":(4,9),"Lead":(5,10),"Manager":(4,9),"Director":(4,8)}
    lo, hi = lo_hi[position]
    k = int(rng.integers(lo, hi + 1))

    weights = np.ones(len(SKILLS), dtype=float)

    boost = {
        "Senior":   ["System Design","AWS","Docker","Kubernetes","MLOps"],
        "Lead":     ["System Design","Kubernetes","MLOps","AWS","Spark"],
        "Manager":  ["System Design","MLOps","AWS"],
        "Director": ["System Design","MLOps","AWS","Azure","GCP"],
        "Junior":   ["Python","SQL"],
        "Mid":      ["Python","Machine Learning","SQL"],
    }

    for s in boost.get(position, []):
        weights[SKILLS.index(s)] *= 2.5

    weights /= weights.sum()
    chosen = rng.choice(SKILLS, size=k, replace=False, p=weights).tolist()

    if "Python" not in chosen and rng.random() < 0.7:
        chosen[rng.integers(0, len(chosen))] = "Python"

    if "SQL" not in chosen and rng.random() < 0.55:
        chosen[rng.integers(0, len(chosen))] = "SQL"

    return sorted(set(chosen))


def _market_salary(rng, years, position, skills, location) -> int:
    base = BASE_BY_POSITION[position]

    if years <= 6:   exp = years * 115_000
    elif years <= 12:exp = 6*115_000 + (years-6)*85_000
    else:            exp = 6*115_000 + 6*85_000 + (years-12)*55_000

    premium = sum(SKILL_PREMIUM.get(s, 0) for s in skills)

    # 🔥 reduce skill dominance slightly
    premium *= 0.75 + 0.25 * np.tanh(len(skills) / 6.0)

    loc_mul = LOCATION_MULTIPLIER[location]

    noise = rng.normal(0.0, 0.06)

    market = (base + exp + premium) * loc_mul * (1.0 + noise)

    return int(round(float(np.clip(market, 350_000, 11_000_000)) / 1000) * 1000)


def _final_salary(rng, market, position) -> int:
    sigma = {"Junior":0.07,"Mid":0.08,"Senior":0.09,"Lead":0.10,"Manager":0.11,"Director":0.12}[position]
    bump = rng.normal(0.02, sigma)
    return int(round(float(np.clip(market * (1 + bump), 300_000, 12_500_000)) / 1000) * 1000)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    rows = []

    for _ in range(cfg.n_rows):
        years    = int(rng.integers(0, 21))
        location = rng.choice(LOCATIONS, p=LOCATION_PROBS)
        position = _sample_position(rng, years)
        skills   = _sample_skills(rng, position)

        market   = _market_salary(rng, years, position, skills, location)
        final    = _final_salary(rng, market, position)

        rows.append(dict(
            years_experience=years,
            position=position,
            skills=skills,
            location=location,
            market_salary=market,
            final_salary=final
        ))

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    df.to_csv(cfg.out_csv, index=False)

    print(f"Generated {len(df)} rows → {cfg.out_csv}")
    print("\nSample:\n", df.head(5).to_string(index=False))

    # 🔥 DEBUG INSIGHT
    print("\nAverage salary by location:")
    print(df.groupby("location")["final_salary"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()