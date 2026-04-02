# CoachIQ Risk Intelligence Engine

A multi-factor athlete injury risk prediction system built in Python 3 — zero dependencies beyond the standard library.

Input an athlete's training loads, RPE scores, sleep data, injury history, and physical profile. The system computes overall injury risk and per-region probability across 8 body regions, generates a risk-adjusted 7-day training plan, and exports full reports to JSON and CSV.

## How to run
```bash
python3 coachiq_risk.py
```

## What this demonstrates
- 9-factor weighted risk model based on Gabbett (2016) ACWR framework
- Training load metrics: ACWR, monotony, strain, load spike detection
- Per-region injury probability across 8 body regions (ankle, knee, hamstring, shoulder, lumbar, groin, elbow, calf)
- SQLite database for longitudinal athlete storage
- JSON and CSV report export with full audit trail
- Evidence-based weekly periodisation plan generator
- Real-time ASCII charts, sparklines, and heatmap in terminal

## References
Gabbett TJ (2016) · Foster C (1998) · Hägglund M (2006) · Milewski M (2014) · Croisier J (2008)

## Built by
Muhammad Ebadur Rahman Siddiqui  
OrthoAI Project · Pre-Medical Student · Karachi, Pakistan
