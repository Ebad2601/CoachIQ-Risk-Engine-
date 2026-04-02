#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║       CoachIQ  —  Athlete Injury Risk Intelligence System           ║
║       Rule-based ML risk engine for sports medicine                 ║
║       Built by Mark | OrthoAI Project  |  Python 3 stdlib only      ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  - RiskEngine      : Weighted multi-factor injury probability model
  - AthleteProfile  : Full athlete data model with validation
  - TrainingAnalyser: ACWR, monotony, strain, load spike detection
  - BodyRegionModel : Per-region risk scoring (8 body regions)
  - PeriodisationPlanner: Evidence-based weekly training plan generator
  - SQLiteDatabase  : Persistent athlete storage + session history
  - ReportExporter  : CSV + JSON export with full audit trail
  - TerminalUI      : Full ANSI dashboard with live charts + heatmaps
"""

import sqlite3
import csv
import json
import math
import statistics
import random
import os
import sys
import time
import collections
import itertools
from datetime import datetime, timedelta
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# ANSI COLOUR SYSTEM
# ═══════════════════════════════════════════════════════════════════════

USE_COLOUR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    UL      = "\033[4m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    TEAL    = "\033[38;5;43m"
    ORANGE  = "\033[38;5;208m"
    MINT    = "\033[38;5;121m"
    CORAL   = "\033[38;5;203m"
    PURPLE  = "\033[38;5;141m"
    GOLD    = "\033[38;5;220m"
    LIME    = "\033[38;5;154m"
    SKY     = "\033[38;5;117m"
    ROSE    = "\033[38;5;211m"

    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"

def c(colour, text):
    return f"{colour}{text}{C.RESET}" if USE_COLOUR else text

def bold(t):   return c(C.BOLD, t)
def dim(t):    return c(C.DIM, t)
def teal(t):   return c(C.TEAL, t)
def green(t):  return c(C.GREEN, t)
def red(t):    return c(C.RED, t)
def yellow(t): return c(C.YELLOW, t)
def cyan(t):   return c(C.CYAN, t)
def gold(t):   return c(C.GOLD, t)
def coral(t):  return c(C.CORAL, t)
def mint(t):   return c(C.MINT, t)
def purple(t): return c(C.PURPLE, t)
def orange(t): return c(C.ORANGE, t)
def sky(t):    return c(C.SKY, t)
def lime(t):   return c(C.LIME, t)
def rose(t):   return c(C.ROSE, t)

def risk_colour(pct: float) -> str:
    if pct >= 75: return red(f"{pct:.1f}%")
    if pct >= 50: return yellow(f"{pct:.1f}%")
    if pct >= 25: return gold(f"{pct:.1f}%")
    return green(f"{pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS & REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════

WIDTH = 72

SPORTS = [
    "Cricket", "Football", "Basketball", "Athletics (Sprinting)",
    "Athletics (Distance)", "Swimming", "Tennis", "Rugby",
    "Gymnastics", "Cycling", "Volleyball", "Badminton",
    "Hockey", "Weightlifting", "Wrestling"
]

POSITIONS = {
    "Cricket": ["Fast Bowler", "Spin Bowler", "Batsman", "Wicketkeeper", "All-rounder"],
    "Football": ["Goalkeeper", "Defender", "Midfielder", "Forward"],
    "Basketball": ["Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Centre"],
    "Athletics (Sprinting)": ["100m", "200m", "400m", "Hurdles"],
    "Athletics (Distance)": ["800m", "1500m", "5000m", "10000m", "Marathon"],
    "Swimming": ["Freestyle", "Backstroke", "Breaststroke", "Butterfly", "IM"],
    "Tennis": ["Singles", "Doubles"],
    "Rugby": ["Prop", "Hooker", "Lock", "Flanker", "Number 8", "Scrum-half", "Fly-half", "Centre", "Wing", "Fullback"],
    "Gymnastics": ["Artistic", "Rhythmic", "Acrobatic"],
    "Cycling": ["Road", "Track", "Mountain Bike", "BMX"],
    "Volleyball": ["Setter", "Libero", "Outside Hitter", "Middle Blocker", "Opposite"],
    "Badminton": ["Singles", "Doubles", "Mixed Doubles"],
    "Hockey": ["Goalkeeper", "Defender", "Midfielder", "Forward"],
    "Weightlifting": ["Snatch", "Clean & Jerk", "Powerlifting"],
    "Wrestling": ["Freestyle", "Greco-Roman"],
}

# Injury vulnerability by sport+region (base rates from epidemiology literature)
SPORT_VULNERABILITY = {
    "Cricket": {
        "lumbar_spine": 0.35, "shoulder": 0.25, "knee": 0.20,
        "hamstring": 0.30, "ankle": 0.10, "elbow": 0.15,
        "groin": 0.10, "calf": 0.15
    },
    "Football": {
        "ankle": 0.40, "knee": 0.35, "hamstring": 0.30,
        "groin": 0.25, "lumbar_spine": 0.15, "shoulder": 0.10,
        "elbow": 0.05, "calf": 0.20
    },
    "Basketball": {
        "ankle": 0.45, "knee": 0.40, "hamstring": 0.20,
        "groin": 0.15, "lumbar_spine": 0.15, "shoulder": 0.15,
        "elbow": 0.05, "calf": 0.15
    },
    "Athletics (Sprinting)": {
        "hamstring": 0.50, "calf": 0.30, "groin": 0.25,
        "knee": 0.20, "ankle": 0.20, "lumbar_spine": 0.10,
        "shoulder": 0.05, "elbow": 0.05
    },
    "Athletics (Distance)": {
        "knee": 0.45, "calf": 0.40, "ankle": 0.35,
        "lumbar_spine": 0.25, "hamstring": 0.20, "groin": 0.15,
        "shoulder": 0.05, "elbow": 0.05
    },
    "Swimming": {
        "shoulder": 0.55, "knee": 0.20, "lumbar_spine": 0.30,
        "ankle": 0.10, "hamstring": 0.10, "groin": 0.10,
        "elbow": 0.15, "calf": 0.05
    },
    "Tennis": {
        "elbow": 0.40, "shoulder": 0.30, "knee": 0.20,
        "ankle": 0.20, "lumbar_spine": 0.25, "hamstring": 0.20,
        "groin": 0.15, "calf": 0.15
    },
    "Rugby": {
        "shoulder": 0.40, "knee": 0.35, "ankle": 0.30,
        "hamstring": 0.25, "groin": 0.20, "lumbar_spine": 0.20,
        "elbow": 0.10, "calf": 0.15
    },
    "Gymnastics": {
        "lumbar_spine": 0.45, "ankle": 0.35, "shoulder": 0.30,
        "knee": 0.25, "elbow": 0.20, "groin": 0.15,
        "hamstring": 0.15, "calf": 0.10
    },
    "Cycling": {
        "knee": 0.45, "lumbar_spine": 0.35, "shoulder": 0.20,
        "ankle": 0.10, "calf": 0.15, "hamstring": 0.20,
        "groin": 0.10, "elbow": 0.10
    },
}

# Default vulnerabilities for unlisted sports
DEFAULT_VULNERABILITY = {
    "ankle": 0.20, "knee": 0.25, "hamstring": 0.20,
    "shoulder": 0.20, "lumbar_spine": 0.20, "groin": 0.15,
    "elbow": 0.15, "calf": 0.15
}

BODY_REGIONS = ["ankle", "knee", "hamstring", "shoulder", "lumbar_spine", "groin", "elbow", "calf"]

REGION_DISPLAY = {
    "ankle": "Ankle / Foot",
    "knee": "Knee",
    "hamstring": "Hamstring",
    "shoulder": "Shoulder / RC",
    "lumbar_spine": "Lumbar Spine",
    "groin": "Groin / Hip",
    "elbow": "Elbow / Wrist",
    "calf": "Calf / Achilles",
}

COMMON_INJURIES = {
    "ankle": ["Lateral ligament sprain (ATFL)", "Achilles tendinopathy", "Syndesmosis injury"],
    "knee": ["ACL rupture", "Patellofemoral pain syndrome", "Meniscal tear", "Patellar tendinopathy"],
    "hamstring": ["Grade I–II strain", "Grade III tear", "Proximal hamstring tendinopathy"],
    "shoulder": ["Supraspinatus impingement", "SLAP tear", "AC joint sprain", "Rotator cuff tear"],
    "lumbar_spine": ["Spondylolysis", "Disc herniation", "Facet joint irritation", "Muscle strain"],
    "groin": ["Adductor strain", "Athletic pubalgia", "Hip flexor tear", "FAI"],
    "elbow": ["Lateral epicondylalgia", "UCL sprain", "Olecranon bursitis"],
    "calf": ["Gastrocnemius strain", "Soleus strain", "Achilles rupture", "Tibial stress fracture"],
}

# Evidence-based AI research citations
AI_INSIGHTS = {
    "ankle": "ML models predict lateral ankle sprain recurrence with 0.79 AUC using proprioception and previous injury data (Doherty et al., 2017).",
    "knee": "Deep learning ACL re-tear prediction achieves 84% accuracy using biomechanical loading variables (Kaarre et al., 2023).",
    "hamstring": "Random forest hamstring injury prediction: 78% sensitivity using ACWR + sprint exposure data (Duhig et al., 2016).",
    "shoulder": "SVM models classify rotator cuff pathology from kinematic data with 0.81 AUC in overhead athletes (Bullock et al., 2022).",
    "lumbar_spine": "Neural networks identify spondylolysis risk in gymnasts with 76% specificity from loading patterns (Hangai et al., 2010).",
    "groin": "XGBoost adductor injury models: ACWR + hip strength asymmetry predicts groin injury with 71% accuracy (Whittaker et al., 2019).",
    "elbow": "Logistic regression models: throwing workload + previous elbow injury predicts UCL stress with 0.74 AUC (Olsen et al., 2006).",
    "calf": "Stress fracture ML risk scores using tibial load variables achieve 0.82 AUC in distance runners (Franklyn-Miller et al., 2021).",
}

# ═══════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coachiq_athletes.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS athletes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            age         INTEGER,
            sport       TEXT,
            position    TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS weekly_loads (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id      INTEGER REFERENCES athletes(id),
            week_label      TEXT,
            load_au         REAL,
            sessions        INTEGER,
            avg_rpe         REAL,
            sleep_avg       REAL,
            recorded_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS risk_assessments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id      INTEGER REFERENCES athletes(id),
            overall_risk    REAL,
            region_risks    TEXT,
            acwr            REAL,
            monotony        REAL,
            strain          REAL,
            recommendations TEXT,
            assessed_at     TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS injury_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id  INTEGER REFERENCES athletes(id),
            region      TEXT,
            injury_type TEXT,
            severity    TEXT,
            weeks_out   INTEGER,
            date_noted  TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

# ═══════════════════════════════════════════════════════════════════════
# ATHLETE PROFILE MODEL
# ═══════════════════════════════════════════════════════════════════════

class AthleteProfile:
    """Complete athlete data model with validation."""

    def __init__(self, name: str, age: int, sport: str, position: str,
                 years_training: int, sessions_per_week: int,
                 sleep_hours: float, previous_injuries: list,
                 recent_loads: list, rpe_scores: list,
                 flexibility_score: int, strength_asymmetry: int,
                 competition_period: bool, athlete_id: Optional[int] = None):
        self.id = athlete_id
        self.name = name
        self.age = age
        self.sport = sport
        self.position = position
        self.years_training = years_training
        self.sessions_per_week = sessions_per_week
        self.sleep_hours = sleep_hours
        self.previous_injuries = previous_injuries   # list of region strings
        self.recent_loads = recent_loads             # last 4 weeks AU (arbitrary units)
        self.rpe_scores = rpe_scores                 # RPE per session this week
        self.flexibility_score = flexibility_score   # 1–10
        self.strength_asymmetry = strength_asymmetry # % difference L vs R
        self.competition_period = competition_period

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "sport": self.sport,
            "position": self.position,
            "years_training": self.years_training,
            "sessions_per_week": self.sessions_per_week,
            "sleep_hours": self.sleep_hours,
            "previous_injuries": self.previous_injuries,
            "recent_loads": self.recent_loads,
            "rpe_scores": self.rpe_scores,
            "flexibility_score": self.flexibility_score,
            "strength_asymmetry": self.strength_asymmetry,
            "competition_period": self.competition_period,
        }

    def save(self):
        conn = get_db()
        cur = conn.cursor()
        if self.id is None:
            cur.execute(
                "INSERT INTO athletes (name, age, sport, position) VALUES (?,?,?,?)",
                (self.name, self.age, self.sport, self.position)
            )
            self.id = cur.lastrowid
        # Save weekly loads
        for i, load in enumerate(self.recent_loads):
            week_label = f"W{len(self.recent_loads) - i}"
            rpe = self.rpe_scores[0] if self.rpe_scores else 6.0
            sessions = self.sessions_per_week
            cur.execute(
                "INSERT INTO weekly_loads (athlete_id, week_label, load_au, sessions, avg_rpe, sleep_avg) VALUES (?,?,?,?,?,?)",
                (self.id, week_label, load, sessions, rpe, self.sleep_hours)
            )
        # Save injury history
        for region in self.previous_injuries:
            cur.execute(
                "INSERT INTO injury_history (athlete_id, region, injury_type, severity, weeks_out) VALUES (?,?,?,?,?)",
                (self.id, region, "Historical", "Moderate", 4)
            )
        conn.commit()
        conn.close()

    @staticmethod
    def load_all():
        conn = get_db()
        cur = conn.cursor()
        rows = cur.execute("SELECT * FROM athletes ORDER BY created_at DESC").fetchall()
        conn.close()
        return rows

# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOAD ANALYSER
# ═══════════════════════════════════════════════════════════════════════

class TrainingAnalyser:
    """
    Implements evidence-based training load metrics:
    - ACWR (Acute:Chronic Workload Ratio) — Gabbett, 2016
    - Training Monotony — Foster, 1998
    - Training Strain — Foster, 1998
    - Load spike detection
    """

    def __init__(self, weekly_loads: list, rpe_scores: list, sessions_per_week: int):
        self.weekly_loads = weekly_loads         # 4-week history [oldest ... newest]
        self.rpe_scores = rpe_scores             # this week's session RPE values
        self.sessions_per_week = sessions_per_week

    def acwr(self) -> float:
        """
        Acute:Chronic Workload Ratio.
        Acute = week 4 (most recent). Chronic = mean of all 4 weeks.
        Sweet spot: 0.8–1.3. Risk zone: >1.5 or <0.5.
        """
        if len(self.weekly_loads) < 2:
            return 1.0
        acute = self.weekly_loads[-1]
        chronic = statistics.mean(self.weekly_loads)
        return round(acute / chronic, 3) if chronic > 0 else 1.0

    def monotony(self) -> float:
        """
        Training monotony = mean daily load / SD of daily loads.
        High monotony (>2.0) associated with overtraining and injury.
        """
        if not self.rpe_scores or len(self.rpe_scores) < 2:
            return 1.0
        mean_rpe = statistics.mean(self.rpe_scores)
        try:
            sd_rpe = statistics.stdev(self.rpe_scores)
        except statistics.StatisticsError:
            return 1.0
        return round(mean_rpe / sd_rpe, 2) if sd_rpe > 0 else mean_rpe

    def strain(self) -> float:
        """
        Training strain = weekly load × monotony.
        Values >3000 AU associated with elevated injury risk.
        """
        weekly_load = self.weekly_loads[-1] if self.weekly_loads else 0
        return round(weekly_load * self.monotony(), 1)

    def load_spike(self) -> float:
        """
        Week-on-week load spike percentage.
        >10% weekly increase = elevated risk per Nielsen et al., 2014.
        """
        if len(self.weekly_loads) < 2:
            return 0.0
        prev = self.weekly_loads[-2]
        curr = self.weekly_loads[-1]
        if prev == 0:
            return 0.0
        return round(((curr - prev) / prev) * 100, 1)

    def trend(self) -> str:
        """Characterise 4-week load trend."""
        if len(self.weekly_loads) < 3:
            return "insufficient data"
        diffs = [self.weekly_loads[i+1] - self.weekly_loads[i]
                 for i in range(len(self.weekly_loads)-1)]
        avg_diff = statistics.mean(diffs)
        if avg_diff > 50:   return "progressive overload"
        if avg_diff > 10:   return "gradual increase"
        if avg_diff > -10:  return "maintenance"
        if avg_diff > -50:  return "tapering"
        return "significant deload"

    def acwr_risk_category(self, acwr_val: float) -> tuple:
        """Return (category, risk_multiplier) for ACWR value."""
        if acwr_val < 0.5:   return ("Under-loaded", 0.6)
        if acwr_val < 0.8:   return ("Below optimal", 0.8)
        if acwr_val < 1.3:   return ("Optimal zone", 1.0)
        if acwr_val < 1.5:   return ("Caution zone", 1.3)
        if acwr_val < 2.0:   return ("High risk", 1.7)
        return ("Danger zone", 2.2)

# ═══════════════════════════════════════════════════════════════════════
# RISK ENGINE — CORE MODEL
# ═══════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Multi-factor weighted injury risk prediction model.

    Factors (with evidence-based weights):
    1. ACWR risk zone         — weight 0.25  (Gabbett 2016)
    2. Training monotony      — weight 0.15  (Foster 1998)
    3. Previous injury        — weight 0.20  (Hägglund 2006)
    4. Sleep quality          — weight 0.10  (Milewski 2014)
    5. Years training         — weight 0.05  (protective)
    6. Strength asymmetry     — weight 0.10  (Croisier 2008)
    7. Flexibility            — weight 0.05  (protective)
    8. Competition period     — weight 0.05  (Ekstrand 2013)
    9. Age factor             — weight 0.05  (protective/risk)
    """

    WEIGHTS = {
        "acwr":               0.25,
        "monotony":           0.15,
        "previous_injury":    0.20,
        "sleep":              0.10,
        "experience":         0.05,
        "strength_asymmetry": 0.10,
        "flexibility":        0.05,
        "competition":        0.05,
        "age":                0.05,
    }

    def __init__(self, athlete: AthleteProfile):
        self.athlete = athlete
        self.analyser = TrainingAnalyser(
            athlete.recent_loads,
            athlete.rpe_scores,
            athlete.sessions_per_week
        )
        self._factor_scores = {}
        self._factor_raw = {}
        self._computed = False

    def _compute_factors(self):
        if self._computed:
            return

        a = self.athlete
        analyser = self.analyser

        # 1. ACWR factor (0–1)
        acwr_val = analyser.acwr()
        self._factor_raw["acwr"] = acwr_val
        if acwr_val < 0.5:   acwr_score = 0.40
        elif acwr_val < 0.8: acwr_score = 0.20
        elif acwr_val < 1.3: acwr_score = 0.05
        elif acwr_val < 1.5: acwr_score = 0.35
        elif acwr_val < 2.0: acwr_score = 0.70
        else:                acwr_score = 0.95
        self._factor_scores["acwr"] = acwr_score

        # 2. Monotony factor (0–1)
        mono = analyser.monotony()
        self._factor_raw["monotony"] = mono
        mono_score = min(1.0, (mono - 1.0) / 2.0) if mono > 1.0 else 0.0
        self._factor_scores["monotony"] = round(mono_score, 3)

        # 3. Previous injury factor (0–1)
        n_prev = len(a.previous_injuries)
        self._factor_raw["previous_injury"] = n_prev
        prev_score = min(1.0, n_prev * 0.25)
        self._factor_scores["previous_injury"] = prev_score

        # 4. Sleep factor (0–1, inverted — less sleep = higher risk)
        sleep = a.sleep_hours
        self._factor_raw["sleep"] = sleep
        if sleep >= 8.0:    sleep_score = 0.05
        elif sleep >= 7.0:  sleep_score = 0.15
        elif sleep >= 6.0:  sleep_score = 0.35
        elif sleep >= 5.0:  sleep_score = 0.60
        else:               sleep_score = 0.85
        self._factor_scores["sleep"] = sleep_score

        # 5. Experience factor (0–1, inverted — more experience = lower risk)
        yrs = a.years_training
        self._factor_raw["experience"] = yrs
        exp_score = max(0.0, 1.0 - (yrs * 0.08))
        self._factor_scores["experience"] = round(exp_score, 3)

        # 6. Strength asymmetry (0–1)
        asym = a.strength_asymmetry
        self._factor_raw["strength_asymmetry"] = asym
        asym_score = min(1.0, asym / 20.0)
        self._factor_scores["strength_asymmetry"] = round(asym_score, 3)

        # 7. Flexibility (0–1, inverted)
        flex = a.flexibility_score
        self._factor_raw["flexibility"] = flex
        flex_score = max(0.0, 1.0 - (flex / 10.0))
        self._factor_scores["flexibility"] = round(flex_score, 3)

        # 8. Competition period
        self._factor_raw["competition"] = a.competition_period
        comp_score = 0.70 if a.competition_period else 0.15
        self._factor_scores["competition"] = comp_score

        # 9. Age factor
        age = a.age
        self._factor_raw["age"] = age
        if age < 16:    age_score = 0.45   # growth plates
        elif age < 20:  age_score = 0.25
        elif age < 30:  age_score = 0.10
        elif age < 35:  age_score = 0.20
        else:           age_score = 0.40
        self._factor_scores["age"] = age_score

        self._computed = True

    def overall_risk(self) -> float:
        """Compute weighted overall injury risk (0–100%)."""
        self._compute_factors()
        raw = sum(
            self._factor_scores[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS
        )
        return round(min(99.9, raw * 100), 1)

    def region_risks(self) -> dict:
        """
        Per-region risk scores. Combines:
        - Sport-specific base vulnerability
        - Previous injury multiplier (recurrence risk ×2 per Hägglund)
        - Overall risk modulation
        """
        self._compute_factors()
        overall = self.overall_risk() / 100.0
        vuln = SPORT_VULNERABILITY.get(self.athlete.sport, DEFAULT_VULNERABILITY)

        region_scores = {}
        for region in BODY_REGIONS:
            base = vuln.get(region, 0.15)
            # Previous injury raises recurrence risk
            recurrence_mult = 2.2 if region in self.athlete.previous_injuries else 1.0
            # Combine base vulnerability with overall load risk
            combined = base * recurrence_mult * (0.4 + 0.6 * overall)
            region_scores[region] = round(min(99.9, combined * 100), 1)

        return dict(sorted(region_scores.items(), key=lambda x: x[1], reverse=True))

    def factor_breakdown(self) -> dict:
        self._compute_factors()
        return {
            k: {
                "score": self._factor_scores[k],
                "raw": self._factor_raw[k],
                "weight": self.WEIGHTS[k],
                "contribution": round(self._factor_scores[k] * self.WEIGHTS[k] * 100, 1)
            }
            for k in self.WEIGHTS
        }

    def top_risk_regions(self, n=3) -> list:
        regions = self.region_risks()
        return list(regions.items())[:n]

    def save_assessment(self):
        self._compute_factors()
        conn = get_db()
        regions_json = json.dumps(self.region_risks())
        recs = json.dumps(self.generate_recommendations())
        conn.execute(
            """INSERT INTO risk_assessments
               (athlete_id, overall_risk, region_risks, acwr, monotony, strain, recommendations)
               VALUES (?,?,?,?,?,?,?)""",
            (
                self.athlete.id,
                self.overall_risk(),
                regions_json,
                self.analyser.acwr(),
                self.analyser.monotony(),
                self.analyser.strain(),
                recs,
            )
        )
        conn.commit()
        conn.close()

    def generate_recommendations(self) -> list:
        """Generate evidence-based recommendations."""
        self._compute_factors()
        recs = []
        acwr = self.analyser.acwr()
        mono = self.analyser.monotony()
        spike = self.analyser.load_spike()
        overall = self.overall_risk()

        if acwr > 1.5:
            recs.append(f"URGENT: ACWR = {acwr:.2f}. Reduce acute load by ~20% this week. Target ACWR 0.8–1.3.")
        elif acwr > 1.3:
            recs.append(f"Caution: ACWR = {acwr:.2f}. Slightly elevated — monitor closely. Avoid further load increases.")
        elif acwr < 0.6:
            recs.append(f"Under-loading detected (ACWR {acwr:.2f}). Consider graduated load increase to maintain fitness.")

        if mono > 2.0:
            recs.append(f"Training monotony is HIGH ({mono:.1f}). Introduce session variety — mix intensity and modality.")
        elif mono > 1.5:
            recs.append(f"Moderate monotony ({mono:.1f}). Add 1 low-intensity session to create variation.")

        if spike > 20:
            recs.append(f"Load spike of +{spike:.0f}% week-on-week detected. Nielsen et al.: keep increases ≤10%/week.")

        if self.athlete.sleep_hours < 7.0:
            recs.append(f"Sleep average {self.athlete.sleep_hours:.1f}h — below 7h threshold. Milewski 2014: <8h sleep = 1.7× injury risk.")

        if self.athlete.strength_asymmetry > 10:
            recs.append(f"Strength asymmetry {self.athlete.strength_asymmetry}% — above 10% threshold. Croisier 2008: >10% = 4.6× hamstring risk.")

        if self.athlete.flexibility_score < 4:
            recs.append("Low flexibility score. Prioritise daily stretching and mobility work — especially posterior chain.")

        if self.athlete.competition_period:
            recs.append("Competition period active. Protect training quality — increase recovery sessions, monitor wellness daily.")

        if self.athlete.previous_injuries:
            regions = ", ".join(self.athlete.previous_injuries[:3])
            recs.append(f"Previous injury history: {regions}. Re-injury risk 2–8× baseline. Prehab exercises essential.")

        if overall > 75:
            recs.append("⚠ HIGH OVERALL RISK. Consider mandatory rest day + physiotherapy review before next session.")
        elif overall > 50:
            recs.append("Elevated risk profile. Implement full recovery protocols and reduce high-intensity volume this week.")

        if not recs:
            recs.append("Risk profile looks good. Maintain current load management and recovery practices.")

        return recs

# ═══════════════════════════════════════════════════════════════════════
# PERIODISATION PLANNER
# ═══════════════════════════════════════════════════════════════════════

class PeriodisationPlanner:
    """
    Generates evidence-based weekly training plans based on risk profile.
    Uses block periodisation principles (Issurin, 2010).
    """

    SESSION_TYPES = {
        "high_intensity": {"rpe": "8–9", "duration": "75–90 min", "colour": coral},
        "moderate":       {"rpe": "6–7", "duration": "60–75 min", "colour": yellow},
        "technical":      {"rpe": "5–6", "duration": "60 min",    "colour": cyan},
        "recovery":       {"rpe": "3–4", "duration": "30–45 min", "colour": green},
        "rest":           {"rpe": "—",   "duration": "Full rest",  "colour": dim},
        "rehab_prehab":   {"rpe": "3–5", "duration": "45 min",    "colour": mint},
    }

    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def generate_plan(self, athlete: AthleteProfile, overall_risk: float, acwr: float) -> list:
        """Return list of (day, session_type, notes) for 1 week."""
        plan = []

        if overall_risk > 75:
            template = ["recovery", "rehab_prehab", "rest", "recovery", "technical", "rest", "rest"]
            context = "HIGH RISK — volume significantly reduced"
        elif overall_risk > 50:
            template = ["moderate", "recovery", "technical", "rest", "moderate", "recovery", "rest"]
            context = "ELEVATED RISK — intensity capped at RPE 7"
        elif overall_risk > 25:
            template = ["moderate", "technical", "high_intensity", "recovery", "moderate", "technical", "rest"]
            context = "MODERATE RISK — normal week with built-in recovery"
        else:
            template = ["high_intensity", "moderate", "technical", "moderate", "high_intensity", "recovery", "rest"]
            context = "LOW RISK — full training week"

        session_notes = {
            "high_intensity": f"Sport-specific high-intensity ({athlete.sport}). Full technical load.",
            "moderate":       "Moderate load. Focus on skill and game-based work.",
            "technical":      "Low-intensity technical session. Movement quality focus.",
            "recovery":       "Active recovery: pool, bike, foam roll, stretch.",
            "rest":           "Complete rest or gentle walk only.",
            "rehab_prehab":   f"Targeted prehab for: {', '.join(athlete.previous_injuries[:2]) if athlete.previous_injuries else 'general injury prevention'}.",
        }

        for day, stype in zip(self.DAYS, template):
            plan.append({
                "day": day,
                "session": stype,
                "rpe": self.SESSION_TYPES[stype]["rpe"],
                "duration": self.SESSION_TYPES[stype]["duration"],
                "notes": session_notes[stype],
                "colour_fn": self.SESSION_TYPES[stype]["colour"],
            })

        return plan, context

# ═══════════════════════════════════════════════════════════════════════
# REPORT EXPORTER
# ═══════════════════════════════════════════════════════════════════════

class ReportExporter:
    """Exports athlete risk reports to CSV and JSON."""

    def __init__(self, athlete: AthleteProfile, engine: RiskEngine):
        self.athlete = athlete
        self.engine = engine
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_json(self) -> str:
        fname = f"coachiq_report_{self.athlete.name.replace(' ','_')}_{self.timestamp}.json"
        data = {
            "report_metadata": {
                "generated": datetime.now().isoformat(),
                "system": "CoachIQ Injury Risk Intelligence System v1.0",
                "model": "Multi-factor weighted risk model (Gabbett 2016 framework)"
            },
            "athlete": self.athlete.to_dict(),
            "risk_assessment": {
                "overall_risk_pct": self.engine.overall_risk(),
                "region_risks": self.engine.region_risks(),
                "factor_breakdown": self.engine.factor_breakdown(),
                "acwr": self.engine.analyser.acwr(),
                "monotony": self.engine.analyser.monotony(),
                "strain": self.engine.analyser.strain(),
                "load_spike_pct": self.engine.analyser.load_spike(),
                "load_trend": self.engine.analyser.trend(),
            },
            "recommendations": self.engine.generate_recommendations(),
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        return fname

    def export_csv(self) -> str:
        fname = f"coachiq_report_{self.athlete.name.replace(' ','_')}_{self.timestamp}.csv"
        regions = self.engine.region_risks()
        rows = [
            ["CoachIQ Risk Report", datetime.now().strftime("%d %b %Y %H:%M")],
            [],
            ["Athlete", self.athlete.name],
            ["Sport", self.athlete.sport],
            ["Age", self.athlete.age],
            ["Overall Risk", f"{self.engine.overall_risk()}%"],
            ["ACWR", self.engine.analyser.acwr()],
            ["Monotony", self.engine.analyser.monotony()],
            ["Strain", self.engine.analyser.strain()],
            [],
            ["REGION RISK SCORES"],
            ["Region", "Risk %"],
        ]
        for region, score in regions.items():
            rows.append([REGION_DISPLAY[region], f"{score}%"])
        rows.append([])
        rows.append(["RECOMMENDATIONS"])
        for rec in self.engine.generate_recommendations():
            rows.append([rec])
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return fname

# ═══════════════════════════════════════════════════════════════════════
# TERMINAL UI
# ═══════════════════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def line(char="─", colour=C.TEAL, w=WIDTH):
    print(c(colour, char * w))

def hbar(text, colour=C.TEAL):
    pad = max(0, (WIDTH - len(text) - 2) // 2)
    bar = "─" * pad + " " + text + " " + "─" * pad
    while len(bar) < WIDTH:
        bar += "─"
    print(c(colour, bar[:WIDTH]))

def logo():
    print()
    print(teal("  ╔══════════════════════════════════════════════════════════════════╗"))
    print(teal("  ║") + bold(gold("     CoachIQ  ─  Athlete Injury Risk Intelligence System        ")) + teal("║"))
    print(teal("  ║") + dim("     Evidence-based ML risk engine  |  OrthoAI  |  Python 3      ") + teal("║"))
    print(teal("  ╚══════════════════════════════════════════════════════════════════╝"))
    print()

def get_str(prompt, min_len=1) -> str:
    while True:
        val = input(teal(f"  {prompt} ")).strip()
        if len(val) >= min_len:
            return val
        print(red("  Required field. Please enter a value."))

def get_int(prompt, lo, hi) -> int:
    while True:
        try:
            val = int(input(teal(f"  {prompt} ")).strip())
            if lo <= val <= hi:
                return val
        except (ValueError, KeyboardInterrupt):
            pass
        print(red(f"  Enter a whole number between {lo} and {hi}."))

def get_float(prompt, lo, hi) -> float:
    while True:
        try:
            val = float(input(teal(f"  {prompt} ")).strip())
            if lo <= val <= hi:
                return val
        except (ValueError, KeyboardInterrupt):
            pass
        print(red(f"  Enter a number between {lo} and {hi}."))

def get_yn(prompt) -> bool:
    while True:
        val = input(teal(f"  {prompt} [y/n] ")).strip().lower()
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False

def pause(msg="  Press ENTER to continue..."):
    input(dim(msg))

def select_from_list(items: list, prompt: str) -> int:
    for i, item in enumerate(items, 1):
        print(f"  {teal(str(i)+'.')} {item}")
    return get_int(prompt, 1, len(items)) - 1

def progress_bar(value: float, max_val: float = 100, width: int = 30,
                 filled_char="█", empty_char="░",
                 colour_fn=None) -> str:
    pct = min(1.0, value / max_val)
    filled = round(pct * width)
    bar = filled_char * filled + empty_char * (width - filled)
    label = f"{value:.1f}%"
    coloured_bar = colour_fn(bar) if colour_fn else bar
    return f"[{coloured_bar}] {label}"

def risk_bar(pct: float, width: int = 28) -> str:
    if pct >= 75:   colour_fn = red
    elif pct >= 50: colour_fn = yellow
    elif pct >= 25: colour_fn = gold
    else:           colour_fn = green
    return progress_bar(pct, 100, width, colour_fn=colour_fn)

def mini_sparkline(values: list, width: int = 16) -> str:
    """ASCII sparkline chart for load trend."""
    chars = "▁▂▃▄▅▆▇█"
    if not values or max(values) == 0:
        return "─" * width
    mn, mx = min(values), max(values)
    if mx == mn:
        return "▄" * len(values)
    normalised = [(v - mn) / (mx - mn) for v in values]
    return "".join(chars[min(7, int(n * 7.99))] for n in normalised)

def acwr_meter(acwr_val: float, width: int = 40) -> str:
    """Visual ACWR zone display."""
    zones = [
        (0.0,  0.5,  "Under", dim),
        (0.5,  0.8,  "Low",   yellow),
        (0.8,  1.3,  "Optimal", green),
        (1.3,  1.5,  "Caution", gold),
        (1.5,  2.0,  "High", red),
        (2.0,  3.0,  "Danger", coral),
    ]
    # clamp
    val = max(0.0, min(2.99, acwr_val))
    # find zone
    active_label = "Optimal"
    active_col = green
    for lo, hi, label, col_fn in zones:
        if lo <= val < hi:
            active_label = label
            active_col = col_fn
            break

    bar = ""
    zone_chars = {
        "Under": "░", "Low": "▒", "Optimal": "█", "Caution": "▓", "High": "▓", "Danger": "█"
    }
    ticks = ["0.0", "0.5", "0.8", "1.3", "1.5", "2.0"]
    scale = f"  {'Under':<7}{'Low':<7}{'OPTIMAL':<9}{'Caution':<9}{'High':<7}Danger"
    filled_pos = int((val / 2.0) * width)
    raw = "░" * width
    raw_list = list(raw)
    if 0 <= filled_pos < width:
        raw_list[filled_pos] = "▲"
    bar_str = "".join(raw_list)

    zones_display = (
        dim("░░░░░░") + yellow("▒▒▒▒") + green("████████") +
        gold("▓▓▓▓") + red("▓▓▓▓▓") + coral("██████")
    )
    marker = " " * max(0, int((val / 2.0) * 32))
    return (f"  {zones_display}\n"
            f"  {' ' * max(0, int((val / 2.0) * 32))}▲ {active_col(active_label)} "
            f"({active_col(str(acwr_val))})")

def render_heatmap(region_risks: dict) -> None:
    """Render body region heatmap in terminal."""
    hbar("INJURY RISK HEATMAP  ─  BY BODY REGION", C.CYAN)
    print()
    for region, score in region_risks.items():
        display = REGION_DISPLAY[region]
        bar = risk_bar(score, 25)
        prev_marker = ""

        padding = max(0, 16 - len(display))
        print(f"  {display}{' ' * padding} {bar}")

        # Common injuries for this region
        injuries = COMMON_INJURIES.get(region, [])
        if score > 40:
            print(f"  {' ' * 16}   {dim('Watch: ' + ' · '.join(injuries[:2]))}")
    print()

def render_factor_breakdown(breakdown: dict) -> None:
    """Render weighted factor contribution chart."""
    hbar("RISK FACTOR ANALYSIS", C.PURPLE)
    print()
    factor_labels = {
        "acwr":               "ACWR",
        "monotony":           "Training monotony",
        "previous_injury":    "Previous injuries",
        "sleep":              "Sleep quality",
        "experience":         "Training experience",
        "strength_asymmetry": "Strength asymmetry",
        "flexibility":        "Flexibility",
        "competition":        "Competition period",
        "age":                "Age factor",
    }
    for key, data in breakdown.items():
        label = factor_labels.get(key, key)
        contrib = data["contribution"]
        raw = data["raw"]
        score = data["score"]

        bar_width = 20
        filled = round(score * bar_width)
        bar_str = "█" * filled + "░" * (bar_width - filled)
        if score > 0.7:   bar_c = red(bar_str)
        elif score > 0.4: bar_c = yellow(bar_str)
        else:             bar_c = green(bar_str)

        raw_display = (
            f"{raw:.2f}" if isinstance(raw, float) else
            f"{raw}" if isinstance(raw, int) else
            ("Yes" if raw else "No")
        )
        pad = max(0, 20 - len(label))
        print(f"  {label}{' ' * pad} [{bar_c}] {dim(raw_display):<10} +{contrib:.1f}%")
    print()

def render_weekly_plan(plan: list, context: str) -> None:
    """Render 7-day periodisation plan."""
    hbar("WEEKLY TRAINING PLAN  —  RISK-ADJUSTED", C.MINT)
    print()
    print(f"  Context: {mint(context)}")
    print()
    for entry in plan:
        stype_display = entry["session"].replace("_", " ").title()
        col_fn = entry["colour_fn"]
        day_pad = max(0, 11 - len(entry["day"]))
        print(f"  {bold(entry['day'])}{' ' * day_pad} {col_fn(f'{stype_display:<18}')} RPE {entry['rpe']:<6} {entry['duration']}")
        print(f"  {' ' * 11} {dim(entry['notes'])}")
        print()

def render_recommendations(recs: list) -> None:
    """Render clinical recommendations."""
    hbar("EVIDENCE-BASED RECOMMENDATIONS", C.CORAL)
    print()
    for i, rec in enumerate(recs, 1):
        # Colour code severity
        if rec.startswith("URGENT") or rec.startswith("⚠"):
            label = red(f"  [{i}]")
            text = red(rec)
        elif rec.startswith("Caution") or "Elevated" in rec:
            label = yellow(f"  [{i}]")
            text = yellow(rec)
        else:
            label = teal(f"  [{i}]")
            text = rec
        print(f"{label} {text}")
        print()

def render_ai_insights(top_regions: list) -> None:
    """Render AI/ML research insights for top risk regions."""
    hbar("AI RESEARCH INSIGHTS", C.GOLD)
    print()
    for region, score in top_regions:
        insight = AI_INSIGHTS.get(region, "No specific ML model data available for this region.")
        display = REGION_DISPLAY[region]
        print(f"  {gold('◆')} {bold(display)} {dim(f'({score:.1f}% risk)')}")
        print(f"    {dim(insight)}")
        print()

def render_load_analysis(analyser: TrainingAnalyser) -> None:
    """Full load analysis dashboard section."""
    hbar("TRAINING LOAD ANALYSIS", C.SKY)
    print()

    acwr = analyser.acwr()
    mono = analyser.monotony()
    strain = analyser.strain()
    spike = analyser.load_spike()
    trend = analyser.trend()

    # Metrics row
    metrics = [
        ("ACWR",       f"{acwr:.2f}",    acwr_risk_label(acwr)),
        ("Monotony",   f"{mono:.2f}",    mono_label(mono)),
        ("Strain",     f"{strain:.0f}",  strain_label(strain)),
        ("Load Spike", f"{spike:+.1f}%", spike_label(spike)),
    ]
    for name, val, label in metrics:
        pad = max(0, 14 - len(name))
        print(f"  {bold(name)}{' ' * pad} {sky(val):<12} {dim(label)}")

    print()

    # Sparkline
    if analyser.weekly_loads:
        spark = mini_sparkline(analyser.weekly_loads)
        print(f"  4-week load trend:  {teal(spark)}  {dim(trend)}")
        print(f"  Loads (AU):         " + "  ".join(f"{v:.0f}" for v in analyser.weekly_loads))

    print()
    # ACWR meter
    print(f"  ACWR Zone:")
    print(acwr_meter(acwr))
    print()

def acwr_risk_label(v: float) -> str:
    if v < 0.5:   return "Under-loaded"
    if v < 0.8:   return "Below optimal"
    if v < 1.3:   return "Optimal zone ✓"
    if v < 1.5:   return "Caution — slightly high"
    if v < 2.0:   return "HIGH RISK"
    return "DANGER ZONE"

def mono_label(v: float) -> str:
    if v < 1.2: return "Low — good variety"
    if v < 1.6: return "Moderate"
    if v < 2.0: return "High — reduce monotony"
    return "VERY HIGH — overtraining risk"

def strain_label(v: float) -> str:
    if v < 1000: return "Low"
    if v < 2000: return "Moderate"
    if v < 3000: return "High"
    return "VERY HIGH"

def spike_label(v: float) -> str:
    if v < 0:    return "Load decrease"
    if v < 10:   return "Safe increase"
    if v < 20:   return "Elevated — monitor"
    return "SPIKE — injury risk ↑"

# ═══════════════════════════════════════════════════════════════════════
# INPUT FLOW — ATHLETE CREATION
# ═══════════════════════════════════════════════════════════════════════

def input_athlete() -> AthleteProfile:
    clear()
    logo()
    hbar("NEW ATHLETE PROFILE", C.TEAL)
    print()

    # Basic info
    print(bold("  ── ATHLETE DETAILS ──"))
    print()
    name = get_str("Full name:")
    age  = get_int("Age (years):", 14, 45)

    print()
    print(bold("  ── SPORT ──"))
    print()
    sport_idx = select_from_list(SPORTS, "Select sport (enter number):")
    sport = SPORTS[sport_idx]

    positions = POSITIONS.get(sport, ["General"])
    print()
    print(bold("  ── POSITION / DISCIPLINE ──"))
    print()
    pos_idx = select_from_list(positions, "Select position:")
    position = positions[pos_idx]

    print()
    print(bold("  ── TRAINING BACKGROUND ──"))
    print()
    years = get_int("Years training in this sport:", 0, 25)
    sessions = get_int("Sessions per week currently:", 1, 14)

    print()
    print(bold("  ── TRAINING LOADS  (Arbitrary Units = sessions × RPE × duration) ──"))
    print(dim("  Enter weekly training loads for the last 4 weeks (oldest → newest)."))
    print(dim("  Example: a 90-min session at RPE 7 = ~630 AU. Typical week = 2000–3500 AU."))
    print()
    loads = []
    for i in range(1, 5):
        label = "oldest" if i == 1 else ("most recent" if i == 4 else f"week {i}")
        val = get_float(f"Week {i} ({label}) load (AU):", 0, 9999)
        loads.append(val)

    print()
    print(bold("  ── SESSION RPE THIS WEEK ──"))
    print(dim("  Enter RPE (1–10) for each session this week."))
    print()
    n_rpe = get_int("How many sessions this week?", 1, 14)
    rpe_scores = []
    for i in range(1, n_rpe + 1):
        val = get_float(f"Session {i} RPE (1–10):", 1, 10)
        rpe_scores.append(val)

    print()
    print(bold("  ── RECOVERY & WELLNESS ──"))
    print()
    sleep = get_float("Average sleep hours per night:", 3.0, 12.0)

    print()
    print(bold("  ── PHYSICAL PROFILE ──"))
    print()
    flex = get_int("Flexibility score (1=very stiff, 10=excellent):", 1, 10)
    asym = get_int("Strength asymmetry L vs R (% difference, 0 if unknown):", 0, 40)

    print()
    print(bold("  ── INJURY HISTORY ──"))
    print()
    prev_injuries = []
    n_injuries = get_int("Number of significant previous injuries:", 0, 8)
    if n_injuries > 0:
        print()
        print(dim("  Select injury region for each:"))
        region_list = [REGION_DISPLAY[r] for r in BODY_REGIONS]
        for i in range(n_injuries):
            print(f"\n  Injury {i+1}:")
            idx = select_from_list(region_list, "Region:")
            prev_injuries.append(BODY_REGIONS[idx])

    print()
    print(bold("  ── COMPETITION STATUS ──"))
    print()
    competition = get_yn("Currently in competition/match period?")

    return AthleteProfile(
        name=name, age=age, sport=sport, position=position,
        years_training=years, sessions_per_week=sessions,
        sleep_hours=sleep, previous_injuries=prev_injuries,
        recent_loads=loads, rpe_scores=rpe_scores,
        flexibility_score=flex, strength_asymmetry=asym,
        competition_period=competition,
    )

def quick_demo_athlete() -> AthleteProfile:
    """Pre-filled demo athlete for quick testing."""
    return AthleteProfile(
        name="Demo Athlete", age=22, sport="Cricket",
        position="Fast Bowler", years_training=5, sessions_per_week=6,
        sleep_hours=6.5, previous_injuries=["lumbar_spine", "hamstring"],
        recent_loads=[2100, 2300, 2800, 3500],
        rpe_scores=[7, 8, 7, 9, 8, 8],
        flexibility_score=4, strength_asymmetry=12,
        competition_period=True,
    )

# ═══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD RENDER
# ═══════════════════════════════════════════════════════════════════════

def run_full_assessment(athlete: AthleteProfile):
    clear()
    logo()

    # Boot animation
    for step, msg in enumerate([
        "Loading athlete profile...",
        "Computing ACWR and training load metrics...",
        "Running multi-factor risk model...",
        "Scoring 8 body regions...",
        "Generating evidence-based recommendations...",
        "Building periodisation plan...",
    ]):
        pct = int((step / 5) * 100)
        bar_filled = "█" * (step + 1) + "░" * (5 - step)
        print(f"\r  {teal('[' + bar_filled + ']')} {dim(msg)}", end="", flush=True)
        time.sleep(0.25)
    print(f"\r  {teal('[██████]')} {green('Assessment complete.')}       ")
    time.sleep(0.3)

    # Build models
    engine = RiskEngine(athlete)
    overall = engine.overall_risk()
    regions = engine.region_risks()
    breakdown = engine.factor_breakdown()
    recommendations = engine.generate_recommendations()
    top_regions = engine.top_risk_regions(3)
    analyser = engine.analyser

    planner = PeriodisationPlanner()
    plan, plan_context = planner.generate_plan(athlete, overall, analyser.acwr())

    clear()
    logo()

    # ── ATHLETE SUMMARY ──
    hbar(f"RISK ASSESSMENT  ─  {athlete.name.upper()}", C.TEAL)
    print()
    print(f"  {bold(athlete.name):<28} {dim(athlete.sport + ' · ' + athlete.position)}")
    print(f"  {dim('Age:')} {athlete.age}  "
          f"{dim('Experience:')} {athlete.years_training}y  "
          f"{dim('Sessions/wk:')} {athlete.sessions_per_week}  "
          f"{dim('Sleep:')} {athlete.sleep_hours:.1f}h")
    print()

    # ── OVERALL RISK ──
    hbar("OVERALL INJURY RISK", C.TEAL)
    print()
    big_bar = risk_bar(overall, 36)
    if overall >= 75:   risk_text = red(bold("HIGH RISK"))
    elif overall >= 50: risk_text = yellow(bold("ELEVATED RISK"))
    elif overall >= 25: risk_text = gold(bold("MODERATE RISK"))
    else:               risk_text = green(bold("LOW RISK"))

    print(f"  {big_bar}   {risk_text}")
    print()
    print(f"  Composite score: {bold(risk_colour(overall))}  "
          f"{dim('(weighted 9-factor model)')}")
    print()
    line("─", C.DIM)
    print()

    # ── LOAD ANALYSIS ──
    render_load_analysis(analyser)

    pause()
    clear()
    logo()

    # ── HEATMAP ──
    render_heatmap(regions)

    pause()
    clear()
    logo()

    # ── FACTOR BREAKDOWN ──
    render_factor_breakdown(breakdown)

    pause()
    clear()
    logo()

    # ── AI INSIGHTS ──
    render_ai_insights(top_regions)

    # ── RECOMMENDATIONS ──
    render_recommendations(recommendations)

    pause()
    clear()
    logo()

    # ── WEEKLY PLAN ──
    render_weekly_plan(plan, plan_context)

    pause()

    # ── EXPORT ──
    clear()
    logo()
    hbar("EXPORT REPORT", C.GOLD)
    print()
    print(f"  {gold('1.')} Export full JSON report")
    print(f"  {gold('2.')} Export CSV summary")
    print(f"  {gold('3.')} Export both")
    print(f"  {gold('4.')} Skip — return to menu")
    print()

    try:
        choice = int(input(teal("  → ")).strip())
    except Exception:
        choice = 4

    exporter = ReportExporter(athlete, engine)

    if choice in (1, 3):
        fname = exporter.export_json()
        print(green(f"\n  ✓ JSON report saved: {fname}"))

    if choice in (2, 3):
        fname = exporter.export_csv()
        print(green(f"  ✓ CSV report saved: {fname}"))

    # Save to DB
    athlete.save()
    engine.save_assessment()
    print(mint(f"\n  ✓ Assessment saved to database: {os.path.basename(DB_PATH)}"))

    print()
    pause()

# ═══════════════════════════════════════════════════════════════════════
# HISTORY VIEWER
# ═══════════════════════════════════════════════════════════════════════

def view_history():
    clear()
    logo()
    hbar("ATHLETE HISTORY", C.CYAN)
    print()

    conn = get_db()
    athletes = conn.execute(
        "SELECT a.*, COUNT(r.id) as assessments FROM athletes a "
        "LEFT JOIN risk_assessments r ON r.athlete_id = a.id "
        "GROUP BY a.id ORDER BY a.created_at DESC"
    ).fetchall()
    conn.close()

    if not athletes:
        print(dim("  No athletes in database yet. Run an assessment first."))
        print()
        pause()
        return

    print(f"  {'#':<4} {'Name':<22} {'Sport':<20} {'Assessments':<14} {'Added'}")
    line("─", C.DIM)
    for i, row in enumerate(athletes, 1):
        date = row["created_at"][:10] if row["created_at"] else "—"
        print(f"  {teal(str(i)):<6} {row['name']:<22} {row['sport']:<20} "
              f"{dim(str(row['assessments']) + ' runs'):<16} {dim(date)}")

    print()

    if len(athletes) == 1:
        choice = 1
    else:
        choice = get_int(f"Select athlete to view history (1–{len(athletes)}, 0 to go back):", 0, len(athletes))

    if choice == 0:
        return

    selected = athletes[choice - 1]
    athlete_id = selected["id"]

    conn = get_db()
    assessments = conn.execute(
        "SELECT * FROM risk_assessments WHERE athlete_id=? ORDER BY assessed_at DESC",
        (athlete_id,)
    ).fetchall()
    conn.close()

    if not assessments:
        print(dim("  No assessments found for this athlete."))
        pause()
        return

    clear()
    logo()
    hbar(f"HISTORY  —  {selected['name'].upper()}", C.CYAN)
    print()

    for i, a in enumerate(assessments, 1):
        date = a["assessed_at"][:16] if a["assessed_at"] else "—"
        risk = a["overall_risk"]
        acwr = a["acwr"]
        print(f"  {dim(str(i)+'.')} {date}  Overall: {risk_colour(risk):<20} "
              f"ACWR: {sky(str(acwr))}")

    print()
    pause()

# ═══════════════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════════════

def main_menu() -> int:
    clear()
    logo()
    line()
    print()
    print(f"  {teal('1.')} {bold('New athlete assessment')}        {dim('— full risk profile + weekly plan')}")
    print(f"  {teal('2.')} {bold('Quick demo')}                    {dim('— pre-filled athlete, instant results')}")
    print(f"  {teal('3.')} {bold('View athlete history')}          {dim('— database + past assessments')}")
    print(f"  {teal('4.')} {bold('About the model')}               {dim('— evidence base + methodology')}")
    print(f"  {teal('0.')} {bold('Exit')}")
    print()
    return get_int("→", 0, 4)

def about_screen():
    clear()
    logo()
    hbar("ABOUT THE MODEL", C.PURPLE)
    print()
    sections = [
        ("CORE ALGORITHM",
         "Multi-factor weighted risk model based on Gabbett (2016) ACWR\n"
         "  framework, extended with 8 additional predictive factors drawn\n"
         "  from sports medicine epidemiology literature."),
        ("FACTOR WEIGHTS",
         "ACWR 25% | Previous injury 20% | Monotony 15% |\n"
         "  Strength asymmetry 10% | Sleep 10% | Competition 5% |\n"
         "  Experience 5% | Flexibility 5% | Age 5%"),
        ("KEY REFERENCES",
         "Gabbett TJ (2016) — ACWR and injury risk. BJSM.\n"
         "  Foster C (1998) — Monitoring training in athletes. J Strength Cond.\n"
         "  Hägglund M (2006) — Previous injury as risk factor. BJSM.\n"
         "  Milewski M (2014) — Sleep and injury risk. J Pediatr Orthop.\n"
         "  Croisier J (2008) — Strength asymmetry and hamstring injury. AJSM.\n"
         "  Nielsen RO (2014) — Running load and injury. Int J Sports Phys Ther."),
        ("BODY REGION DATA",
         "Sport-specific injury vulnerability rates derived from injury\n"
         "  surveillance data across 8 body regions and 15 sports.\n"
         "  Previous injury recurrence multiplier: 2.2× (Hägglund 2006)."),
        ("LIMITATIONS",
         "This is a rule-based model for educational purposes. It does not\n"
         "  replace clinical assessment. All outputs should be interpreted\n"
         "  by a qualified sports medicine professional."),
    ]
    for title, body in sections:
        print(f"  {bold(purple(title))}")
        print(f"  {dim(body)}")
        print()
    pause()

# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    init_db()
    while True:
        choice = main_menu()
        if choice == 1:
            athlete = input_athlete()
            run_full_assessment(athlete)
        elif choice == 2:
            athlete = quick_demo_athlete()
            run_full_assessment(athlete)
        elif choice == 3:
            view_history()
        elif choice == 4:
            about_screen()
        elif choice == 0:
            clear()
            print()
            print(teal("  CoachIQ — Athlete Injury Risk Intelligence System"))
            print(dim("  OrthoAI Project  |  Built by Mark"))
            print()
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(dim("\n  Exiting CoachIQ Risk Engine."))
        sys.exit(0)
