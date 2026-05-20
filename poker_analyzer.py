"""
Poker Session Analyzer
======================
Reads a poker session log in the established Excel format and produces a full
suite of statistics, charts, and an HTML report.

The Excel layout (rows 0-9 are summary cells managed by hand):
    Row 10 (header): Title | Outcome | Buy in | Outcome in buying | Notes
    Row 11+ : one row per session.

Usage:
    python poker_analyzer.py
    python poker_analyzer.py Poker.xlsx
    python poker_analyzer.py Poker.xlsx --outdir ./report

What it computes
----------------
* Headline totals: nominal winnings, "real" winnings (Truly adjustments,
  forgiven debts removed, money given away netted out), forgiven money.
* Per-game-type breakdown (Heads-up, 3-way, 5-way, fish games, etc.).
* Per-buy-in breakdown (ROI in buy-ins).
* Per-opponent breakdown (parsed from titles/notes: Giorgos, Oussama, ...).
* Recent form: last 5 / 10 / 20 sessions, cumulative curve, rolling averages.
* Streaks: longest winning / losing streak, current streak.
* Best & worst sessions, with the relevant notes attached.
* Sentiment-flavoured "best/worst games by feel" using simple keyword scoring
  over the notes column (lucky vs unlucky, good play vs mistakes, etc.).
* Volatility: standard deviation, max drawdown on the cumulative curve.

Outputs
-------
* report/index.html  -- full report (open in a browser)
* report/charts/*.png -- every chart, also embedded in the HTML
* report/sessions_clean.csv -- the cleaned per-session table
* Console summary -- key headline numbers
"""

from __future__ import annotations

import argparse
import base64
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — tune these if your shorthand evolves
# ---------------------------------------------------------------------------

HEADER_ROW = 11  # 0-indexed: row 12 in Excel (rows 1-11 are summary cells)

# Known opponents/regulars referenced in titles or notes. Extend freely.
KNOWN_PEOPLE = [
    "Giorgos", "Georgios", "Vassil", "Egemen", "Oussama", "Baran",
    "Oskar", "Nohan", "Saade", "Mike",
]

# Sentiment keywords used to score sessions from the Notes column.
POSITIVE_WORDS = [
    "lucky", "good play", "great", "solid", "decent play", "on point",
    "incredible", "nut", "good folds", "good reads", "owning",
]
NEGATIVE_WORDS = [
    "unlucky", "horrible", "mistake", "stupid", "bad call", "embarrassing",
    "regret", "rough", "too greedy", "couldn't trust", "didn't trust",
    "bluffed into", "gave up", "no hands",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def load_sessions(path: Path) -> pd.DataFrame:
    """Load the session table from the Excel file, skipping summary rows."""
    raw = pd.read_excel(path, sheet_name=0, header=HEADER_ROW)
    raw.columns = [str(c).strip() for c in raw.columns]
    # Standardise column names regardless of slight spacing differences.
    rename = {}
    for c in raw.columns:
        lc = c.lower()
        if "title" in lc:
            rename[c] = "Title"
        elif "outcome in buying" in lc or "in buying" in lc:
            rename[c] = "OutcomeInBuyins"
        elif "outcome" in lc:
            rename[c] = "Outcome"
        elif "buy" in lc:
            rename[c] = "BuyIn"
        elif "note" in lc:
            rename[c] = "Notes"
    df = raw.rename(columns=rename)
    # Keep only rows that look like real sessions (have a Title and a numeric Outcome).
    df = df[df["Title"].notna()].copy()
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    df["BuyIn"] = pd.to_numeric(df["BuyIn"], errors="coerce")
    df["OutcomeInBuyins"] = pd.to_numeric(df["OutcomeInBuyins"], errors="coerce")
    df = df[df["Outcome"].notna()].reset_index(drop=True)
    df["Notes"] = df["Notes"].fillna("").astype(str)
    df["SessionNum"] = np.arange(1, len(df) + 1)
    return df


# A "truly" note means the real result is less than the nominal one because
# some of the chips on the table never actually belonged to me. e.g. "(Truly
# +20, lost 40euro heads or tails 10buyin)" — nominal outcome was 60, real is
# 20.
TRULY_RE = re.compile(r"truly\s*\+?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
# "covered" usually means someone else bought my buy-in, so the real outcome
# is nominal minus that buy-in.
COVERED_RE = re.compile(r"covered\s+buy[\s\-]?in", re.IGNORECASE)
# "forgiven" means I never collected the money — it's a gift to the opponent.
FORGIVEN_RE = re.compile(r"forgiven", re.IGNORECASE)
# Explicit gifts: "gave back 20", "given to Baran", "to compensate his loss"
GAVE_BACK_RE = re.compile(r"gave\s+back\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
GIVEN_AMOUNT_RE = re.compile(r"given\s+(?:to\s+\w+\s+)?(\d+(?:\.\d+)?)", re.IGNORECASE)


def compute_real_and_forgiven(row: pd.Series) -> tuple[float, float]:
    """Return (real_outcome, forgiven_amount) for a single session.

    Rules, in order:
    1. If notes contain "forgiven", the entire outcome is forgiven and real=0.
    2. If notes contain a "Truly +X" / "Truly X" annotation, real = that value.
       Forgiven amount = nominal outcome - real outcome (everything above the
       truly figure is money I effectively handed back).
    3. If notes mention "covered" buy-in, real = nominal - buy_in.
    4. Otherwise real = nominal, forgiven = 0.
    Explicit "gave back N" / "given to X N" amounts always add to forgiven
    (and reduce real if rule 2 didn't already account for them).
    """
    notes = row["Notes"]
    nominal = float(row["Outcome"])
    buyin = float(row["BuyIn"]) if not math.isnan(row["BuyIn"]) else 0.0

    if FORGIVEN_RE.search(notes):
        return 0.0, nominal

    truly_match = TRULY_RE.search(notes)
    if truly_match:
        real = float(truly_match.group(1))
        forgiven = max(nominal - real, 0.0)
        return real, forgiven

    if COVERED_RE.search(notes):
        real = nominal - buyin
        forgiven = 0.0  # covered means someone *gave* me money, not the other way
        return real, forgiven

    # Catch explicit "gave back N" when no Truly tag.
    gave_back = sum(float(m.group(1)) for m in GAVE_BACK_RE.finditer(notes))
    if gave_back > 0:
        return nominal - gave_back, gave_back

    return nominal, 0.0


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

HEADS_UP_RE = re.compile(r"\b(heads\s*up|\bhu\b)", re.IGNORECASE)
WAY_RE = re.compile(r"(\d+)[\s\-]*(?:to[\s\-]*\d+[\s\-]*)?way", re.IGNORECASE)
FISH_RE = re.compile(r"fish", re.IGNORECASE)


def classify_game(title: str) -> dict[str, object]:
    """Pull structured info out of the free-form Title field."""
    t = str(title)
    out: dict[str, object] = {"IsHeadsUp": False, "Players": None, "HasFish": False}
    if HEADS_UP_RE.search(t):
        out["IsHeadsUp"] = True
        out["Players"] = 2
    way_match = WAY_RE.search(t)
    if way_match:
        out["Players"] = int(way_match.group(1))
    if FISH_RE.search(t):
        out["HasFish"] = True
    return out


def players_bucket(players: Optional[float]) -> str:
    if players is None or (isinstance(players, float) and math.isnan(players)):
        return "Unknown"
    p = int(players)
    if p <= 2:
        return "Heads-up (2)"
    if p == 3:
        return "3-way"
    if p == 4:
        return "4-way"
    if p in (5, 6):
        return "5-6 way"
    if p in (7, 8):
        return "7-8 way"
    return f"{p}+ way"


def buyin_bucket(buyin: float) -> str:
    if math.isnan(buyin):
        return "Unknown"
    if buyin <= 10:
        return "€10 and under"
    if buyin <= 20:
        return "€11–20"
    if buyin <= 30:
        return "€21–30"
    if buyin <= 50:
        return "€31–50"
    if buyin <= 100:
        return "€51–100"
    return "€100+"


def find_people(text: str) -> list[str]:
    found: list[str] = []
    for name in KNOWN_PEOPLE:
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
            canonical = "Giorgos" if name.lower() in {"giorgos", "georgios"} else name
            if canonical not in found:
                found.append(canonical)
    return found


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------


def feel_score(notes: str) -> int:
    """Tiny heuristic: positives - negatives in the notes."""
    n = notes.lower()
    if not n:
        return 0
    pos = sum(n.count(w) for w in POSITIVE_WORDS)
    neg = sum(n.count(w) for w in NEGATIVE_WORDS)
    return pos - neg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    df: pd.DataFrame
    totals: dict
    by_players: pd.DataFrame
    by_buyin: pd.DataFrame
    by_people: pd.DataFrame
    fish_split: pd.DataFrame
    streaks: dict
    recent: dict
    extremes: dict
    feel: dict
    drawdown: dict


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    classified = df["Title"].apply(classify_game).apply(pd.Series)
    df = pd.concat([df, classified], axis=1)
    real_and_forgiven = df.apply(compute_real_and_forgiven, axis=1, result_type="expand")
    df["RealOutcome"] = real_and_forgiven[0]
    df["Forgiven"] = real_and_forgiven[1]
    df["CumulativeNominal"] = df["Outcome"].cumsum()
    df["CumulativeReal"] = df["RealOutcome"].cumsum()
    df["PlayersBucket"] = df["Players"].apply(players_bucket)
    df["BuyInBucket"] = df["BuyIn"].apply(buyin_bucket)
    df["FeelScore"] = df["Notes"].apply(feel_score)
    df["PeopleMentioned"] = (df["Title"] + " " + df["Notes"]).apply(find_people)
    df["Win"] = df["Outcome"] > 0
    df["RealWin"] = df["RealOutcome"] > 0
    return df


def streak_stats(series: pd.Series) -> dict:
    """Compute longest run lengths above/below zero, and the current run."""
    signs = np.sign(series.values)
    longest_win = longest_loss = current = 0
    current_sign = 0
    for s in signs:
        if s == current_sign and s != 0:
            current += 1
        else:
            current_sign = s
            current = 1 if s != 0 else 0
        if current_sign > 0:
            longest_win = max(longest_win, current)
        elif current_sign < 0:
            longest_loss = max(longest_loss, current)
    return {
        "longest_winning_streak": int(longest_win),
        "longest_losing_streak": int(longest_loss),
        "current_streak": int(current) * int(current_sign),
    }


def max_drawdown(curve: np.ndarray) -> dict:
    """Largest peak-to-trough drop on a cumulative curve."""
    if len(curve) == 0:
        return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0}
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve - peaks
    trough = int(np.argmin(drawdowns))
    peak = int(np.argmax(curve[: trough + 1])) if trough > 0 else 0
    return {
        "max_drawdown": float(drawdowns.min()),
        "peak_idx": peak,
        "trough_idx": trough,
    }


def compute_stats(df: pd.DataFrame) -> Stats:
    totals = {
        "sessions": int(len(df)),
        "nominal_sum": float(df["Outcome"].sum()),
        "real_sum": float(df["RealOutcome"].sum()),
        "forgiven_sum": float(df["Forgiven"].sum()),
        "total_buyins_played": float(df["BuyIn"].sum()),
        "win_rate_nominal": float((df["Outcome"] > 0).mean()),
        "win_rate_real": float((df["RealOutcome"] > 0).mean()),
        "avg_outcome": float(df["Outcome"].mean()),
        "avg_real_outcome": float(df["RealOutcome"].mean()),
        "median_outcome": float(df["Outcome"].median()),
        "stdev_outcome": float(df["Outcome"].std(ddof=0)),
        "avg_buyins_won": float(df["OutcomeInBuyins"].mean()),
        "roi_pct": float(df["Outcome"].sum() / df["BuyIn"].sum() * 100)
        if df["BuyIn"].sum()
        else 0.0,
        "real_roi_pct": float(df["RealOutcome"].sum() / df["BuyIn"].sum() * 100)
        if df["BuyIn"].sum()
        else 0.0,
    }

    by_players = (
        df.groupby("PlayersBucket")
        .agg(
            sessions=("Outcome", "size"),
            nominal=("Outcome", "sum"),
            real=("RealOutcome", "sum"),
            avg=("Outcome", "mean"),
            win_rate=("Win", "mean"),
            avg_buyins=("OutcomeInBuyins", "mean"),
        )
        .sort_values("nominal", ascending=False)
    )

    by_buyin = (
        df.groupby("BuyInBucket")
        .agg(
            sessions=("Outcome", "size"),
            nominal=("Outcome", "sum"),
            real=("RealOutcome", "sum"),
            avg=("Outcome", "mean"),
            win_rate=("Win", "mean"),
            avg_buyins=("OutcomeInBuyins", "mean"),
        )
        .sort_values("nominal", ascending=False)
    )

    # Per-person breakdown: a session contributes to every person mentioned.
    rows = []
    for _, r in df.iterrows():
        for person in r["PeopleMentioned"]:
            rows.append(
                {
                    "Person": person,
                    "Outcome": r["Outcome"],
                    "RealOutcome": r["RealOutcome"],
                    "Win": r["Win"],
                }
            )
    if rows:
        people_df = pd.DataFrame(rows)
        by_people = (
            people_df.groupby("Person")
            .agg(
                sessions=("Outcome", "size"),
                nominal=("Outcome", "sum"),
                real=("RealOutcome", "sum"),
                avg=("Outcome", "mean"),
                win_rate=("Win", "mean"),
            )
            .sort_values("nominal", ascending=False)
        )
    else:
        by_people = pd.DataFrame(
            columns=["sessions", "nominal", "real", "avg", "win_rate"]
        )

    fish_split = (
        df.groupby("HasFish")
        .agg(
            sessions=("Outcome", "size"),
            nominal=("Outcome", "sum"),
            real=("RealOutcome", "sum"),
            avg=("Outcome", "mean"),
            win_rate=("Win", "mean"),
        )
        .rename(index={True: "With fish", False: "No fish noted"})
    )

    streaks = streak_stats(df["Outcome"])

    recent = {}
    for n in (5, 10, 20):
        tail = df.tail(n)
        if len(tail):
            recent[f"last_{n}_nominal"] = float(tail["Outcome"].sum())
            recent[f"last_{n}_real"] = float(tail["RealOutcome"].sum())
            recent[f"last_{n}_winrate"] = float((tail["Outcome"] > 0).mean())

    extremes = {
        "best_session": df.loc[df["Outcome"].idxmax()].to_dict(),
        "worst_session": df.loc[df["Outcome"].idxmin()].to_dict(),
        "best_real_session": df.loc[df["RealOutcome"].idxmax()].to_dict(),
        "worst_real_session": df.loc[df["RealOutcome"].idxmin()].to_dict(),
    }

    notes_df = df[df["Notes"].str.strip() != ""].copy()
    if len(notes_df):
        feel = {
            "best_by_feel": notes_df.loc[notes_df["FeelScore"].idxmax()].to_dict(),
            "worst_by_feel": notes_df.loc[notes_df["FeelScore"].idxmin()].to_dict(),
        }
    else:
        feel = {}

    drawdown = max_drawdown(df["CumulativeNominal"].values)

    return Stats(
        df=df,
        totals=totals,
        by_players=by_players,
        by_buyin=by_buyin,
        by_people=by_people,
        fish_split=fish_split,
        streaks=streaks,
        recent=recent,
        extremes=extremes,
        feel=feel,
        drawdown=drawdown,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_charts(stats: Stats, outdir: Path) -> dict[str, Path]:
    df = stats.df
    charts: dict[str, Path] = {}
    chart_dir = outdir / "charts"

    # 1. Cumulative winnings (nominal vs real)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["SessionNum"], df["CumulativeNominal"], label="Nominal", linewidth=2)
    ax.plot(df["SessionNum"], df["CumulativeReal"], label="Real", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Session #")
    ax.set_ylabel("Cumulative € won")
    ax.set_title("Cumulative winnings over time")
    ax.legend()
    ax.grid(alpha=0.3)
    charts["cumulative"] = _save(fig, chart_dir / "cumulative.png")

    # 2. Per-session outcome bars
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in df["Outcome"]]
    ax.bar(df["SessionNum"], df["Outcome"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Session #")
    ax.set_ylabel("€ won")
    ax.set_title("Per-session outcomes")
    ax.grid(alpha=0.3, axis="y")
    charts["per_session"] = _save(fig, chart_dir / "per_session.png")

    # 3. Outcome distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Outcome"], bins=15, color="#1f77b4", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(df["Outcome"].mean(), color="red", linestyle="--", label=f"Mean €{df['Outcome'].mean():.1f}")
    ax.set_xlabel("Session outcome (€)")
    ax.set_ylabel("# sessions")
    ax.set_title("Distribution of session outcomes")
    ax.legend()
    charts["histogram"] = _save(fig, chart_dir / "histogram.png")

    # 4. Outcomes in buy-ins
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["OutcomeInBuyins"].dropna(), bins=15, color="#9467bd", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Outcome in buy-ins")
    ax.set_ylabel("# sessions")
    ax.set_title("Distribution of buy-in-normalised outcomes")
    charts["buyin_hist"] = _save(fig, chart_dir / "buyin_histogram.png")

    # 5. By players bucket
    if not stats.by_players.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = stats.by_players.sort_values("nominal")
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in bp["nominal"]]
        ax.barh(bp.index, bp["nominal"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Net € won")
        ax.set_title("Winnings by game size")
        ax.grid(alpha=0.3, axis="x")
        charts["by_players"] = _save(fig, chart_dir / "by_players.png")

    # 6. By buy-in bucket
    if not stats.by_buyin.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        bb = stats.by_buyin.sort_values("nominal")
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in bb["nominal"]]
        ax.barh(bb.index, bb["nominal"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Net € won")
        ax.set_title("Winnings by buy-in size")
        ax.grid(alpha=0.3, axis="x")
        charts["by_buyin"] = _save(fig, chart_dir / "by_buyin.png")

    # 7. By people
    if not stats.by_people.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = stats.by_people.sort_values("nominal")
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in bp["nominal"]]
        ax.barh(bp.index, bp["nominal"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Net € won in sessions involving this person")
        ax.set_title("Winnings by opponent presence")
        ax.grid(alpha=0.3, axis="x")
        charts["by_people"] = _save(fig, chart_dir / "by_people.png")

    # 8. Rolling average (window=5 if enough sessions)
    if len(df) >= 5:
        window = min(5, max(3, len(df) // 4))
        roll = df["Outcome"].rolling(window=window).mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["SessionNum"], roll, color="#ff7f0e", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Session #")
        ax.set_ylabel(f"Rolling avg € (window {window})")
        ax.set_title(f"Recent form — {window}-session rolling average")
        ax.grid(alpha=0.3)
        charts["rolling"] = _save(fig, chart_dir / "rolling.png")

    # 9. Fish vs no-fish boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [
        df.loc[df["HasFish"], "Outcome"].values,
        df.loc[~df["HasFish"], "Outcome"].values,
    ]
    ax.boxplot(groups, labels=["With fish", "No fish noted"])
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("€ won")
    ax.set_title("Outcome distribution: fish vs no fish")
    ax.grid(alpha=0.3, axis="y")
    charts["fish_box"] = _save(fig, chart_dir / "fish_box.png")

    # 10. Feel score vs outcome scatter
    if (df["FeelScore"] != 0).any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["FeelScore"], df["Outcome"], alpha=0.7, s=60)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Feel score (notes sentiment)")
        ax.set_ylabel("€ outcome")
        ax.set_title("Does it feel how it pays?")
        ax.grid(alpha=0.3)
        charts["feel_scatter"] = _save(fig, chart_dir / "feel_scatter.png")

    return charts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}" />'


def _fmt_money(v: float) -> str:
    return f"€{v:,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v*100:.1f}%"


def _session_block(label: str, s: dict) -> str:
    notes = s.get("Notes") or "—"
    return (
        f"<div class='session'><strong>{label}:</strong> "
        f"session #{int(s['SessionNum'])} — <em>{s['Title']}</em><br>"
        f"Outcome: {_fmt_money(s['Outcome'])} "
        f"(real {_fmt_money(s['RealOutcome'])}, buy-in {_fmt_money(s['BuyIn'])})<br>"
        f"<span class='notes'>{notes}</span></div>"
    )


def render_html(stats: Stats, charts: dict[str, Path], outdir: Path) -> Path:
    t = stats.totals
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
           margin: 2em auto; color: #222; padding: 0 1em; }
    h1, h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }
    h2 { margin-top: 2.5em; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1em; margin: 1em 0; }
    .kpi { background: #f7f7f9; padding: 1em; border-radius: 8px;
           border-left: 4px solid #4a7eff; }
    .kpi .label { font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi .value { font-size: 1.6em; font-weight: 600; margin-top: 0.3em; }
    .kpi.good { border-left-color: #2ca02c; }
    .kpi.bad { border-left-color: #d62728; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; }
    th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid #eee; }
    th:first-child, td:first-child { text-align: left; }
    th { background: #f7f7f9; }
    img { max-width: 100%; height: auto; margin: 0.5em 0; }
    .session { background: #fafafa; padding: 0.8em 1em; border-radius: 6px; margin: 0.6em 0; }
    .notes { color: #555; font-size: 0.9em; }
    .twocol { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
    @media (max-width: 720px) { .twocol { grid-template-columns: 1fr; } }
    """

    def kpi(label: str, value: str, klass: str = "") -> str:
        return f"<div class='kpi {klass}'><div class='label'>{label}</div><div class='value'>{value}</div></div>"

    headline = "".join(
        [
            kpi("Sessions", str(t["sessions"])),
            kpi("Nominal won", _fmt_money(t["nominal_sum"]),
                "good" if t["nominal_sum"] >= 0 else "bad"),
            kpi("Real won", _fmt_money(t["real_sum"]),
                "good" if t["real_sum"] >= 0 else "bad"),
            kpi("Forgiven money", _fmt_money(t["forgiven_sum"]),
                "bad" if t["forgiven_sum"] > 0 else ""),
            kpi("Win rate", _fmt_pct(t["win_rate_nominal"])),
            kpi("ROI", f"{t['roi_pct']:.1f}%"),
            kpi("Avg / session", _fmt_money(t["avg_outcome"])),
            kpi("Volatility (σ)", _fmt_money(t["stdev_outcome"])),
        ]
    )

    streak_html = "".join(
        [
            kpi("Longest winning streak", f"{stats.streaks['longest_winning_streak']} sessions"),
            kpi("Longest losing streak", f"{stats.streaks['longest_losing_streak']} sessions"),
            kpi(
                "Current streak",
                f"{abs(stats.streaks['current_streak'])} sessions "
                f"{'winning' if stats.streaks['current_streak'] > 0 else 'losing' if stats.streaks['current_streak'] < 0 else ''}",
            ),
            kpi("Max drawdown", _fmt_money(stats.drawdown["max_drawdown"]), "bad"),
        ]
    )

    recent_html = "".join(
        kpi(
            f"Last {n}",
            f"{_fmt_money(stats.recent[f'last_{n}_nominal'])} ({_fmt_pct(stats.recent[f'last_{n}_winrate'])} win)",
            "good" if stats.recent[f"last_{n}_nominal"] >= 0 else "bad",
        )
        for n in (5, 10, 20)
        if f"last_{n}_nominal" in stats.recent
    )

    extremes_html = "".join(
        [
            _session_block("🏆 Best session (nominal)", stats.extremes["best_session"]),
            _session_block("💀 Worst session (nominal)", stats.extremes["worst_session"]),
            _session_block("🏆 Best session (real)", stats.extremes["best_real_session"]),
            _session_block("💀 Worst session (real)", stats.extremes["worst_real_session"]),
        ]
    )

    feel_html = ""
    if stats.feel:
        feel_html = "".join(
            [
                _session_block("😊 Best by feel (notes)", stats.feel["best_by_feel"]),
                _session_block("😣 Worst by feel (notes)", stats.feel["worst_by_feel"]),
            ]
        )

    def df_table(d: pd.DataFrame, money_cols=("nominal", "real", "avg"),
                 pct_cols=("win_rate",)) -> str:
        if d.empty:
            return "<p><em>No data.</em></p>"
        d = d.copy()
        for c in money_cols:
            if c in d.columns:
                d[c] = d[c].map(_fmt_money)
        for c in pct_cols:
            if c in d.columns:
                d[c] = d[c].map(_fmt_pct)
        if "avg_buyins" in d.columns:
            d["avg_buyins"] = d["avg_buyins"].map(lambda v: f"{v:.2f}x")
        return d.to_html()

    sessions_view = stats.df[
        ["SessionNum", "Title", "Outcome", "RealOutcome", "Forgiven",
         "BuyIn", "OutcomeInBuyins", "PlayersBucket", "BuyInBucket", "Notes"]
    ].copy()
    sessions_view["Outcome"] = sessions_view["Outcome"].map(_fmt_money)
    sessions_view["RealOutcome"] = sessions_view["RealOutcome"].map(_fmt_money)
    sessions_view["Forgiven"] = sessions_view["Forgiven"].map(_fmt_money)
    sessions_view["BuyIn"] = sessions_view["BuyIn"].map(_fmt_money)
    sessions_view["OutcomeInBuyins"] = sessions_view["OutcomeInBuyins"].map(
        lambda v: f"{v:.2f}x" if pd.notna(v) else ""
    )

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Poker analysis</title>
<style>{css}</style></head><body>
<h1>♠ Poker analysis report</h1>
<p>Generated from <code>{Path(outdir).name}</code>. {t['sessions']} sessions analysed.</p>

<h2>Headline numbers</h2>
<div class='grid'>{headline}</div>
<p><em>"Real" winnings adjust nominal outcomes for <code>Truly</code> annotations,
forgiven debts, and covered buy-ins as noted in the spreadsheet.</em></p>

<h2>Cumulative curve</h2>
{_img_tag(charts['cumulative'])}

<h2>Streaks & drawdown</h2>
<div class='grid'>{streak_html}</div>

<h2>Recent form</h2>
<div class='grid'>{recent_html}</div>
{_img_tag(charts['rolling']) if 'rolling' in charts else ''}

<h2>Per-session</h2>
{_img_tag(charts['per_session'])}
<div class='twocol'>
  <div>{_img_tag(charts['histogram'])}</div>
  <div>{_img_tag(charts['buyin_hist'])}</div>
</div>

<h2>By game size (# of players)</h2>
{_img_tag(charts['by_players']) if 'by_players' in charts else ''}
{df_table(stats.by_players)}

<h2>By buy-in size</h2>
{_img_tag(charts['by_buyin']) if 'by_buyin' in charts else ''}
{df_table(stats.by_buyin)}

<h2>By opponent</h2>
{_img_tag(charts['by_people']) if 'by_people' in charts else ''}
{df_table(stats.by_people)}

<h2>Fish at the table?</h2>
{_img_tag(charts['fish_box'])}
{df_table(stats.fish_split)}

<h2>Best & worst sessions</h2>
{extremes_html}

<h2>Best & worst by feel (note sentiment)</h2>
{feel_html if feel_html else '<p><em>Not enough notes to score.</em></p>'}
{_img_tag(charts['feel_scatter']) if 'feel_scatter' in charts else ''}

<h2>All sessions</h2>
{sessions_view.to_html(index=False)}

</body></html>
"""
    out = outdir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_console_summary(stats: Stats) -> None:
    t = stats.totals
    print("\n" + "=" * 60)
    print("POKER ANALYSIS SUMMARY".center(60))
    print("=" * 60)
    print(f"Sessions:              {t['sessions']}")
    print(f"Nominal winnings:      {_fmt_money(t['nominal_sum'])}")
    print(f"Real winnings:         {_fmt_money(t['real_sum'])}")
    print(f"Forgiven money:        {_fmt_money(t['forgiven_sum'])}")
    print(f"Total buy-ins played:  {_fmt_money(t['total_buyins_played'])}")
    print(f"ROI (nominal):         {t['roi_pct']:.1f}%")
    print(f"ROI (real):            {t['real_roi_pct']:.1f}%")
    print(f"Win rate:              {_fmt_pct(t['win_rate_nominal'])}")
    print(f"Avg / session:         {_fmt_money(t['avg_outcome'])}")
    print(f"Volatility (σ):        {_fmt_money(t['stdev_outcome'])}")
    print(f"Longest winning run:   {stats.streaks['longest_winning_streak']}")
    print(f"Longest losing run:    {stats.streaks['longest_losing_streak']}")
    print(f"Max drawdown:          {_fmt_money(stats.drawdown['max_drawdown'])}")
    for n in (5, 10, 20):
        key = f"last_{n}_nominal"
        if key in stats.recent:
            print(f"Last {n:>2} sessions:      "
                  f"{_fmt_money(stats.recent[key])} "
                  f"(real {_fmt_money(stats.recent[f'last_{n}_real'])})")
    best = stats.extremes["best_session"]
    worst = stats.extremes["worst_session"]
    print(f"\n🏆 Best:  #{int(best['SessionNum'])} {best['Title']} -> {_fmt_money(best['Outcome'])}")
    print(f"💀 Worst: #{int(worst['SessionNum'])} {worst['Title']} -> {_fmt_money(worst['Outcome'])}")
    print("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse a poker session log.")
    p.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).parent / "Poker.xlsx"),
        help="Path to the Poker .xlsx file (default: Poker.xlsx next to this script)",
    )
    p.add_argument("--outdir", default="poker_report",
                   help="Output directory for HTML + charts (default: poker_report)")
    args = p.parse_args()

    src = Path(args.path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_sessions(src)
    df = enrich(df)
    stats = compute_stats(df)

    # Save the cleaned table
    df.drop(columns=["PeopleMentioned"]).to_csv(outdir / "sessions_clean.csv", index=False)

    charts = make_charts(stats, outdir)
    html_path = render_html(stats, charts, outdir)

    print_console_summary(stats)
    print(f"\nReport written to: {html_path.resolve()}")
    print(f"Cleaned CSV:       {(outdir / 'sessions_clean.csv').resolve()}")


if __name__ == "__main__":
    main()
