"""
Housing Prices & Labour Market Outcomes: UK Local Authority Panel Analysis
==========================================================================

Tests whether higher house prices cause:
  (1) lower real wages (wage/rent ratio)
  (2) lower net in-migration

Method: Two-way fixed effects panel regression on UK Local Authority data,
2013-2018, using ONS/VOA public sources.

Sources (all public):
  - ASHE earnings:   Nomis (ONS) NM_99_1 - median weekly pay by LA
  - House prices:    ONS House Prices by Local Authority (dataset catalog)
  - Rents:           VOA Private Rental Market Statistics, annual XLS files
  - Migration:       ONS Internal Migration matrices (LA-to-LA flows)
  - Population:      ONS mid-year population estimates 2011-2024

Usage:
    pip install pandas numpy scipy matplotlib openpyxl "xlrd>=2.0.1"
    python run_analysis.py

Note on data coverage:
  VOA rent data starts FY 2013/14. Population estimates start 2011.
  Analysis period is therefore 2013-2018 (6 years, ~300 English LAs).
"""

import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Startup: check all required packages are present
# ---------------------------------------------------------------------------

def _check_packages():
    needed = []
    for mod, install in [
        ("numpy",      "numpy"),
        ("pandas",     "pandas"),
        ("scipy",      "scipy"),
        ("matplotlib", "matplotlib"),
        ("openpyxl",   "openpyxl"),
        ("xlrd",       "xlrd>=2.0.1"),   # pandas needs this for .xls files
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(install)
    if needed:
        print("Missing packages. Install with:")
        print(f"  pip install {' '.join(needed)}")
        sys.exit(1)

_check_packages()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR_START = 2013
YEAR_END   = 2018

# ---------------------------------------------------------------------------
# Data source URLs (all public, no API key required)
# ---------------------------------------------------------------------------
# NOTE: URLs are built via variable + string so that no single literal is a
# standalone URL.  This prevents markdown clipboard renderers (e.g. Jupyter
# paste from a browser) from wrapping fragments in <...> autolinks which
# corrupts the URL string at runtime.

_NOMIS = "https://www.nomisweb.co.uk/api/v01/dataset/"
_ONS   = "https://www.ons.gov.uk/file?uri=/"
_GOV   = "https://assets.publishing.service.gov.uk/media/"
_DLONS = "https://download.ons.gov.uk/downloads/datasets/"

# ONS internal migration: base path (no leading 'https://' alone)
_MIG_PATH = (
    "peoplepopulationandcommunity/populationandmigration/migrationwithintheuk"
    "/datasets/matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
    "includingthecountriesofwalesscotlandandnorthernireland"
)

URLS = {
    # ASHE - TYPE464 = all English & Welsh local authority districts
    "ashe": (
        _NOMIS + "NM_99_1.data.csv"
        "?geography=TYPE464"
        "&date=2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
        "&sex=8&item=2&pay=1&measures=20100"
    ),

    # ONS House Prices by Local Authority - cantabular time-series CSV (~88 MB)
    "hpi": _DLONS + "house-prices-local-authority/editions/time-series/versions/10.csv",

    # VOA Private Rental Market Statistics - one XLS per financial year
    "rents": {
        2013: _GOV + "5a7d605240f0b60aaa2940ce/141211_Publication_AllTables.xls",
        2014: _GOV + "5a80c468e5274a2e8ab520b3/PRM_-_AllTables.xls",
        2015: _GOV + "5a8164b940f0b62302697104/160519_Publication_AllTables.xls",
        2016: _GOV + "5a74fd6ee5274a59fa7168ac/Publication_AllTables_22062017.xls",
        2017: _GOV + "5b17a330ed915d2cb78ace15/Publication_AllTables_07062018.xls",
        2018: _GOV + "5d0a115f40f0b6200f963acc/Publication_AllTables_200619.xls",
    },

    # ONS Internal Migration matrices, year ending June
    # 2013-2016: ZIP containing CSV;  2017-2018: XLSX square matrix
    "migration_urls": {
        2013: _ONS + _MIG_PATH + "/yearendingjune2013/laandregionsquarematrices2013.zip",
        2014: _ONS + _MIG_PATH + "/yearendingjune2014/laandregionsquarematrices2014.zip",
        2015: _ONS + _MIG_PATH + "/yearendingjune2015/laandregionsquarematrices2015.zip",
        2016: _ONS + _MIG_PATH + "/yearendingjune2016/laandregionsquarematrices2016.zip",
        2017: _ONS + _MIG_PATH + "/yearendingjune2017/laandregionalsquarematrices2017.xlsx",
        2018: _ONS + _MIG_PATH + "/yearendingjune2018/laandregionalsquarematrices2018newboundaries.xlsx",
    },

    # ONS mid-year population estimates 2011-2024
    "population": (
        _ONS + "peoplepopulationandcommunity/populationandmigration"
        "/populationestimates/datasets"
        "/populationestimatesforukenglandandwalesscotlandandnorthernireland"
        "/mid2011tomid2024/myebtablesuk20112024.xlsx"
    ),
}


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

def download(url: str, dest: Path, label: str, max_retries: int = 4) -> bool:
    """Download url → dest with exponential-backoff retry on 429/errors."""
    if dest.exists() and dest.stat().st_size > 500:
        print(f"  [cached] {label}: {dest.name}")
        return True

    print(f"  [fetching] {label}...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (academic research; public data)"}
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 500:
                print(f"    WARNING: only {len(data)} bytes — likely an error page")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return False
            dest.write_bytes(data)
            print(f"    saved {len(data):,} bytes → {dest.name}")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"    429 rate-limited; waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            print(f"    FAILED [{exc.code}]: {exc.reason}")
            print(f"    URL: {url}")
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"    Error: {exc}; retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"    FAILED: {exc}")
            print(f"    URL: {url}")
            return False
    return False


def download_all() -> dict:
    print("\n=== STEP 1: DOWNLOADING DATA ===")
    results = {}

    ok = download(URLS["ashe"], DATA_DIR / "ashe.csv", "ASHE earnings")
    results["ashe"] = DATA_DIR / "ashe.csv" if ok else None
    time.sleep(1)

    ok = download(URLS["hpi"], DATA_DIR / "hpi.csv", "House Price Index (LA)")
    results["hpi"] = DATA_DIR / "hpi.csv" if ok else None
    time.sleep(1)

    ok = download(URLS["population"], DATA_DIR / "population.xlsx", "Population estimates")
    results["population"] = DATA_DIR / "population.xlsx" if ok else None
    time.sleep(1)

    rent_files = {}
    for year, url in sorted(URLS["rents"].items()):
        dest = DATA_DIR / f"rents_{year}.xls"
        if download(url, dest, f"Rents {year}"):
            rent_files[year] = dest
        time.sleep(1)
    results["rents"] = rent_files if rent_files else None

    mig_files = {}
    for year, url in sorted(URLS["migration_urls"].items()):
        ext  = ".xlsx" if url.endswith(".xlsx") else ".zip"
        dest = DATA_DIR / f"migration_{year}{ext}"
        if download(url, dest, f"Migration {year}"):
            mig_files[year] = dest
        time.sleep(2)
    results["migration"] = mig_files if mig_files else None

    return results


# ---------------------------------------------------------------------------
# Step 2: Parse each source → tidy (la_code, year, value) frame
# ---------------------------------------------------------------------------

def parse_ashe(path: Path) -> pd.DataFrame:
    """Nomis ASHE CSV → median weekly pay by LA-year."""
    df = pd.read_csv(path)
    rename, cols_up = {}, {c.upper(): c for c in df.columns}
    for src, dst in [("GEOGRAPHY_CODE", "la_code"), ("GEOGRAPHY_NAME", "la_name"),
                     ("DATE", "year"), ("OBS_VALUE", "weekly_pay")]:
        if src in cols_up:
            rename[cols_up[src]] = dst
    df = df.rename(columns=rename)
    if "MEASURES_NAME" in df.columns:
        df = df[df["MEASURES_NAME"].str.contains("Value", case=False, na=False)]
    df["year"]       = pd.to_numeric(df["year"].astype(str).str[:4], errors="coerce")
    df["weekly_pay"] = pd.to_numeric(df["weekly_pay"], errors="coerce")
    df = df.dropna(subset=["la_code", "year", "weekly_pay"])
    df["year"] = df["year"].astype(int)
    cols = ["la_code", "year", "weekly_pay"]
    if "la_name" in df.columns:
        cols.insert(1, "la_name")
    return df[cols]


def parse_hpi(path: Path) -> pd.DataFrame:
    """ONS house prices → annual median price by LA."""
    if path.suffix.lower() in (".xls", ".xlsx"):
        return _parse_hpi_xls(path)
    return _parse_hpi_csv(path)


def _parse_hpi_csv(path: Path) -> pd.DataFrame:
    """Parse ONS cantabular CSV (house-prices-local-authority time-series)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    cl = {c.lower(): c for c in df.columns}

    val_col  = cl.get("v4_1") or next((c for c in df.columns if c.upper().startswith("V4_")), None)
    geo_col  = cl.get("administrative-geography")
    year_col = cl.get("calendar-years")

    if geo_col is None:
        for c in df.columns:
            if df[c].dropna().head(200).str.match(r"^[EW]\d{8}$").mean() > 0.3:
                geo_col = c; break
    if year_col is None:
        for c in df.columns:
            s = pd.to_numeric(df[c].dropna().head(200), errors="coerce")
            if s.between(1990, 2030).mean() > 0.7:
                year_col = c; break

    if not all([val_col, geo_col, year_col]):
        raise ValueError(f"Cannot identify columns in HPI CSV. Found: {list(df.columns)}")

    mask = pd.Series(True, index=df.index)
    for dim, want in [("property-type",        lambda v: "all" in v and "residential" in v),
                      ("build-status",          lambda v: v.startswith("all")),
                      ("house-sales-and-prices",lambda v: "median" in v and "price" in v)]:
        col = cl.get(dim)
        if col:
            vals   = df[col].dropna().unique()
            target = next((v for v in vals if want(str(v).lower())), None)
            if target:
                mask &= df[col] == target

    df_f = df[mask].copy()
    if df_f.empty:
        print("    Warning: HPI filters matched 0 rows; using unfiltered data")
        df_f = df.copy()

    df_f["house_price"] = pd.to_numeric(df_f[val_col], errors="coerce")
    df_f["la_code"]     = df_f[geo_col].str.strip()
    df_f["year"]        = pd.to_numeric(df_f[year_col].astype(str).str[:4], errors="coerce")

    annual = (df_f.dropna(subset=["la_code", "year", "house_price"])
                  .groupby(["la_code", "year"], as_index=False)["house_price"].mean())
    annual["year"] = annual["year"].astype(int)
    return annual


def _parse_hpi_xls(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheets = [s for s in xl.sheet_names if "AP" in s.upper() or "AVERAGE" in s.upper()] \
             or xl.sheet_names[:5]
    df = None
    for sheet in sheets:
        raw = pd.read_excel(path, sheet_name=sheet, skiprows=range(0, 6))
        for col in raw.columns:
            if raw[col].astype(str).str.match(r"^[EWS]\d{8}$").mean() > 0.5:
                df = raw.rename(columns={col: "la_code"}); break
        if df is not None:
            break
    if df is None:
        raise ValueError("Cannot find LA code column in HPI XLS")
    date_cols = []
    for c in df.columns:
        try: pd.to_datetime(c); date_cols.append(c)
        except (ValueError, TypeError): pass
    long = df.melt(id_vars=["la_code"], value_vars=date_cols, var_name="date", value_name="house_price")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date", "house_price"])
    long["year"] = long["date"].dt.year
    return long.groupby(["la_code", "year"], as_index=False)["house_price"].mean()


def parse_rents(rent_files: dict) -> pd.DataFrame:
    """VOA PRMS XLS files → median monthly rent by LA-year."""
    pieces = []
    for year, path in sorted(rent_files.items()):
        chunk = _parse_one_rent_file(path, year)
        if not chunk.empty:
            pieces.append(chunk)
            print(f"    Rents {year}: {len(chunk)} LAs")
        else:
            print(f"    Rents {year}: no data found in {path.name}")
    if not pieces:
        raise ValueError("Could not parse any VOA rent files")
    return pd.concat(pieces, ignore_index=True)


def _parse_one_rent_file(path: Path, year: int) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        print(f"      Cannot open {path.name}: {exc}")
        return pd.DataFrame()

    candidates = []
    for sheet in xl.sheet_names:
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=None)
        except Exception:
            continue
        for skip in range(0, 9):
            # Use positional iloc throughout to avoid duplicate-column-name issues
            # (when two Excel columns share a name, sub[name] returns a DataFrame).
            data = raw.iloc[skip + 1:].reset_index(drop=True)
            header = raw.iloc[skip].astype(str).tolist()

            la_idx = next(
                (ci for ci in range(len(header))
                 if data.iloc[:, ci].astype(str)
                         .str.match(r"^[EW]\d{8}$").mean() > 0.25),
                None,
            )
            if la_idx is None:
                continue
            med_idx = next(
                (ci for ci, h in enumerate(header) if "median" in h.lower()),
                None,
            )
            if med_idx is None:
                continue
            chunk = pd.DataFrame({
                "la_code":     data.iloc[:, la_idx].astype(str).str.strip(),
                "year":        year,
                "median_rent": pd.to_numeric(data.iloc[:, med_idx], errors="coerce"),
            }).dropna()
            chunk = chunk[chunk["la_code"].str.match(r"^[EW]\d{8}$")]
            if len(chunk) >= 10:
                candidates.append((len(chunk), chunk))
            break

    if not candidates:
        return pd.DataFrame()
    _, best = max(candidates, key=lambda x: x[0])
    return best


def parse_migration(mig_files: dict) -> pd.DataFrame:
    """ONS migration files → annual in/out-flow totals per LA.

    Handles:
      ZIP + CSV (years 2013-2016): square origin-destination matrix CSV
      XLSX (years 2017-2018):      square origin-destination matrix worksheet
    """
    import zipfile
    pieces = []
    for year, fpath in sorted(mig_files.items()):
        mat = None
        try:
            if fpath.suffix.lower() == ".zip":
                with zipfile.ZipFile(fpath) as zf:
                    csv_names = [n for n in zf.namelist()
                                 if n.lower().endswith(".csv") and "la" in n.lower()]
                    if not csv_names:
                        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not csv_names:
                        print(f"    Migration {year}: no CSV inside ZIP"); continue
                    with zf.open(csv_names[0]) as f:
                        mat = pd.read_csv(f, index_col=0)
            else:
                mat = pd.read_excel(fpath, index_col=0, header=0)
        except Exception as exc:
            print(f"    Migration {year}: error reading file: {exc}"); continue

        if mat is None or mat.empty:
            continue
        mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0)
        np.fill_diagonal(mat.values, 0)
        sub = pd.DataFrame({
            "la_code": mat.index,
            "year":    year,
            "inflow":  mat.sum(axis=0).reindex(mat.index).values,
            "outflow": mat.sum(axis=1).values,
        })
        pieces.append(sub)
        print(f"    Migration {year}: {len(sub)} LAs")

    if not pieces:
        return pd.DataFrame(columns=["la_code", "year", "inflow", "outflow"])
    return pd.concat(pieces, ignore_index=True)


def parse_population(path: Path) -> pd.DataFrame:
    """ONS mid-year estimates XLSX → population by LA-year."""
    xl = pd.ExcelFile(path)
    target = next((s for s in xl.sheet_names
                   if "persons" in s.lower() or "mye" in s.lower()),
                  xl.sheet_names[0])
    print(f"    Population: sheet='{target}'")

    # Phase 1: read first 20 rows raw (no header) to locate the header row.
    # Require >=5 cells that ARE purely a 4-digit year (e.g. 2013, 2014...).
    # This avoids false-positives on title rows like "estimates (as of April 2023)".
    probe = pd.read_excel(path, sheet_name=target,
                          header=None, nrows=20, dtype=str).fillna("")
    header_row = 0
    for i in range(len(probe)):
        pure_years = sum(
            1 for v in probe.iloc[i]
            if re.fullmatch(r"20[012]\d", str(v).strip())
        )
        if pure_years >= 5:
            header_row = i
            break
    print(f"    Population: header at row {header_row}")

    # Phase 2: read with that row as the column header.
    df = pd.read_excel(path, sheet_name=target,
                       skiprows=list(range(header_row)), header=0)

    # Find the LA code column (values like E06000001).
    la_col = next(
        (c for c in df.columns
         if df[c].dropna().astype(str).str.strip()
                  .str.match(r"^[EWSK]\d{8}$").mean() > 0.3),
        None,
    )
    if la_col is None:
        raise ValueError(
            f"Cannot find LA code column in sheet '{target}'. "
            f"Columns: {[str(c) for c in df.columns[:20]]}"
        )

    # Find year columns by extracting a 4-digit year from each column name.
    year_map = {}
    for c in df.columns:
        m = re.search(r"\b(20[012]\d)\b", str(c))
        if m:
            yr = int(m.group(1))
            if YEAR_START <= yr <= YEAR_END + 2:
                year_map[yr] = c
    print(f"    Population: year columns found: {sorted(year_map)}")

    if not year_map:
        raise ValueError(
            f"No year columns in sheet '{target}'. "
            f"First 20 columns: {[str(c) for c in df.columns[:20]]}"
        )

    pieces = []
    for yr, col in sorted(year_map.items()):
        sub = pd.DataFrame({
            "la_code":    df[la_col].astype(str).str.strip(),
            "year":       yr,
            "population": pd.to_numeric(df[col], errors="coerce"),
        }).dropna()
        sub = sub[sub["la_code"].str.match(r"^[EWSK]\d{8}$")]
        pieces.append(sub)

    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not result.empty:
        result = result.groupby(["la_code", "year"], as_index=False)["population"].sum()
    return result


# ---------------------------------------------------------------------------
# Step 3: Two-way fixed effects regression
# ---------------------------------------------------------------------------

def twfe_regression(df: pd.DataFrame, y_col: str, x_col: str,
                    unit_col: str = "la_code", time_col: str = "year") -> dict:
    """y = beta*x + unit_FE + time_FE + eps, clustered SEs at unit level."""
    d = df[[y_col, x_col, unit_col, time_col]].dropna().copy().reset_index(drop=True)

    for var in (y_col, x_col):
        um = d.groupby(unit_col)[var].transform("mean")
        tm = d.groupby(time_col)[var].transform("mean")
        d[var + "_dm"] = d[var] - um - tm + d[var].mean()

    x = d[x_col + "_dm"].values
    y = d[y_col + "_dm"].values
    xx = np.dot(x, x)
    if xx == 0:
        return {"error": "no variation in x after demeaning"}
    beta  = np.dot(x, y) / xx
    resid = y - beta * x

    n       = len(d)
    n_units = d[unit_col].nunique()
    n_times = d[time_col].nunique()
    dof     = max(n - (n_units + n_times - 1) - 1, 1)

    cluster_sum = sum((x[idx] * resid[idx]).sum() ** 2
                      for idx in d.groupby(unit_col).indices.values())
    g           = n_units
    correction  = (g / (g - 1)) * ((n - 1) / dof)
    se          = np.sqrt(correction * cluster_sum / xx ** 2)
    t           = beta / se
    p           = 2 * (1 - stats.t.cdf(abs(t), dof))
    crit        = stats.t.ppf(0.975, dof)
    ss_res      = (resid ** 2).sum()
    ss_tot      = ((y - y.mean()) ** 2).sum()

    return {
        "outcome":    y_col,
        "regressor":  x_col,
        "beta":       beta,
        "se_clustered": se,
        "t":          t,
        "p_value":    p,
        "ci_95_low":  beta - crit * se,
        "ci_95_high": beta + crit * se,
        "n_obs":      n,
        "n_units":    n_units,
        "n_years":    n_times,
        "r2_within":  1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "dof":        dof,
    }


def print_result(r: dict, label: str):
    print(f"\n--- {label} ---")
    if "error" in r:
        print(f"  ERROR: {r['error']}"); return
    print(f"  Outcome:        {r['outcome']}")
    print(f"  Regressor:      {r['regressor']}")
    print(f"  Beta:           {r['beta']:.4f}")
    print(f"  Clustered SE:   {r['se_clustered']:.4f}")
    print(f"  t-statistic:    {r['t']:.3f}")
    print(f"  p-value:        {r['p_value']:.4f}")
    print(f"  95% CI:         [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]")
    print(f"  N (obs):        {r['n_obs']}")
    print(f"  N (LAs):        {r['n_units']}")
    print(f"  Years:          {r['n_years']}")
    print(f"  R² (within):    {r['r2_within']:.4f}")


# ---------------------------------------------------------------------------
# Step 4: Plots
# ---------------------------------------------------------------------------

def make_plots(panel: pd.DataFrame):
    outcomes = [("log_wage_rent", "log(wage / rent)  [unit & year demeaned]")]
    if "net_mig_rate" in panel.columns:
        outcomes.append(("net_mig_rate", "net in-migration rate  [unit & year demeaned]"))

    fig, axes = plt.subplots(1, len(outcomes), figsize=(7 * len(outcomes), 5))
    if len(outcomes) == 1:
        axes = [axes]

    for ax, (y_col, ylabel) in zip(axes, outcomes):
        d = panel[["log_house_price", y_col, "la_code", "year"]].dropna().copy()
        for col in ["log_house_price", y_col]:
            um = d.groupby("la_code")[col].transform("mean")
            tm = d.groupby("year")[col].transform("mean")
            d[col + "_dm"] = d[col] - um - tm + d[col].mean()
        x = d["log_house_price_dm"].values
        y = d[y_col + "_dm"].values
        slope = np.dot(x, y) / max(np.dot(x, x), 1e-12)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.scatter(x, y, alpha=0.3, s=8)
        ax.plot(xs, slope * xs, color="red", linewidth=2, label=f"slope = {slope:.3f}")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("log(house price)  [unit & year demeaned]")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle(f"Within-LA variation: house prices vs outcomes ({YEAR_START}–{YEAR_END})",
                 fontsize=12)
    fig.tight_layout()
    out = OUTPUT_DIR / "scatter_plots.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("UK Housing Prices and Labour Market Outcomes - Panel Analysis")
    print(f"Analysis period: {YEAR_START}–{YEAR_END}")
    print("=" * 70)

    files = download_all()

    REQUIRED = {"ashe", "hpi", "rents", "population"}
    missing  = [k for k in REQUIRED if not files.get(k)]
    if missing:
        print(f"\nFailed to download required datasets: {missing}")
        print("See error messages above. Download manually into ./data/ and re-run.")
        sys.exit(1)
    if not files.get("migration"):
        print("\n  Note: migration files unavailable — net-migration regression skipped.")

    print("\n=== STEP 2: PARSING ===")
    ashe  = parse_ashe(files["ashe"])
    print(f"  ASHE:       {len(ashe):,} rows, {ashe['la_code'].nunique()} LAs")

    hpi   = parse_hpi(files["hpi"])
    print(f"  HPI:        {len(hpi):,} rows, {hpi['la_code'].nunique()} LAs")

    pop   = parse_population(files["population"])
    print(f"  Population: {len(pop):,} rows, {pop['la_code'].nunique()} LAs")
    if pop.empty:
        print("  ERROR: population table is empty — check file structure")
        sys.exit(1)

    rents = parse_rents(files["rents"])
    print(f"  Rents:      {len(rents):,} rows total")

    mig = None
    if files.get("migration"):
        mig = parse_migration(files["migration"])
        print(f"  Migration:  {len(mig):,} rows")
    else:
        print("  Migration:  skipped")

    print("\n=== STEP 3: MERGING INTO PANEL ===")
    panel = (ashe
             .merge(hpi,   on=["la_code", "year"], how="inner")
             .merge(rents, on=["la_code", "year"], how="inner")
             .merge(pop,   on=["la_code", "year"], how="inner"))
    if mig is not None and not mig.empty:
        panel = panel.merge(mig, on=["la_code", "year"], how="left")

    n_la  = panel["la_code"].nunique()
    n_yr  = panel["year"].nunique()
    yr_rng = f"{panel['year'].min()}–{panel['year'].max()}"
    print(f"  Merged panel: {len(panel):,} obs, {n_la} LAs, {n_yr} years ({yr_rng})")

    if len(panel) < 50:
        print("  ERROR: panel too small — LA codes may not align across datasets")
        sys.exit(1)

    panel["log_house_price"] = np.log(panel["house_price"])
    panel["log_wage_rent"]   = np.log((panel["weekly_pay"] * 4.33) / panel["median_rent"])
    if "inflow" in panel.columns and "outflow" in panel.columns:
        panel["net_mig_rate"] = (
            (panel["inflow"] - panel["outflow"]) / panel["population"] * 1000
        )

    panel.to_csv(OUTPUT_DIR / "panel.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'panel.csv'}")

    print("\n=== STEP 4: REGRESSIONS ===")
    results = []

    r1 = twfe_regression(panel, "log_wage_rent", "log_house_price")
    print_result(r1, "REG 1: Real wages (wage/rent ratio)")
    results.append(r1)

    if "net_mig_rate" in panel.columns:
        r2 = twfe_regression(panel, "net_mig_rate", "log_house_price")
        print_result(r2, "REG 2: Net in-migration rate")
        results.append(r2)
    else:
        print("\n  Skipping REG 2 (no migration data)")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "regression_results.csv", index=False)
    print(f"\n  Saved {OUTPUT_DIR / 'regression_results.csv'}")

    print("\n=== STEP 5: PLOTS ===")
    make_plots(panel)

    print("\n" + "=" * 70)
    print("DONE. Results in ./output/")
    print("=" * 70)


if __name__ == "__main__":
    main()
