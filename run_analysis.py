"""
Housing Prices & Labour Market Outcomes: UK Local Authority Panel Analysis
==========================================================================

Tests whether higher house prices cause:
  (1) lower real wages (wage/rent ratio)
  (2) lower net in-migration

Method: Two-way fixed effects panel regression on UK Local Authority data,
2013-2018, using ONS/VOA public sources.

Sources (all public):
  - ASHE earnings:        Nomis (ONS) - median weekly pay by LA (NM_99_1)
  - House prices:         ONS House Prices by Local Authority (dataset catalog)
  - Rents:                VOA Private Rental Market Statistics, annual files
  - Migration:            ONS Internal Migration matrices (LA-to-LA flows)
  - Population:           ONS mid-year population estimates 2011-2024

Usage:
    python run_analysis.py

Requires: pandas, numpy, scipy, matplotlib, requests, openpyxl
Optional but recommended: statsmodels, linearmodels (for cleaner output)

Note on data coverage:
  VOA rent data is available from FY 2013/14 onwards at LA level.
  Population estimates cover 2011 onwards in the bulk historical file.
  Analysis period is therefore 2013-2018 (6 years, ~300 LAs).
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR_START = 2013
YEAR_END = 2018

# Data source URLs (all public, no API key required).
# If any URL returns 404, the script will print the failure with instructions.
URLS = {
    # ASHE - Nomis NM_99_1, TYPE464 = all local authority districts (England & Wales)
    # sex=8 (all), item=2 (all employees), pay=1 (gross weekly pay), measures=20100 (value)
    "ashe": (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_99_1.data.csv"
        "?geography=TYPE464"
        "&date=2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
        "&sex=8&item=2&pay=1&measures=20100"
    ),

    # ONS House Prices by Local Authority - cantabular time-series CSV
    # Covers median/mean price for all property types, all years, all English+Welsh LAs
    "hpi": (
        "https://download.ons.gov.uk/downloads/datasets/"
        "house-prices-local-authority/editions/time-series/versions/10.csv"
    ),

    # VOA Private Rental Market Statistics - one file per financial year
    # Each covers April-March (or Oct-Sep for oldest). Year label = FY start.
    "rents": {
        2013: (
            "https://assets.publishing.service.gov.uk/media/"
            "5a7d605240f0b60aaa2940ce/141211_Publication_AllTables.xls"
        ),
        2014: (
            "https://assets.publishing.service.gov.uk/media/"
            "5a80c468e5274a2e8ab520b3/PRM_-_AllTables.xls"
        ),
        2015: (
            "https://assets.publishing.service.gov.uk/media/"
            "5a8164b940f0b62302697104/160519_Publication_AllTables.xls"
        ),
        2016: (
            "https://assets.publishing.service.gov.uk/media/"
            "5a74fd6ee5274a59fa7168ac/Publication_AllTables_22062017.xls"
        ),
        2017: (
            "https://assets.publishing.service.gov.uk/media/"
            "5b17a330ed915d2cb78ace15/Publication_AllTables_07062018.xls"
        ),
        2018: (
            "https://assets.publishing.service.gov.uk/media/"
            "5d0a115f40f0b6200f963acc/Publication_AllTables_200619.xls"
        ),
    },

    # ONS Internal Migration matrices (LA-to-LA), year ending June.
    # Filenames changed between releases: ZIPs for 2013-2016, XLSX for 2017+.
    "migration_urls": {
        2013: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2013/laandregionsquarematrices2013.zip"
        ),
        2014: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2014/laandregionsquarematrices2014.zip"
        ),
        2015: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2015/laandregionsquarematrices2015.zip"
        ),
        2016: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2016/laandregionsquarematrices2016.zip"
        ),
        2017: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2017/laandregionalsquarematrices2017.xlsx"
        ),
        2018: (
            "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
            "populationandmigration/migrationwithintheuk/datasets/"
            "matricesofinternalmigrationmovesbetweenlocalauthoritiesandregions"
            "includingthecountriesofwalesscotlandandnorthernireland/"
            "yearendingjune2018/laandregionalsquarematrices2018newboundaries.xlsx"
        ),
    },

    # ONS Mid-year population estimates 2011-2024 by LA
    "population": (
        "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/"
        "populationandmigration/populationestimates/datasets/"
        "populationestimatesforukenglandandwalesscotlandandnorthernireland/"
        "mid2011tomid2024/myebtablesuk20112024.xlsx"
    ),
}


# ----------------------------------------------------------------------------
# Step 1: Download
# ----------------------------------------------------------------------------

def download(url: str, dest: Path, label: str, max_retries: int = 4) -> bool:
    """Download url to dest with retry on 429. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 500:
        print(f"  [cached] {label}: {dest.name}")
        return True

    print(f"  [fetching] {label}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (academic research; public data)"},
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 500:
                print(f"    WARNING: only {len(data)} bytes received — likely an error page")
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


def download_all():
    """Download every dataset. Returns dict of {name: Path or None}."""
    print("\n=== STEP 1: DOWNLOADING DATA ===")
    results = {}

    # ASHE - all LAs, all years 2010-2019 in one request
    ok = download(URLS["ashe"], DATA_DIR / "ashe.csv", "ASHE earnings")
    results["ashe"] = DATA_DIR / "ashe.csv" if ok else None
    time.sleep(1)

    # HPI - ONS cantabular CSV (88 MB, covers all English+Welsh LAs)
    ok = download(URLS["hpi"], DATA_DIR / "hpi.csv", "House Price Index (LA)")
    results["hpi"] = DATA_DIR / "hpi.csv" if ok else None
    time.sleep(1)

    # Population estimates 2011-2024
    ok = download(URLS["population"], DATA_DIR / "population.xlsx", "Population estimates")
    results["population"] = DATA_DIR / "population.xlsx" if ok else None
    time.sleep(1)

    # Rents - one file per year from VOA
    rent_files = {}
    for year, url in sorted(URLS["rents"].items()):
        dest = DATA_DIR / f"rents_{year}.xls"
        if download(url, dest, f"Rents {year}"):
            rent_files[year] = dest
        time.sleep(1)
    results["rents"] = rent_files if rent_files else None

    # Migration - one file per year; format varies (ZIP 2013-2016, XLSX 2017-2018)
    migration_files = {}
    for year, url in sorted(URLS["migration_urls"].items()):
        ext = ".xlsx" if url.endswith(".xlsx") else ".zip"
        dest = DATA_DIR / f"migration_{year}{ext}"
        if download(url, dest, f"Migration {year}"):
            migration_files[year] = dest
        time.sleep(2)  # be polite to ONS servers
    results["migration"] = migration_files if migration_files else None

    return results


# ----------------------------------------------------------------------------
# Step 2: Parse each source into a tidy (la_code, year, value) frame
# ----------------------------------------------------------------------------

def parse_ashe(path: Path) -> pd.DataFrame:
    """ASHE Nomis CSV → tidy panel of median weekly pay by LA-year."""
    df = pd.read_csv(path)
    # Standard Nomis column names
    rename = {}
    cols_lower = {c.upper(): c for c in df.columns}
    for src, dst in [("GEOGRAPHY_CODE", "la_code"),
                     ("GEOGRAPHY_NAME", "la_name"),
                     ("DATE", "year"),
                     ("OBS_VALUE", "weekly_pay")]:
        if src in cols_lower:
            rename[cols_lower[src]] = dst
    df = df.rename(columns=rename)

    # Keep only the plain value rows (drop confidence intervals etc.)
    if "MEASURES_NAME" in df.columns:
        df = df[df["MEASURES_NAME"].str.contains("Value", case=False, na=False)]

    df["year"] = pd.to_numeric(
        df["year"].astype(str).str[:4], errors="coerce")
    df["weekly_pay"] = pd.to_numeric(df["weekly_pay"], errors="coerce")
    df = df.dropna(subset=["la_code", "year", "weekly_pay"])
    df["year"] = df["year"].astype(int)
    return df[["la_code", "la_name", "year", "weekly_pay"]] if "la_name" in df.columns \
        else df[["la_code", "year", "weekly_pay"]]


def parse_hpi(path: Path) -> pd.DataFrame:
    """ONS House Prices LA dataset → annual median price by LA.

    Handles:
      • New ONS cantabular CSV (house-prices-local-authority dataset)
      • Legacy XLS (HPISSA workbooks)
    """
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        return _parse_hpi_xls(path)
    return _parse_hpi_csv(path)


def _parse_hpi_csv(path: Path) -> pd.DataFrame:
    """Parse ONS cantabular CSV (house-prices-local-authority time-series)."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    col_lower = {c.lower(): c for c in df.columns}

    # Observation value column: V4_1 (cantabular convention)
    val_col = col_lower.get("v4_1") or next(
        (c for c in df.columns if c.upper().startswith("V4_")), None)

    # Geography: column whose values look like ONS LA codes (E06…/W06…)
    geo_col = col_lower.get("administrative-geography")
    if geo_col is None:
        for c in df.columns:
            sample = df[c].dropna().head(200)
            if sample.str.match(r"^[EW]\d{8}$").mean() > 0.3:
                geo_col = c
                break

    # Year: column with 4-digit year-like values
    year_col = col_lower.get("calendar-years")
    if year_col is None:
        for c in df.columns:
            sample = pd.to_numeric(df[c].dropna().head(200), errors="coerce")
            if sample.between(1990, 2030).mean() > 0.7:
                year_col = c
                break

    if not all([val_col, geo_col, year_col]):
        raise ValueError(
            f"Could not identify required columns in HPI CSV.\n"
            f"  Found: {list(df.columns)}\n"
            f"  val={val_col}, geo={geo_col}, year={year_col}"
        )

    # Build filter mask to keep median price for all-residential, all build statuses
    mask = pd.Series(True, index=df.index)

    ptype_col = col_lower.get("property-type")
    if ptype_col:
        vals = df[ptype_col].dropna().unique()
        # prefer "all-residential-property" > anything starting "all"
        target = (
            next((v for v in vals if "all" in str(v).lower()
                  and "residential" in str(v).lower()), None)
            or next((v for v in vals if str(v).lower().startswith("all")), None)
        )
        if target:
            mask &= df[ptype_col] == target

    bstatus_col = col_lower.get("build-status")
    if bstatus_col:
        vals = df[bstatus_col].dropna().unique()
        target = next((v for v in vals if str(v).lower().startswith("all")), None)
        if target:
            mask &= df[bstatus_col] == target

    stat_col = col_lower.get("house-sales-and-prices")
    if stat_col:
        vals = df[stat_col].dropna().unique()
        target = next(
            (v for v in vals
             if "median" in str(v).lower() and "price" in str(v).lower()),
            None,
        )
        if target:
            mask &= df[stat_col] == target

    df_f = df[mask].copy()
    if df_f.empty:
        print("    Warning: HPI filters matched 0 rows; using all rows")
        df_f = df.copy()

    df_f["house_price"] = pd.to_numeric(df_f[val_col], errors="coerce")
    df_f["la_code"] = df_f[geo_col].str.strip()
    df_f["year"] = pd.to_numeric(
        df_f[year_col].astype(str).str[:4], errors="coerce")

    annual = (
        df_f.dropna(subset=["la_code", "year", "house_price"])
            .groupby(["la_code", "year"], as_index=False)["house_price"]
            .mean()
    )
    annual["year"] = annual["year"].astype(int)
    return annual


def _parse_hpi_xls(path: Path) -> pd.DataFrame:
    """Parse legacy ONS HPI HPISSA workbooks (annual LA average price)."""
    xl = pd.ExcelFile(path)
    candidate_sheets = [s for s in xl.sheet_names
                        if "AP" in s.upper() or "AVERAGE" in s.upper()] \
                       or xl.sheet_names[:5]

    df = None
    for sheet in candidate_sheets:
        raw = pd.read_excel(path, sheet_name=sheet, skiprows=range(0, 6))
        for col in raw.columns:
            vals = raw[col].astype(str)
            if vals.str.match(r"^[EWS]\d{8}$").mean() > 0.5:
                raw = raw.rename(columns={col: "la_code"})
                df = raw
                break
        if df is not None:
            break
    if df is None:
        raise ValueError("Could not locate LA code column in HPI XLS")

    date_cols = []
    for c in df.columns:
        try:
            pd.to_datetime(c)
            date_cols.append(c)
        except (ValueError, TypeError):
            pass

    long = df.melt(id_vars=["la_code"], value_vars=date_cols,
                   var_name="date", value_name="house_price")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date", "house_price"])
    long["year"] = long["date"].dt.year
    annual = (long.groupby(["la_code", "year"], as_index=False)["house_price"]
                  .mean())
    return annual


def parse_rents(rent_files: dict) -> pd.DataFrame:
    """VOA PRMS files → median monthly rent by LA-year.

    Args:
        rent_files: dict mapping calendar year (int) to Path of XLS file.
    """
    pieces = []
    for year, path in sorted(rent_files.items()):
        chunk = _parse_one_rent_file(path, year)
        if not chunk.empty:
            pieces.append(chunk)
            print(f"    Rents {year}: {len(chunk)} LAs")
        else:
            print(f"    Rents {year}: no usable data found in {path.name}")

    if not pieces:
        raise ValueError("Could not parse any VOA rent files")
    return pd.concat(pieces, ignore_index=True)


def _parse_one_rent_file(path: Path, year: int) -> pd.DataFrame:
    """Extract median monthly rent by LA from a single VOA PRMS XLS file."""
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        print(f"      Could not open {path.name}: {exc}")
        return pd.DataFrame()

    candidates = []
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=None)
        except Exception:
            continue

        # Try different header row offsets (0-8) to find the real column headers
        for skip in range(0, 9):
            sub = df.iloc[skip:].reset_index(drop=True)
            sub.columns = sub.iloc[0].astype(str)
            sub = sub.iloc[1:].reset_index(drop=True)

            # Locate LA code column
            la_col = None
            for col in sub.columns:
                sample = sub[col].astype(str)
                if sample.str.match(r"^[EW]\d{8}$").mean() > 0.25:
                    la_col = col
                    break
            if la_col is None:
                continue

            # Locate median rent column
            median_col = None
            for col in sub.columns:
                if "median" in str(col).lower():
                    median_col = col
                    break
            if median_col is None:
                continue

            chunk = pd.DataFrame({
                "la_code": sub[la_col].astype(str).str.strip(),
                "year": year,
                "median_rent": pd.to_numeric(sub[median_col], errors="coerce"),
            }).dropna()
            chunk = chunk[chunk["la_code"].str.match(r"^[EW]\d{8}$")]

            if len(chunk) >= 10:
                candidates.append((len(chunk), chunk))
            break  # found valid header row for this sheet

    if not candidates:
        return pd.DataFrame()

    # Prefer the candidate with the most LAs (avoids regional-only sheets)
    _, best = max(candidates, key=lambda x: x[0])
    return best


def parse_migration(mig_files: dict) -> pd.DataFrame:
    """Migration files → annual in-/out-migration totals per LA.

    Handles ZIP+CSV (years 2013-2016) and XLSX (years 2017-2018).
    Both formats are square origin×destination matrices with LA codes
    as row and column labels.
    """
    import zipfile
    pieces = []
    for year, fpath in sorted(mig_files.items()):
        mat = None
        try:
            if fpath.suffix.lower() == ".zip":
                with zipfile.ZipFile(fpath) as zf:
                    csv_names = [n for n in zf.namelist()
                                 if n.lower().endswith(".csv")
                                 and "la" in n.lower()]
                    if not csv_names:
                        csv_names = [n for n in zf.namelist()
                                     if n.lower().endswith(".csv")]
                    if not csv_names:
                        print(f"    Migration {year}: no CSV inside ZIP")
                        continue
                    with zf.open(csv_names[0]) as f:
                        mat = pd.read_csv(f, index_col=0)
            else:
                mat = pd.read_excel(fpath, index_col=0, header=0)
        except Exception as exc:
            print(f"    Migration {year}: error reading file: {exc}")
            continue

        if mat is None or mat.empty:
            continue

        mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0)
        np.fill_diagonal(mat.values, 0)
        out_flow = mat.sum(axis=1)
        in_flow  = mat.sum(axis=0)
        sub = pd.DataFrame({
            "la_code": mat.index,
            "year": year,
            "inflow":  in_flow.reindex(mat.index).values,
            "outflow": out_flow.values,
        })
        pieces.append(sub)
        print(f"    Migration {year}: {len(sub)} LAs")

    if not pieces:
        return pd.DataFrame(columns=["la_code", "year", "inflow", "outflow"])
    return pd.concat(pieces, ignore_index=True)


def parse_population(path: Path) -> pd.DataFrame:
    """Population workbook → mid-year population by LA-year."""
    xl = pd.ExcelFile(path)
    target_sheet = None
    for s in xl.sheet_names:
        if "persons" in s.lower() or "mye" in s.lower():
            target_sheet = s
            break
    target_sheet = target_sheet or xl.sheet_names[0]

    # Try a range of skiprows; ONS files differ between releases
    df = None
    for skip in range(3, 12):
        raw = pd.read_excel(path, sheet_name=target_sheet, skiprows=range(0, skip))
        # Find LA code column
        la_col = None
        for col in raw.columns:
            sample = raw[col].dropna().astype(str)
            if sample.str.match(r"^[EWSK]\d{8}$").mean() > 0.4:
                la_col = col
                break
        if la_col is not None:
            df = raw
            break

    if df is None or la_col is None:
        raise ValueError("Could not find LA code column in population file")

    year_cols = [c for c in df.columns
                 if str(c).isdigit() and YEAR_START <= int(str(c)) <= YEAR_END + 2]
    if not year_cols:
        # Fall back: numeric columns in the right range
        year_cols = [c for c in df.columns
                     if str(c).isdigit() and 2010 <= int(str(c)) <= 2025]

    long = df.melt(id_vars=[la_col], value_vars=year_cols,
                   var_name="year", value_name="population")
    long = long.rename(columns={la_col: "la_code"})
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")
    long["population"] = pd.to_numeric(long["population"], errors="coerce")
    return long.dropna().query(f"{YEAR_START} <= year <= {YEAR_END + 2}")


# ----------------------------------------------------------------------------
# Step 3: Two-way fixed effects regression (within transformation)
# ----------------------------------------------------------------------------

def twfe_regression(df: pd.DataFrame, y_col: str, x_col: str,
                    unit_col: str = "la_code", time_col: str = "year"):
    """y = beta*x + alpha_unit + gamma_time + eps via within transform.

    Standard errors clustered at the unit (LA) level (Liang–Zeger CR1).
    Returns a result dict.
    """
    d = df[[y_col, x_col, unit_col, time_col]].dropna().copy()
    d = d.reset_index(drop=True)

    y_unit = d.groupby(unit_col)[y_col].transform("mean")
    y_time = d.groupby(time_col)[y_col].transform("mean")
    y_grand = d[y_col].mean()
    d["y_dm"] = d[y_col] - y_unit - y_time + y_grand

    x_unit = d.groupby(unit_col)[x_col].transform("mean")
    x_time = d.groupby(time_col)[x_col].transform("mean")
    x_grand = d[x_col].mean()
    d["x_dm"] = d[x_col] - x_unit - x_time + x_grand

    x = d["x_dm"].values
    y = d["y_dm"].values
    xx = np.dot(x, x)
    if xx == 0:
        return {"error": "no variation in x after demeaning"}
    beta = np.dot(x, y) / xx
    resid = y - beta * x

    n = len(d)
    n_units = d[unit_col].nunique()
    n_times = d[time_col].nunique()
    k = n_units + n_times - 1
    dof = max(n - k - 1, 1)

    # Cluster-robust (CR1) SE
    cluster_sum = 0.0
    for _, idx in d.groupby(unit_col).indices.items():
        cluster_sum += (x[idx] * resid[idx]).sum() ** 2
    g = n_units
    correction = (g / (g - 1)) * ((n - 1) / dof)
    var_beta = correction * cluster_sum / (xx ** 2)
    se = np.sqrt(var_beta)
    t = beta / se
    p = 2 * (1 - stats.t.cdf(abs(t), dof))

    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_within = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    crit = stats.t.ppf(0.975, dof)

    return {
        "outcome": y_col,
        "regressor": x_col,
        "beta": beta,
        "se_clustered": se,
        "t": t,
        "p_value": p,
        "ci_95_low": beta - crit * se,
        "ci_95_high": beta + crit * se,
        "n_obs": n,
        "n_units": n_units,
        "n_years": n_times,
        "r2_within": r2_within,
        "dof": dof,
    }


def print_result(r: dict, label: str):
    print(f"\n--- {label} ---")
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
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


# ----------------------------------------------------------------------------
# Step 4: Plots
# ----------------------------------------------------------------------------

def make_plots(panel: pd.DataFrame):
    """Two scatter plots: house prices vs each outcome (within-demean)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (y_col, ylabel) in zip(
        axes,
        [("log_wage_rent", "log(wage / rent)  [unit & year demeaned]"),
         ("net_mig_rate",  "net in-migration rate  [unit & year demeaned]")],
    ):
        d = panel[["log_house_price", y_col, "la_code", "year"]].dropna()
        for col in ["log_house_price", y_col]:
            um = d.groupby("la_code")[col].transform("mean")
            tm = d.groupby("year")[col].transform("mean")
            d = d.copy()
            d[col + "_dm"] = d[col] - um - tm + d[col].mean()
        ax.scatter(d["log_house_price_dm"], d[y_col + "_dm"],
                   alpha=0.3, s=8)
        x = d["log_house_price_dm"].values
        y = d[y_col + "_dm"].values
        slope = np.dot(x, y) / max(np.dot(x, x), 1e-12)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs, color="red", linewidth=2,
                label=f"slope = {slope:.3f}")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("log(house price)  [unit & year demeaned]")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.suptitle(
        f"Within-LA variation: house prices vs labour outcomes "
        f"({YEAR_START}–{YEAR_END})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_plots.png", dpi=120)
    plt.close(fig)
    print(f"\n  Saved {OUTPUT_DIR / 'scatter_plots.png'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("UK Housing Prices and Labour Market Outcomes - Panel Analysis")
    print(f"Analysis period: {YEAR_START}–{YEAR_END}")
    print("=" * 70)

    # ---- Download ----
    files = download_all()

    # Migration is optional: if it fails we skip the net-migration regression.
    REQUIRED = {"ashe", "hpi", "rents", "population"}
    missing = [k for k in REQUIRED if not files.get(k)]
    if missing:
        print(f"\nFailed to download required datasets: {missing}")
        print("See URLs/errors above. Download manually into ./data/ and re-run.")
        sys.exit(1)
    if not files.get("migration"):
        print("\n  Note: migration files unavailable — net-migration regression will be skipped.")

    # ---- Parse ----
    print("\n=== STEP 2: PARSING ===")
    ashe = parse_ashe(files["ashe"])
    print(f"  ASHE:       {len(ashe):,} rows, {ashe['la_code'].nunique()} LAs")

    hpi = parse_hpi(files["hpi"])
    print(f"  HPI:        {len(hpi):,} rows, {hpi['la_code'].nunique()} LAs")

    pop = parse_population(files["population"])
    print(f"  Population: {len(pop):,} rows")

    rents = parse_rents(files["rents"])
    print(f"  Rents:      {len(rents):,} rows total")

    if files["migration"]:
        mig = parse_migration(files["migration"])
        print(f"  Migration:  {len(mig):,} rows")
    else:
        print("  Migration:  no files downloaded — omitting net-migration regression")
        mig = None

    # ---- Merge ----
    print("\n=== STEP 3: MERGING INTO PANEL ===")
    panel = (ashe
             .merge(hpi,   on=["la_code", "year"], how="inner")
             .merge(rents, on=["la_code", "year"], how="inner")
             .merge(pop,   on=["la_code", "year"], how="inner"))
    if mig is not None and not mig.empty:
        panel = panel.merge(mig, on=["la_code", "year"], how="left")

    print(f"  Merged panel: {len(panel):,} obs, "
          f"{panel['la_code'].nunique()} LAs, "
          f"{panel['year'].nunique()} years "
          f"({panel['year'].min()}–{panel['year'].max()})")

    if len(panel) < 50:
        print("\n  ERROR: panel is too small after merging. "
              "Check that LA codes are consistent across datasets.")
        sys.exit(1)

    # ---- Construct variables ----
    panel["log_house_price"] = np.log(panel["house_price"])
    panel["log_wage_rent"] = np.log(
        (panel["weekly_pay"] * 4.33) / panel["median_rent"]
    )
    if "inflow" in panel.columns and "outflow" in panel.columns:
        panel["net_mig_rate"] = (
            (panel["inflow"] - panel["outflow"]) / panel["population"] * 1000
        )

    panel.to_csv(OUTPUT_DIR / "panel.csv", index=False)
    print(f"  Saved {OUTPUT_DIR / 'panel.csv'}")

    # ---- Regressions ----
    print("\n=== STEP 4: REGRESSIONS ===")
    r1 = twfe_regression(panel, "log_wage_rent", "log_house_price")
    print_result(r1, "REG 1: Real wages (wage/rent ratio)")

    results = [r1]

    if "net_mig_rate" in panel.columns:
        r2 = twfe_regression(panel, "net_mig_rate", "log_house_price")
        print_result(r2, "REG 2: Net in-migration rate")
        results.append(r2)
    else:
        print("\n  Skipping REG 2 (no migration data merged)")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "regression_results.csv", index=False)
    print(f"\n  Saved {OUTPUT_DIR / 'regression_results.csv'}")

    # ---- Plots ----
    print("\n=== STEP 5: PLOTS ===")
    make_plots(panel)

    print("\n" + "=" * 70)
    print("DONE. See ./output/ for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
