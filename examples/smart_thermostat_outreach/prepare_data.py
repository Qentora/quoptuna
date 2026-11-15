"""Build the connected-thermostat outreach dataset from RECS 2020 microdata.

CE application: a utility (or a thermostat maker running a utility's
bring-your-own-thermostat program) prioritises enrollment invitations. It has
no device inventory, so it scores households for the probability that they
already run an Internet-connected ("smart") thermostat and invites the top of
the list within a fixed offer budget. Fairness question: among households that
DO own one (potential participants), are lower-income owners invited as often
as higher-income owners (equal opportunity across income groups)?

Three disjoint parts, written to results/data/ next to this script:
  audit.csv        50% of households, stratified on income group x owner, so it
                   keeps the survey's prevalence; NWEIGHT applies. Scored after
                   the selection rules were frozen.
  calibration.csv  2,500 further households, same stratification; sets the
                   offer-budget score cutoff and the mitigation cutoffs.
  search.csv       2,000 households drawn from what is left, enriched to 30%
                   owners (all remaining lower-income owners), split into
                   Quoptuna train / val / test.

Everything the example produces lives under results/ (data, per-run trial
exports, Optuna databases, logs, summary.json).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RECS_URL = "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv"
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATA = RESULTS_DIR / "data"
RAW = DATA / "recs2020_public_v7.csv"

SEED = 2026
AUDIT_SHARE = 0.5
CALIBRATION_N = 2500
SEARCH_N = 2000
SEARCH_POS_SHARE = 0.30

# Twelve household features; income and race are deliberately NOT inputs.
FEATURES = [
    "housing_type",  # TYPEHUQ 1-5
    "year_built",  # YEARMADERANGE 1-9
    "tenure",  # KOWNRENT 1 own, 2 rent, 3 occupy without rent
    "square_feet",  # TOTSQFT_EN
    "household_size",  # NHSLDMEM
    "householder_age",  # HHAGE
    "education",  # EDUCATION 1-5
    "urbanicity",  # UATYP10 R < C < U
    "climate_zone",  # BA_climate, ordinal cold -> hot
    "smart_speakers",  # SMARTSPK (-2 no internet -> 0)
    "streaming_devices",  # INTSTREAM (-2 no TV -> 0)
    "smartphones",  # NUMSMPHONE
]
CLIMATE_ORDER = {
    "Subarctic": 0,
    "Very-Cold": 1,
    "Cold": 2,
    "Marine": 3,
    "Mixed-Dry": 4,
    "Mixed-Humid": 5,
    "Hot-Dry": 6,
    "Hot-Humid": 7,
}


def income_group(moneypy: pd.Series) -> pd.Series:
    """RECS MONEYPY brackets -> three groups: <$35k (1-9), $35k-$100k (10-14), >=$100k (15-16)."""
    return pd.cut(moneypy, [0, 9, 14, 16], labels=["lt35k", "35to100k", "ge100k"]).astype(str)


def load_raw() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    if not RAW.exists():
        urllib.request.urlretrieve(RECS_URL, RAW)  # noqa: S310 - fixed EIA URL
    return pd.read_csv(RAW)


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    # TYPETHERM: 3 smart/Internet-connected, 2 programmable, 1 manual,
    # 0 no thermostat, -2 not applicable (dropped: no heating or cooling).
    d = raw[raw["TYPETHERM"] >= 0].copy()
    out = pd.DataFrame(
        {
            "DOEID": d["DOEID"],
            "housing_type": d["TYPEHUQ"],
            "year_built": d["YEARMADERANGE"],
            "tenure": d["KOWNRENT"],
            "square_feet": d["TOTSQFT_EN"],
            "household_size": d["NHSLDMEM"],
            "householder_age": d["HHAGE"],
            "education": d["EDUCATION"],
            "urbanicity": d["UATYP10"].map({"R": 0, "C": 1, "U": 2}),
            "climate_zone": d["BA_climate"].map(CLIMATE_ORDER),
            "smart_speakers": d["SMARTSPK"].clip(lower=0),
            "streaming_devices": d["INTSTREAM"].clip(lower=0),
            "smartphones": d["NUMSMPHONE"],
            "income_group": income_group(d["MONEYPY"]),
            "income_bracket": d["MONEYPY"],
            "nweight": d["NWEIGHT"],
            "smart_thermostat": np.where(d["TYPETHERM"] == 3, "yes", "no"),
        }
    )
    if out[FEATURES].isna().any().any():
        msg = f"unmapped feature values: {out[FEATURES].isna().sum()[lambda s: s > 0].to_dict()}"
        raise ValueError(msg)
    return out.reset_index(drop=True)


def stratified_take(frame: pd.DataFrame, n: float, rng: np.random.Generator) -> pd.Index:
    """Proportional stratified sample on income group x owner (n rows, or a share if < 1)."""
    share = n if n < 1 else n / len(frame)
    picked = []
    for _, idx in frame.groupby(["income_group", "smart_thermostat"]).groups.items():
        idx = np.asarray(idx)
        picked.append(rng.choice(idx, round(share * len(idx)), replace=False))
    return pd.Index(np.sort(np.concatenate(picked)))


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    audit_idx = stratified_take(frame, AUDIT_SHARE, rng)
    pool = frame.drop(index=audit_idx)
    cal_idx = stratified_take(pool, CALIBRATION_N, rng)
    pool = pool.drop(index=cal_idx)

    # Enriched search sample: lower-income owners are rare (146 in the survey),
    # so every one left in the pool goes in; otherwise validation would hold a
    # handful and the in-search equal-opportunity gap would be noise.
    is_pos = pool["smart_thermostat"] == "yes"
    low_pos = pool.index[is_pos & (pool["income_group"] == "lt35k")].to_numpy()
    other_pos = pool.index[is_pos & (pool["income_group"] != "lt35k")].to_numpy()
    neg = pool.index[~is_pos].to_numpy()
    n_pos = round(SEARCH_N * SEARCH_POS_SHARE)
    take = np.concatenate(
        [
            low_pos,
            rng.choice(other_pos, n_pos - len(low_pos), replace=False),
            rng.choice(neg, SEARCH_N - n_pos, replace=False),
        ]
    )
    search = frame.loc[np.sort(take)].copy()
    # Train / val / test = 65 / 25 / 10, stratified on owner x income group. The
    # internal test split is only logged by Quoptuna (calibration and audit sets
    # do the real evaluation), so it is kept small to give validation, where the
    # in-search equal-opportunity gap is measured, as many owners as possible.
    strata = search["smart_thermostat"] + "|" + search["income_group"]
    role = pd.Series("train", index=search.index)
    for _, idx in search.groupby(strata).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_test = round(0.10 * len(idx))
        n_val = round(0.25 * len(idx))
        role[idx[:n_test]] = "test"
        role[idx[n_test : n_test + n_val]] = "val"
    search["split"] = role
    return search, frame.loc[cal_idx].copy(), frame.loc[audit_idx].copy()


def main() -> None:
    frame = build_frame(load_raw())
    search, calibration, audit = split(frame)
    parts = {"search": search, "calibration": calibration, "audit": audit}
    ids = [set(p["DOEID"]) for p in parts.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1 :]):
        raise RuntimeError("splits overlap")
    for name, part in parts.items():
        part.to_csv(DATA / f"{name}.csv", index=False)
        tab = pd.crosstab(part["income_group"], part["smart_thermostat"])
        print(f"{name}: n={len(part)} owners={int((part.smart_thermostat == 'yes').sum())}\n{tab}")
    print(pd.crosstab(search["split"], [search["income_group"], search["smart_thermostat"]]))


if __name__ == "__main__":
    main()
