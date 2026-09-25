"""Build compact, reproducible fairness benchmarks for the offline picker.

Each artifact retains the documented demographic columns so they can be selected
as sensitive features. Samples are seeded and target-stratified; they are ML
benchmarks, not population-weighted estimates.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from ucimlrepo import fetch_ucirepo

from quoptuna.datasets import normalize_uci_targets

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "quoptuna" / "datasets"
SAMPLE_ROWS = 2_000
RANDOM_STATE = 0
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
ACS_CALIFORNIA_PERSON_URL = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pca.zip"
)


def stratified_sample(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a seeded sample that preserves source target prevalence."""
    shares = frame[target].value_counts(normalize=True)
    counts = {label: int(SAMPLE_ROWS * share) for label, share in shares.items()}
    remainder = SAMPLE_ROWS - sum(counts.values())

    def fractional(label: str) -> float:
        return SAMPLE_ROWS * shares[label] - counts[label]

    for label in sorted(counts, key=fractional, reverse=True)[:remainder]:
        counts[label] += 1
    samples = [
        rows.sample(n=counts[label], random_state=RANDOM_STATE)
        for label, rows in frame.groupby(target)
    ]
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def prepare_uci(dataset_id: int) -> pd.DataFrame:
    """Build a complete-case UCI fairness dataset with canonical labels."""
    dataset = fetch_ucirepo(id=dataset_id)
    targets = normalize_uci_targets(dataset_id, dataset.data.targets)
    return pd.concat([dataset.data.features, targets], axis=1).dropna().reset_index(drop=True)


def prepare_compas() -> pd.DataFrame:
    """Build the standard ProPublica two-year-recidivism analysis cohort."""
    request = Request(COMPAS_URL, headers={"User-Agent": "quoptuna dataset builder"})
    raw = pd.read_csv(io.BytesIO(urlopen(request).read()))
    cohort = raw[
        raw["days_b_screening_arrest"].between(-30, 30)
        & (raw["is_recid"] != -1)
        & (raw["c_charge_degree"] != "O")
        & (raw["score_text"] != "N/A")
    ]
    frame = cohort[
        [
            "age",
            "age_cat",
            "sex",
            "race",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            "c_charge_degree",
            "two_year_recid",
        ]
    ].rename(columns={"two_year_recid": "target"})
    frame["target"] = frame["target"].map({0: "no_recidivism", 1: "recidivism"})
    return stratified_sample(frame.dropna(), "target")


def acs_person_frame() -> pd.DataFrame:
    """Load the California 2023 ACS PUMS person extract used by ACS tasks."""
    request = Request(
        ACS_CALIFORNIA_PERSON_URL,
        headers={"User-Agent": "quoptuna dataset builder"},
    )
    archive = zipfile.ZipFile(io.BytesIO(urlopen(request).read()))
    member = next(
        name
        for name in archive.namelist()
        if name.lower().startswith("psam_p") and name.lower().endswith(".csv")
    )
    columns = [
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELSHIPP",
        "WKHP",
        "SEX",
        "RAC1P",
        "ESR",
        "JWMNP",
    ]
    return pd.read_csv(archive.open(member), usecols=columns, low_memory=False)


def prepare_acs_employment(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the Folktables-style employment task from ACS PUMS."""
    selected = frame[frame["AGEP"].between(16, 90)].copy()
    selected["target"] = selected.pop("ESR").eq(1).map({True: "employed", False: "not_employed"})
    selected = selected.drop(columns="JWMNP")
    return stratified_sample(selected.dropna(), "target")


def prepare_acs_travel_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the Folktables-style commute-over-20-minutes task without leakage."""
    selected = frame[(frame["AGEP"] >= 16) & frame["JWMNP"].notna()].copy()
    selected["target"] = (
        selected.pop("JWMNP").gt(20).map({True: "over_20_minutes", False: "20_minutes_or_less"})
    )
    selected = selected.drop(columns="ESR")
    return stratified_sample(selected.dropna(), "target")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    acs = acs_person_frame()
    prepared = {
        "uci_2_adult_income": prepare_uci(2),
        "uci_144_german_credit": prepare_uci(144),
        "compas_two_year_recidivism": prepare_compas(),
        "acs_2023_california_employment": prepare_acs_employment(acs),
        "acs_2023_california_travel_time": prepare_acs_travel_time(acs),
    }
    for name, frame in prepared.items():
        output = DATASETS_DIR / f"{name}.csv.gz"
        frame.to_csv(output, index=False)
        target = "target" if "target" in frame else frame.columns[-1]
        print(
            f"{name}: {len(frame)} rows, {len(frame.columns) - 1} features, "
            f"{frame[target].value_counts().to_dict()}"
        )


if __name__ == "__main__":
    main()
