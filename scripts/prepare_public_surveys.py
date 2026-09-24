"""Build compact, reproducible public-survey datasets for the picker.

These artifacts intentionally differ from the full-source UCI bundle: their
source microdata is large, so each is a seeded, target-stratified extract with
transparent feature recoding. They are machine-learning benchmarks, not
weighted population estimates.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "quoptuna" / "datasets"
SAMPLE_ROWS = 2_000
RANDOM_STATE = 0
NTIA_URL = "https://www.ntia.gov/sites/default/files/data/2023-survey-data/nov23-cps-csv.zip"
ACS_URL = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_hus.zip"


def download_zip(url: str) -> zipfile.ZipFile:
    request = Request(url, headers={"User-Agent": "quoptuna dataset builder"})
    return zipfile.ZipFile(io.BytesIO(urlopen(request).read()))


def stratified_sample(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a seeded sample that retains the source target distribution."""
    shares = frame[target].value_counts(normalize=True)
    counts = {label: int(SAMPLE_ROWS * share) for label, share in shares.items()}
    remainder = SAMPLE_ROWS - sum(counts.values())
    fractional = lambda label: SAMPLE_ROWS * shares[label] - counts[label]
    for label in sorted(counts, key=fractional, reverse=True)[:remainder]:
        counts[label] += 1
    samples = [
        rows.sample(n=counts[label], random_state=RANDOM_STATE)
        for label, rows in frame.groupby(target)
    ]
    return pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def prepare_ntia() -> pd.DataFrame:
    """Build adult wearable-use classification data from NTIA CPS Nov 2023."""
    columns = [
        "prtage", "prpertyp", "pewearab", "hefaminc", "peeduca", "pesex", "ptdtrace",
        "pehspnon", "prdisflg", "gtmetsta",
    ]
    archive = download_zip(NTIA_URL)
    frame = pd.read_csv(archive.open("nov23-cps.csv"), usecols=columns, low_memory=False)
    # NTIA's public Stata script defines this universe as persons aged 15+ who
    # are not group-quarter residents. Codes 1/2 answer yes/no for wearables.
    frame = frame[
        (frame["prtage"] >= 15)
        & (frame["prpertyp"] != 3)
        & frame["pewearab"].isin([1, 2])
    ].copy()
    frame["target"] = frame.pop("pewearab").map({1: "uses_wearable", 2: "does_not_use_wearable"})
    frame["age_group"] = pd.cut(frame.pop("prtage"), [14, 24, 44, 64, float("inf")], labels=["15_24", "25_44", "45_64", "65_plus"])
    frame["income_bracket"] = pd.cut(frame.pop("hefaminc"), [0, 7, 11, 13, 14, 16], labels=["under_25k", "25k_49k", "50k_74k", "75k_99k", "100k_plus"])
    frame["education"] = pd.cut(frame.pop("peeduca"), [0, 38, 39, 42, float("inf")], labels=["no_diploma", "high_school", "some_college", "college_plus"])
    frame["sex"] = frame.pop("pesex").map({1: "male", 2: "female"})
    frame["race"] = frame.pop("ptdtrace").astype("string")
    frame.loc[frame.pop("pehspnon") == 1, "race"] = "hispanic"
    frame["disability"] = frame.pop("prdisflg").map({1: "disabled", 2: "not_disabled"}).fillna("unknown")
    frame["metro"] = frame.pop("gtmetsta").map({1: "metropolitan", 2: "non_metropolitan", 3: "unknown"})
    frame = frame.drop(columns="prpertyp")
    return stratified_sample(frame.dropna(), "target")


def prepare_acs() -> pd.DataFrame:
    """Build household paid-internet-access classification data from ACS 2023."""
    columns = [
        "ACCESSINET",
        "COMPOTHX",
        "SMARTPHONE",
        "TABLET",
        "HINCP",
        "HHLDRAGEP",
        "HHLDRRAC1P",
        "WGTP",
    ]
    archive = download_zip(ACS_URL)
    parts = []
    for member in (name for name in archive.namelist() if name.endswith(".csv")):
        for raw_chunk in pd.read_csv(archive.open(member), usecols=columns, chunksize=100_000):
            chunk = raw_chunk[raw_chunk["ACCESSINET"].isin([1, 2, 3])].copy()
            if not chunk.empty:
                parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True)
    frame["target"] = frame.pop("ACCESSINET").map(
        {1: "paid_internet", 2: "unpaid_internet", 3: "no_internet"}
    )
    for column in ["COMPOTHX", "SMARTPHONE", "TABLET"]:
        frame[column] = frame[column].map({1: "yes", 2: "no"})
    frame["income"] = frame.pop("HINCP").clip(lower=-60_000)
    frame["householder_age"] = frame.pop("HHLDRAGEP")
    frame["householder_race"] = frame.pop("HHLDRRAC1P").astype("string")
    frame = frame.drop(columns="WGTP")
    return stratified_sample(frame.dropna(), "target")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, prepare in (("ntia_2023_wearable_use", prepare_ntia), ("acs_2023_internet_access", prepare_acs)):
        frame = prepare()
        output = DATASETS_DIR / f"{name}.csv.gz"
        frame.to_csv(output, index=False)
        print(f"{name}: {len(frame)} rows, {len(frame.columns) - 1} features, {frame['target'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
