"""Build the bundled RECS 2020 housing-tenure CSV from the EIA public microdata.

Source: U.S. Energy Information Administration, Residential Energy Consumption
Survey (RECS) 2020 public use microdata, v7:

    https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv

Produced by a U.S. federal agency, so the data is public domain and
redistributable.

The raw file is 54 MB with 18,496 rows and 799 columns. That is neither
committable nor usable here: the kernel models build an n x n quantum kernel,
which at 18k rows is ~342M circuit evaluations. This script therefore derives a
compact, interpretable classification task and writes it in two sizes:

* **Target** ``tenure`` - ``owned`` vs ``rented``, from ``KOWNRENT``. The 182
  "occupied without payment of rent" households are dropped: they are neither,
  and a three-way split on 1% of rows would only add noise.
* **Features** - twelve household and dwelling characteristics, with the survey's
  numeric codes decoded to their codebook labels so SHAP plots read in English.
  ``MONEYPY`` stays an ordinal bracket index (1 = lowest income band,
  16 = highest) because its sixteen labels are long and genuinely ordered.
* **Full** (``recs2020_housing_tenure_full.csv.gz``) - every eligible household,
  18,314 rows. Fine for the classical models; expect the kernel-based quantum
  models to be impractical at this size.
* **Subset** (``recs2020_housing_tenure.csv.gz``) - a seeded, class-stratified
  sample of 2,000, for quantum runs and quick iteration. Stratified, *not*
  balanced: it preserves the population's ~73/27 owner/renter split, because
  rebalancing here would also rebalance the held-out test split and make the
  reported metrics describe a population that does not exist. Use the run's
  ``resampling`` option, which rebalances the training split only.

Both are gzipped: ``pandas`` infers the codec from the suffix, so every reader
is unchanged, and the pair costs ~240 KB instead of ~1.9 MB.

.. warning::

   These are **unweighted**. RECS ships ``NWEIGHT`` replicate weights that are
   required for nationally representative estimates; this is a machine-learning
   benchmark, not a survey estimate. Do not use it to make claims about U.S.
   housing.

Usage::

    python scripts/prepare_recs2020.py                      # downloads the raw CSV
    python scripts/prepare_recs2020.py path/to/recs2020_public_v7.csv

Re-run this only when refreshing the committed files; they live in
``src/quoptuna/datasets/``.
"""

import sys
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv"

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "quoptuna" / "datasets"
# Two artifacts: every eligible household, and a seeded stratified sample for
# quick runs. Gzipped because pandas reads it transparently from the suffix and
# it costs nothing at the call sites.
FULL_OUTPUT = DATASETS_DIR / "recs2020_housing_tenure_full.csv.gz"
SUBSET_OUTPUT = DATASETS_DIR / "recs2020_housing_tenure.csv.gz"

SAMPLE_ROWS = 2000
RANDOM_STATE = 0

TARGET_COLUMN = "tenure"
# KOWNRENT: 1 owned, 2 rented, 3 occupied without payment of rent (dropped).
TENURE_LABELS = {1: "owned", 2: "rented"}

# Decoded from the RECS 2020 codebook. Columns absent from this mapping are kept
# as their raw numeric value.
CODEBOOK = {
    "TYPEHUQ": {
        1: "Mobile home",
        2: "Single-family detached",
        3: "Single-family attached",
        4: "Apartment 2-4 units",
        5: "Apartment 5+ units",
    },
    "YEARMADERANGE": {
        1: "Before 1950",
        2: "1950-1959",
        3: "1960-1969",
        4: "1970-1979",
        5: "1980-1989",
        6: "1990-1999",
        7: "2000-2009",
        8: "2010-2015",
        9: "2016-2020",
    },
    "UATYP10": {"U": "Urban area", "C": "Urban cluster", "R": "Rural"},
    "EDUCATION": {
        1: "Less than high school",
        2: "High school or GED",
        3: "Some college",
        4: "Bachelor's degree",
        5: "Graduate degree",
    },
}

# source column -> output column
FEATURES = {
    "TYPEHUQ": "housing_type",
    "YEARMADERANGE": "year_built",
    "BEDROOMS": "bedrooms",
    "TOTROOMS": "total_rooms",
    "TOTSQFT_EN": "square_feet",
    "NHSLDMEM": "household_size",
    "HHAGE": "householder_age",
    "EDUCATION": "education",
    "MONEYPY": "income_bracket",
    "UATYP10": "urban_rural",
    "BA_climate": "climate_zone",
    "NUMFRIG": "refrigerators",
}


def main(raw_path: str = SOURCE_URL) -> None:
    raw = pd.read_csv(raw_path, low_memory=False)

    missing = [c for c in [*FEATURES, "KOWNRENT"] if c not in raw.columns]
    if missing:
        raise SystemExit(f"Source file is missing expected columns: {missing}")

    data = raw[[*FEATURES, "KOWNRENT"]].copy()
    data = data[data["KOWNRENT"].isin(TENURE_LABELS)]
    data[TARGET_COLUMN] = data.pop("KOWNRENT").map(TENURE_LABELS)

    for column, labels in CODEBOOK.items():
        unknown = set(data[column].unique()) - set(labels)
        if unknown:
            raise SystemExit(f"{column} has codes outside the codebook: {sorted(unknown)}")
        data[column] = data[column].map(labels)

    # RECS encodes refusals/not-applicable as negative codes; none of the
    # selected columns should carry them, so fail rather than train on -2.
    numeric = data.select_dtypes("number")
    negative = numeric.columns[(numeric < 0).any()].tolist()
    if negative:
        raise SystemExit(f"Columns contain RECS missing-value codes: {negative}")
    if data.isna().any().any():
        raise SystemExit("Selected columns contain missing values; expected none.")

    data = data.rename(columns=FEATURES)[[*FEATURES.values(), TARGET_COLUMN]]

    # Stratified so the sample keeps the population's owner/renter split.
    shares = data[TARGET_COLUMN].value_counts(normalize=True)
    sampled = [
        rows.sample(n=round(SAMPLE_ROWS * shares[label]), random_state=RANDOM_STATE)
        for label, rows in data.groupby(TARGET_COLUMN)
    ]
    subset = (
        pd.concat(sampled).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for frame, output in ((data.reset_index(drop=True), FULL_OUTPUT), (subset, SUBSET_OUTPUT)):
        frame.to_csv(output, index=False)
        size_kb = output.stat().st_size / 1024
        print(
            f"Wrote {output.name} ({len(frame)} rows, {len(frame.columns)} columns, "
            f"{size_kb:.0f} KB gzipped)"
        )
        print(frame[TARGET_COLUMN].value_counts().to_dict())


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit(__doc__)
    main(sys.argv[1] if len(sys.argv) == 2 else SOURCE_URL)
