"""Build the full-source UCI datasets bundled by quoptuna.

Each output is a complete UCI source dataset after only documented schema cleanup:
IDs are removed, target columns are named ``target``, and source missing-value
markers are converted to deterministic per-column modes. No UCI dataset is
sampled. Run this script when refreshing the committed CSV artifacts.

Sources are the static UCI archive ZIPs listed in ``SOURCES``. UCI dataset
metadata defines the original feature and target meaning; this script only
normalizes files into the common picker format.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "quoptuna" / "datasets"
SOURCES = {
    15: "https://archive.ics.uci.edu/static/public/15/breast+cancer+wisconsin+original.zip",
    30: "https://archive.ics.uci.edu/static/public/30/contraceptive+method+choice.zip",
    53: "https://archive.ics.uci.edu/static/public/53/iris.zip",
    161: "https://archive.ics.uci.edu/static/public/161/mammographic+mass.zip",
    236: "https://archive.ics.uci.edu/static/public/236/seeds.zip",
    257: "https://archive.ics.uci.edu/static/public/257/user+knowledge+modeling.zip",
    267: "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
    357: "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
    523: "https://archive.ics.uci.edu/static/public/523/exasens.zip",
    545: "https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip",
    850: "https://archive.ics.uci.edu/static/public/850/raisin.zip",
}


def archive(dataset_id: int) -> zipfile.ZipFile:
    """Download one UCI source archive with an ordinary browser user agent."""
    request = Request(SOURCES[dataset_id], headers={"User-Agent": "quoptuna dataset builder"})
    return zipfile.ZipFile(io.BytesIO(urlopen(request).read()))


def read_csv(zf: zipfile.ZipFile, filename: str, **kwargs: object) -> pd.DataFrame:
    """Read a CSV-like member without extracting transient files."""
    return pd.read_csv(zf.open(filename), **kwargs)


def complete(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace source missing values with deterministic column modes."""
    frame = frame.replace("?", pd.NA)
    for column in frame.columns:
        if frame[column].isna().any():
            mode = frame[column].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"{column} has no observed values for imputation")
            frame[column] = frame[column].fillna(mode.iloc[0])
    if frame.isna().any().any():
        raise ValueError("output retains missing values")
    return frame


def parse_arff(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    """Parse UCI's simple numeric/categorical ARFF without another dependency."""
    lines = zf.read(filename).decode("utf-8").splitlines()
    attributes = [
        line.split(maxsplit=2)[1] for line in lines if line.lower().startswith("@attribute")
    ]
    data_start = next(i for i, line in enumerate(lines) if line.lower() == "@data") + 1
    return pd.read_csv(io.StringIO("\n".join(lines[data_start:])), names=attributes)


def load_dataset(dataset_id: int) -> pd.DataFrame:
    """Return the normalized full source dataset for one catalog id."""
    zf = archive(dataset_id)
    if dataset_id == 15:
        names = [
            "sample_id",
            "clump_thickness",
            "cell_size_uniformity",
            "cell_shape_uniformity",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitoses",
            "target",
        ]
        return complete(
            read_csv(zf, "breast-cancer-wisconsin.data", names=names).drop(columns="sample_id")
        )
    if dataset_id == 30:
        names = [
            "wife_age",
            "wife_education",
            "husband_education",
            "children",
            "wife_religion",
            "wife_working",
            "husband_occupation",
            "standard_of_living",
            "media_exposure",
            "target",
        ]
        return complete(read_csv(zf, "cmc.data", names=names))
    if dataset_id == 53:
        return complete(
            read_csv(
                zf,
                "iris.data",
                names=["sepal_length", "sepal_width", "petal_length", "petal_width", "target"],
            ).dropna()
        )
    if dataset_id == 161:
        # UCI documents BI-RADS as non-predictive, so it is intentionally excluded.
        names = ["birads", "age", "shape", "margin", "density", "target"]
        return complete(
            read_csv(zf, "mammographic_masses.data", names=names).drop(columns="birads")
        )
    if dataset_id == 236:
        names = [
            "area",
            "perimeter",
            "compactness",
            "kernel_length",
            "kernel_width",
            "asymmetry",
            "groove_length",
            "target",
        ]
        return complete(read_csv(zf, "seeds_dataset.txt", sep=r"\s+", names=names))
    if dataset_id == 257:
        sheets = pd.read_excel(
            io.BytesIO(zf.read("Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls")),
            sheet_name=["Training_Data", "Test_Data"],
        )
        frame = pd.concat(sheets.values(), ignore_index=True).iloc[:, :6]
        frame = frame.rename(columns={" UNS": "target", "UNS": "target"})
        frame["target"] = frame["target"].str.strip().str.lower().str.replace(" ", "_")
        return complete(frame)
    if dataset_id == 267:
        return complete(
            read_csv(
                zf,
                "data_banknote_authentication.txt",
                names=["variance", "skewness", "curtosis", "entropy", "target"],
            )
        )
    if dataset_id == 357:
        parts = [
            read_csv(zf, name).drop(columns=["date", '"date"'], errors="ignore")
            for name in ("datatraining.txt", "datatest.txt", "datatest2.txt")
        ]
        return complete(pd.concat(parts, ignore_index=True).rename(columns={"Occupancy": "target"}))
    if dataset_id == 523:
        raw = read_csv(zf, "Exasens.csv", skiprows=2)
        raw = raw.iloc[:, :8]
        raw.columns = [
            "target",
            "sample_id",
            "imaginary_min",
            "imaginary_avg",
            "real_min",
            "real_avg",
            "gender",
            "age",
        ]
        return complete(raw.drop(columns="sample_id"))
    if dataset_id == 545:
        return complete(
            parse_arff(zf, "Rice_Cammeo_Osmancik.arff").rename(columns={"Class": "target"})
        )
    if dataset_id == 850:
        nested = zipfile.ZipFile(io.BytesIO(zf.read("Raisin_Dataset.zip")))
        frame = pd.read_excel(
            nested.open("Raisin_Dataset/Raisin_Dataset.xlsx"),
            engine="openpyxl",
        )
        return complete(frame.rename(columns={"Class": "target"}))
    raise ValueError(f"Unsupported UCI dataset id: {dataset_id}")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_id in SOURCES:
        frame = load_dataset(dataset_id)
        output = DATASETS_DIR / f"uci_{dataset_id}.csv.gz"
        frame.to_csv(output, index=False)
        print(
            f"{dataset_id}: {len(frame)} rows, {len(frame.columns) - 1} features, {frame['target'].value_counts().to_dict()}"
        )


if __name__ == "__main__":
    main()
