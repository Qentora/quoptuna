"""Build the bundled Wireless Indoor Localization CSV from the raw UCI file.

Source: UCI ML Repository, "Wireless Indoor Localization" (id 422). The raw
download is a headerless, tab-separated ``wifi_localization.txt`` with 2000 rows
and 8 columns: seven WiFi signal strengths plus the room label (1-4).

Usage::

    python scripts/prepare_wireless_indoor_localization.py path/to/wifi_localization.txt

Re-run this only when refreshing the committed CSV; the generated file lives at
``src/quoptuna/datasets/wireless_indoor_localization.csv``.
"""

import sys
from pathlib import Path

import pandas as pd

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "quoptuna"
    / "datasets"
    / "wireless_indoor_localization.csv"
)

# 8 attribute names: wifi from 1 to 7 and room
cols_label = []
for i in range(1, 8):
    cols_label.append("wifi_" + str(i))
cols_label.append("room")


def main(raw_path: Path, output: Path = OUTPUT) -> None:
    data = pd.read_csv(raw_path, sep="\t", names=cols_label, header=None)

    if list(data.columns) != cols_label:
        raise SystemExit(f"Unexpected columns: {list(data.columns)}")
    if data.isna().any().any():
        raise SystemExit("Raw file contains missing values; expected none.")
    if sorted(data["room"].unique()) != [1, 2, 3, 4]:
        raise SystemExit(f"Unexpected room labels: {sorted(data['room'].unique())}")

    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    print(f"Wrote {output} ({len(data)} rows, {len(data.columns)} columns)")
    print(data.head())
    print(data["room"].value_counts().sort_index())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
