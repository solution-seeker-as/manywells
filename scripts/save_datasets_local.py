"""
Save HuggingFace ManyWells datasets to ./data for local scripts.

Compatible with:
  - scripts/load_well_from_dataset.py
  - scripts/ml_examples/data_loader.py
"""
from pathlib import Path

import datasets

REPO = "solution-seeker-as/manywells"
DATA_ROOT = Path("./data")

# (folder under data/, HF data config name, HF config config name)
DATASETS = [
    ("manywells-sol", "manywells-sol-1", "manywells-sol-1-config"),
    ("manywells-nsol", "manywells-nsol-1", "manywells-nsol-1-config"),
    ("manywells-nscl", "manywells-nscl-1", "manywells-nscl-1-config"),
]


def save_one(version: str, data_name: str, config_name: str) -> None:
    dest = DATA_ROOT / version
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_name}...")
    data = datasets.load_dataset(REPO, name=data_name)
    df = data["train"].to_pandas()
    data_zip = dest / f"{data_name}.zip"
    df.to_csv(data_zip, index=False, compression="zip")
    print(f"  Wrote {data_zip} ({len(df):,} rows)")

    print(f"Loading {config_name}...")
    config = datasets.load_dataset(REPO, name=config_name)
    df_config = config["train"].to_pandas()
    config_zip = dest / f"{data_name}_config.zip"  # note: underscore, not hyphen
    df_config.to_csv(config_zip, index=False, compression="zip")
    print(f"  Wrote {config_zip} ({len(df_config):,} rows)")


if __name__ == "__main__":
    # Download all three datasets:
    for version, data_name, config_name in DATASETS:
        save_one(version, data_name, config_name)

    # Or only manywells-sol-1:
    # save_one("manywells-sol", "manywells-sol-1", "manywells-sol-1-config")

    print("Done.")