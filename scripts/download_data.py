"""
Download Stata example datasets from the Stata-Press data repository.

Usage:
    uv run scripts/download_data.py auto
    uv run scripts/download_data.py auto nlswork
    uv run scripts/download_data.py --list

Supported dataset sources:
    - sysuse: Stata built-in datasets (auto, census, lifeexp, nlsw88, etc.)
    - webuse: Stata web datasets (nlswork, union, wages, etc.)

Datasets are saved to .local/data/
"""

import argparse
import urllib.request
from pathlib import Path

STATA_BASE_URL = "https://www.stata-press.com/data/r19"

SYSUSE_DATASETS = [
    "auto", "autornd", "bplong", "bpwide", "cancer", "census",
    "citytemp", "educ99gdp", "fruit", "gdppc", "lifeexp", "lnf",
    "nlsw88", "pop2000", "sp500", "states", "transpl", "tsltstck",
    "uslifeexp", "voter", "xmpl1", "xmpl2", "xmpl3",
]

WEBUSE_DATASETS = [
    "nlswork", "abdata", "air2", "auto", "consumption", "cancer",
    "catcathlab", "cigtax", "consump", "cps1", "drugtr", "educ99gdp",
    "gasoline", "grunfeld", "hbank", "hmda", "houseprice", "inelastic",
    "laborsup", "lifeexp", "margarin", "mus02psid92", "mus03cel",
    "mus04uibModal", "nhanes2", "nlswork", "oil", "panel101", "petris",
    "plow", "rdc", "rdc2", "rental", "robdata", "sbux", "ship", "smokes",
    "sp500", "stockmark", "surface", "texashs", "texhsgprc", "tofan",
    "traffic", "tslbond", "turan", "uganda", "union", "uslifeexp",
    "wages", "wpi1",
]

DATA_DIR = Path(__file__).resolve().parent.parent / ".local" / "data"


def download(name: str) -> Path | None:
    """Download a single dataset. Returns file path on success, None on failure."""
    filename = f"{name}.dta"
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    url = f"{STATA_BASE_URL}/{filename}"
    print(f"  Downloading {url} ...", end=" ")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        print("OK")
        return dest
    except Exception as e:
        print(f"Failed ({e})")
        if dest.exists():
            dest.unlink()
        return None


def list_datasets():
    """List all supported datasets."""
    all_datasets = sorted(set(SYSUSE_DATASETS + WEBUSE_DATASETS))
    print("Available datasets:")
    for name in all_datasets:
        marker = " [downloaded]" if (DATA_DIR / f"{name}.dta").exists() else ""
        print(f"  {name}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Download Stata example datasets")
    parser.add_argument("datasets", nargs="*", help="Dataset names (e.g., auto nlswork)")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.datasets:
        parser.print_help()
        return

    targets = sorted(set(SYSUSE_DATASETS + WEBUSE_DATASETS)) if "all" in args.datasets else args.datasets
    for name in targets:
        download(name)


if __name__ == "__main__":
    main()
