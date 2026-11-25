"""Download the three files we need from the Building Data Genome 2 (BDG2) open dataset.

BDG2: Miller et al., "The Building Data Genome Project 2", Scientific Data (2020).
Hourly meter readings for 1,636 non-residential buildings, 2016-2017.
We use the CHILLED-WATER meters: the cooling energy delivered to each building's HVAC system.
"""
from pathlib import Path
import urllib.request

BASE = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {
    "chilledwater_cleaned.csv": f"{BASE}/meters/cleaned/chilledwater_cleaned.csv",
    "weather.csv": f"{BASE}/weather/weather.csv",
    "metadata.csv": f"{BASE}/metadata/metadata.csv",
}

def main():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = out / name
        if dest.exists():
            print(f"already have {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    print("done ->", out)

if __name__ == "__main__":
    main()
