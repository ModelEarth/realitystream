"""
Pull recent forest canopy cover for every US county from Google Data Commons and write
the two files the Forest Canopy parameters use:

    forest-canopy-timeseries-<first>-<last>.csv   Fips, Name, <year>, <year>, ...   (percent forest cover)
    forest-canopy-target.csv                      Fips, Target                     (1 = growth >= threshold)

This replaces the one-off Forest Canopy colab described in github.com/ModelEarth/tree-canopy:
variable LandCoverFraction_Forest, relative growth between the first and last year of the
window, counties at or above the threshold labelled 1.

    python process/canopy/canopy-process.py                      # last 5 available years, 5 percent
    python process/canopy/canopy-process.py --years 4 --threshold 5 --out ../tree-canopy/input/targets

Needs DATACOMMONS_API_KEY (environment, OS credential store, or the .env named in
automation/paths.yaml; see api/keystore.md). Request a key at https://apikeys.datacommons.org/.
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "api"))
from run_models import require_env  # noqa: E402

VARIABLE = "LandCoverFraction_Forest"


def fetch_county_forest_cover(api_key):
    """Long table: Fips, Name, year, value for every US county and every year Data Commons has."""
    from datacommons_client import DataCommonsClient

    client = DataCommonsClient(api_key=api_key)
    obs = client.observations_dataframe(
        variable_dcids=[VARIABLE], date="all", parent_entity="country/USA", entity_type="County"
    )
    if obs is None or obs.empty:
        raise RuntimeError(f"Data Commons returned no {VARIABLE} observations for US counties.")
    obs = obs.rename(columns={"entity": "Fips", "entity_name": "Name"})
    obs["Fips"] = obs["Fips"].astype(str).str.replace("geoId/", "", regex=False).str.zfill(5)
    obs["year"] = obs["date"].astype(str).str[:4].astype(int)
    if "Name" not in obs.columns:
        obs["Name"] = ""
    obs["Name"] = obs["Name"].astype(str).str.replace(" County", "", regex=False)
    return obs[["Fips", "Name", "year", "value"]]


def build_outputs(obs, years, threshold):
    """(timeseries wide table, target table, first_year, last_year) from the long observations."""
    available = sorted(obs["year"].unique())
    window = available[-years:] if years else available
    first, last = window[0], window[-1]
    wide = obs[obs["year"].isin(window)].pivot_table(index=["Fips", "Name"], columns="year", values="value",
                                                     aggfunc="median").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    start, recent = wide[str(first)], wide[str(last)]
    growth = (recent - start) / start.replace(0, pd.NA) * 100
    target = pd.DataFrame({"Fips": wide["Fips"], "Target": (growth >= threshold).fillna(False).astype(int)})
    return wide, target, first, last


def main():
    parser = argparse.ArgumentParser(description="Forest canopy timeseries and target from Data Commons")
    parser.add_argument("--years", type=int, default=5, help="how many of the most recent years to keep (default 5)")
    parser.add_argument("--threshold", type=float, default=5.0, help="percent growth labelled 1 (default 5)")
    parser.add_argument("--out", default=os.path.join(HERE, "output"), help="output folder")
    args = parser.parse_args()

    obs = fetch_county_forest_cover(require_env("DATACOMMONS_API_KEY"))
    wide, target, first, last = build_outputs(obs, args.years, args.threshold)

    os.makedirs(args.out, exist_ok=True)
    ts_path = os.path.join(args.out, f"forest-canopy-timeseries-{first}-{last}.csv")
    target_path = os.path.join(args.out, "forest-canopy-target.csv")
    wide.to_csv(ts_path, index=False)
    target.to_csv(target_path, index=False)
    print(f"{len(wide)} counties, years {first} to {last}")
    print(f"{int(target['Target'].sum())} counties at or above {args.threshold:g} percent growth")
    print(f"wrote {ts_path}\nwrote {target_path}")


if __name__ == "__main__":
    main()
