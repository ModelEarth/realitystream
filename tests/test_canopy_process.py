"""Offline check of process/canopy/canopy-process.py (the Data Commons pull itself needs a key)."""
import importlib.util
import os

import pandas as pd

path = os.path.join(os.path.dirname(__file__), "..", "process", "canopy", "canopy-process.py")
spec = importlib.util.spec_from_file_location("canopy_process", path)
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def test_growth_window_and_threshold():
    rows = []
    for fips, name, cover in (("01001", "Autauga", [40, 41, 42, 43]), ("01003", "Baldwin", [50, 50, 50, 50]),
                               ("01005", "Barbour", [20, 20, 20, 19])):
        for year, value in zip((2016, 2017, 2018, 2019), cover):
            rows.append({"Fips": fips, "Name": name, "year": year, "value": value})
    obs = pd.DataFrame(rows)
    wide, target, first, last = cp.build_outputs(obs, years=3, threshold=4.0)
    assert (first, last) == (2017, 2019)
    assert list(wide.columns) == ["Fips", "Name", "2017", "2018", "2019"]
    assert dict(zip(target["Fips"], target["Target"])) == {"01001": 1, "01003": 0, "01005": 0}  # 41->43 is 4.9%
