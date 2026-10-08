"""Row-independence test for 01_data_cleaning.ipynb.

Run from the project folder (takes about a minute):

    .venv\\Scripts\\python tests\\test_row_independence.py

What it checks
    Cleaning must be row-local: adding or changing one inspection must not change any other row's cleaned values.
    The test executes the notebook's own code cells (the real implementation, not a copy of its logic) twice:
      1. on DATASET_2026.csv as delivered;
      2. on a modified copy in which, for a sample of restaurants, the LATEST inspection is changed and a new,
         LATER inspection is appended.
    Every untouched original row must come out identical in every cleaned column.

Safety
    Both runs read their dataset from a temporary folder and write the notebook's outputs (the save cell runs, behind
    the notebook's own readiness checks) to a temporary folder. The lookups are read from raw/ read-only. The project's
    raw/ and clean/ files are hashed before and after and must be unchanged.
    Only three explicit substitutions are made to the notebook code (dataset path, output folder, and the fixed
    35,000-row assertion); the test fails if any of them no longer matches exactly once.

Exit code 0 = pass, 1 = fail.
"""
import contextlib
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "01_data_cleaning.ipynb"
N_RESTAURANTS_RANDOM = 300  # on top of every restaurant whose earlier rows have a bad static value
SEED = 0

# Values written into the changed / added later inspections (all valid, so they would be usable donors)
NEW_VALUES = {
    "EMPLOYEE_COUNT": "77", "MEDIAN_EMPLOYEE_AGE": "45.5", "MEDIAN_EMPLOYEE_TENURE": "9.25",
    "RESTAURANT_LOCATION": "Changed Location", "RESTAURANT_CATEGORY": "Buffet", "CITY": "Henderson",
    "ZIP": "89052-1234", "ZIP5": "89052.0", "LATITUDE": "36.0", "LONGITUDE": "115.0",
    "LAT_LONG_RAW": "(36.0, 115.0)", "VIOLATIONS_RAW": "202,211,230", "NUMBER_OF_VIOLATIONS": "3.0",
    "CURRENT_GRADE": "B", "Comments": "tot dem 9 // 3 viol noted // grade b posted",
}


def file_hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*")) if p.is_file()}


def notebook_code(dataset_path, out_dir):
    """The notebook's code cells with the three verified substitutions applied."""
    cells = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"]
    joined = "\n".join(cells)
    substitutions = [
        ('RAW / "DATASET_2026.csv"', f'Path(r"{dataset_path}")'),
        ('OUT = ROOT / "clean"', f'OUT = Path(r"{out_dir}")'),
        ("assert len(df) == 35_000 and ", "assert "),
    ]
    for old, _ in substitutions:
        if joined.count(old) != 1:
            raise RuntimeError(f"notebook changed: expected {old!r} exactly once, found {joined.count(old)}")
    for old, new in substitutions:
        cells = [c.replace(old, new) for c in cells]
    return cells


def clean(raw_frame, workdir, tag):
    """Run the cleaning notebook's code on raw_frame; returns (cleaned df, all_passed, out_dir)."""
    dataset = workdir / f"{tag}.csv"
    out_dir = workdir / f"clean_{tag}"
    raw_frame.to_csv(dataset, index=False)
    ns = {"display": lambda *a, **k: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for src in notebook_code(dataset, out_dir):
            exec(compile(src, f"01_data_cleaning[{tag}]", "exec"), ns)
    return ns["df"], ns["all_passed"], out_dir


def missing(v):
    return v is None or (not isinstance(v, (list, tuple, str)) and bool(pd.isna(v)))


def same(x, y):
    if missing(x) or missing(y):
        return missing(x) and missing(y)
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        return isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and list(x) == list(y)
    return x == y


def differing_cells(a, b, rows):
    out = {}
    for col in a.columns:
        n = sum(not same(x, y) for x, y in zip(a.loc[rows, col].tolist(), b.loc[rows, col].tolist()))
        if n:
            out[col] = n
    return out


def perturb(raw):
    """Change the latest inspection of sampled restaurants and append a later inspection for each."""
    when = pd.to_datetime(raw["INSPECTION_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dated = raw.assign(_t=when).dropna(subset=["_t"])
    multi = dated.groupby("RESTAURANT_SERIAL_NUMBER").filter(lambda g: len(g) >= 2)
    bad_static = (~pd.to_numeric(raw["EMPLOYEE_COUNT"], errors="coerce").between(1, 500)
                  | raw["RESTAURANT_LOCATION"].str.strip().isin(["", "###", "--", "NULL", "N/A", "n/a", "TBD", "unknown"])
                  | raw["RESTAURANT_CATEGORY"].str.strip().isin(["", "0", "--", "NULL", "N/A", "n/a", "TBD", "unknown"]))
    targeted = sorted(set(raw.loc[bad_static, "RESTAURANT_SERIAL_NUMBER"]) & set(multi["RESTAURANT_SERIAL_NUMBER"]))
    others = sorted(set(multi["RESTAURANT_SERIAL_NUMBER"]) - set(targeted))
    chosen = targeted + list(np.random.default_rng(SEED).choice(others, N_RESTAURANTS_RANDOM, replace=False))
    latest = (multi[multi["RESTAURANT_SERIAL_NUMBER"].isin(chosen)]
              .sort_values("_t").groupby("RESTAURANT_SERIAL_NUMBER").tail(1))

    changed = raw.copy()
    for col, value in NEW_VALUES.items():
        changed.loc[latest.index, col] = value
    added = raw.loc[latest.index].copy()
    added["INSPECTION_TIME"] = (latest["_t"] + pd.Timedelta(days=30)).dt.strftime("%Y-%m-%d %H:%M:%S").values
    for col, value in NEW_VALUES.items():
        added[col] = value
    added["EMPLOYEE_COUNT"] = "12"
    return pd.concat([changed, added], ignore_index=True), latest.index, len(chosen), len(targeted), len(added)


def main():
    os.chdir(PROJECT)  # the notebook resolves raw/ relative to the working directory
    before = {"raw": file_hashes(PROJECT / "raw"), "clean": file_hashes(PROJECT / "clean")}
    raw = pd.read_csv(PROJECT / "raw" / "DATASET_2026.csv", dtype=str, keep_default_na=False)
    perturbed, changed_rows, n_chosen, n_targeted, n_added = perturb(raw)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        base, base_passed, base_out = clean(raw, work, "base")
        after, after_passed, after_out = clean(perturbed, work, "perturbed")
        base_saved = (base_out / "inspections_clean.pkl").exists()
        reference = pd.read_pickle(PROJECT / "clean" / "inspections_clean.pkl")

        untouched = raw.index.difference(changed_rows)
        diffs = differing_cells(base, after, untouched)
        matches_project = not differing_cells(base, reference, base.index) and base.shape == reference.shape

    unchanged_files = before == {"raw": file_hashes(PROJECT / "raw"), "clean": file_hashes(PROJECT / "clean")}

    print(f"restaurants perturbed: {n_chosen} ({n_targeted} targeted) | latest inspections changed: {len(changed_rows)} "
          f"| later inspections added: {n_added}")
    print(f"untouched rows compared: {len(untouched):,} x {base.shape[1]} columns")
    print("untouched rows whose cleaned values changed:", diffs if diffs else "none")
    print(f"notebook readiness checks passed: base run {base_passed}, perturbed run {after_passed} "
          f"(outputs written only to a temporary folder: {base_saved})")
    print(f"base run reproduces clean/inspections_clean.pkl exactly: {matches_project}")
    print(f"project raw/ and clean/ files unchanged by the test: {unchanged_files}")

    ok = not diffs and base_passed and after_passed and unchanged_files
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
