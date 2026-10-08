# Tests

Run from the project folder with the project environment:

```
.venv\Scripts\python tests\test_row_independence.py
```

| Test | What it proves | Runtime |
|---|---|---|
| `test_row_independence.py` | Cleaning is row-local: changing a restaurant's latest inspection or adding a later one leaves every other row's cleaned values unchanged. Runs the real `01_data_cleaning.ipynb` code on temporary copies; never writes to `raw/` or `clean/` (checked by file hashes). Also reports whether the run reproduces `clean/inspections_clean.pkl` exactly. | ~35 s |

Exit code 0 means pass.
