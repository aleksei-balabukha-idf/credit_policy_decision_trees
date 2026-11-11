# Interactive Decision Tree (PyQt6)

Desktop app for bank risk analysts: load a CSV, build a decision tree
interactively, every split is logged and visualised.

## Run from source

```bash
pip install -r requirements.txt
cd src
python main.py
```

## Build the .exe

```bash
pip install pyinstaller
pyinstaller app.spec
# -> dist/DecisionTree.exe
```

## Workflow

1. **Open CSV…** — "Sniff & Preview" dialog: first 20 rows, guessed
   column types (# numeric, A categorical, 📅 date), NaN cells in red,
   mixed-type cells in amber. Read options (encoding, delimiter,
   decimal separator) can be changed and re-previewed.
2. **Tree settings** — pick the target, the features (with per-column
   type switch), the split metric (Gini / Entropy for classification,
   Variance-MSE for regression), min leaf size and max depth.
3. The **root node** appears with rows count and metric value.
   Every node has two buttons:
   - **split_best** — exhaustive search over all features (progress bar
     + Cancel for large data). The best split creates child nodes.
   - **split** — manual: cut-off for numerics, value subset for
     categoricals, or NaN-vs-filled. NaN routing: left / right /
     separate child / ignore. The dialog shows how many NaN rows the
     feature has.
4. **🔒 on a node** blocks features from split_best on that node.
5. **Navigation**: mouse wheel = zoom, drag = pan, toolbar Fit/Zoom.
6. **Export PNG…** saves the whole tree. **Save session…** writes a
   `.tree` file (ZIP: `session.json` + `data.csv.gz`) that can be
   reopened later.

Dates are converted to int YYYYMMDD (e.g. 20251229) and treated as numeric.

## Project layout

```
src/core/     — no Qt imports: csv_loader, dtypes, metrics, splitter, tree, session
src/ui/       — PyQt6 widgets: preview, canvas, node cards, dialogs, log panel
tests/        — unittest (also runs under pytest)
data/         — sample CSV
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Logging

- File: `~/.dtree_logs/dtree.log` (DEBUG+, rotating 5 MB) — load params,
  split candidates, NaN stats, tracebacks.
- In-app Log panel / stderr: INFO+ user-facing events.
