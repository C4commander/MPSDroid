# MPSDroid

MPSDroid is a **static Android malware detection** pipeline based on:
- **Call graph extraction** from APKs (`.apk -> .gexf`)
- **Graph centrality features** on sensitive APIs (from `APIChecker_dot.txt`)
- **Sensitive API sequence extraction** (from `APSAPI.txt`)
- **Laplace smoothing / entropy-style statistics** on sensitive API usage (train set)
- **Sequence merging** (merge similar sequences into sets)
- **Clustering** of sensitive API sets
- **Feature concatenation** (centrality + clustering distribution)
- **Classification & evaluation**

This repository provides **two execution modes** with the *same internal steps/files*, but **different orchestration code**:
1. `MPSDroid/10-fold/` — **10-fold cross validation**
2. `MPSDroid/test-train/` — **fixed train/test split** (train set + test set)

> If you only want a straightforward “train on train, test on test” run, start with **`MPSDroid/test-train/`**.

---

## Repository Structure

- `MPSDroid/test-train/`  
  Fixed train/test split workflow

- `MPSDroid/10-fold/`  
  10-fold workflow (fold-based outputs + per-fold evaluation + average metrics)

Each directory typically contains:
- `CallGraphExtraction.py`
- `FeatureExtraction.py`
- `gexfToSequences.py`
- `Laplace_smoothing.py` (Laplace smoothing / statistics)
- `sequences_merge.py`
- `clustering.py`
- `concat.py`
- `classification.py`
- `utils.py`
- `APIChecker_dot.txt`
- `APSAPI.txt` (present in `test-train`; 10-fold may use alternative lists depending on your setup)

---

## Requirements / Environment Setup

This project is **100% Python**.

### Install dependencies

Use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Tip: It is recommended to install in a virtual environment (venv/conda) to avoid dependency conflicts.

---

## Data Layout (Recommended)

Many scripts recursively search for `benign/` and `malware/` directories. A common layout is:

```text
DATASET_ROOT/
  benign/
    <app1>.apk
    <app2>.apk
    ...
  malware/
    <appX>.apk
    <appY>.apk
    ...
```

After call graph extraction you will have:

```text
GRAPH_ROOT/
  benign/
    <sha_or_name>.gexf
  malware/
    <sha_or_name>.gexf
```

After sequence extraction you will have:

```text
SEQUENCE_ROOT/
  benign/
    <sha_or_name>.txt
  malware/
    <sha_or_name>.txt
```

---

## Pipeline Overview (What Each Script Does)

### 1) CallGraphExtraction.py — APK → `.gexf`
Extracts call graphs from APKs and writes NetworkX GEXF files.

### 2) FeatureExtraction.py — `.gexf` → centrality features (CSV)
Reads `.gexf` / `.gexf.gz` graphs and computes centrality features for nodes that match the sensitive API list in `APIChecker_dot.txt`.

### 3) gexfToSequences.py — `.gexf` → sensitive API sequences (TXT)
Converts graph nodes to Java-like method names, contracts to sensitive APIs based on `APSAPI.txt`, and outputs sequences per APK.

### 4) Laplace_smoothing.py — training statistics (entropy/weights)
Computes training-set statistics for sensitive API usage (used to compute “maliciousness value” / weights for sequences).

### 5) sequences_merge.py — merge similar sequences
Merges similar sequences into sets (to reduce redundancy).

### 6) clustering.py — cluster sensitive API sets
Clusters the merged sensitive API sets and produces per-file cluster distribution features.

### 7) concat.py — concatenate features
Concatenates centrality features and cluster distribution features.

### 8) classification.py — train/evaluate classifier
Trains and evaluates a classifier and outputs metrics (ACC/Precision/Recall/F1, etc.).

---

## Mode A: Fixed Train/Test Split (`MPSDroid/test-train`)

> `run_all.py` has been removed. Please run each stage manually in order.

### Run step-by-step

```bash
cd MPSDroid/test-train

# 1) (Optional) extract call graphs from APKs
python CallGraphExtraction.py --help

# 2) extract centrality features from gexf graphs
python FeatureExtraction.py --help

# 3) extract sensitive API sequences from gexf graphs
python gexfToSequences.py --help

# 4) compute Laplace smoothing / statistics
python Laplace_smoothing.py --help

# 5) merge similar sequences
python sequences_merge.py --help

# 6) cluster
python clustering.py --help

# 7) concatenate features
# IMPORTANT: For test-train mode, concat.py needs to be run TWICE:
#   - once for the TRAIN dataset feature concatenation
#   - once for the TEST dataset feature concatenation
python concat.py --help

# 8) classify
python classification.py --help
```

> Notes:
> - Some scripts have default paths hard-coded for the author's environment. Use `--help` and update arguments/paths to match your local dataset.
> - `classification.py` in `test-train` expects a `./statistic/train/` and `./statistic/test/` layout by default (see its `--root-dir` argument).

---

## Mode B: 10-Fold Cross Validation (`MPSDroid/10-fold`)

The 10-fold mode uses fold-based outputs (e.g., `fold_01`, `fold_02`, …). The exact fold generation and expected directory layout may depend on your dataset preparation.

General usage pattern:

```bash
cd MPSDroid/10-fold

# run each stage as needed (see --help for paths)
python CallGraphExtraction.py --help
python FeatureExtraction.py --help
python gexfToSequences.py --help
python Laplace_smoothing.py --help
python sequences_merge.py --help
python clustering.py --help
python concat.py --help

# evaluate per-fold and average metrics
python classification.py --help
```

`classification.py` in `10-fold` evaluates across `fold_*` directories and computes average metrics across folds.

---

## Classification Models

Both modes support multiple models, including:
- Random Forest (`rf`)
- Extra Trees (`et`)
- Gradient Boosting (`gbdt`)
- XGBoost (`xgb`, optional dependency)
- KNN (`knn`)

Select via CLI arguments in `classification.py` (see `--help`).

---

## Reproducibility Tips

- Keep consistent sensitive API lists:
  - `APIChecker_dot.txt` for centrality features
  - `APSAPI.txt` for sequence extraction
- Ensure `benign/` and `malware/` directory naming is consistent (scripts infer labels from these directory names).
- If you compress graphs as `.gexf.gz`, `FeatureExtraction.py` is designed to handle both `.gexf` and `.gexf.gz`.

---

## License
