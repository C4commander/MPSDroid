import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier  # Added: KNN

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def select_model(model_type: str, seed: int):
    if model_type == 'rf':
        return RandomForestClassifier(
            random_state=seed,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
    elif model_type == 'et':
        return ExtraTreesClassifier(
            random_state=seed,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
    elif model_type == 'gbdt':
        return GradientBoostingClassifier(
            random_state=seed,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0
        )
    elif model_type == 'xgb':
        if not HAS_XGB:
            raise ImportError("xgboost is not installed. Please select another model or install xgboost")
        return xgb.XGBClassifier(
            random_state=seed,
            colsample_bytree=0.8,  # 0.8
            reg_lambda=1,  # 0.8
            n_jobs=-1,
            n_estimators=300,  # 300
            learning_rate=0.08,  # 0.08
            max_depth=6,  # 6
            subsample=0.8,  # 0.8
            eval_metric='logloss'
        )
    elif model_type == 'knn':
        # Added: KNN model, this is a common default configuration.
        # If you have custom requirements, set n_neighbors, weights, etc. as needed.
        return KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',   # or 'uniform'
            metric='minkowski',   # p=2 is Euclidean distance
            p=2
        )
    else:
        raise ValueError("model_type must be 'rf', 'et', 'gbdt', 'xgb', or 'knn'")


def nanmean_fmt(arr):
    arr = np.array(arr, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def determine_feature_columns(df_tr: pd.DataFrame, df_te: pd.DataFrame, quiet: bool = False):
    """
    Feature column logic:
    - Assume first column is sha256 (not a feature)
    - Assume last column is label (not a feature)
    - All other columns are used as features
    Robustness:
    - If label is not the last column, still exclude the "label" column by name
    - If the first column is not sha256, still exclude the first column by position and show a warning
    """
    cols_tr = list(df_tr.columns)

    if "label" not in cols_tr:
        raise KeyError("Label column missing in training set")
    if "label" not in df_te.columns:
        raise KeyError("Label column missing in test set")

    if len(cols_tr) < 3:
        raise ValueError("Not enough columns: Requires sha256, at least one feature, and label")

    first_col = cols_tr[0]
    last_col = cols_tr[-1]

    if not quiet:
        if first_col.lower() != "sha256":
            print(f"[WARN] The first column is not sha256 but '{first_col}', still excluding by position")
        if last_col != "label":
            print(f"[WARN] The last column is not label but '{last_col}', will also remove the true label column by name")

    # Exclude the first and last columns by position
    feat_cols = cols_tr[1:-1]

    # If label is not the last column, also exclude 'label'
    if last_col != "label" and "label" in feat_cols:
        feat_cols = [c for c in feat_cols if c != "label"]

    # Also ensure sha256 is not included in the features (if it appears in the middle)
    feat_cols = [c for c in feat_cols if c.lower() != "sha256"]

    if not feat_cols:
        raise ValueError("No feature columns found (all removed after excluding the first and last columns).")

    if not quiet:
        print(f"[INFO] Number of feature columns used: {len(feat_cols)}, first 5: {feat_cols[:5]}")
    return feat_cols


def eval_one_fold(train_csv: str, test_csv: str, seed: int, model_type: str, quiet: bool = False):
    df_tr = pd.read_csv(train_csv, encoding="utf-8-sig")
    df_te = pd.read_csv(test_csv, encoding="utf-8-sig")

    if "label" not in df_tr.columns or "label" not in df_te.columns:
        raise KeyError(f"Label column missing in: {train_csv} or {test_csv}")

    # Only keep labels 0/1
    df_tr = df_tr[df_tr["label"].isin([0, 1])].copy()
    df_te = df_te[df_te["label"].isin([0, 1])].copy()

    # Feature column logic
    feat_cols = determine_feature_columns(df_tr, df_te, quiet=quiet)

    # Align test set to fitted features: fill missing with zero, drop extra
    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = 0
    X_tr = df_tr[feat_cols].values
    y_tr = df_tr["label"].values.astype(int)
    # Only keep the sequence from training for test
    X_te = df_te[feat_cols].values
    y_te = df_te["label"].values.astype(int)

    # Skip if training set is single class
    uniq_tr = np.unique(y_tr)
    if uniq_tr.size < 2:
        if not quiet:
            print(f"[WARN] Training set contains only one class {uniq_tr.tolist()}, skipping this fold: {train_csv}")
        return None

    clf = select_model(model_type, seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        TN = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        FP = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        FN = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    else:
        TN, FP, FN, TP = cm.ravel()

    tpr = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    tnr = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan
    fnr = FN / (FN + TP) if (FN + TP) > 0 else np.nan
    acc = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    return {
        "ACC": float(acc), "TPR": float(tpr), "TNR": float(tnr), "FPR": float(fpr), "FNR": float(fnr),
        "Precision": float(precision), "Recall": float(recall), "F1": float(f1)
    }


def external_folds_evaluation(folds_root: str, seed: int, model_type: str, quiet: bool = False):
    folds = sorted([p for p in glob.glob(os.path.join(folds_root, "fold_*")) if os.path.isdir(p)])
    if not folds:
        raise FileNotFoundError(f"No fold directories found: {folds_root}/fold_*")

    if not quiet:
        print(f"Found {len(folds)} folds, starting external fold evaluation...")

    per_fold = []
    for fdir in folds:
        # train_csv = os.path.join(fdir, "train", "file_cluster_distribution_merge.csv")
        # test_csv = os.path.join(fdir, "test", "file_cluster_distribution_merge.csv")
        train_csv = os.path.join(fdir, "train", "file_cluster_distribution.csv")
        test_csv = os.path.join(fdir, "test", "file_cluster_distribution.csv")
        if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
            if not quiet:
                print(f"[WARN] Missing train/test distribution CSV, skipping this fold: {fdir}")
            continue

        metrics = eval_one_fold(train_csv, test_csv, seed, model_type, quiet=quiet)
        if metrics is None:
            per_fold.append({
                "fold": os.path.basename(fdir),
                "status": "skipped_single_class_train"
            })
            continue

        row = {"fold": os.path.basename(fdir), "status": "ok"}
        row.update(metrics)
        per_fold.append(row)

        if not quiet:
            print(
                f"{row['fold']}: "
                f"ACC={row['ACC']:.4f}, "
                f"TPR={row['TPR']:.4f}, TNR={row['TNR']:.4f}, FPR={row['FPR']:.4f}, FNR={row['FNR']:.4f}, "
                f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}"
            )

    ok_rows = [r for r in per_fold if r.get("status") == "ok"]
    avg = {}
    if ok_rows:
        for k in ["ACC", "TPR", "TNR", "FPR", "FNR", "Precision", "Recall", "F1"]:
            avg[k] = nanmean_fmt([r[k] for r in ok_rows])

    return per_fold, avg


def main():
    parser = argparse.ArgumentParser(
        description="External fold testing and averaging for each fold (features=all columns except first/last)."
    )
    parser.add_argument("--folds-root", default="/mnt/data2/wb2024/Methodology/MyWay2.0/fold_outputs-mc", help="Root directory containing subdirs fold_01, fold_02, etc.")
    parser.add_argument("--model-type", choices=["rf", "et", "gbdt", "xgb", "knn"], default="xgb", help="Classification model type") 
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (affects tree models, KNN is deterministic)")
    parser.add_argument("--report-json", type=str, default=None, help="Evaluation results JSON path (default: folds-root/model_eval_summary_merge.json)")
    parser.add_argument("--report-csv", type=str, default=None, help="Evaluation results per fold in CSV (default: folds-root/model_eval_per_fold_merge.csv)")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output")
    args = parser.parse_args()

    per_fold, avg = external_folds_evaluation(
        folds_root=args.folds_root,
        seed=args.random_state,
        model_type=args.model_type,
        quiet=args.quiet
    )

    if avg:
        print("\nAverage metrics:")
        print(
            f"F1={avg.get('F1', float('nan')):.4f}, "
            f"Precision={avg.get('Precision', float('nan')):.4f}, "
            f"Recall={avg.get('Recall', float('nan')):.4f}, "
            f"ACC={avg.get('ACC', float('nan')):.4f}, "
            f"TPR={avg.get('TPR', float('nan')):.4f}, "
            f"TNR={avg.get('TNR', float('nan')):.4f}, "
            f"FPR={avg.get('FPR', float('nan')):.4f}, "
            f"FNR={avg.get('FNR', float('nan')):.4f}"
        )
    else:
        print("\nNo valid folds for averaging (possibly all skipped).")

    folds_root = args.folds_root
    if args.report_json is None:
        args.report_json = os.path.join(folds_root, "model_eval_summary_merge.json")
    if args.report_csv is None:
        args.report_csv = os.path.join(folds_root, "model_eval_per_fold_merge.csv")

    try:
        summary = {
            "folds_root": folds_root,
            "model_type": args.model_type,
            "random_state": args.random_state,
            "per_fold": per_fold,
            "average": avg
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Evaluation summary JSON written: {args.report_json}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON: {e}")

    try:
        ok_rows = [r for r in per_fold if r.get("status") == "ok"]
        if ok_rows:
            pd.DataFrame(ok_rows).to_csv(args.report_csv, index=False, encoding="utf-8-sig")
            print(f"Per fold evaluation results written to CSV: {args.report_csv}")
    except Exception as e:
        print(f"[WARN] Failed to write CSV: {e}")


if __name__ == "__main__":
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")