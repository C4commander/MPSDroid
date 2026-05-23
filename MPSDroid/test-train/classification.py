import os
import json
import argparse
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier  # KNN

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
            raise ImportError("xgboost not installed, please choose another model or install xgboost")
        return xgb.XGBClassifier(
            random_state=seed,
            colsample_bytree=0.8,
            reg_lambda=1,
            n_jobs=-1,
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            eval_metric='logloss'
        )
    elif model_type == 'knn':
        return KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='minkowski',
            p=2
        )
    else:
        raise ValueError("model_type must be one of 'rf', 'et', 'gbdt', 'xgb', or 'knn'")


def determine_feature_columns(df_tr: pd.DataFrame, df_te: pd.DataFrame, quiet: bool = False):
    """
    Logic for selecting feature columns:
    - Assume the first column is sha256 (not a feature)
    - Assume the last column is label (not a feature)
    - All other columns are features
    Robust logic:
    - If label is not the last column, still exclude columns named label
    - If the first column is not sha256, still exclude the first column and print a warning
    """
    cols_tr = list(df_tr.columns)

    if "label" not in cols_tr:
        raise KeyError("label column is missing in training set")
    if "label" not in df_te.columns:
        raise KeyError("label column is missing in test set")

    if len(cols_tr) < 3:
        raise ValueError("Insufficient number of columns: at least sha256, one feature column, and label are needed")

    first_col = cols_tr[0]
    last_col = cols_tr[-1]

    if not quiet:
        if first_col.lower() != "sha256":
            print(f"[WARN] The first column is not named sha256 but '{first_col}', will exclude first column based on position")
        if last_col != "label":
            print(f"[WARN] The last column is not named label but '{last_col}', will still exclude the real label column by name")

    # Exclude first and last columns by position
    feat_cols = cols_tr[1:-1]

    # If label is not last, additionally exclude it
    if last_col != "label" and "label" in feat_cols:
        feat_cols = [c for c in feat_cols if c != "label"]

    # Also ensure sha256 does not get included by mistake
    feat_cols = [c for c in feat_cols if c.lower() != "sha256"]

    if not feat_cols:
        raise ValueError("No feature columns found (all removed after excluding first and last column).")

    if not quiet:
        print(f"[INFO] Number of feature columns: {len(feat_cols)}, first 5: {feat_cols[:5]}")
    return feat_cols


def eval_one_split(train_csv: str, test_csv: str, seed: int, model_type: str, quiet: bool = False):
    df_tr = pd.read_csv(train_csv, encoding="utf-8-sig")
    df_te = pd.read_csv(test_csv, encoding="utf-8-sig")

    if "label" not in df_tr.columns or "label" not in df_te.columns:
        raise KeyError(f"Missing label column: {train_csv} or {test_csv}")

    # Only keep labels 0/1
    df_tr = df_tr[df_tr["label"].isin([0, 1])].copy()
    df_te = df_te[df_te["label"].isin([0, 1])].copy()

    # Get feature columns
    feat_cols = determine_feature_columns(df_tr, df_te, quiet=quiet)

    # Align feature columns (add 0 for missing, drop extras)
    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = 0
    X_tr = df_tr[feat_cols].values
    y_tr = df_tr["label"].values.astype(int)
    X_te = df_te[feat_cols].values
    y_te = df_te["label"].values.astype(int)

    uniq_tr = np.unique(y_tr)
    if uniq_tr.size < 2:
        if not quiet:
            print(f"[WARN] Training set contains only one class {uniq_tr.tolist()}, cannot train a valid classifier")
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classifier on train/test distribution files in a specified directory (no cross-validation used)."
    )
    parser.add_argument(
        "--root-dir",
        default="./statistic",
        help="Root dir containing train/ and test/ subdirs; should contain train/file_cluster_distribution.csv and test/file_cluster_distribution.csv"
    )
    parser.add_argument(
        "--model-type",
        choices=["rf", "et", "gbdt", "xgb", "knn"],
        default="xgb",
        help="Type of classifier"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (effective for tree models, KNN has no randomness)"
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Evaluation result JSON output path (default: root-dir/model_eval_summary.json)"
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default=None,
        help="Evaluation result CSV output path (default: root-dir/model_eval_single.csv)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output"
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    train_csv = os.path.join(root_dir, "train", "file_cluster_distribution_merge.csv")
    test_csv = os.path.join(root_dir, "test", "file_cluster_distribution_merge.csv")

    if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
        raise FileNotFoundError(
            f"Train/test distribution CSV not found:\n  train: {train_csv}\n  test : {test_csv}"
        )

    if not args.quiet:
        print(f"[INFO] Using root dir: {root_dir}")
        print(f"[INFO] Train file: {train_csv}")
        print(f"[INFO] Test file: {test_csv}")
        print(f"[INFO] Model type: {args.model_type}, random_state={args.random_state}")

    metrics = eval_one_split(
        train_csv=train_csv,
        test_csv=test_csv,
        seed=args.random_state,
        model_type=args.model_type,
        quiet=args.quiet
    )

    per_split = []
    if metrics is None:
        status = "skipped_single_class_train"
        if not args.quiet:
            print("[WARN] Training set only contains a single class, this evaluation is skipped.")
        per_split.append({
            "root": os.path.basename(os.path.normpath(root_dir)),
            "status": status
        })
        avg = {}
    else:
        status = "ok"
        row = {"root": os.path.basename(os.path.normpath(root_dir)), "status": status}
        row.update(metrics)
        per_split.append(row)
        avg = metrics

        if not args.quiet:
            print("\nEvaluation:")
            print(
                f"ACC={row['ACC']:.4f}, "
                f"TPR={row['TPR']:.4f}, TNR={row['TNR']:.4f}, "
                f"FPR={row['FPR']:.4f}, FNR={row['FNR']:.4f}, "
                f"Precision={row['Precision']:.4f}, "
                f"Recall={row['Recall']:.4f}, "
                f"F1={row['F1']:.4f}"
            )

    if args.report_json is None:
        args.report_json = os.path.join(root_dir, "model_eval_summary.json")
    if args.report_csv is None:
        args.report_csv = os.path.join(root_dir, "model_eval_single.csv")

    # Write JSON
    try:
        summary = {
            "root_dir": root_dir,
            "model_type": args.model_type,
            "random_state": args.random_state,
            "result": per_split[0] if per_split else {},
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"Evaluation summary written to JSON: {args.report_json}")
    except Exception as e:
        print(f"[WARN] Failed to write JSON: {e}")

    # Write CSV (single row)
    try:
        ok_rows = [r for r in per_split if r.get("status") == "ok"]
        if ok_rows:
            pd.DataFrame(ok_rows).to_csv(args.report_csv, index=False, encoding="utf-8-sig")
            if not args.quiet:
                print(f"Evaluation results written to CSV: {args.report_csv}")
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