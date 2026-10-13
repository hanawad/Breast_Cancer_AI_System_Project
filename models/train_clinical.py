"""
Train the clinical model (StandardScaler + RandomForestClassifier) on REAL data.

Reads : dataset/Breast_cancer_dataset.xlsx   (or .csv - auto-detected)
Writes: models/scaler.pkl          fitted StandardScaler
        models/clinical_model.pkl  trained RandomForestClassifier
        models/model_meta.json     feature order, metrics, importances, cleaning info

"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent          # .../models
ROOT = HERE.parent                              # project root
DATASET_DIR = ROOT / "dataset"

# Default feature list = the 30 numeric columns of the Wisconsin breast cancer dataset.
# The backend imports this name. The real feature list used for training is detected
# from the file at run time and saved into model_meta.json.
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]
LABEL = "label"

LABEL_CANDIDATES = ["diagnosis", "label", "target", "class", "outcome", "result", "malignant"]
ID_LIKE = {"id", "patient_id", "patientid", "sample_id", "sampleid", "index", "no", "serial"}
POSITIVE_TOKENS = {"m", "malignant", "1", "1.0", "yes", "y", "true", "cancer", "positive", "p"}
NEGATIVE_TOKENS = {"b", "benign", "0", "0.0", "no", "n", "false", "normal", "negative", "healthy"}


# Reading
def find_dataset(path=None) -> Path:
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        return p
    for name in ("Breast_cancer_dataset.xlsx", "Breast_cancer_dataset.csv"):
        p = DATASET_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No Breast_cancer_dataset.xlsx / .csv found in {DATASET_DIR}. "
        "Pass the file explicitly with --data."
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as exc:  # try next encoding
            last_err = exc
    raise ValueError(f"Could not read {path}: {last_err}")


# Cleaning
def detect_label_column(df: pd.DataFrame, hint=None) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    if hint:
        if hint in df.columns:
            return hint
        if hint.lower() in cols_lower:
            return cols_lower[hint.lower()]
        raise ValueError(f"--label '{hint}' not found. Columns: {list(df.columns)}")
    for cand in LABEL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    raise ValueError(
        "Could not detect the label column automatically. "
        f"Columns are: {list(df.columns)}. Use --label <column_name>."
    )


def encode_label(series: pd.Series):
    """Return (y as 0/1 ints, mapping dict). Positive class (1) = malignant."""
    s = series.astype(str).str.strip().str.lower()
    uniques = sorted(s.unique())
    if len(uniques) != 2:
        raise ValueError(f"Label must be binary, found {len(uniques)} values: {uniques}")

    mapping = {}
    for u in uniques:
        if u in POSITIVE_TOKENS:
            mapping[u] = 1
        elif u in NEGATIVE_TOKENS:
            mapping[u] = 0
    if len(mapping) != 2 or len(set(mapping.values())) != 2:
        # Unknown wording: fall back to alphabetical order and say so loudly.
        mapping = {uniques[0]: 0, uniques[1]: 1}
        print(f"[clinical] WARNING: unrecognised label values {uniques}; "
              f"assuming {uniques[0]!r}=0 and {uniques[1]!r}=1. Check this is correct!")
    return s.map(mapping).astype(int), mapping


def clean_dataframe(df: pd.DataFrame, label_hint=None):
    """Return (X dataframe, y series, info dict)."""
    info = {}
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    info["raw_shape"] = list(df.shape)

    # 1. Drop empty / unnamed junk columns (e.g. trailing comma -> 'Unnamed: 32')
    junk = [c for c in df.columns if c.lower().startswith("unnamed") or df[c].isna().all()]
    df = df.drop(columns=junk)
    info["dropped_empty_columns"] = junk

    # 2. Label
    label_col = detect_label_column(df, label_hint)
    info["label_column"] = label_col
    df = df.dropna(subset=[label_col])

    # 3. Drop ID-like columns (they would leak/overfit)
    id_cols = [c for c in df.columns if c.lower() in ID_LIKE and c != label_col]
    df = df.drop(columns=id_cols)
    info["dropped_id_columns"] = id_cols

    # 4. Duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["duplicates_removed"] = n_before - len(df)

    y, label_map = encode_label(df[label_col])
    info["label_mapping"] = label_map
    X = df.drop(columns=[label_col])

    # 5. Types: convert numeric-looking text (e.g. '12,5' or ' 3.2 ') to numbers
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            converted = pd.to_numeric(
                X[c].astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce"
            )
            if converted.notna().sum() >= 0.9 * X[c].notna().sum():
                X[c] = converted

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    info["numeric_columns"] = numeric_cols
    info["categorical_columns"] = categorical_cols

    # 6. Drop constant columns (no information)
    constant = [c for c in numeric_cols if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant)
    numeric_cols = [c for c in numeric_cols if c not in constant]
    info["dropped_constant_columns"] = constant

    # 7. Categorical: missing -> explicit 'missing' category, then one-hot encode
    for c in categorical_cols:
        X[c] = X[c].fillna("missing").astype(str).str.strip().replace({"nan": "missing", "": "missing"})
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, dtype=float)

    X = X.astype(float)
    info["missing_values_before_imputation"] = {
        c: int(v) for c, v in X.isna().sum().items() if v > 0
    }
    return X, y, info


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_save(
    output_dir=HERE,
    csv_path=None,
    n_samples: int = 0,        # kept for backwards compatibility (unused: no synthetic data)
    seed: int = 42,
    verbose: bool = True,
    label=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset(csv_path)
    raw = read_table(data_path)
    X, y, info = clean_dataframe(raw, label)
    feature_names = list(X.columns)

    if verbose:
        print(f"[clinical] file          : {data_path}")
        print(f"[clinical] raw shape     : {info['raw_shape']}")
        print(f"[clinical] label column  : {info['label_column']}  mapping={info['label_mapping']}")
        print(f"[clinical] dropped empty : {info['dropped_empty_columns']}")
        print(f"[clinical] dropped id    : {info['dropped_id_columns']}")
        print(f"[clinical] duplicates    : {info['duplicates_removed']}")
        print(f"[clinical] n features    : {len(feature_names)}")
        print(f"[clinical] class balance : { {int(k): int(v) for k, v in y.value_counts().sort_index().items()} }")
        if info["missing_values_before_imputation"]:
            print(f"[clinical] missing values: {info['missing_values_before_imputation']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Impute missing numeric values with the TRAIN median (no test leakage)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    def make_rf():
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    # Cross-validated estimate on the training portion (more reliable than one split)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(
        make_pipeline(StandardScaler(), make_rf()), X_train, y_train, cv=cv, scoring="roc_auc"
    )

    scaler = StandardScaler().fit(X_train)
    model = make_rf().fit(scaler.transform(X_train), y_train)

    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    report = classification_report(
        y_test, pred, target_names=["benign(0)", "malignant(1)"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, pred)

    joblib.dump(model, output_dir / "clinical_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    meta = {
        "features": feature_names,
        "source": f"real:{data_path.name}",
        "is_demo": False,
        "n_samples": int(len(X)),
        "label_column": info["label_column"],
        "label_mapping": info["label_mapping"],
        "roc_auc_test": auc,
        "accuracy_test": float(report["accuracy"]),
        "recall_malignant_test": float(report["malignant(1)"]["recall"]),
        "precision_malignant_test": float(report["malignant(1)"]["precision"]),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "confusion_matrix": cm.tolist(),
        "impute_medians": {k: float(v) for k, v in medians.items()},
        "cleaning": {
            k: info[k]
            for k in (
                "dropped_empty_columns", "dropped_id_columns", "dropped_constant_columns",
                "duplicates_removed", "categorical_columns",
            )
        },
        "feature_importances": {
            f: float(v) for f, v in zip(feature_names, model.feature_importances_)
        },
    }
    (output_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if verbose:
        print(f"\n[clinical] 5-fold CV ROC-AUC (train): {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
        print(f"[clinical] TEST accuracy: {report['accuracy']:.3f} | ROC-AUC: {auc:.3f}")
        print(f"[clinical] TEST malignant recall: {report['malignant(1)']['recall']:.3f} "
              f"precision: {report['malignant(1)']['precision']:.3f}")
        print(f"[clinical] confusion matrix [[TN FP],[FN TP]]:\n{cm}")
        top = sorted(meta["feature_importances"].items(), key=lambda kv: -kv[1])[:5]
        print("[clinical] top-5 features:", ", ".join(f"{k} ({v:.3f})" for k, v in top))
        print(f"[clinical] saved: clinical_model.pkl, scaler.pkl, model_meta.json -> {output_dir}")
    return meta


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Train the clinical Random Forest on real data")
    parser.add_argument("--data", default=None, help="Path to .xlsx/.csv (default: auto-detect in dataset/)")
    parser.add_argument("--label", default=None, help="Label column name (default: auto-detect)")
    parser.add_argument("--out", default=str(HERE), help="Output directory (default: models/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_save(args.out, args.data, seed=args.seed, label=args.label)


if __name__ == "__main__":
    main()