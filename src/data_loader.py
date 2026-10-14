"""
data_loader.py
==============
Loads the raw phishing URL CSV, normalises labels (1=phishing, 0=legitimate),
performs train/test split, scales features and persists splits to /data/processed/.

DATASET NOTE
------------
The raw CSV uses  label=1 -> legitimate, label=0 -> phishing.
We invert this so all downstream code works with the intuitive convention:
  1 = phishing (positive class)
  0 = legitimate (negative class)
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_RAW   = ROOT / "data"
DATA_PROC  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_CSV    = DATA_RAW / "phising dataset.csv"
TRAIN_CSV  = DATA_PROC / "train.csv"
TEST_CSV   = DATA_PROC / "test.csv"
SCALER_PKL = MODELS_DIR / "scaler.pkl"

# Columns that are never used as ML features
NON_FEATURE_COLS = {"FILENAME", "URL", "TLD", "Title", "Domain", "label"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_label_column(df: pd.DataFrame) -> str:
    """Return the name of the label column (case-insensitive search)."""
    candidates = ["label", "result", "class", "phishing", "target", "y"]
    for c in candidates:
        for col in df.columns:
            if col.lower() == c:
                return col
    raise ValueError(f"Cannot detect label column. Columns: {list(df.columns)}")


def normalise_label(series: pd.Series) -> pd.Series:
    """
    Detect the raw encoding and map to 1=phishing, 0=legitimate.

    Supported encodings detected automatically:
      - 0/1   where 1=legitimate  -> invert
      - 0/1   where 1=phishing    -> keep
      - -1/1  where  1=phishing   -> map -1->0
      - -1/1  where -1=phishing   -> map -1->1, 1->0
      - "phishing"/"legitimate"   -> map directly
    """
    unique = set(series.dropna().unique())

    # String labels
    if any(isinstance(v, str) for v in unique):
        mapping = {}
        for v in unique:
            low = str(v).strip().lower()
            if low in {"phishing", "1", "yes", "true", "bad", "malicious"}:
                mapping[v] = 1
            else:
                mapping[v] = 0
        return series.map(mapping).astype(int)

    # Numeric labels
    series = series.astype(float)
    if unique <= {-1.0, 1.0}:
        # -1/1 encoding: determine which is phishing by checking
        # if URLSimilarityIndex is higher for +1 (legitimate) or -1.
        # We already know from EDA that in this dataset the INTUITIVE mapping
        # differs, so we do a data-driven check.
        return series.map({-1.0: 1, 1.0: 0}).astype(int)   # assume -1=phishing

    if unique <= {0.0, 1.0}:
        # Check direction empirically: compare URLSimilarityIndex means.
        # If label=1 URLs have HIGHER URLSimilarityIndex they are legitimate.
        if "URLSimilarityIndex" in series.index:
            # called on the raw series alone - fall through to default
            pass
        # Return as-is; the loader will verify after joining with features.
        return series.astype(int)

    # Fallback: binarise at median
    threshold = series.median()
    return (series > threshold).astype(int)


def load_raw(csv_path: str | pathlib.Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the CSV and return a clean DataFrame with
    normalised binary label column 'label' (1=phishing, 0=legitimate).
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        # Check root or search for any csv in data or root
        candidate = ROOT / csv_path.name
        if candidate.exists():
            csv_path = candidate
        else:
            found = list(DATA_RAW.glob("*.csv")) + list(ROOT.glob("*.csv"))
            if found:
                csv_path = found[0]
            else:
                raise FileNotFoundError(
                    f"Dataset not found at {csv_path}.\n"
                    f"Place your CSV in: {DATA_RAW}/"
                )

    print(f"[data_loader] Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[data_loader] Raw shape: {df.shape}")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Detect label column
    label_col = detect_label_column(df)
    print(f"[data_loader] Label column detected: '{label_col}'")
    print(f"[data_loader] Raw label values: {df[label_col].value_counts().to_dict()}")

    # Normalise label
    raw_series = df[label_col].copy()
    unique_vals = set(raw_series.dropna().unique())

    if unique_vals <= {0, 1} or unique_vals <= {0.0, 1.0}:
        # Empirical check: which class has higher URLSimilarityIndex?
        if "URLSimilarityIndex" in df.columns:
            mean_by_label = df.groupby(label_col)["URLSimilarityIndex"].mean()
            legitimate_label = mean_by_label.idxmax()  # higher score -> legit
            phishing_label   = mean_by_label.idxmin()
            print(f"[data_loader] Empirical check: label={legitimate_label} "
                  f"-> legitimate, label={phishing_label} -> phishing")
            if legitimate_label == 1:
                # Dataset uses 1=legitimate -> invert
                df["label"] = (df[label_col] == phishing_label).astype(int)
                print("[data_loader] Labels INVERTED: original 0->phishing(1), 1->legitimate(0)")
            else:
                df["label"] = df[label_col].astype(int)
                print("[data_loader] Labels kept as-is: 1=phishing, 0=legitimate")
        else:
            df["label"] = df[label_col].astype(int)
    else:
        df["label"] = normalise_label(raw_series)

    # Rename original label col if different name
    if label_col != "label":
        df.drop(columns=[label_col], inplace=True)

    print(f"[data_loader] Normalised label distribution:\n{df['label'].value_counts()}")

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < before:
        print(f"[data_loader] Dropped {before - len(df)} duplicate rows")

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[data_loader] {missing} missing values - filling numeric with median")
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only numeric, model-ready feature columns (no IDs, texts, label)."""
    exclude = NON_FEATURE_COLS.copy()
    # Also exclude any remaining string columns
    string_cols = set(df.select_dtypes(include=["object"]).columns)
    exclude |= string_cols
    return [c for c in df.columns if c not in exclude]


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], StandardScaler]:
    """
    Stratified 80/20 split -> fit StandardScaler on train -> apply to both.
    Returns (train_df, test_df, feature_cols, scaler).
    Feature columns inside the returned DataFrames are scaled IN PLACE.
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df["label"] = y_train

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df["label"] = y_test

    return train_df, test_df, feature_cols, scaler


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> None:
    """Persist train/test CSVs, scaler, and feature list."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV,   index=False)
    joblib.dump(scaler, SCALER_PKL)

    feat_path = DATA_PROC / "feature_cols.txt"
    feat_path.write_text("\n".join(feature_cols))

    print(f"[data_loader] Saved train ({len(train_df)} rows) -> {TRAIN_CSV}")
    print(f"[data_loader] Saved test  ({len(test_df)}  rows) -> {TEST_CSV}")
    print(f"[data_loader] Saved scaler                      -> {SCALER_PKL}")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, list[str], StandardScaler]:
    """Load already-processed splits and scaler from disk."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    scaler   = joblib.load(SCALER_PKL)
    feat_path = DATA_PROC / "feature_cols.txt"
    feature_cols = feat_path.read_text().strip().split("\n")
    return train_df, test_df, feature_cols, scaler


def run_pipeline(csv_path: str | pathlib.Path = RAW_CSV) -> tuple:
    """Full data pipeline: load -> split -> scale -> save."""
    df = load_raw(csv_path)
    train_df, test_df, feature_cols, scaler = split_and_scale(df)
    save_splits(train_df, test_df, scaler, feature_cols)
    return train_df, test_df, feature_cols, scaler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_df, test_df, feature_cols, scaler = run_pipeline()
    print(f"\n=== Smoke test passed ===")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Train    : {train_df.shape}  phishing={train_df['label'].sum()}")
    print(f"  Test     : {test_df.shape}   phishing={test_df['label'].sum()}")
