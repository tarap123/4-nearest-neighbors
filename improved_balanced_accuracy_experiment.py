import functools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split


RANDOM_STATE = 42
VALIDATION_MODE = os.environ.get("VALIDATION_MODE", "time").strip().lower()
MAKE_SUBMISSION = os.environ.get("MAKE_SUBMISSION", "0").strip() == "1"
print = functools.partial(print, flush=True)
TRAIN_PATH = Path("train.csv")
TEST_PATH = Path("test.csv")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")
TARGET_COL = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"


def fail_if_lfs_pointer(path):
    first_line = path.read_text(errors="ignore").splitlines()[0]
    if first_line.startswith("version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer. Run `git lfs pull` first."
        )


def parse_incident_date(series):
    return pd.to_datetime(series, format="%m/%d/%y", errors="coerce")


def time_to_minutes_after_midnight(series):
    parsed = pd.to_datetime(series.astype("string").str.strip(), format="%H:%M", errors="coerce")
    return (parsed.dt.hour * 60 + parsed.dt.minute).astype("float64")


def parse_count_like_feature(series):
    text = series.astype("string").str.strip().str.lower()
    parsed = pd.to_numeric(text, errors="coerce")
    range_map = {
        "2-10": 6,
        "10-feb": 6,
        "11-100": 55,
        "more than 100": 101,
        "over 100": 101,
    }
    return parsed.fillna(text.map(range_map)).astype("float64")


def base_clean(train_df, test_df):
    missing_placeholders = ["UNKNOWN", "ZZZZ"]
    train_df = train_df.replace(missing_placeholders, np.nan).copy()
    test_df = test_df.replace(missing_placeholders, np.nan).copy()

    missing_summary = train_df.isna().mean().sort_values(ascending=False)
    high_missing_cols = missing_summary[missing_summary > 0.95].index.tolist()

    manual_drop_cols = [
        "BIRD_BAND_NUMBER",
        "REMARKS",
        "COMMENTS",
        "LUPDATE",
        "TRANSFER",
        "REG",
        "LOCATION",
    ]
    model_only_drop_cols = [ID_COL, "INCIDENT_MONTH", "INCIDENT_YEAR"]
    cols_to_drop = sorted(set(high_missing_cols + manual_drop_cols + model_only_drop_cols))

    X = train_df.drop(columns=[TARGET_COL] + cols_to_drop, errors="ignore")
    y = train_df[TARGET_COL].astype(int)
    X_test = test_df.drop(columns=cols_to_drop, errors="ignore")
    train_ids = train_df[ID_COL].copy()
    test_ids = test_df[ID_COL].copy()

    for feature_df in (X, X_test):
        if "RUNWAY" in feature_df.columns:
            runway_is_blank = feature_df["RUNWAY"].isna() | feature_df["RUNWAY"].astype(str).str.strip().eq("")
            feature_df["airborne"] = runway_is_blank.astype(int)
            feature_df.drop(columns=["RUNWAY"], inplace=True)

    row_missing_fraction = X.isna().mean(axis=1)
    rows_to_keep = row_missing_fraction <= 0.50
    X = X.loc[rows_to_keep].copy()
    y = y.loc[rows_to_keep].copy()
    train_ids = train_ids.loc[rows_to_keep].copy()

    return X, y, X_test, train_ids, test_ids, cols_to_drop


def add_datetime_features(feature_df):
    engineered = feature_df.copy()

    if "INCIDENT_DATE" in engineered.columns:
        incident_datetime = parse_incident_date(engineered["INCIDENT_DATE"])
        engineered["incident_date_missing"] = incident_datetime.isna().astype(int)
        engineered["incident_year"] = incident_datetime.dt.year.astype("float64")
        engineered["incident_month"] = incident_datetime.dt.month.astype("float64")
        engineered["incident_dayofyear"] = incident_datetime.dt.dayofyear.astype("float64")
        engineered["incident_quarter"] = incident_datetime.dt.quarter.astype("float64")
        engineered["incident_month_sin"] = np.sin(2 * np.pi * engineered["incident_month"] / 12)
        engineered["incident_month_cos"] = np.cos(2 * np.pi * engineered["incident_month"] / 12)
        engineered["INCIDENT_DATE"] = (incident_datetime.astype("int64") // 10**9).where(
            incident_datetime.notna(), np.nan
        ).astype("float64")

    if "TIME" in engineered.columns:
        minutes = time_to_minutes_after_midnight(engineered["TIME"])
        engineered["time_missing"] = minutes.isna().astype(int)
        engineered["incident_hour"] = np.floor(minutes / 60).where(minutes.notna())
        engineered["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
        engineered["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
        engineered["TIME"] = minutes

    return engineered


def add_missingness_and_domain_features(feature_df):
    engineered = add_datetime_features(feature_df)

    for col in ["HEIGHT", "SPEED", "DISTANCE", "WARNED", "NUM_SEEN", "NUM_STRUCK", "SIZE"]:
        if col in engineered.columns:
            engineered[f"{col.lower()}_missing"] = engineered[col].isna().astype(int)

    for col in ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE", "AC_MASS", "NUM_ENGS"]:
        if col in engineered.columns:
            engineered[col] = pd.to_numeric(engineered[col], errors="coerce")

    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        if col in engineered.columns:
            values = pd.to_numeric(engineered[col], errors="coerce")
            engineered[f"log1p_{col.lower()}"] = np.log1p(values.clip(lower=0))

    if "NUM_SEEN" in engineered.columns:
        engineered["num_seen_numeric"] = parse_count_like_feature(engineered["NUM_SEEN"])
    if "NUM_STRUCK" in engineered.columns:
        engineered["num_struck_numeric"] = parse_count_like_feature(engineered["NUM_STRUCK"])

    if "SIZE" in engineered.columns:
        size_map = {"Small": 1, "Medium": 2, "Large": 3}
        engineered["size_ordered"] = engineered["SIZE"].map(size_map).astype("float64")

    if {"HEIGHT", "SPEED"}.issubset(engineered.columns):
        engineered["height_speed_interaction"] = engineered["HEIGHT"] * engineered["SPEED"]
    if {"size_ordered", "num_struck_numeric"}.issubset(engineered.columns):
        engineered["size_x_num_struck"] = engineered["size_ordered"] * engineered["num_struck_numeric"]
    if {"AC_MASS", "NUM_ENGS"}.issubset(engineered.columns):
        engineered["engine_count_x_mass"] = engineered["AC_MASS"] * engineered["NUM_ENGS"]
    if "HEIGHT" in engineered.columns:
        engineered["on_ground"] = (engineered["HEIGHT"] == 0).astype(int)

    if "PHASE_OF_FLIGHT" in engineered.columns:
        phase = engineered["PHASE_OF_FLIGHT"].astype("string").str.lower()
        engineered["phase_ground"] = phase.str.contains("landing roll|taxi|parked", na=False).astype(int)
        engineered["phase_takeoff_climb"] = phase.str.contains("take-off|takeoff|climb", na=False).astype(int)
        engineered["phase_approach_landing"] = phase.str.contains("approach|landing", na=False).astype(int)

    return engineered


def fit_oof_target_encoding_preprocessor(X_fit_raw, y_fit, low_cardinality_threshold=5, smoothing=50, n_splits=3):
    X_fit = add_missingness_and_domain_features(X_fit_raw)

    numeric_cols = X_fit.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_fit.columns.difference(numeric_cols).tolist()

    numeric_fill_values = X_fit[numeric_cols].median().fillna(0)
    categorical_fill_values = X_fit[categorical_cols].agg(
        lambda col: col.mode(dropna=True).iloc[0] if not col.mode(dropna=True).empty else "Missing"
    )

    X_imputed = X_fit.copy()
    X_imputed[numeric_cols] = X_imputed[numeric_cols].fillna(numeric_fill_values)
    X_imputed[categorical_cols] = X_imputed[categorical_cols].fillna(categorical_fill_values)

    # Clip heavy-tailed numeric fields using training-only quantiles.
    clip_bounds = {}
    for col in numeric_cols:
        lo, hi = X_imputed[col].quantile([0.001, 0.999])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            clip_bounds[col] = (float(lo), float(hi))
            X_imputed[col] = X_imputed[col].clip(lo, hi)

    low_cardinality_cols = []
    high_cardinality_cols = []
    for col in X_imputed.select_dtypes(include=["object", "string"]).columns:
        if X_imputed[col].nunique(dropna=False) <= low_cardinality_threshold:
            low_cardinality_cols.append(col)
        else:
            high_cardinality_cols.append(col)

    global_target_mean = float(y_fit.mean())
    X_encoded = X_imputed.copy()
    target_encoding_maps = {}

    for col in high_cardinality_cols:
        oof_values = pd.Series(global_target_mean, index=X_imputed.index, dtype="float64")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_pos, valid_pos in skf.split(X_imputed, y_fit):
            train_index = X_imputed.index[train_pos]
            valid_index = X_imputed.index[valid_pos]
            stats = pd.DataFrame(
                {"category": X_imputed.loc[train_index, col], "target": y_fit.loc[train_index]}
            ).groupby("category")["target"].agg(["sum", "count"])
            fold_map = (stats["sum"] + smoothing * global_target_mean) / (stats["count"] + smoothing)
            oof_values.loc[valid_index] = X_imputed.loc[valid_index, col].map(fold_map).fillna(global_target_mean)

        full_stats = pd.DataFrame({"category": X_imputed[col], "target": y_fit}).groupby("category")["target"].agg(
            ["sum", "count"]
        )
        target_encoding_maps[col] = (full_stats["sum"] + smoothing * global_target_mean) / (
            full_stats["count"] + smoothing
        )
        X_encoded[col] = oof_values

    X_encoded = pd.get_dummies(X_encoded, columns=low_cardinality_cols, drop_first=False)

    return {
        "feature_columns": X_fit_raw.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "clip_bounds": clip_bounds,
        "low_cardinality_cols": low_cardinality_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "target_encoding_maps": target_encoding_maps,
        "global_target_mean": global_target_mean,
        "encoded_columns": X_encoded.columns.tolist(),
        "X_fit_encoded_oof": X_encoded,
    }


def transform_target_encoded_features(X_raw, preprocessor):
    X_transformed = X_raw.reindex(columns=preprocessor["feature_columns"]).copy()
    X_transformed = add_missingness_and_domain_features(X_transformed)

    numeric_cols = preprocessor["numeric_cols"]
    categorical_cols = preprocessor["categorical_cols"]
    X_transformed[numeric_cols] = X_transformed[numeric_cols].fillna(preprocessor["numeric_fill_values"])
    X_transformed[categorical_cols] = X_transformed[categorical_cols].fillna(
        preprocessor["categorical_fill_values"]
    )

    for col, (lo, hi) in preprocessor["clip_bounds"].items():
        if col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].clip(lo, hi)

    for col in preprocessor["high_cardinality_cols"]:
        X_transformed[col] = X_transformed[col].map(preprocessor["target_encoding_maps"][col]).fillna(
            preprocessor["global_target_mean"]
        )

    X_transformed = pd.get_dummies(
        X_transformed,
        columns=preprocessor["low_cardinality_cols"],
        drop_first=False,
    )
    X_transformed = X_transformed.reindex(columns=preprocessor["encoded_columns"], fill_value=0)
    return X_transformed


def make_time_aware_split(X_raw, y, validation_fraction=0.20):
    dates = parse_incident_date(X_raw["INCIDENT_DATE"])
    split_rank = dates.rank(method="first", na_option="bottom")
    cutoff = split_rank.quantile(1 - validation_fraction)
    valid_idx = X_raw.index[split_rank > cutoff]
    train_idx = X_raw.index.difference(valid_idx)
    return train_idx, valid_idx


def evaluate_thresholds(model_name, y_true, probabilities, thresholds=np.arange(0.05, 0.76, 0.01)):
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                "accuracy": accuracy_score(y_true, predictions),
                "damage_precision": precision_score(y_true, predictions, pos_label=1, zero_division=0),
                "damage_recall": recall_score(y_true, predictions, pos_label=1, zero_division=0),
                "damage_f1": f1_score(y_true, predictions, pos_label=1, zero_division=0),
            }
        )
    results = pd.DataFrame(rows)
    return results.loc[results["balanced_accuracy"].idxmax()]


def tune_random_forest(X_train, y_train, X_valid, y_valid):
    sample_size = min(30000, len(y_train))
    sample_idx, _ = train_test_split(
        y_train.index,
        train_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    param_grid = list(
        ParameterGrid(
            {
                "n_estimators": [150],
                "max_depth": [None, 35],
                "min_samples_leaf": [3],
                "max_features": ["sqrt"],
                "class_weight": ["balanced_subsample", {0: 1, 1: 6}],
            }
        )
    )
    best = None
    for params in param_grid:
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
        probabilities = model.predict_proba(X_valid)[:, 1]
        result = evaluate_thresholds("RandomForest", y_valid, probabilities).to_dict()
        result["params"] = params
        if best is None or result["balanced_accuracy"] > best["balanced_accuracy"]:
            best = result

    final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best["params"])
    final_model.fit(X_train, y_train)
    probabilities = final_model.predict_proba(X_valid)[:, 1]
    final_result = evaluate_thresholds("RandomForest full train", y_valid, probabilities).to_dict()
    final_result["params"] = best["params"]
    return final_model, probabilities, final_result


def tune_hist_gradient_boosting(X_train, y_train, X_valid, y_valid):
    sample_size = min(40000, len(y_train))
    sample_idx, _ = train_test_split(
        y_train.index,
        train_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    param_grid = list(
        ParameterGrid(
            {
                "max_iter": [200],
                "learning_rate": [0.05, 0.08],
                "max_leaf_nodes": [31],
                "min_samples_leaf": [30],
                "l2_regularization": [0.0, 0.1],
                "class_weight": ["balanced", {0: 1, 1: 6}],
            }
        )
    )
    best = None
    for params in param_grid:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            **params,
        )
        model.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
        probabilities = model.predict_proba(X_valid)[:, 1]
        result = evaluate_thresholds("HistGradientBoosting", y_valid, probabilities).to_dict()
        result["params"] = params
        if best is None or result["balanced_accuracy"] > best["balanced_accuracy"]:
            best = result

    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        **best["params"],
    )
    final_model.fit(X_train, y_train)
    probabilities = final_model.predict_proba(X_valid)[:, 1]
    final_result = evaluate_thresholds("HistGradientBoosting full train", y_valid, probabilities).to_dict()
    final_result["params"] = best["params"]
    return final_model, probabilities, final_result


def tune_two_model_ensemble(y_valid, rf_probabilities, hgb_probabilities):
    best = None
    for rf_weight in np.arange(0, 1.01, 0.05):
        probabilities = rf_weight * rf_probabilities + (1 - rf_weight) * hgb_probabilities
        result = evaluate_thresholds("RF + HGB ensemble", y_valid, probabilities).to_dict()
        result["rf_weight"] = float(rf_weight)
        result["hgb_weight"] = float(1 - rf_weight)
        if best is None or result["balanced_accuracy"] > best["balanced_accuracy"]:
            best = result
    return best


def make_submission(X, y, X_test, test_ids, model_result, model_name, output_path):
    final_preprocessor = fit_oof_target_encoding_preprocessor(X, y)
    X_full = final_preprocessor["X_fit_encoded_oof"]
    X_test_final = transform_target_encoded_features(X_test, final_preprocessor)

    if model_name == "rf":
        final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **model_result["params"])
    elif model_name == "hgb":
        final_model = HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            **model_result["params"],
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    final_model.fit(X_full, y)
    probabilities = final_model.predict_proba(X_test_final)[:, 1]
    predictions = (probabilities >= model_result["threshold"]).astype(int)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET_COL: predictions})
    submission = submission[sample_submission.columns]
    assert submission[ID_COL].equals(sample_submission[ID_COL])
    submission.to_csv(output_path, index=False)
    return output_path


def make_rf_hgb_ensemble_submission(X, y, X_test, test_ids, output_path):
    """Train the best validation RF+HGB ensemble on all data and save Kaggle predictions."""
    final_preprocessor = fit_oof_target_encoding_preprocessor(X, y)
    X_full = final_preprocessor["X_fit_encoded_oof"]
    X_test_final = transform_target_encoded_features(X_test, final_preprocessor)

    rf_params = {
        "class_weight": {0: 1, 1: 6},
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "n_estimators": 150,
    }
    hgb_params = {
        "class_weight": {0: 1, 1: 6},
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
    }
    ensemble_threshold = 0.19
    rf_weight = 0.55
    hgb_weight = 0.45

    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    rf_model.fit(X_full, y)
    rf_probabilities = rf_model.predict_proba(X_test_final)[:, 1]

    hgb_model = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        **hgb_params,
    )
    hgb_model.fit(X_full, y)
    hgb_probabilities = hgb_model.predict_proba(X_test_final)[:, 1]

    ensemble_probabilities = rf_weight * rf_probabilities + hgb_weight * hgb_probabilities
    predictions = (ensemble_probabilities >= ensemble_threshold).astype(int)

    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET_COL: predictions})
    submission = submission[sample_submission.columns]
    assert len(submission) == len(sample_submission)
    assert submission[ID_COL].equals(sample_submission[ID_COL])
    submission.to_csv(output_path, index=False)
    return output_path, submission[TARGET_COL].value_counts(normalize=True).to_dict()


def main():
    fail_if_lfs_pointer(TRAIN_PATH)
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)
    X, y, X_test, train_ids, test_ids, cols_to_drop = base_clean(train_df, test_df)

    if MAKE_SUBMISSION:
        output_path, prediction_fraction = make_rf_hgb_ensemble_submission(
            X,
            y,
            X_test,
            test_ids,
            "submission_rf_hgb_improved_ensemble.csv",
        )
        print(f"Saved {output_path}")
        print("Prediction fraction:", prediction_fraction)
        return

    if VALIDATION_MODE == "random":
        train_idx, valid_idx = train_test_split(
            X.index,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    else:
        train_idx, valid_idx = make_time_aware_split(X, y, validation_fraction=0.20)
    X_train_raw = X.loc[train_idx].copy()
    X_valid_raw = X.loc[valid_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_valid = y.loc[valid_idx].copy()

    print("Cleaned training rows:", len(X))
    print("Validation mode:", VALIDATION_MODE)
    print("Train rows:", len(X_train_raw))
    print("Validation rows:", len(X_valid_raw))
    print("Validation incident years:")
    print(parse_incident_date(X_valid_raw["INCIDENT_DATE"]).dt.year.describe())
    print("Train damage rate:", float(y_train.mean()))
    print("Validation damage rate:", float(y_valid.mean()))

    preprocessor = fit_oof_target_encoding_preprocessor(X_train_raw, y_train)
    X_train_encoded = preprocessor["X_fit_encoded_oof"]
    X_valid_encoded = transform_target_encoded_features(X_valid_raw, preprocessor)

    print("Encoded train shape:", X_train_encoded.shape)
    print("Encoded validation shape:", X_valid_encoded.shape)
    print("OOF target-encoded columns:", preprocessor["high_cardinality_cols"])
    print("One-hot columns:", preprocessor["low_cardinality_cols"])

    rf_model, rf_probabilities, rf_result = tune_random_forest(
        X_train_encoded, y_train, X_valid_encoded, y_valid
    )
    print("RF result:")
    print(json.dumps(rf_result, indent=2, default=str))

    hgb_model, hgb_probabilities, hgb_result = tune_hist_gradient_boosting(
        X_train_encoded, y_train, X_valid_encoded, y_valid
    )
    print("HGB result:")
    print(json.dumps(hgb_result, indent=2, default=str))

    ensemble_result = tune_two_model_ensemble(y_valid, rf_probabilities, hgb_probabilities)
    print("Ensemble result:")
    print(json.dumps(ensemble_result, indent=2, default=str))

    best_model = max([rf_result, hgb_result, ensemble_result], key=lambda row: row["balanced_accuracy"])
    print("Best validation result:")
    print(json.dumps(best_model, indent=2, default=str))

    results = pd.DataFrame([rf_result, hgb_result, ensemble_result])
    results.to_csv("improved_balanced_accuracy_results.csv", index=False)


if __name__ == "__main__":
    main()
