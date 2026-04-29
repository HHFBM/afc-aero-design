import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as onp
import pandas as pd


KULFAN_UPPER_COLUMNS = [f"kulfan_upper_weights_{i}" for i in range(8)]
KULFAN_LOWER_COLUMNS = [f"kulfan_lower_weights_{i}" for i in range(8)]

GEOMETRY_COLUMNS = [
    *KULFAN_UPPER_COLUMNS,
    *KULFAN_LOWER_COLUMNS,
    "leading_edge_weight",
    "TE_thickness",
]

CONDITION_COLUMNS = [
    "alpha",
    "Re",
    "Mach",
    "n_crit",
    "xtr_upper",
    "xtr_lower",
]

AFC_COLUMNS = [
    "C_mu",
    "x_jet",
    "theta_jet",
]

OUTPUT_COLUMNS = [
    "CL",
    "CD",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
]

METADATA_COLUMNS = [
    "converged",
    "source",
    "confidence_label",
]

AFC_DATASET_COLUMNS = [
    *GEOMETRY_COLUMNS,
    *CONDITION_COLUMNS,
    *AFC_COLUMNS,
    *OUTPUT_COLUMNS,
    *METADATA_COLUMNS,
]

NUMERIC_COLUMNS = [
    *GEOMETRY_COLUMNS,
    *CONDITION_COLUMNS,
    *AFC_COLUMNS,
    *OUTPUT_COLUMNS,
    "confidence_label",
]

RANGE_LIMITS = {
    **{column: (-5.0, 5.0) for column in [*KULFAN_UPPER_COLUMNS, *KULFAN_LOWER_COLUMNS]},
    "leading_edge_weight": (-2.0, 2.0),
    "TE_thickness": (0.0, 0.30),
    "alpha": (-180.0, 180.0),
    "Re": (1e3, 1e9),
    "Mach": (0.0, 5.0),
    "n_crit": (0.0, 20.0),
    "xtr_upper": (0.0, 1.0),
    "xtr_lower": (0.0, 1.0),
    "C_mu": (0.0, 1.0),
    "x_jet": (0.0, 1.0),
    "theta_jet": (-180.0, 180.0),
    "CD": (0.0, 10.0),
    "Top_Xtr": (0.0, 1.0),
    "Bot_Xtr": (0.0, 1.0),
    "confidence_label": (0.0, 1.0),
}

AFC_DATASET_SCHEMA_VERSION = "afc_2d_v1"
AFC_DATASET_MANIFEST_VERSION = 1
AFC_DATASET_INPUT_COLUMNS = [
    *GEOMETRY_COLUMNS,
    *CONDITION_COLUMNS,
    *AFC_COLUMNS,
]


class AFCDatasetValidationError(ValueError):
    """Raised when an AFC airfoil dataset does not match the expected schema."""


class AFCDatasetDuplicateError(ValueError):
    """Raised when duplicate AFC input samples are found and duplicates are disallowed."""


def empty_afc_dataframe() -> pd.DataFrame:
    """
    Returns an empty AFC dataset table with the canonical column order.

    The schema is intentionally flat so the same file can be stored as CSV or Parquet. Kulfan arrays are represented as
    `kulfan_upper_weights_0..7` and `kulfan_lower_weights_0..7`.
    """
    return pd.DataFrame(columns=AFC_DATASET_COLUMNS)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise AFCDatasetValidationError(
            f"AFC dataset is missing required columns: {missing}"
        )


def validate_afc_dataframe(
    df: pd.DataFrame,
    *,
    require_all_columns: bool = True,
    check_ranges: bool = True,
) -> pd.DataFrame:
    """
    Validates an AFC 2D surrogate dataset.

    Args:
        df: Dataset as a pandas DataFrame.
        require_all_columns: If True, every canonical AFC schema column must be present.
        check_ranges: If True, run basic unit/range checks on dimensional and nondimensional fields.

    Returns:
        The input DataFrame, for convenient call chaining.

    Raises:
        AFCDatasetValidationError: If the schema, dtypes, finite values, or value ranges are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if require_all_columns:
        _require_columns(df, AFC_DATASET_COLUMNS)
    else:
        _require_columns(df, NUMERIC_COLUMNS)

    for column in NUMERIC_COLUMNS:
        try:
            values = pd.to_numeric(df[column])
        except Exception as e:
            raise AFCDatasetValidationError(
                f'Column "{column}" must be numeric.'
            ) from e

        if values.isna().any():
            bad_count = int(values.isna().sum())
            raise AFCDatasetValidationError(
                f'Column "{column}" contains {bad_count} NaN values.'
            )

        if not onp.all(onp.isfinite(values.to_numpy(dtype=float))):
            raise AFCDatasetValidationError(
                f'Column "{column}" contains non-finite values.'
            )

    if "converged" in df.columns:
        invalid_converged = ~df["converged"].isin([True, False, 0, 1])
        if invalid_converged.any():
            raise AFCDatasetValidationError(
                '`converged` must contain booleans or 0/1 values.'
            )

    if "source" in df.columns:
        invalid_source = df["source"].isna() | (df["source"].astype(str).str.len() == 0)
        if invalid_source.any():
            raise AFCDatasetValidationError(
                '`source` must contain non-empty source labels such as "SU2", "RANS", "XFoil", or "experiment".'
            )

    if check_ranges:
        for column, (lower, upper) in RANGE_LIMITS.items():
            values = pd.to_numeric(df[column])
            invalid = (values < lower) | (values > upper)
            if invalid.any():
                examples = values[invalid].head(5).to_list()
                raise AFCDatasetValidationError(
                    f'Column "{column}" is outside [{lower}, {upper}]. Examples: {examples}'
                )

    return df


def _json_default(value):
    if isinstance(value, (onp.integer,)):
        return int(value)
    if isinstance(value, (onp.floating,)):
        return float(value)
    if isinstance(value, (onp.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _column_range(df: pd.DataFrame, column: str) -> Dict[str, Optional[float]]:
    if column not in df.columns or len(df) == 0:
        return {"min": None, "max": None}
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().all():
        return {"min": None, "max": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
    }


def make_afc_dataset_manifest(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    source: Optional[Union[str, Sequence[str]]] = None,
    created_at: Optional[str] = None,
    schema_version: str = AFC_DATASET_SCHEMA_VERSION,
    split_method: str = "not_split",
    filters: Optional[Mapping[str, object]] = None,
    known_limitations: Optional[Sequence[str]] = None,
    validate: bool = True,
) -> Dict[str, object]:
    """
    Builds a JSON-serializable manifest for an AFC 2D dataset.

    The manifest is intended to live next to the CSV/Parquet file and records provenance, ranges, filtering choices,
    and known limitations. It does not contain raw data.
    """
    if validate:
        validate_afc_dataframe(df)

    if source is None:
        if "source" in df.columns:
            source = sorted(df["source"].astype(str).unique().tolist())
        else:
            source = "unknown"

    return {
        "manifest_version": AFC_DATASET_MANIFEST_VERSION,
        "dataset_id": dataset_id,
        "source": source,
        "created_at": created_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schema_version": schema_version,
        "n_cases": int(len(df)),
        "input_ranges": {
            column: _column_range(df, column)
            for column in AFC_DATASET_INPUT_COLUMNS
            if column in df.columns
        },
        "split_method": split_method,
        "filters": dict(filters or {}),
        "known_limitations": list(known_limitations or []),
    }


def write_afc_dataset_manifest(
    manifest: Mapping[str, object],
    filepath: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    """
    Writes an AFC dataset manifest as JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dict(manifest), f, indent=indent, default=_json_default)
    return filepath


def read_afc_dataset_manifest(filepath: Union[str, Path]) -> Dict[str, object]:
    """
    Reads an AFC dataset manifest JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_fields = [
        "dataset_id",
        "source",
        "created_at",
        "schema_version",
        "n_cases",
        "input_ranges",
        "split_method",
        "filters",
        "known_limitations",
    ]
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        raise AFCDatasetValidationError(
            f"AFC dataset manifest is missing required fields: {missing}"
        )
    return manifest


def find_duplicate_afc_samples(
    df: pd.DataFrame,
    *,
    subset: Optional[Sequence[str]] = None,
    decimals: Optional[int] = None,
    keep: Union[str, bool] = False,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Returns rows whose input sample definition is duplicated.

    By default duplicates are checked over geometry, operating condition, and AFC input columns. `decimals` can be used
    to round numeric columns before comparison when data comes from text formats with small formatting noise.
    """
    if validate:
        validate_afc_dataframe(df)

    subset = list(subset or AFC_DATASET_INPUT_COLUMNS)
    _require_columns(df, subset)
    comparison = df[subset].copy()
    if decimals is not None:
        for column in comparison.columns:
            if pd.api.types.is_numeric_dtype(comparison[column]):
                comparison[column] = comparison[column].round(decimals)

    duplicated = comparison.duplicated(keep=keep)
    return df.loc[duplicated].copy()


def check_duplicate_afc_samples(
    df: pd.DataFrame,
    *,
    subset: Optional[Sequence[str]] = None,
    decimals: Optional[int] = None,
    raise_on_duplicates: bool = False,
    validate: bool = True,
) -> Dict[str, object]:
    """
    Checks duplicate AFC input samples and optionally raises on duplicates.
    """
    duplicates = find_duplicate_afc_samples(
        df,
        subset=subset,
        decimals=decimals,
        keep=False,
        validate=validate,
    )
    report = {
        "n_rows": int(len(df)),
        "n_duplicate_rows": int(len(duplicates)),
        "duplicate_fraction": float(len(duplicates) / len(df)) if len(df) else 0.0,
        "subset": list(subset or AFC_DATASET_INPUT_COLUMNS),
    }
    if raise_on_duplicates and len(duplicates) > 0:
        raise AFCDatasetDuplicateError(
            f"Found {len(duplicates)} duplicate AFC input sample rows."
        )
    return report


def read_afc_dataset(
    filepath: Union[str, Path],
    *,
    validate: bool = True,
    manifest_path: Optional[Union[str, Path]] = None,
    return_manifest: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[Dict[str, object]]]]:
    """
    Reads an AFC dataset from `.parquet` or `.csv`.

    Parquet is preferred for large datasets because it preserves dtypes and is faster. CSV is supported for small
    exchange files and environments without a Parquet engine.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(filepath)
        except ImportError as e:
            raise ImportError(
                "Reading Parquet AFC datasets requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(
            f'Unsupported AFC dataset extension "{filepath.suffix}". Use `.parquet` or `.csv`.'
        )

    if validate:
        validate_afc_dataframe(df)

    manifest = None
    if manifest_path is not None:
        manifest = read_afc_dataset_manifest(manifest_path)

    if return_manifest:
        return df, manifest
    return df


def write_afc_dataset(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    *,
    validate: bool = True,
    index: bool = False,
    manifest: Optional[Mapping[str, object]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Writes an AFC dataset to `.parquet` or `.csv`.

    Args:
        df: Dataset as a pandas DataFrame.
        filepath: Output path. The suffix selects the format.
        validate: If True, validate before writing.
        index: Whether to include the DataFrame index in the output file.

    Returns:
        The written path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if validate:
        validate_afc_dataframe(df)

    df = df[AFC_DATASET_COLUMNS]
    suffix = filepath.suffix.lower()

    if suffix == ".parquet":
        try:
            df.to_parquet(filepath, index=index)
        except ImportError as e:
            raise ImportError(
                "Writing Parquet AFC datasets requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".csv":
        df.to_csv(filepath, index=index)
    else:
        raise ValueError(
            f'Unsupported AFC dataset extension "{filepath.suffix}". Use `.parquet` or `.csv`.'
        )

    if manifest_path is not None:
        if manifest is None:
            manifest = make_afc_dataset_manifest(
                df,
                dataset_id=filepath.stem,
                validate=False,
            )
        write_afc_dataset_manifest(manifest, manifest_path)

    return filepath


def split_afc_dataset(
    df: pd.DataFrame,
    *,
    train_fraction: float = 0.80,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    random_seed: int = 0,
    shuffle: bool = True,
    validate: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits an AFC dataset into train/validation/test DataFrames.
    """
    if validate:
        validate_afc_dataframe(df)

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-12:
        raise ValueError(
            "`train_fraction + val_fraction + test_fraction` must equal 1."
        )

    n = len(df)
    indices = onp.arange(n)
    if shuffle:
        rng = onp.random.default_rng(random_seed)
        rng.shuffle(indices)

    n_train = int(onp.floor(train_fraction * n))
    n_val = int(onp.floor(val_fraction * n))

    train = df.iloc[indices[:n_train]].reset_index(drop=True)
    val = df.iloc[indices[n_train : n_train + n_val]].reset_index(drop=True)
    test = df.iloc[indices[n_train + n_val :]].reset_index(drop=True)

    return train, val, test


def afc_dataset_statistics(
    df: pd.DataFrame,
    *,
    validate: bool = True,
) -> Dict[str, object]:
    """
    Computes a compact statistics report for an AFC dataset.
    """
    if validate:
        validate_afc_dataframe(df)

    numeric_summary = df[NUMERIC_COLUMNS].describe().transpose()

    return {
        "n_cases": int(len(df)),
        "n_converged": int(df["converged"].astype(bool).sum()),
        "converged_fraction": float(df["converged"].astype(bool).mean())
        if len(df) > 0
        else onp.nan,
        "sources": df["source"].value_counts().to_dict(),
        "confidence_label_mean": float(df["confidence_label"].mean())
        if len(df) > 0
        else onp.nan,
        "numeric_summary": numeric_summary,
    }


def afc_dataset_grouped_statistics(
    df: pd.DataFrame,
    *,
    by: Sequence[str] = ("source",),
    round_digits: Optional[Mapping[str, int]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Computes grouped statistics for source/Re/C_mu/alpha slices or other columns.

    Continuous grouping columns can be rounded before grouping using `round_digits`, e.g.
    `{"Re": 0, "C_mu": 4, "alpha": 2}`.
    """
    if validate:
        validate_afc_dataframe(df)

    by = list(by)
    _require_columns(df, by)
    round_digits = dict(round_digits or {})

    grouped_df = df.copy()
    for column, digits in round_digits.items():
        if column in grouped_df.columns:
            grouped_df[column] = pd.to_numeric(grouped_df[column]).round(digits)

    table = (
        grouped_df.groupby(by, dropna=False)
        .agg(
            n_cases=("CL", "size"),
            converged_fraction=("converged", lambda s: float(s.astype(bool).mean())),
            confidence_label_mean=("confidence_label", "mean"),
            CL_mean=("CL", "mean"),
            CL_std=("CL", "std"),
            CD_mean=("CD", "mean"),
            CD_std=("CD", "std"),
            CM_mean=("CM", "mean"),
            CM_std=("CM", "std"),
        )
        .reset_index()
    )
    return table


def afc_dataset_standard_statistics_tables(
    df: pd.DataFrame,
    *,
    validate: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns standard grouped statistics tables by source, Re, C_mu, alpha, and their combined slices.
    """
    if validate:
        validate_afc_dataframe(df)

    return {
        "by_source": afc_dataset_grouped_statistics(df, by=("source",), validate=False),
        "by_Re": afc_dataset_grouped_statistics(
            df, by=("Re",), round_digits={"Re": 0}, validate=False
        ),
        "by_C_mu": afc_dataset_grouped_statistics(
            df, by=("C_mu",), round_digits={"C_mu": 5}, validate=False
        ),
        "by_alpha": afc_dataset_grouped_statistics(
            df, by=("alpha",), round_digits={"alpha": 3}, validate=False
        ),
        "by_source_Re_C_mu_alpha": afc_dataset_grouped_statistics(
            df,
            by=("source", "Re", "C_mu", "alpha"),
            round_digits={"Re": 0, "C_mu": 5, "alpha": 3},
            validate=False,
        ),
    }


def make_fake_afc_dataset(
    n_cases: int = 200,
    *,
    random_seed: int = 0,
    source: str = "synthetic_placeholder",
) -> pd.DataFrame:
    """
    Generates a fake AFC dataset that follows the canonical schema.

    This is for pipeline and schema testing only. The aerodynamic outputs are smooth synthetic values and should not be
    used as physics data.
    """
    rng = onp.random.default_rng(random_seed)

    alpha = rng.uniform(-8, 16, n_cases)
    log_Re = rng.uniform(onp.log(1e5), onp.log(1e7), n_cases)
    Re = onp.exp(log_Re)
    Mach = rng.uniform(0.0, 0.35, n_cases)
    C_mu = rng.uniform(0.0, 0.08, n_cases)
    x_jet = rng.uniform(0.03, 0.35, n_cases)
    theta_jet = rng.uniform(10.0, 70.0, n_cases)

    upper = rng.normal(
        loc=onp.array([0.18, 0.20, 0.18, 0.14, 0.10, 0.07, 0.04, 0.02]),
        scale=0.025,
        size=(n_cases, 8),
    )
    lower = rng.normal(
        loc=onp.array([-0.10, -0.09, -0.08, -0.06, -0.04, -0.025, -0.015, -0.005]),
        scale=0.020,
        size=(n_cases, 8),
    )
    leading_edge_weight = rng.normal(0.0, 0.02, n_cases)
    TE_thickness = rng.uniform(0.0, 0.02, n_cases)

    camber_proxy = onp.mean(upper + lower, axis=1)
    thickness_proxy = onp.mean(upper - lower, axis=1)
    afc_effectiveness = (
        (1 - onp.exp(-60 * C_mu))
        * onp.exp(-((x_jet - 0.10) / 0.20) ** 2)
        * (0.5 + 0.5 * onp.cos(onp.radians(theta_jet - 30))) ** 2
    )

    CL = (
        2 * onp.pi * onp.radians(alpha + 2.0)
        + 3.5 * camber_proxy
        + 0.75 * afc_effectiveness
    )
    CD = (
        0.006
        + 0.03 * CL**2
        + 0.08 * (thickness_proxy - 0.20) ** 2
        + 0.004 * Mach**2
        + 0.002 * afc_effectiveness
    )
    CM = -0.05 - 0.22 * camber_proxy - 0.15 * afc_effectiveness * (1 - x_jet)

    Top_Xtr = onp.clip(0.8 - 0.04 * alpha - 1.5 * C_mu, 0, 1)
    Bot_Xtr = onp.clip(0.7 + 0.03 * alpha - 0.4 * C_mu, 0, 1)
    converged = rng.uniform(0, 1, n_cases) > 0.04
    confidence_label = onp.clip(
        0.95
        - 0.20 * onp.abs(alpha) / 20
        - 0.10 * Mach
        - 0.30 * C_mu
        - 0.60 * (~converged),
        0,
        1,
    )

    data = {
        **{KULFAN_UPPER_COLUMNS[i]: upper[:, i] for i in range(8)},
        **{KULFAN_LOWER_COLUMNS[i]: lower[:, i] for i in range(8)},
        "leading_edge_weight": leading_edge_weight,
        "TE_thickness": TE_thickness,
        "alpha": alpha,
        "Re": Re,
        "Mach": Mach,
        "n_crit": 9.0 * onp.ones(n_cases),
        "xtr_upper": onp.ones(n_cases),
        "xtr_lower": onp.ones(n_cases),
        "C_mu": C_mu,
        "x_jet": x_jet,
        "theta_jet": theta_jet,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        "converged": converged,
        "source": source,
        "confidence_label": confidence_label,
    }

    df = pd.DataFrame(data, columns=AFC_DATASET_COLUMNS)
    validate_afc_dataframe(df)
    return df
