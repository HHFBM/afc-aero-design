from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as onp
import pandas as pd

from aerosandbox.aerodynamics.aero_2D.afc_dataset import (
    AFC_DATASET_COLUMNS,
    KULFAN_LOWER_COLUMNS,
    KULFAN_UPPER_COLUMNS,
    make_afc_dataset_manifest,
    validate_afc_dataframe,
    write_afc_dataset,
)


FLOW_CONTROL_LAB_NACA0018_ASSUMPTIONS = {
    "airfoil": "NACA0018 represented by AeroSandbox KulfanAirfoil('naca0018').",
    "x_jet": "Not provided per sample in the public FCL text tables; set by converter argument.",
    "theta_jet": "Not provided per sample in the public FCL text tables; set by converter argument in degrees.",
    "Top_Xtr": "Transition labels are not provided; set by converter argument.",
    "Bot_Xtr": "Transition labels are not provided; set by converter argument.",
    "n_crit": "Not an experimental field; set by converter argument for NeuralFoil compatibility.",
    "xtr_upper": "Forced transition input, not measured; set by converter argument.",
    "xtr_lower": "Forced transition input, not measured; set by converter argument.",
    "Mach": "Estimated from U_infty divided by converter speed_of_sound.",
    "C_mu": "Converted from FCL Cmu_[%] by dividing by 100.",
}


def _parse_reynolds_from_header(header: str) -> Optional[float]:
    match = re.search(r"Re\s*=\s*([0-9.+\-eE]+)", header)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_measurement_id(header: str, filepath: Path) -> str:
    match = re.search(r"measurement\s*#?\s*([0-9]+)", header, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1)
    match = re.search(r"#([0-9]+)", filepath.stem)
    if match is not None:
        return match.group(1)
    return filepath.stem


def _get_column(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    normalized = {str(column).strip(): column for column in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return df[normalized[candidate]]
    raise KeyError(f"Could not find any of columns {list(candidates)} in FCL table.")


def _naca0018_kulfan_parameters() -> Dict[str, object]:
    import aerosandbox as asb

    airfoil = asb.KulfanAirfoil("naca0018")
    return {
        "upper_weights": onp.asarray(airfoil.upper_weights, dtype=float),
        "lower_weights": onp.asarray(airfoil.lower_weights, dtype=float),
        "leading_edge_weight": float(airfoil.leading_edge_weight),
        "TE_thickness": float(airfoil.TE_thickness),
    }


def read_flow_control_lab_txt_file(
    filepath: Union[str, Path],
    *,
    include_dynamic: bool = False,
    speed_of_sound: float = 340.294,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    x_jet: float = 0.10,
    theta_jet: float = 30.0,
    Top_Xtr: float = 1.0,
    Bot_Xtr: float = 1.0,
    confidence_label: float = 0.85,
    source: str = "FlowControlLab_NACA0018_quasi_steady_pressure",
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Converts one Flow Control Lab NACA0018 text file into the canonical AFC 2D schema.

    The public FCL text files do not contain all NeuralFoil/AFC fields. Missing quantities are filled using explicit
    converter arguments and recorded in the dataset manifest by `convert_flow_control_lab_naca0018_directory()`.
    """
    filepath = Path(filepath)
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        return pd.DataFrame(columns=AFC_DATASET_COLUMNS)

    header = lines[0].strip()
    is_quasi_steady = "quasi-steady" in header.lower()
    if not include_dynamic and not is_quasi_steady:
        return pd.DataFrame(columns=AFC_DATASET_COLUMNS)

    raw = pd.read_csv(filepath, sep="\t", skiprows=1)
    raw = raw.rename(columns=lambda column: str(column).strip())

    alpha = pd.to_numeric(_get_column(raw, ["AoA_[deg]", "AoA  [°]", "AoA_[°]"]))
    velocity = pd.to_numeric(
        _get_column(raw, ["U_infty_[m/s]", "U∞  [m/s]", "U_infty [m/s]"])
    )
    CL = pd.to_numeric(_get_column(raw, ["Cl", "CL"]))
    CD = pd.to_numeric(_get_column(raw, ["Cd", "CD"]))
    CM = pd.to_numeric(_get_column(raw, ["Cm", "CM"]))
    C_mu = pd.to_numeric(_get_column(raw, ["Cmu_[%]", "Cµ  [%]", "C_mu_[%]"])) / 100

    re_from_header = _parse_reynolds_from_header(header)
    if "Re" in raw.columns:
        Re = pd.to_numeric(raw["Re"])
    elif re_from_header is not None:
        Re = pd.Series(re_from_header * onp.ones(len(raw)))
    else:
        raise ValueError(f"Could not infer Reynolds number from {filepath}.")

    kulfan = _naca0018_kulfan_parameters()
    n = len(raw)
    data = {
        **{KULFAN_UPPER_COLUMNS[i]: kulfan["upper_weights"][i] for i in range(8)},
        **{KULFAN_LOWER_COLUMNS[i]: kulfan["lower_weights"][i] for i in range(8)},
        "leading_edge_weight": kulfan["leading_edge_weight"],
        "TE_thickness": kulfan["TE_thickness"],
        "alpha": alpha,
        "Re": Re,
        "Mach": velocity / speed_of_sound,
        "n_crit": n_crit,
        "xtr_upper": xtr_upper,
        "xtr_lower": xtr_lower,
        "C_mu": C_mu,
        "x_jet": x_jet,
        "theta_jet": theta_jet,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        "converged": True,
        "source": source,
        "confidence_label": confidence_label,
    }
    df = pd.DataFrame(data, columns=AFC_DATASET_COLUMNS)

    if drop_invalid:
        numeric = df[["alpha", "Re", "Mach", "C_mu", "CL", "CD", "CM"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = onp.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        valid &= numeric["CD"].to_numpy(dtype=float) > 0
        df = df.loc[valid].reset_index(drop=True)

    measurement_id = _parse_measurement_id(header, filepath)
    if not is_quasi_steady:
        df["source"] = f"{source}_dynamic_measurement_{measurement_id}"

    validate_afc_dataframe(df)
    return df


def convert_flow_control_lab_naca0018_directory(
    input_directory: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    include_dynamic: bool = False,
    file_glob: str = "**/*.txt",
    dataset_id: str = "fcl_naca0018_quasi_steady_afc_schema",
    x_jet: float = 0.10,
    theta_jet: float = 30.0,
    Top_Xtr: float = 1.0,
    Bot_Xtr: float = 1.0,
    confidence_label: float = 0.85,
    filters: Optional[Dict[str, object]] = None,
    known_limitations: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Converts a Flow Control Lab NACA0018 directory of `.txt` files into the AFC dataset schema.

    Args:
        input_directory: Directory containing the extracted FCL pressure-data text files.
        output_path: Optional CSV/Parquet output path.
        manifest_path: Optional JSON manifest output path.
        include_dynamic: If False, only quasi-steady pitch measurements are converted.
    """
    input_directory = Path(input_directory)
    files = sorted(input_directory.glob(file_glob))
    frames = [
        read_flow_control_lab_txt_file(
            file,
            include_dynamic=include_dynamic,
            x_jet=x_jet,
            theta_jet=theta_jet,
            Top_Xtr=Top_Xtr,
            Bot_Xtr=Bot_Xtr,
            confidence_label=confidence_label,
        )
        for file in files
    ]
    frames = [frame for frame in frames if len(frame) > 0]
    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=AFC_DATASET_COLUMNS)
    )
    validate_afc_dataframe(df)

    manifest = make_afc_dataset_manifest(
        df,
        dataset_id=dataset_id,
        source=sorted(df["source"].astype(str).unique().tolist()) if len(df) else [],
        split_method="unsplit; train/val/test split happens in training script",
        filters={
            "include_dynamic": include_dynamic,
            "file_glob": file_glob,
            "drop_invalid": True,
            **dict(filters or {}),
        },
        known_limitations=list(
            known_limitations
            or [
                "Only NACA0018 public pressure-data coefficients are represented.",
                "x_jet and theta_jet are fixed assumptions unless provided externally.",
                "Top_Xtr and Bot_Xtr are fixed placeholder labels; the FCL tables do not provide transition locations.",
                "The converter estimates Mach from U_infty and does not reconstruct tunnel blockage or uncertainty.",
            ]
        ),
        validate=False,
    )
    manifest["assumptions"] = FLOW_CONTROL_LAB_NACA0018_ASSUMPTIONS

    if output_path is not None:
        write_afc_dataset(df, output_path, manifest=manifest, manifest_path=manifest_path)
    elif manifest_path is not None:
        from aerosandbox.aerodynamics.aero_2D.afc_dataset import write_afc_dataset_manifest

        write_afc_dataset_manifest(manifest, manifest_path)

    return df

