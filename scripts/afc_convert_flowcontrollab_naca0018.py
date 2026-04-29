from __future__ import annotations

import argparse
from pathlib import Path

import aerosandbox as asb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Flow Control Lab NACA0018 pressure-data text files to the AFC 2D dataset schema."
    )
    parser.add_argument(
        "--input-directory",
        required=True,
        help="Directory containing extracted Flow Control Lab .txt files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .csv or .parquet path. Large external outputs should stay outside the source tree.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional output manifest .json path. Defaults to <output>.manifest.json.",
    )
    parser.add_argument("--include-dynamic", action="store_true")
    parser.add_argument("--x-jet", type=float, default=0.10)
    parser.add_argument("--theta-jet", type=float, default=30.0)
    parser.add_argument("--top-xtr", type=float, default=1.0)
    parser.add_argument("--bot-xtr", type=float, default=1.0)
    parser.add_argument("--confidence-label", type=float, default=0.85)
    parser.add_argument(
        "--dataset-id",
        default="fcl_naca0018_quasi_steady_afc_schema",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    manifest = Path(args.manifest) if args.manifest else output.with_suffix(
        output.suffix + ".manifest.json"
    )

    df = asb.convert_flow_control_lab_naca0018_directory(
        input_directory=args.input_directory,
        output_path=output,
        manifest_path=manifest,
        include_dynamic=args.include_dynamic,
        dataset_id=args.dataset_id,
        x_jet=args.x_jet,
        theta_jet=args.theta_jet,
        Top_Xtr=args.top_xtr,
        Bot_Xtr=args.bot_xtr,
        confidence_label=args.confidence_label,
    )

    duplicate_report = asb.check_duplicate_afc_samples(df, decimals=8)
    print(f"Wrote {len(df)} AFC rows to {output}")
    print(f"Wrote manifest to {manifest}")
    print(f"Duplicate rows: {duplicate_report['n_duplicate_rows']}")
    print("Assumptions:")
    for key, value in asb.FLOW_CONTROL_LAB_NACA0018_ASSUMPTIONS.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

