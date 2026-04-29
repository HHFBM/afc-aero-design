"""
Run the AFC validation suite and write acceptance reports.

Large CFD/RANS/experimental validation datasets should stay outside the repository and be passed with
`--config` or `--external-2d-dataset`.
"""

import argparse
from pathlib import Path

import aerosandbox as asb


def parse_args():
    parser = argparse.ArgumentParser(description="Run AFC validation and regression checks.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Validation config file, e.g. configs/afc/validation_config.example.yaml",
    )
    parser.add_argument(
        "--external-2d-dataset",
        type=str,
        default=None,
        help="Optional external CSV/Parquet AFC validation dataset following the canonical schema.",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default=None,
        help="Directory where JSON and Markdown reports will be written.",
    )
    parser.add_argument(
        "--synthetic-2d-cases",
        type=int,
        default=12,
        help="Number of synthetic cases used when no external dataset is supplied.",
    )
    parser.add_argument(
        "--include-avl",
        action="store_true",
        help="Also attempt AVL baseline comparison if AVL is available in the environment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config is not None:
        config = asb.read_afc_config(
            args.config,
            required_fields=[
                "afc_config_version",
                "config_type",
                "benchmark.model_size",
                "report.output_directory",
            ],
        )
        if args.external_2d_dataset is not None:
            config.setdefault("external_data", {})
            config["external_data"]["external_2d_dataset"] = args.external_2d_dataset
        output_directory = Path(args.output_directory or config["report"]["output_directory"])
        report = asb.generate_afc_validation_acceptance_report(
            config,
            config_path=args.config,
            output_directory=output_directory,
        )
        paths = asb.write_afc_validation_report(report, output_directory=output_directory)
    else:
        output_directory = Path(args.output_directory or "validation_outputs/afc")
        report = asb.run_afc_validation_suite(
            external_2d_dataset=args.external_2d_dataset,
            synthetic_2d_cases=args.synthetic_2d_cases,
            include_avl=args.include_avl,
        )
        paths = asb.write_afc_validation_report(
            report,
            output_directory=output_directory,
        )

    print(f"AFC validation status: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"JSON report     : {paths['json']}")
    print(f"Markdown report : {paths['markdown']}")

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
