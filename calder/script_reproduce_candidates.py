import argparse

from reproduce_candidates import build_reproduction_report


def main(args):
    report = build_reproduction_report(
        out_dir=args.out_dir,
        out_format=args.out_format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        metrics_dip_threshold=args.metrics_dip_threshold,
    )

    columns = [
        "source",
        "source_id",
        "mag_bin",
        "detected",
        "detection_details",
        "g_n_peaks",
        "v_n_peaks",
        "matches_expected",
    ]
    existing = [c for c in columns if c in report.columns]
    print(report[existing].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the targeted reproduction search and summarize results.")
    parser.add_argument("--out-dir", default="./results_test", help="Directory for peak_results output")
    parser.add_argument("--out-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--n-workers", type=int, default=10, help="ProcessPool worker count for naive_dip_finder")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per chunk flush for CSV output")
    parser.add_argument("--metrics-dip-threshold", type=float, default=0.3, help="Dip threshold for run_metrics")

    main(parser.parse_args())
