"""Import an all-bin cluster bundle and point its passing rows at local LCs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from malca.io.table_io import read_feature_table, write_feature_table
from malca.products.feature_layers import with_feature_columns
from malca.products.run_bundle import import_bundle_zip


def prepare(bundle: Path, run_dir: Path) -> None:
    run_dir = run_dir.expanduser().resolve()
    results = run_dir / "results"
    if (run_dir / "review/review.db").exists() or any(results.glob("lc_events_characterized*")):
        raise RuntimeError("Local enrichment has already started; resume home without importing again.")
    import_bundle_zip(bundle, run_dir, show_progress=True)
    originals = run_dir / "cluster_originals"
    originals.mkdir(exist_ok=True)
    local_lcs = run_dir / "bundle_assets/lightcurves"
    for name in ("run_params.json", "run_summary.json"):
        source = run_dir / name
        if source.is_file() and not (originals / name).exists():
            shutil.copy2(source, originals / name)
    for stem in ("lc_events_filtered", "lc_events_enriched"):
        tagged = results / f"{stem}_all.parquet"
        original = originals / tagged.name
        if not original.exists():
            shutil.copy2(tagged, original)
        frame = read_feature_table(original)
        decisions = with_feature_columns(frame, ["failed_any"])["failed_any"].astype("boolean")
        if decisions.isna().any():
            raise ValueError(f"Missing filter decisions in {original}")
        paths = frame["lc_path"].map(lambda value: local_lcs / Path(str(value)).name)
        available = paths.map(Path.is_file)
        if (~decisions & ~available).any():
            raise FileNotFoundError(f"Bundle lacks {int((~decisions & ~available).sum())} passing LCs")
        frame.loc[available, "lc_path"] = paths.loc[available].astype(str)
        write_feature_table(frame, tagged)
        shutil.copy2(tagged, results / f"{stem}.parquet")
        print(f"{stem}: {int((~decisions).sum())} passers, local LC paths verified")
    print(f"Ready for --stage home --output-dir {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bundle.expanduser(), args.run_dir)
