"""Back up the original enriched table and publish completed local statistics."""
from argparse import ArgumentParser
from pathlib import Path
import time

from malca.products.enriched_refresh import backup_original_enriched, promote_local_features
from malca.products.stage_state import read_stage_state


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backup-only", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait up to six hours for the current statistics job to finish")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if args.backup_only:
        print(f"Original enriched table backed up: {backup_original_enriched(run_dir)}")
    else:
        if args.wait:
            print("Waiting for the current statistics job to finish; the original standard table remains in place.", flush=True)
            deadline = time.monotonic() + 6 * 3600
            while True:
                state = read_stage_state(run_dir / "results/local_ml_features/LOCAL_FEATURES_STAGE.json")
                if state is None or state["result"]["status"] != "running":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Statistics did not finish within six hours; the standard table was not replaced")
                time.sleep(30)
        print(f"Standard enriched table updated: {promote_local_features(run_dir)}")
