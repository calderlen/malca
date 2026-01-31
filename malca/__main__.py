"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    python -m malca manifest [options]    # Build source_id → path index
    python -m malca detect [options]      # Run event detection
    python -m malca validate [options]    # Validate on known objects
    python -m malca injection [options]   # Run injection-recovery tests
    python -m malca detection_rate [options]  # Measure detection rate
    python -m malca plot [options]        # Plot light curves
    python -m malca post_filter [options] # Apply post-filters
    python -m malca postprocess [options] # Plot passing candidates
"""

import sys
import argparse


def main():
    # Check if user is calling a subcommand with --help
    # If so, forward directly to the submodule
    if len(sys.argv) >= 2 and sys.argv[1] in [
        "manifest", "detect", "reproduce", "injection", 
        "detection_rate", "validate", "plot", "postprocess", "post_filter"
    ]:
        command = sys.argv[1]
        remaining = sys.argv[2:]
        
        # Dispatch to appropriate module (--help will be handled by that module)
        if command == "manifest":
            from malca import manifest
            sys.argv = [sys.argv[0]] + remaining
            manifest.main()
        elif command == "detect":
            from malca import detect
            sys.argv = [sys.argv[0]] + remaining
            detect.main()
        elif command == "reproduce":
            from tests import reproduce
            sys.argv = [sys.argv[0]] + remaining
            reproduce.main()
        elif command == "injection":
            from malca import injection
            sys.argv = [sys.argv[0]] + remaining
            injection.main()
        elif command == "detection_rate":
            from malca import detection_rate
            sys.argv = [sys.argv[0]] + remaining
            detection_rate.main()
        elif command == "plot":
            from malca import plot
            sys.argv = [sys.argv[0]] + remaining
            plot.main()
        elif command == "postprocess":
            from malca import postprocess
            sys.argv = [sys.argv[0]] + remaining
            postprocess.main()
        elif command == "post_filter":
            from malca import post_filter
            sys.argv = [sys.argv[0]] + remaining
            post_filter.main()
        elif command == "validate":
            from tests import validation
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
        return 0
    
    # If no subcommand or just --help for main, show main help
    parser = argparse.ArgumentParser(
        prog="malca",
        description="MALCA: Multi-timescale ASAS-SN Light Curve Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("manifest", help="Build manifest (source_id → path index)")
    subparsers.add_parser("detect", help="Run event detection pipeline")
    subparsers.add_parser("reproduce", help="Re-run detection on known objects (needs raw data)")
    subparsers.add_parser("injection", help="Run injection-recovery tests")
    subparsers.add_parser("detection_rate", help="Measure detection rate")
    subparsers.add_parser("validate", help="Validate results against known candidates")
    subparsers.add_parser("plot", help="Plot light curves with events")
    subparsers.add_parser("postprocess", help="Plot passing candidates from post-filter output")
    subparsers.add_parser("post_filter", help="Apply quality post-filters")
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
