"""
MALCA - Multi-timescale ASAS-SN Light Curve Analysis

Unified command-line interface for the MALCA pipeline.

Usage:
    python -m malca manifest [options]    # Build source_id → path index
    python -m malca detect [options]      # Run event detection
    python -m malca validate [options]    # Validate on known objects
    python -m malca plot [options]        # Plot light curves
    python -m malca score [options]       # Score events
    python -m malca filter [options]      # Apply filters
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="malca",
        description="MALCA: Multi-timescale ASAS-SN Light Curve Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Manifest command
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Build manifest (source_id → path index)",
        description="Build a manifest that maps ASAS-SN IDs to their light-curve paths"
    )
    manifest_parser.add_argument("--help-full", action="store_true", help="Show full help for manifest command")
    
    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run event detection pipeline",
        description="Run Bayesian event detection on light curves"
    )
    detect_parser.add_argument("--help-full", action="store_true", help="Show full help for detect command")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate detection on known objects",
        description="Run reproduction/validation on known candidates"
    )
    validate_parser.add_argument("--help-full", action="store_true", help="Show full help for validate command")
    
    # Validation command (results-based, no raw data needed)
    validation_parser = subparsers.add_parser(
        "validation",
        help="Validate results against known candidates",
        description="Compare detection results to known candidates without raw data"
    )
    validation_parser.add_argument("--help-full", action="store_true", help="Show full help for validation command")
    
    # Plot command
    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot light curves with events",
        description="Generate light curve plots with event overlays"
    )
    plot_parser.add_argument("--help-full", action="store_true", help="Show full help for plot command")
    
    # Score command
    score_parser = subparsers.add_parser(
        "score",
        help="Score detected events",
        description="Compute event quality scores for dips or microlensing"
    )
    score_parser.add_argument("--help-full", action="store_true", help="Show full help for score command")
    
    # Filter command (pre/post)
    filter_parser = subparsers.add_parser(
        "filter",
        help="Apply quality filters",
        description="Apply pre-filters or post-filters to candidates"
    )
    filter_parser.add_argument("--help-full", action="store_true", help="Show full help for filter command")
    
    args, remaining = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Dispatch to appropriate module
    if args.command == "manifest":
        if hasattr(args, 'help_full') and args.help_full:
            from malca import manifest
            manifest.parse_args(['--help'])
        else:
            from malca import manifest
            sys.argv = [sys.argv[0]] + remaining
            manifest.main()
    
    elif args.command == "detect":
        if hasattr(args, 'help_full') and args.help_full:
            from malca import events_filtered
            sys.argv = [sys.argv[0], '--help']
            events_filtered.main()
        else:
            from malca import events_filtered
            sys.argv = [sys.argv[0]] + remaining
            events_filtered.main()
    
    elif args.command == "validate":
        if hasattr(args, 'help_full') and args.help_full:
            from tests import reproduction
            sys.argv = [sys.argv[0], '--help']
            reproduction.main()
        else:
            from tests import reproduction
            sys.argv = [sys.argv[0]] + remaining
            reproduction.main()
    
    elif args.command == "plot":
        if hasattr(args, 'help_full') and args.help_full:
            from malca import plot
            sys.argv = [sys.argv[0], '--help']
            plot.main()
        else:
            from malca import plot
            sys.argv = [sys.argv[0]] + remaining
            plot.main()
    
    elif args.command == "score":
        if hasattr(args, 'help_full') and args.help_full:
            from malca import score
            sys.argv = [sys.argv[0], '--help']
            score.main()
        else:
            from malca import score
            sys.argv = [sys.argv[0]] + remaining
            score.main()
    
    elif args.command == "filter":
        if hasattr(args, 'help_full') and args.help_full:
            from malca import post_filter
            sys.argv = [sys.argv[0], '--help']
            post_filter.main()
        else:
            from malca import post_filter
            sys.argv = [sys.argv[0]] + remaining
            post_filter.main()
    
    elif args.command == "validation":
        if hasattr(args, 'help_full') and args.help_full:
            from tests import validation
            sys.argv = [sys.argv[0], '--help']
            validation.main()
        else:
            from tests import validation
            sys.argv = [sys.argv[0]] + remaining
            validation.main()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
