"""
Command-line script to generate Pareto plots of execution_time vs total_area
from one or more design_point_summary JSON files.

Usage (single file):
    python3 -m src.inverse_pass.plot_pareto path/to/design_point_summary_iter_0.json

Usage (multiple files, overlaid):
    python3 -m src.inverse_pass.plot_pareto file1.json file2.json --labels run_a run_b

Options:
    --output-dir DIR      Directory to save the PNG (default: directory of first input file)
    --output-filename STR Filename for the saved PNG (default: pareto.png for multi-file,
                          <stem>_pareto.png for single file)
    --labels LABEL ...    Legend labels, one per file (default: filename stems)
    --show                Display interactively instead of saving
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto curve(s) of execution_time vs total_area."
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        metavar="JSON",
        help="One or more design_point_summary JSON files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Legend label for each file (must match number of JSON files).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the plot (default: directory of first input file).",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Output PNG filename (default: <stem>_pareto.png for single file, pareto.png for multi).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of saving.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.json_paths):
        parser.error(f"--labels count ({len(args.labels)}) must match number of JSON files ({len(args.json_paths)})")

    from src.inverse_pass.utils import plot_pareto_execution_time_vs_area

    output_dir = None if args.show else (
        args.output_dir or os.path.dirname(os.path.abspath(args.json_paths[0]))
    )

    kwargs = dict(
        json_path=args.json_paths if len(args.json_paths) > 1 else args.json_paths[0],
        labels=args.labels,
        output_dir=output_dir,
    )
    if args.output_filename:
        kwargs["output_filename"] = args.output_filename

    plot_pareto_execution_time_vs_area(**kwargs)

    if output_dir:
        print(f"Saved to {output_dir}/")

if __name__ == "__main__":
    main()
