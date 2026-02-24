"""Regenerate all visualisation plots from a previous codesign log directory."""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.inverse_pass.utils import regenerate_plots_from_log_dir

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log_dir", help="Path to the log directory containing .pkl result files")
parser.add_argument("--obj-type", default="edp", help="Objective type for axis labels (default: edp)")
parser.add_argument("--top-percent", type=float, default=1.0, help="Fraction of top designs to visualise (default: 1.0 = all)")
args = parser.parse_args()

regenerate_plots_from_log_dir(args.log_dir, obj_type=args.obj_type, top_percent=args.top_percent)
