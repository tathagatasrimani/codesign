"""Analyze how pipeline overhead scales with the base (no-pipeline) quantity.

Usage:
    python3 -m src.analysis.pipeline_overhead_plot <design_point_summary.json> [--out <output_dir>]

For each functional unit type (e.g. Mult16, Add16), produces one plot per metric
(delay, dynamic energy, leakage power, area) showing:
  - X axis: base value with no pipelining (e.g. base area)
  - Left Y axis: total value including pipeline overhead (log scale)
  - Right Y axis: pipeline overhead percentage (overhead / total * 100)
  - Vertical line: where DFF setup time equals the clock period (degenerate boundary)
"""

import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# (key_base, key_tot, y_label, filename_tag)
_METRICS = [
    ("delay_comb",   "delay_tot",   "Total delay (cycles)", "delay"),
    ("energy_base",  "energy_tot",  "Total energy (J)",     "energy"),
    ("leakage_base", "leakage_tot", "Total leakage (mW)",   "leakage"),
    ("area_base",    "area_tot",    "Total area (µm²)",     "area"),
]


def load_fu_data(summary_path):
    """Return {fu_name: [dict of per-point fields, ...]} across all valid design points."""
    with open(summary_path) as f:
        design_points = json.load(f)

    fu_data = defaultdict(list)
    for dp in design_points:
        if not math.isfinite(dp.get("obj_value", math.inf)):
            continue
        clk_period = dp.get("clk_period")
        if clk_period is None or clk_period <= 0:
            continue
        for fu_name, fu in dp.get("fu_logic", {}).items():
            if fu.get("delay_comb", 0) <= 0 or fu.get("delay_tot", 0) <= 0:
                continue
            fu_data[fu_name].append({**fu, "clk_period": clk_period})

    return fu_data


def _degenerate_x_cycles(delay_comb_cyc, delay_tot_cyc):
    """Combinational delay (cycles) at which DFF_DELAY = clk_period.

    In cycle units: delay_tot = delay_comb / (1 - DFF_DELAY_cycles)
    k = DFF_DELAY_cycles / delay_comb_cycles is constant (both ∝ t).
    Degenerate point: delay_comb* = 1/k.
    """
    k = np.median((1 - delay_comb_cyc / delay_tot_cyc) / delay_comb_cyc)
    assert k > 0, f"DFF/comb ratio must be positive, got {k}"
    return 1.0 / k


def _plot_metric(fu_name, delay_comb_cyc, base_vals, tot_vals, degen_x,
                 y_label, filename_tag, out_dir):
    overhead_pct = (tot_vals - base_vals) / tot_vals * 100

    order = np.argsort(delay_comb_cyc)
    x = delay_comb_cyc[order]
    base = base_vals[order]
    tot = tot_vals[order]
    pct = overhead_pct[order]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color_main = "#1f77b4"
    color_overhead = "#d62728"
    color_degen = "#2ca02c"

    ax1.plot(x, base, color="gray", linewidth=1.0, linestyle=":", label="No overhead (base)")
    ax1.plot(x, tot, color=color_main, linewidth=1.5, label=f"Total {filename_tag}")
    ax1.scatter(x, tot, color=color_main, s=12, zorder=3)
    ax1.axvline(degen_x, color=color_degen, linewidth=1.2, linestyle="-.",
                label="DFF setup time = clk period")
    ax1.set_xlabel("Combinational delay (cycles)")
    ax1.set_ylabel(y_label, color=color_main)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color_main)

    ax2 = ax1.twinx()
    ax2.plot(x, pct, color=color_overhead, linewidth=1.5, linestyle="--", label="Pipeline overhead")
    ax2.scatter(x, pct, color=color_overhead, s=12, zorder=3)
    ax2.set_ylabel(f"Pipeline overhead (% of total {filename_tag})", color=color_overhead)
    ax2.tick_params(axis="y", labelcolor=color_overhead)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title(f"Pipeline overhead ({filename_tag}) — {fu_name}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline_overhead_{fu_name}_{filename_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved {out_path}")


def plot_fu(fu_name, points, out_dir):
    clk_periods = np.array([p["clk_period"] for p in points])
    delay_comb_cyc = np.array([p["delay_comb"] for p in points]) / clk_periods
    delay_tot_cyc = np.array([p["delay_tot"] for p in points]) / clk_periods

    degen_x = _degenerate_x_cycles(delay_comb_cyc, delay_tot_cyc)

    for key_base, key_tot, y_label, tag in _METRICS:
        if key_base == "delay_comb":
            base_vals = delay_comb_cyc
            tot_vals = delay_tot_cyc
        else:
            if key_base not in points[0]:
                continue
            base_vals = np.array([p[key_base] for p in points])
            tot_vals = np.array([p[key_tot] for p in points])

        _plot_metric(fu_name, delay_comb_cyc, base_vals, tot_vals, degen_x,
                     y_label, tag, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="Path to design_point_summary_iter_N.json")
    parser.add_argument("--out", default=None, help="Output directory (default: same dir as summary)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.summary)), "pipeline_overhead_plots")

    print(f"Loading {args.summary}...")
    fu_data = load_fu_data(args.summary)

    if not fu_data:
        print("No fu_logic data found — ensure the summary was generated with pipeline breakdown fields.")
        return

    print(f"Found {len(fu_data)} functional unit types: {sorted(fu_data)}")
    for fu_name, points in sorted(fu_data.items()):
        print(f"  Plotting {fu_name} ({len(points)} design points)...")
        plot_fu(fu_name, points, out_dir)

    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
