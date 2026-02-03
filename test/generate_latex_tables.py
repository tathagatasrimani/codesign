#!/usr/bin/env python3
"""
Generate LaTeX tables from codesign regression results.

This script parses through regression result directories and creates
formatted LaTeX tables with configurable metrics.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Default metrics mapping: JSON key -> (LaTeX column header, unit, format)
DEFAULT_METRICS = {
    "supply voltage": ("$V_{dd}$", "V", ".2f"),
    "effective threshold voltage": ("$V_{th}$", "V", ".2f"),
    "gate length": ("$L_g$", "nm", ".0f"),  # multiply by 1e9
    #"gate width": ("$W_g$", "nm", ".0f"),   # multiply by 1e9
    "eot_corrected": ("$t_{oxTeq}$", "nm", ".1f"),      # multiply by 1e9
    #"k_gate": ("$k_{gate}$", "", ".1f"),
    "tsemi": ("$t_{semi}$", "nm", ".0f"),
    "scale length": ("$L_{scale}$", "nm", ".0f"),
    #"MUL": ("$MUL$", "", ".0f"),
    #"GEO": ("$GEO$", "", ".0f"),
}

# Additional metrics for diff_benchmark experiments
DIFF_BENCHMARK_METRICS = {
    "supply voltage": ("$V_{dd}$", "V", ".2f"),
    "effective threshold voltage": ("$V_{th}$", "V", ".2f"),
    "gate length": ("$L_g$", "nm", ".0f"),
    "eot_corrected": ("$t_{oxTeq}$", "nm", ".1f"),
    "tsemi": ("$t_{semi}$", "nm", ".0f"),
    "scale length": ("$L_{scale}$", "nm", ".0f"),
    #"MUL": ("$MUL$", "", ".0f"),
    #"GEO": ("$GEO$", "", ".0f"),
    "max_wire_length": ("$L_{wire,max}$", "mm", ".2f"),
}

# Scale factors for converting to display units
SCALE_FACTORS = {
    "gate length": 1e9,  # m -> nm
    "gate width": 1e9,   # m -> nm
    "t_ox": 1e9,         # m -> nm
    "scale length": 1e9, # m -> nm
    "k_gate": 1,       # m -> nm
    "tsemi": 1e9,        # m -> nm
    "eot_corrected": 1e9,         # m -> nm
    "MUL": 1,           # multiplier
    "GEO": 1,           # geometry factor
    "max_wire_length": 1e3,  # m -> mm
}


def get_latest_log_dir(log_path: str) -> Optional[str]:
    """Get the most recent log directory based on timestamp in name."""
    if not os.path.exists(log_path):
        return None

    log_dirs = [d for d in os.listdir(log_path)
                if os.path.isdir(os.path.join(log_path, d))]

    if not log_dirs:
        return None

    # Sort by name (timestamps sort correctly alphabetically)
    log_dirs.sort(reverse=True)
    return os.path.join(log_path, log_dirs[0])


def load_param_data(results_dir: str, load_wire_lengths: bool = False) -> Optional[Dict]:
    """Load the final parameter data from a results directory."""
    param_path = os.path.join(results_dir, "figs", "plot_param_data.json")

    if not os.path.exists(param_path):
        print(f"Warning: {param_path} not found")
        return None

    try:
        with open(param_path, "r") as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: {param_path} is empty")
                return None
            param_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse {param_path}: {e}")
        return None

    # Return the last (final) iteration
    result = param_data[-1] if param_data else None

    # Optionally load max wire length for diff_benchmark experiments
    if result and load_wire_lengths:
        wire_path = os.path.join(results_dir, "wire_lengths_0.json")
        if os.path.exists(wire_path):
            try:
                with open(wire_path, "r") as f:
                    wire_lengths = json.load(f)
                    if wire_lengths:
                        result["max_wire_length"] = max(wire_lengths)
                        print(f"Max wire length: {result['max_wire_length']}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to parse {wire_path}: {e}")

    return result


def format_value(value: float, metric_key: str, metrics: Dict) -> str:
    """Format a value with appropriate scaling and formatting."""
    if value is None:
        return "—"

    # Apply scale factor if needed
    scale = SCALE_FACTORS.get(metric_key, 1.0)
    scaled_value = value * scale

    # Get format string
    _, unit, fmt = metrics.get(metric_key, ("", "", ".2f"))

    # Format the value
    formatted = f"{scaled_value:{fmt}}"

    # Add unit if present
    if unit:
        return f"{formatted}{unit}"
    return formatted


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    # Escape underscores (but not in math mode)
    return text.replace("_", r"\_")


def clean_benchmark_name(dir_name: str) -> str:
    """Clean up benchmark name by removing 'test1', 'streamhls', 'vitis', and underscores."""
    name = dir_name

    # Remove 'test1' (case insensitive, with optional leading underscore)
    name = re.sub(r'_?test1$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_?test1_', '_', name, flags=re.IGNORECASE)

    # Remove 'streamhls' (case insensitive, with optional surrounding underscores)
    name = re.sub(r'^streamhls_?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_?streamhls_?', '_', name, flags=re.IGNORECASE)

    # Remove 'vitis' (case insensitive, with optional surrounding underscores)
    name = re.sub(r'^vitis_?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_?vitis_?', '_', name, flags=re.IGNORECASE)

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Clean up any double spaces and strip
    name = ' '.join(name.split())

    return name


def extract_objective_name(dir_name: str, use_full_name: bool = False, is_diff_benchmark: bool = False) -> str:
    """Extract a human-readable objective name from directory name."""
    # For diff_benchmark experiments, clean up the benchmark name
    if is_diff_benchmark:
        return clean_benchmark_name(dir_name)

    # Common patterns: gemm_delay, gemm_energy, gemm_edp, gemm_ed2
    # Also: gemm_100_bulk, gemm_1000_dg_ns
    parts = dir_name.split("_")

    # Map common objectives to LaTeX-friendly names
    obj_map = {
        "delay": r"\textit{delay}",
        "energy": r"\textit{energy}",
        "edp": r"$\textit{energy}\cdot\textit{delay}$",
        "ed2": r"$\textit{energy}\cdot\textit{delay}^2$",
    }

    if use_full_name or len(parts) > 2:
        # For names like gemm_100_bulk, use the full suffix after benchmark
        suffix = "_".join(parts[1:]) if len(parts) > 1 else dir_name
        # Check if the last part is a known objective
        if parts[-1] in obj_map:
            return obj_map[parts[-1]]
        # Escape underscores for LaTeX
        return escape_latex(suffix)

    if len(parts) >= 2:
        obj = parts[-1]
        return obj_map.get(obj, obj)

    return dir_name


def collect_results_from_experiment_dir(experiment_dir: str, is_diff_benchmark: bool = False) -> List[Tuple[str, str, Dict]]:
    """
    Collect results from an experiment directory.

    Returns list of tuples: (run_name, display_name, param_data)
    """
    results = []

    if not os.path.exists(experiment_dir):
        print(f"Warning: {experiment_dir} does not exist")
        return results

    # Get all run directories
    run_dirs = sorted([d for d in os.listdir(experiment_dir)
                       if os.path.isdir(os.path.join(experiment_dir, d))])

    # Check if we need full names (to avoid duplicates)
    short_names = [extract_objective_name(d, use_full_name=False, is_diff_benchmark=is_diff_benchmark) for d in run_dirs]
    use_full_name = len(short_names) != len(set(short_names))

    for run_dir in run_dirs:
        run_path = os.path.join(experiment_dir, run_dir)
        log_path = os.path.join(run_path, "log")

        # Get latest log directory
        latest_log = get_latest_log_dir(log_path)
        if not latest_log:
            print(f"Warning: No log directory found in {run_path}")
            continue

        # Load parameter data (include wire lengths for diff_benchmark)
        param_data = load_param_data(latest_log, load_wire_lengths=is_diff_benchmark)
        if param_data is None:
            continue

        display_name = extract_objective_name(run_dir, use_full_name=use_full_name, is_diff_benchmark=is_diff_benchmark)
        results.append((run_dir, display_name, param_data))

    return results


def generate_latex_table(
    results: List[Tuple[str, str, Dict]],
    metrics: Dict,
    caption: str = "Final technology configurations",
    benchmark_name: str = "",
    max_dsp: Optional[int] = None,
    include_caption: bool = False,
    first_column_header: str = "Objective",
) -> str:
    """Generate a LaTeX table from the results."""

    # Build column headers
    metric_keys = list(metrics.keys())
    headers = [metrics[k][0] for k in metric_keys]

    # Number of columns: 1 for objective/benchmark + len(metrics)
    num_cols = len(headers) + 1
    col_spec = "l" + "c" * len(headers)

    # Build caption
    if benchmark_name:
        full_caption = f"Final technology configurations for \\textit{{{benchmark_name}}}"
        if max_dsp:
            full_caption += f": (maximum DSP: {max_dsp})"
    else:
        full_caption = caption

    # Start building the table
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
    ]
    if include_caption:
        lines.append(f"\\caption{{{full_caption}.}}")
    lines.extend([
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ])

    # Add multicolumn header
    title_text = f"Final technology configurations for \\textit{{{benchmark_name}}}" if benchmark_name else caption
    lines.append(f"\\multicolumn{{{num_cols}}}{{c}}{{\\textbf{{{title_text}}}}} \\\\")
    lines.append(r"\midrule")

    # Add column headers
    header_row = f"\\textbf{{{first_column_header}}} & " + " & ".join(headers) + r" \\"
    lines.append(header_row)
    lines.append(r"\midrule")

    # Add data rows
    for run_name, display_name, param_data in results:
        row_values = [display_name]
        for metric_key in metric_keys:
            value = param_data.get(metric_key)
            formatted = format_value(value, metric_key, metrics)
            row_values.append(formatted)

        row = " & ".join(row_values) + r" \\"
        lines.append(row)

    # Close the table
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_power_density_table(
    results: List[Tuple[str, str, Dict]],
    metrics: Dict,
    caption: str = "Technology configurations by power density constraint",
    benchmark_name: str = "",
    include_caption: bool = False,
) -> str:
    """Generate a LaTeX table for power density experiments with device type on the side."""

    # Parse results to extract power density and device type
    bulk_results = []
    dg_ns_results = []

    for run_name, display_name, param_data in results:
        # Parse folder name like "gemm_1000_bulk" or "gemm_100_dg_ns"
        parts = run_name.split("_")
        if len(parts) >= 3:
            # Extract power density value (e.g., "1000", "100", "10", "1", "05", "0001")
            power_density_str = parts[1]
            device_type = "_".join(parts[2:])  # "bulk" or "dg_ns"

            # Parse power density: "05" -> 0.5, "0001" -> 0.001, "1000" -> 1000
            if power_density_str.startswith("0") and len(power_density_str) > 1:
                # Leading zero indicates decimal: "05" -> "0.5", "0001" -> "0.001"
                power_density_val = float(f"0.{power_density_str[1:]}")
            else:
                power_density_val = float(power_density_str)

            # Format power density for display
            if power_density_val < 1:
                power_str = f"{power_density_val:g} W/cm$^2$"
            else:
                power_str = f"{int(power_density_val)} W/cm$^2$"

            if "bulk" in device_type:
                bulk_results.append((run_name, power_str, param_data, power_density_val))
            elif "dg" in device_type or "ns" in device_type:
                dg_ns_results.append((run_name, power_str, param_data, power_density_val))

    # Sort by power density (descending)
    bulk_results.sort(key=lambda x: x[3], reverse=True)
    dg_ns_results.sort(key=lambda x: x[3], reverse=True)

    # Build column headers
    metric_keys = list(metrics.keys())
    headers = [metrics[k][0] for k in metric_keys]

    # Number of columns: 1 for device type + 1 for power density + len(metrics)
    num_cols = len(headers) + 2
    col_spec = "ll" + "c" * len(headers)

    # Build caption
    if benchmark_name:
        full_caption = f"Technology configurations for \\textit{{{benchmark_name}}} by power density constraint"
    else:
        full_caption = caption

    # Start building the table
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
    ]
    if include_caption:
        lines.append(f"\\caption{{{full_caption}.}}")
    lines.extend([
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ])

    # Add column headers
    header_row = r"\textbf{Device} & \textbf{Power Density} & " + " & ".join(headers) + r" \\"
    lines.append(header_row)
    lines.append(r"\midrule")

    # Add Bulk section with multirow
    if bulk_results:
        n_bulk = len(bulk_results)
        for i, (run_name, power_str, param_data, _) in enumerate(bulk_results):
            row_values = []
            if i == 0:
                # First row gets the multirow device label
                row_values.append(f"\\multirow{{{n_bulk}}}{{*}}{{Bulk}}")
            else:
                row_values.append("")  # Empty cell for subsequent rows
            row_values.append(power_str)
            for metric_key in metric_keys:
                value = param_data.get(metric_key)
                formatted = format_value(value, metric_key, metrics)
                row_values.append(formatted)
            row = " & ".join(row_values) + r" \\"
            lines.append(row)

    # Add separator between sections
    if bulk_results and dg_ns_results:
        lines.append(r"\midrule")

    # Add Double Gate / Nanosheet section with multirow
    if dg_ns_results:
        n_dg = len(dg_ns_results)
        for i, (run_name, power_str, param_data, _) in enumerate(dg_ns_results):
            row_values = []
            if i == 0:
                # First row gets the multirow device label
                row_values.append(f"\\multirow{{{n_dg}}}{{*}}{{DG/NS}}")
            else:
                row_values.append("")  # Empty cell for subsequent rows
            row_values.append(power_str)
            for metric_key in metric_keys:
                value = param_data.get(metric_key)
                formatted = format_value(value, metric_key, metrics)
                row_values.append(formatted)
            row = " & ".join(row_values) + r" \\"
            lines.append(row)

    # Close the table
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def render_latex_to_image(latex_code: str, output_path: str) -> bool:
    """Render LaTeX table to an image using tectonic or pdflatex."""

    # Create a complete LaTeX document
    full_doc = r"""
\documentclass[preview,border={10pt 60pt 10pt 10pt}]{standalone}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{multirow}
\begin{document}
""" + latex_code + r"""
\end{document}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "table.tex")
        pdf_file = os.path.join(tmpdir, "table.pdf")

        # Write LaTeX file
        with open(tex_file, "w") as f:
            f.write(full_doc)

        # Try tectonic first, then pdflatex
        compiled = False
        for compiler in ["tectonic", "pdflatex"]:
            try:
                if compiler == "tectonic":
                    result = subprocess.run(
                        ["tectonic", "-o", tmpdir, tex_file],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                else:
                    result = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                if os.path.exists(pdf_file):
                    compiled = True
                    break
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                continue

        if not compiled:
            print("No LaTeX compiler available (tried tectonic, pdflatex)")
            return False

        # Convert PDF to PNG using ImageMagick convert or pdftoppm
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", pdf_file, os.path.join(tmpdir, "table")],
                capture_output=True,
                timeout=30
            )
            # pdftoppm creates table-1.png
            png_file = os.path.join(tmpdir, "table-1.png")
            if os.path.exists(png_file):
                import shutil
                shutil.copy(png_file, output_path)
                return True
        except FileNotFoundError:
            pass

        # Fallback: try convert (ImageMagick)
        try:
            subprocess.run(
                ["convert", "-density", "300", pdf_file, "-quality", "100", output_path],
                capture_output=True,
                timeout=30
            )
            return os.path.exists(output_path)
        except FileNotFoundError:
            # Copy PDF as fallback
            import shutil
            pdf_output = output_path.replace(".png", ".pdf")
            shutil.copy(pdf_file, pdf_output)
            print(f"PNG conversion not available. PDF saved to: {pdf_output}")
            return True

    return False


def display_table_matplotlib(
    results: List[Tuple[str, str, Dict]],
    metrics: Dict,
    title: str = "Final Technology Configurations",
    output_path: Optional[str] = None,
):
    """Display table using matplotlib with proper formatting."""
    import matplotlib.pyplot as plt
    import numpy as np

    metric_keys = list(metrics.keys())
    headers = [metrics[k][0].replace("$", "").replace("_{", " ").replace("}", "")
               for k in metric_keys]

    # Prepare data
    cell_text = []
    row_labels = []

    for run_name, display_name, param_data in results:
        row_values = []
        for metric_key in metric_keys:
            value = param_data.get(metric_key)
            formatted = format_value(value, metric_key, metrics)
            row_values.append(formatted)
        cell_text.append(row_values)
        # Clean up display name for matplotlib
        clean_name = display_name.replace(r"\textit{", "").replace("}", "")
        clean_name = clean_name.replace(r"\cdot", "·").replace("$", "").replace("^2", "²")
        row_labels.append(clean_name)

    # Create figure
    fig, ax = plt.subplots(figsize=(2 * len(headers) + 2, 0.6 * len(results) + 1.5))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        rowLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9ecef')
        if col == -1:
            cell.set_text_props(weight='bold', style='italic')
            cell.set_facecolor('#f8f9fa')

    ax.set_title(title, fontweight='bold', fontsize=12, pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Table image saved to: {output_path}")

    plt.close(fig)


def process_regression_results(
    regression_dir: str,
    metrics: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    display: bool = True,
):
    """
    Process all experiment directories in a regression results folder.

    Args:
        regression_dir: Path to the regression results directory (e.g., most_things.list)
        metrics: Dictionary of metrics to include (uses DEFAULT_METRICS if None)
        output_dir: Where to save output files (creates 'latex_tables' subdir if None)
        display: Whether to display the tables
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if output_dir is None:
        output_dir = os.path.join(regression_dir, "latex_tables")

    os.makedirs(output_dir, exist_ok=True)

    # Get all experiment directories (e.g., diff_objective, power_density, wires)
    experiment_dirs = [d for d in os.listdir(regression_dir)
                       if os.path.isdir(os.path.join(regression_dir, d))
                       and not d.startswith(".")]

    all_tables = []

    for exp_dir_name in sorted(experiment_dirs):
        exp_dir_path = os.path.join(regression_dir, exp_dir_name)

        # Skip non-experiment directories
        if exp_dir_name in ["latex_tables", "aggregate_results"]:
            continue

        print(f"\nProcessing experiment: {exp_dir_name}")

        # Detect if this is a diff_benchmark experiment
        is_diff_benchmark = exp_dir_name.startswith("diff_bench")

        results = collect_results_from_experiment_dir(exp_dir_path, is_diff_benchmark=is_diff_benchmark)

        if not results:
            print(f"  No results found in {exp_dir_name}")
            continue

        # Extract benchmark name from the first result
        if is_diff_benchmark:
            benchmark_name = "different benchmarks"
        else:
            benchmark_name = results[0][0].split("_")[0] if results else exp_dir_name

        # Generate LaTeX table (use specialized format for power_density)
        if exp_dir_name.startswith("power_density"):
            latex_table = generate_power_density_table(
                results,
                metrics,
                benchmark_name=benchmark_name,
            )
        else:
            # Use "Benchmark" as first column header for diff_benchmark experiments
            # Also use DIFF_BENCHMARK_METRICS which includes max_wire_length
            first_col_header = "Benchmark" if is_diff_benchmark else "Objective"
            table_metrics = DIFF_BENCHMARK_METRICS if is_diff_benchmark else metrics
            latex_table = generate_latex_table(
                results,
                table_metrics,
                benchmark_name=benchmark_name,
                first_column_header=first_col_header,
            )

        all_tables.append((exp_dir_name, latex_table))

        # Save LaTeX to file
        tex_file = os.path.join(output_dir, f"{exp_dir_name}_table.tex")
        with open(tex_file, "w") as f:
            f.write(latex_table)
        print(f"  LaTeX saved to: {tex_file}")

        # Try to render LaTeX to image
        png_file = os.path.join(output_dir, f"{exp_dir_name}_table.png")
        if render_latex_to_image(latex_table, png_file):
            print(f"  LaTeX rendered to: {png_file}")
        else:
            # Fallback to matplotlib rendering
            mpl_png = os.path.join(output_dir, f"{exp_dir_name}_table_mpl.png")
            display_table_matplotlib(
                results,
                metrics,
                title=f"Final Technology Configurations ({exp_dir_name})",
                output_path=mpl_png,
            )

    # Print all tables to console
    if display:
        print("\n" + "=" * 80)
        print("GENERATED LATEX TABLES")
        print("=" * 80)

        for exp_name, latex in all_tables:
            print(f"\n--- {exp_name} ---\n")
            print(latex)
            print()

    return all_tables


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from codesign regression results"
    )
    parser.add_argument(
        "regression_dir",
        help="Path to regression results directory (e.g., regression_results/most_things.list)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for tables (default: <regression_dir>/latex_tables)"
    )
    parser.add_argument(
        "-m", "--metrics",
        nargs="+",
        help="Metrics to include (default: supply voltage, effective threshold voltage, gate length, gate width, t_ox, k_gate)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't print tables to console"
    )

    args = parser.parse_args()

    # Build metrics dict if custom metrics specified
    metrics = None
    if args.metrics:
        metrics = {m: DEFAULT_METRICS.get(m, (m, "", ".2f")) for m in args.metrics}

    process_regression_results(
        args.regression_dir,
        metrics=metrics,
        output_dir=args.output,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
