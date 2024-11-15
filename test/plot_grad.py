import os
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def calculate_similarity_matrix(python_delta_y, c_delta_y):
    """
    Calculates the similarity of two gradients.
    A score of 100 indicates same magnitude and sign, while
    -100 indicates same magnitude and opposite sign.

    delta_x is the same for both runs (python and c).

    Inputs:
    python_grad : float
        The gradient calculated using Python.
    c_grad : float
        The gradient calculated using C.

    Returns:
    float or str
        The similarity score or "NA" if the C gradient is
        zero. If both gradients are zero, returns 100.
    """

    # Handle the case where both values are 0
    if python_delta_y == 0 and c_delta_y == 0:
        return 100
    elif c_delta_y == 0:
        return "NA"

    # Calculate similarity
    magnitude = 100 * (1 - np.abs(python_delta_y - c_delta_y) / np.abs(c_delta_y))
    sign = np.sign(python_delta_y * c_delta_y)
    greater_than_1 = (
        np.abs(python_delta_y - c_delta_y) / np.abs(c_delta_y) > 1
    )  # essentially if they are the same sign, but doesn't fall in the -1 to 1 range

    similarity = magnitude * sign * -1 if greater_than_1 else magnitude * sign
    return similarity


def create_custom_colormap():
    """
    Trying to convey accuracy when both values are of the same magnitude and sign.
    It is really bad when the signs are off.
    Still bad, but less bad when the signs are right but magnitudes are off.
    """
    # Custom colormap: red at -100 and 300, white at 0 and 200, green at 100
    colors = [
        (0.8, 0.0, 0.2),  # Red at -100
        (1.0, 1.0, 1.0),  # White at 0
        (0.0, 0.5, 0.3),  # Green at 100
        (1.0, 1.0, 1.0),  # White at 200
        (0.8, 0.0, 0.2),  # Red at 300
    ]
    nodes = [-100, 0, 100, 200, 300]
    normalized_nodes = [
        (node - min(nodes)) / (max(nodes) - min(nodes)) for node in nodes
    ]
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", list(zip(normalized_nodes, colors))
    )
    return cmap


def plot_partial_derivative_similarity_matrix(df, config_name, tech_size, sign_only=False):
    custom_cmap = create_custom_colormap()

    tmp_sqr = df.pivot(index="x_name", columns="y_name", values="similarity")

    title = f"Python partial d compared to C partial d for {config_name} ({tech_size}nm)"
    file_save_name = f"{config_name}_{tech_size}_grad_similarity"

    plt.figure(figsize=(12, 8))  # Make the figure narrower
    if sign_only:
        ax = sns.heatmap(
            np.sign(tmp_sqr),
            cmap=mpl.colormaps['RdYlGn'],
            annot=True,
            fmt=".0f",
            annot_kws={"size": 10},
            cbar_kws={"shrink": 1},  # Adjust color bar size
            linewidths=0.5,  # Add grid lines for better separation)
        )
        title += " (Sign Only)"
        file_save_name += "_sign"
    else:
        ax = sns.heatmap(
            tmp_sqr,
            cmap=custom_cmap,  # sns.diverging_palette(0, 120, s=60, as_cmap=True)
            center=100,
            vmin=-100,  # Minimum value for the colormap
            vmax=300,  # Maximum value for the colormap
            annot=True,
            fmt=".0f",
            annot_kws={"size": 10},
            cbar_kws={"shrink": 1},  # Adjust color bar size
            linewidths=0.5,  # Add grid lines for better separation)
        )
    plt.title(title)
    ax.set_xlabel("Memory Output Values", fontsize=16)
    ax.set_ylabel("Technology Input Parameters", fontsize=16)

    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "figs",
            f"{file_save_name}.png",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify results file (--file)")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path or Name to the results file; Assumed to be in codesign/test/results",
    )
    parser.add_argument(
        "-s",
        "--sign",
        action="store_true",
        help="Plot the sign of the gradient similarity",
    )

    args = parser.parse_args()

    if not args.file:
        raise ValueError("Please provide a results file to plot.")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", args.file))

    # Adjust 'python delta y' based on 'y_name'
    df.loc[df['y_name'].isin(['read_dynamic', 'write_dynamic']), 'python delta y'] *= 1e9
    df.loc[df['y_name'] == 'read_leakage', 'python delta y'] *= 1e3

    # Recalculate similarity using the new function
    df['similarity'] = df.apply(
        lambda row: calculate_similarity_matrix(row['python delta y'], row['c delta y']),
        axis=1
    )

    # Replace 'NA' with np.nan to handle non-numeric values properly
    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')

    # Extract the configuration name from the file name
    cfg_details = args.file.replace("_grad_results.csv", "")
    config_name = "_".join(cfg_details.split("_")[0:-1])
    tech_size = cfg_details.split("_")[-1]
    print(f"Plotting gradient similarity for {config_name} ({tech_size}nm)")

    plot_partial_derivative_similarity_matrix(df, config_name, tech_size, args.sign)

    print(f"Done Plotting.")
