import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap


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

def plot_partial_derivative_similarity_matrix(df, config_name, tech_size):
    custom_cmap = create_custom_colormap()

    tmp_sqr = df.pivot(index="x_name", columns="y_name", values="similarity")

    plt.figure(figsize=(12, 8))  # Make the figure narrower
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
    plt.title(
        f"Python partial d compared to C partial d for {config_name} ({tech_size}nm)"
    )
    ax.set_xlabel("Memory Output Values", fontsize=16)
    ax.set_ylabel("Technology Input Parameters", fontsize=16)

    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            "figs",
            f"{config_name}_{tech_size}_grad_similarity.png",
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

    args = parser.parse_args()

    if not args.file:
        raise ValueError("Please provide a results file to plot.")

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", args.file))

    # Extract the configuration name from the file name
    cfg_details = args.file.replace("_grad_results.csv", "")
    config_name = "_".join(cfg_details.split("_")[0:-1])
    tech_size = cfg_details.split("_")[-1]
    print(f"Plotting gradient similarity for {config_name} ({tech_size}nm)")

    plot_partial_derivative_similarity_matrix(df, config_name, tech_size)

    print(f"Done Plotting.")
