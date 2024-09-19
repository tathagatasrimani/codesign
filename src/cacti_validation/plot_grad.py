import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import LinearSegmentedColormap

def calculate_similarity_matrix(experimental, expected):
    return 100 * (1 - np.abs(experimental - expected) / np.abs(expected))

def parse_csv_to_dict(file_path):
    # Read the CSV file with headers
    df = pd.read_csv(file_path)

    # Initialize an empty dictionary to hold the data
    data = {}

    for index, row in df.iterrows():
        # Extract the key and replace "False" with "MainMem" and "True" with "Cache"
        key = row['Tech Config'].strip()
        if "False" in key:
            key = key.replace("Cache=False", "MainMem")
        elif "True" in key:
            key = key.replace("Cache=True", "Cache")

        # Clean up the key format by removing commas
        key = key.replace(',', '').strip()

        param = row['Var Name'].strip()
        similarity = row['Similarities']

        # Handle NA or NaN values
        if pd.isna(similarity) or similarity == 'NA':
            similarity = np.nan
        else:
            try:
                similarity = float(similarity)
            except ValueError:
                similarity = np.nan  # Handle non-convertible values

        # Initialize the key in the dictionary if not present
        if key not in data:
            data[key] = []

        # Append the parameter and its associated similarity to the dictionary
        data[key].append((param, similarity))

    return data

def plot_diff(csv_file_path=None, show_numbers=True, square=False, width=6, name="similarity_matrix",
              xlabel_fontsize=12, ylabel_fontsize=12, title_fontsize=14):
    # If a CSV file path is provided, parse it
    if csv_file_path:
        data = parse_csv_to_dict(csv_file_path)
    else:
        # Default data if no CSV file is provided
        data = {
            'cache 90': [('Vdd', 92.51),
                          ('Vth', 91.71),
                          ('C_g_ideal', 86.23)],
            'cache 45': [('Vdd', 70), ('Vth', 60), ('C_g_ideal', 50), ('Additional_param', 40)],
            'mem 90': [('Vdd', 30), ('Vth', 20), ('C_g_ideal', 10)]
        }

    # Extract x and y labels
    x_labels = list(data.keys())
    y_labels = sorted({y_label for entries in data.values() for y_label, _ in entries})

    # Convert the dictionary data to a 2D numpy array
    matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    for col_idx, x_value in enumerate(x_labels):
        for y_value, value in data[x_value]:
            row_idx = y_labels.index(y_value)
            matrix[row_idx, col_idx] = value

    # Adjust the number of labels to match the matrix dimensions
    y_labels = y_labels[:matrix.shape[0]]
    x_labels = x_labels[:matrix.shape[1]]

    # Custom colormap: red at -100 and 300, white at 0 and 200, green at 100
    colors = [
        (0.8, 0.0, 0.2),  # Red at -100
        (1.0, 1.0, 1.0),    # White at 0
        (0.0, 0.5, 0.3),  # Green at 100
        (1.0, 1.0, 1.0),    # White at 200
        (0.8, 0.0, 0.2)   # Red at 300
    ]
    nodes = [-100, 0, 100, 200, 300]
    normalized_nodes = [(node - min(nodes)) / (max(nodes) - min(nodes)) for node in nodes]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(normalized_nodes, colors)))

    # Create the heatmap with adjusted figure size
    plt.figure(figsize=(width, 8))  # Make the figure narrower
    ax = sns.heatmap(
        matrix, 
        annot=show_numbers, 
        fmt=".2f" if show_numbers else "", 
        cmap=cmap, 
        center=100,  # Center the colormap at 0 for white
        vmin=-100,  # Minimum value for the colormap
        vmax=300,  # Maximum value for the colormap
        annot_kws={"size": 10}, 
        cbar_kws={"shrink": 1},  # Adjust color bar size
        linewidths=0.5,  # Add grid lines for better separation
        square=square
    )

    # Set x and y labels with correct number of labels
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)

    # Set x label font sizes based on the content
    for label in ax.get_xticklabels():
        if label.get_text().startswith("End of"):
            label.set_fontsize(4)  # Smaller fontsize for labels starting with "End of"
        else:
            label.set_fontsize(8)  # Default fontsize for other labels

    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=xlabel_fontsize)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=ylabel_fontsize)

    # Set the title
    title = name.replace('no_num_', '').replace('No_Num_', '')
    title = title.replace('_', ' ')
    title = title.title()
    title += " Matrix"
    plt.title(title, fontsize=20)

    # Add x and y labels with customizable font size
    ax.set_xlabel('Memory Configuration', fontsize=16)
    ax.set_ylabel('Technology Parameters', fontsize=16)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # Adjust spacing

    # Save the plot
    output_dir = 'src/cacti_validation/figs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f'grad_{name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def merge_csv_files(input_files, output_file):
    # Create an empty DataFrame to hold the merged data
    merged_df = pd.DataFrame()

    for i, (file, label) in enumerate(input_files):
        # Read each CSV file
        df = pd.read_csv(file)

        # Modify the Tech Config by appending the label at the beginning
        df['Tech Config'] = label + ", " + df['Tech Config']

        # Append the actual data to the merged DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True)

        # Create a buffer DataFrame with the same number of rows, only if it's not the last file
        if i < len(input_files) - 1:
            buffer_df = df.copy()
            buffer_df['Tech Config'] = f"End of {label}"
            buffer_df['Similarities'] = 0

            # Append the buffer DataFrame to the merged DataFrame
            merged_df = pd.concat([merged_df, buffer_df], ignore_index=True)

    # Write the merged DataFrame to the output file
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify config (--config)")
    parser.add_argument("-c", "--config", type=str, default="base_cache", help="Path or Name to the configuration file; don't append src/cacti/ or .cfg")

    args = parser.parse_args()
    cfg_name = args.config

    current_directory = os.path.dirname(__file__)

    # INDIVIDUAL PLOTS
    access_time_file = os.path.join(current_directory, 'results', f"{cfg_name}_access_time_grad_results.csv")
    plot_diff(access_time_file, name=f"{cfg_name}_access_time_similarity")

    read_energy_file = os.path.join(current_directory, 'results', f'{cfg_name}_read_dynamic_grad_results.csv')
    plot_diff(read_energy_file, name=f"{cfg_name}_read_dynamic_similarity")

    write_energy_file = csv_file_path = os.path.join(
        current_directory, "results", f"{cfg_name}_write_dynamic_grad_results.csv"
    )
    plot_diff(write_energy_file, name=f"{cfg_name}_write_dynamic_similarity")

    leakage_power_file = os.path.join(current_directory, 'results', f'{cfg_name}_read_leakage_grad_results.csv')
    plot_diff(leakage_power_file, name=f"{cfg_name}_read_leakage_similarity")

    # COMBINED PLOT
    csv_files = [
        (access_time_file, "Access Time"),
        (
            read_energy_file,
            "Read Dynamic",
        ),
        (
            write_energy_file,
            "Write Dynamic",
        ),
        (
            leakage_power_file,
            "Leakage",
        ),
    ]
    combined_csv_path = os.path.join(current_directory, f'results/{cfg_name}_combined.csv')
    merge_csv_files(csv_files, combined_csv_path)
    plot_diff(combined_csv_path, False, False, 16, name=f"{cfg_name}_combined_gradient_similarity")
