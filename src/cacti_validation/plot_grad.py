import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

def plot_diff(csv_file_path=None, show_numbers=True, name="similarity_matrix"):
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

    # Set up the color map: green for 100, red for -100
    cmap = sns.diverging_palette(10, 150, as_cmap=True)

    # Create the heatmap with adjusted figure size
    plt.figure(figsize=(6, 8))  # Make the figure narrower
    ax = sns.heatmap(
        matrix, 
        annot=show_numbers, 
        fmt=".2f" if show_numbers else "", 
        cmap=cmap, 
        center=0, 
        vmin=-100, 
        vmax=100, 
        annot_kws={"size": 10}, 
        cbar_kws={"shrink": 1},  # Adjust color bar size
        linewidths=0.5,  # Add grid lines for better separation
        square=(not show_numbers)
    )

    # Set x and y labels with correct number of labels
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(y_labels, rotation=0)

    title = name.replace('no_num_', '').replace('No_Num_', '')
    title = title.replace('_', ' ')
    title = title.title()
    title += " Matrix"
    plt.title(title)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # Adjust spacing

    # Save the plot
    output_dir = 'grad_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f'{name}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    csv_file_path = 'results/access_time_grad_results.csv'
    plot_diff(csv_file_path, name="access_time_similarity")

    csv_file_path = 'results/read_dynamic_grad_results.csv'
    plot_diff(csv_file_path, name="read_dynamic_similarity")

    csv_file_path = 'results/write_dynamic_grad_results.csv'
    plot_diff(csv_file_path, name="write_dynamic_similarity")

    csv_file_path = 'results/read_leakage_grad_results.csv'
    plot_diff(csv_file_path, name="read_leakage_similarity")

    # no num
    csv_file_path = 'results/access_time_grad_results.csv'
    plot_diff(csv_file_path, False, name="access_time_no_num_similarity")

    csv_file_path = 'results/read_dynamic_grad_results.csv'
    plot_diff(csv_file_path, False, name="read_dynamic_no_num_similarity")

    csv_file_path = 'results/write_dynamic_grad_results.csv'
    plot_diff(csv_file_path, False, name="write_dynamic_no_num_similarity")

    csv_file_path = 'results/read_leakage_grad_results.csv'
    plot_diff(csv_file_path, False, name="read_leakage_no_num_similarity")

    

