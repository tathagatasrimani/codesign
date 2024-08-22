import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_similarity_matrix(experimental, expected):
    return 100 * (1 - np.abs(experimental - expected) / np.abs(expected))

def parse_csv_to_dict(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None, skip_blank_lines=False)
    
    # Initialize an empty dictionary to hold the data
    data = {}
    
    current_key = None
    for index, row in df.iterrows():
        if isinstance(row[0], str) and "Cache" in row[0]:
            # Retain the entire key with "Cache=" and ","
            key = row[0].strip()
            current_key = key
            data[current_key] = []
        elif current_key and isinstance(row[0], str):
            # Extract the parameter and value pairs
            param_values = row[0].split(";")
            for param_value in param_values:
                if ':' in param_value:
                    param, value = param_value.split(":")
                    param = param.strip()
                    value = value.strip()
                    try:
                        value = float(value)
                    except ValueError:
                        value = np.nan  # Handle NaN values
                    data[current_key].append((param, value))
    
    return data

def plot_diff(csv_file_path=None):
    # If a CSV file path is provided, parse it
    if csv_file_path:
        data = parse_csv_to_dict(csv_file_path)
    else:
        # Default data if no CSV file is provided
        data = {
            'cache 90': [('Vdd', calculate_similarity_matrix(0.830501978176752, 0.7817400000000001)),
                          ('Vth', 91.71),
                          ('C_g_ideal', 86.23)],
            'cache 45': [('Vdd', 70), ('Vth', 60), ('C_g_ideal', 50), ('Additional_param', 40)],
            'mem 90': [('Vdd', 30), ('Vth', 20), ('C_g_ideal', 10)]
        }

    # Extract x and y labels
    x_labels = list(data.keys())
    y_labels = sorted({y_label for entries in data.values() for y_label, _ in entries})

    # Convert the dictionary data to a 2D numpy array
    matrix = np.zeros((len(y_labels), len(x_labels)))

    for col_idx, x_value in enumerate(x_labels):
        for y_value, value in data[x_value]:
            row_idx = y_labels.index(y_value)
            matrix[row_idx, col_idx] = value

    # Adjust the number of labels to match the matrix dimensions
    y_labels = y_labels[:matrix.shape[0]]
    x_labels = x_labels[:matrix.shape[1]]

    # Set up the color map: green for close to 100%, red for close to 0%
    cmap = sns.diverging_palette(10, 150, as_cmap=True)

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Adjust the size to accommodate more data
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, center=50, vmin=0, vmax=100)

    # Set x and y labels with correct number of labels
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0)

    plt.title("Similarity Matrix")
    plt.show()

if __name__ == "__main__":
    csv_file_path = 'grad_results.csv'
    plot_diff(csv_file_path)