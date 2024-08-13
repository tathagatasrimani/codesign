import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to read CSV file and load data into a DataFrame
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to compute errors between result and validate values
def compute_errors(df):
    df['access_time_error'] = df['access_time (ns)'] - df['validate_access_time (ns)']
    df['read_dynamic_error'] = df['result_read_dynamic (nJ)'] - df['validate_read_dynamic (nJ)']
    df['write_dynamic_error'] = df['result_write_dynamic (nJ)'] - df['validate_write_dynamic (nJ)']
    df['leakage_error'] = df['result_leakage (mW)'] - df['validate_leakage (mW)']
    return df

# Plot actual errors as dots, organized by configuration
def plot_errors_dots(df):
    df['config'] = df['transistor_size (um)'].astype(str) + ', Cache=' + df['is_cache'].astype(str)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['config'], df['access_time_error'], color='blue')
    plt.xlabel('Configuration (Transistor Size, Cache)')
    plt.ylabel('Access Time Error (ns)')
    plt.title('Access Time Error by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(df['config'], df['read_dynamic_error'], color='green')
    plt.xlabel('Configuration (Transistor Size, Cache)')
    plt.ylabel('Read Dynamic Error (nJ)')
    plt.title('Read Dynamic Error by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.scatter(df['config'], df['write_dynamic_error'], color='red')
    plt.xlabel('Configuration (Transistor Size, Cache)')
    plt.ylabel('Write Dynamic Error (nJ)')
    plt.title('Write Dynamic Error by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.scatter(df['config'], df['leakage_error'], color='purple')
    plt.xlabel('Configuration (Transistor Size, Cache)')
    plt.ylabel('Leakage Error (mW)')
    plt.title('Leakage Error by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot box and whisker of error, organized by configuration
# def plot_box_and_whisker_access(df, plot_times=False):
#     df['config'] = df['transistor_size (um)'].astype(str) + ', Cache=' + df['is_cache'].astype(str)
#     plt.figure(figsize=(12, 8))

#     ax = plt.gca()
#     df.boxplot(column='access_time_error', by='config', grid=False, showfliers=False, widths=0.6, ax=ax)

#     unique_configs = df['config'].unique()
#     positions = range(1, len(unique_configs) + 1) 

#     if plot_times:
#         for position, config in zip(positions, unique_configs):
#             config_data = df[df['config'] == config]
#             plt.scatter([position] * len(config_data), config_data['access_time (ns)'], color='blue', label='Access Time' if position == 1 else "", alpha=0.6)
#             plt.scatter([position] * len(config_data), config_data['validate_access_time (ns)'], color='red', label='Validate Access Time' if position == 1 else "", alpha=0.6)

#     for position, config in zip(positions, unique_configs):
#         config_data = df[df['config'] == config]
#         plt.scatter([position] * len(config_data), config_data['access_time_error'], color='green', label='Error (Access Time - Validate)' if position == 1 else "", alpha=0.6)

#     plt.title('Access Time Error and Actual Values by Configuration')
#     plt.suptitle('')  
#     plt.xlabel('Configuration (Transistor Size, Cache)')
#     plt.ylabel('Error Time (ns)')
#     plt.xticks(positions, unique_configs, rotation=45, ha='right')
#     plt.grid(True)

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())

#     plt.tight_layout()
#     plt.show()

# Plot box and whisker of error depending on selector, organized by configuration
def plot_box_and_whisker_single(df, selector="access_time", plot_times=False):
    df['config'] = df['transistor_size (um)'].astype(str) + ', Cache=' + df['is_cache'].astype(str)
    plt.figure(figsize=(12, 8))

    if selector == 'access_time':
        error_column = 'access_time_error'
        result_column = 'access_time (ns)'
        validate_column = 'validate_access_time (ns)'
        ylabel = 'Error Time (ns)'
        title = 'Access Time Error and Actual Values by Configuration'
    elif selector == 'read_dynamic':
        error_column = 'read_dynamic_error'
        result_column = 'result_read_dynamic (nJ)'
        validate_column = 'validate_read_dynamic (nJ)'
        ylabel = 'Error Energy (nJ)'
        title = 'Read Dynamic Error and Actual Values by Configuration'
    elif selector == 'write_dynamic':
        error_column = 'write_dynamic_error'
        result_column = 'result_write_dynamic (nJ)'
        validate_column = 'validate_write_dynamic (nJ)'
        ylabel = 'Error Energy (nJ)'
        title = 'Write Dynamic Error and Actual Values by Configuration'
    elif selector == 'leakage':
        error_column = 'leakage_error'
        result_column = 'result_leakage (mW)'
        validate_column = 'validate_leakage (mW)'
        ylabel = 'Error Power (mW)'
        title = 'Leakage Error and Actual Values by Configuration'
    else:
        raise ValueError("Invalid selector. Choose from 'access_time', 'read_dynamic', 'write_dynamic', 'leakage'.")

    ax = plt.gca()
    df.boxplot(column=error_column, by='config', grid=False, showfliers=False, widths=0.6, ax=ax)

    unique_configs = df['config'].unique()
    positions = range(1, len(unique_configs) + 1)

    if plot_times:
        for position, config in zip(positions, unique_configs):
            config_data = df[df['config'] == config]
            plt.scatter([position] * len(config_data), config_data[result_column], color='blue', label=result_column.replace('_', ' ').title() if position == 1 else "", alpha=0.6)
            plt.scatter([position] * len(config_data), config_data[validate_column], color='red', label=validate_column.replace('_', ' ').title() if position == 1 else "", alpha=0.6)

    for position, config in zip(positions, unique_configs):
        config_data = df[df['config'] == config]
        plt.scatter([position] * len(config_data), config_data[error_column], color='green', label=f"Error ({result_column.replace('_', ' ').title()} - {validate_column.replace('_', ' ').title()})" if position == 1 else "", alpha=0.6)

    plt.title(title)
    plt.suptitle('')
    plt.xlabel('Configuration (Transistor Size, Cache)')
    plt.ylabel(ylabel)
    plt.xticks(positions, unique_configs, rotation=45, ha='right')
    plt.grid(True)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()

# Helper for plot all box and whisker plots
def plot_box_and_whisker_helper(df, selector="access_time", plot_times=False):
    df['config'] = df['transistor_size (um)'].astype(str) + ', Cache=' + df['is_cache'].astype(str)

    if selector == 'access_time':
        error_column = 'access_time_error'
        result_column = 'access_time (ns)'
        validate_column = 'validate_access_time (ns)'
        ylabel = 'Error Time (ns)'
        title = 'Access Time Error and Actual Values by Configuration'
    elif selector == 'read_dynamic':
        error_column = 'read_dynamic_error'
        result_column = 'result_read_dynamic (nJ)'
        validate_column = 'validate_read_dynamic (nJ)'
        ylabel = 'Error Energy (nJ)'
        title = 'Read Dynamic Error and Actual Values by Configuration'
    elif selector == 'write_dynamic':
        error_column = 'write_dynamic_error'
        result_column = 'result_write_dynamic (nJ)'
        validate_column = 'validate_write_dynamic (nJ)'
        ylabel = 'Error Energy (nJ)'
        title = 'Write Dynamic Error and Actual Values by Configuration'
    elif selector == 'leakage':
        error_column = 'leakage_error'
        result_column = 'result_leakage (mW)'
        validate_column = 'validate_leakage (mW)'
        ylabel = 'Error Power (mW)'
        title = 'Leakage Error and Actual Values by Configuration'
    else:
        raise ValueError("Invalid selector. Choose from 'access_time', 'read_dynamic', 'write_dynamic', 'leakage'.")

    ax = plt.gca()
    df.boxplot(column=error_column, by='config', grid=False, showfliers=False, widths=0.6, ax=ax)

    unique_configs = df['config'].unique()
    positions = range(1, len(unique_configs) + 1)

    if plot_times:
        for position, config in zip(positions, unique_configs):
            config_data = df[df['config'] == config]
            ax.scatter([position] * len(config_data), config_data[result_column], color='blue', label=result_column.replace('_', ' ').title() if position == 1 else "", alpha=0.6)
            ax.scatter([position] * len(config_data), config_data[validate_column], color='red', label=validate_column.replace('_', ' ').title() if position == 1 else "", alpha=0.6)

    for position, config in zip(positions, unique_configs):
        config_data = df[df['config'] == config]
        ax.scatter([position] * len(config_data), config_data[error_column], color='green', label=f"Error ({result_column.replace('_', ' ').title()} - {validate_column.replace('_', ' ').title()})" if position == 1 else "", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('Configuration (Transistor Size, Cache)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_configs, rotation=45, ha='right')
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

# PLot all box and whisker plots
def plot_all_box_and_whisker(df, plot_times=False):
    plt.figure(figsize=(20, 16))

    plt.subplot(2, 2, 1)
    plot_box_and_whisker_helper(df, selector='access_time', plot_times=plot_times)
    
    plt.subplot(2, 2, 2)
    plot_box_and_whisker_helper(df, selector='read_dynamic', plot_times=plot_times)
    
    plt.subplot(2, 2, 3)
    plot_box_and_whisker_helper(df, selector='write_dynamic', plot_times=plot_times)
    
    plt.subplot(2, 2, 4)
    plot_box_and_whisker_helper(df, selector='leakage', plot_times=plot_times)

    plt.tight_layout()
    plt.show()

# Main function to run the script
def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, 'validate_results.csv')
    df = load_data(file_path)
    df = compute_errors(df)
    plot_all_box_and_whisker(df)
    plot_errors_dots(df)

if __name__ == "__main__":
    main()
