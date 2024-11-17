import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Plots box and whisker plots of error (result - validation) for each metric category.
"""

def plot_overall_box_and_whisker(csv_file):
    '''
    Reads and plots deltas from the entire CSV file, so all data ever generated.
    Not just from the most recent run.
    '''
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Define the categories to calculate differences
    categories = [
        ("access_time (ns)", "validate_access_time (ns)"),
        ("result_read_dynamic (nJ)", "validate_read_dynamic (nJ)"),
        ("result_write_dynamic (nJ)", "validate_write_dynamic (nJ)"),
        ("result_leakage (mW)", "validate_leakage (mW)"),
    ]

    io_categories = [
        ("result_io_timing_margin", "validate_io_timing"),
        ("result_io_dynamic_power", "validate_io_power_dynamic"),
        ("result_io_phy_power", "validate_io_power_phy"),
        ("result_io_termination_power", "validate_io_power_termination_and_bias"),
    ]

    # Calculate differences for each category
    diff_data = {}
    print("Percentage Differences for Main Categories:")
    for result_col, validate_col in categories:
        # Convert columns to numeric, forcing invalid values to NaN
        df[result_col] = pd.to_numeric(df[result_col], errors="coerce")
        df[validate_col] = pd.to_numeric(df[validate_col], errors="coerce")

        # Subtract validation values from result values
        diff_col_name = result_col.replace("result_", "").replace(" (ns)", "").replace(" (nJ)", "").replace(" (mW)", "")
        diff_data[diff_col_name] = ((df[result_col] - df[validate_col]) / df[validate_col]) * 100

        # Display actual values for each category
        print(f"{diff_col_name}: {diff_data[diff_col_name].values}")

    # Convert the differences into a DataFrame
    diff_df = pd.DataFrame(diff_data)

    # Plot all box and whisker plots on the same figure
    plt.figure(figsize=(12, 8))
    diff_df.plot(kind="box", title="Box and Whisker Plots for Main Categories")
    plt.ylabel("% Difference (Result - Validate)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Ensure the 'figs' directory exists before saving the plot
    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Save the plot
    print("Plot saved to figs/absolute_validation.png")
    plt.savefig(os.path.join(figs_dir, "absolute_validation.png"))

    # Calculate differences for each I/O category
    io_diff_data = {}
    print("\nPercentage Differences for I/O Categories:")
    for result_col, validate_col in io_categories:
        # Convert columns to numeric, forcing invalid values to NaN
        df[result_col] = pd.to_numeric(df[result_col], errors="coerce")
        df[validate_col] = pd.to_numeric(df[validate_col], errors="coerce")

        # Subtract validation values from result values
        io_diff_col_name = result_col.replace("result_", "").replace(" (ns)", "").replace(" (nJ)", "").replace(" (mW)", "")
        io_diff_data[io_diff_col_name] = ((df[result_col] - df[validate_col]) / df[result_col]) * 100

        # Display actual values for each I/O category
        print(f"{io_diff_col_name}: {io_diff_data[io_diff_col_name].values}")

    # Convert the differences into a DataFrame
    io_diff_df = pd.DataFrame(io_diff_data)

    # Plot all box and whisker plots on the same figure for I/O categories
    plt.figure(figsize=(12, 8))
    io_diff_df.plot(kind="box", title="Box and Whisker Plots for I/O Categories")
    plt.ylabel("% Difference (Result - Validate)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot for I/O categories
    print("Plot saved to figs/absolute_validation_io.png")
    plt.savefig(os.path.join(figs_dir, "absolute_validation_io.png"))

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    csv_file_path = os.path.join(current_directory, "results", "abs_validate_results.csv")
    plot_overall_box_and_whisker(csv_file_path)
