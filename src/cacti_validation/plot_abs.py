import os
import pandas as pd
import matplotlib.pyplot as plt

'''
Plots box and whisker plots of error (result - validation) for each metric category.
'''
def plot_overall_box_and_whisker(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Define the categories to calculate differences
    categories = [
        ('access_time (ns)', 'validate_access_time (ns)'),
        ('result_read_dynamic (nJ)', 'validate_read_dynamic (nJ)'),
        ('result_write_dynamic (nJ)', 'validate_write_dynamic (nJ)'),
        ('result_leakage (mW)', 'validate_leakage (mW)'),
        ('result_io_area', 'validate_io_area'),
        ('result_io_timing_margin', 'validate_io_timing'),
        ('result_io_dynamic_power', 'validate_io_power_dynamic'),
        ('result_io_phy_power', 'validate_io_power_phy'),
        ('result_io_termination_power', 'validate_io_power_termination_and_bias')
    ]
    
    # Calculate differences for each category
    diff_data = {}
    for result_col, validate_col in categories:
        # Convert columns to numeric, forcing invalid values to NaN
        df[result_col] = pd.to_numeric(df[result_col], errors='coerce')
        df[validate_col] = pd.to_numeric(df[validate_col], errors='coerce')
        
        # Subtract validation values from result values
        diff_data[result_col.replace('result_', '').replace(' (ns)', '').replace(' (nJ)', '').replace(' (mW)', '')] = df[result_col] - df[validate_col]
    
    # Convert the differences into a DataFrame
    diff_df = pd.DataFrame(diff_data)
    
    # Plot all box and whisker plots on the same figure
    plt.figure(figsize=(12, 8))
    diff_df.plot(kind='box', title='Box and Whisker Plots for Various Categories')
    plt.ylabel('Difference (Result - Validate)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure the 'figs' directory exists before saving the plot
    figs_dir = os.path.join(os.path.dirname(__file__), 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(figs_dir, 'absolute_validation.png'))
    # plt.show()

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    csv_file_path = os.path.join(current_directory, 'results', 'abs_validate_results.csv')
    plot_overall_box_and_whisker(csv_file_path)
