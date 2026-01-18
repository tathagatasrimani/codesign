import copy
import matplotlib.pyplot as plt
import os
from src.sim_util import xreplace_safe
from src.sim_util import get_latest_log_dir
import json
import logging
import math
logger = logging.getLogger(__name__)

from test.visualize_block_vectors import visualize_all_block_vectors

DEBUG = False

def log_info(msg):
    if DEBUG:
        logger.info(msg)

def log_warning(msg):
    if DEBUG:
        logger.warning(msg)

class TrendPlot:
    def __init__(self, codesign_module, params_over_generations, param_names, obj_over_generations, lag_factor_over_generations, wire_lengths_over_generations, wire_delays_over_generations, device_delays_over_generations, sensitivities_over_generations, constraint_slack_over_generations, save_dir, obj="Energy Delay Product", units="nJ*ns", obj_fn="edp"):
        self.codesign_module = codesign_module
        self.params_over_generations = params_over_generations
        self.param_names = param_names
        self.plot_list = set(self.codesign_module.hw.obj_sub_exprs.values())
        self.plot_list_exclude = set(["execution_time", "passive power", "active power"])
        log_info(f"plot list: {self.plot_list}")
        self.plot_list_labels = {param: label for label, param in self.codesign_module.hw.obj_sub_exprs.items()}
        self.plot_list_names = self.codesign_module.hw.obj_sub_plot_names
        self.obj_over_generations = obj_over_generations
        self.lag_factor_over_generations = lag_factor_over_generations
        self.wire_lengths_over_generations = wire_lengths_over_generations
        self.wire_delays_over_generations = wire_delays_over_generations
        self.device_delays_over_generations = device_delays_over_generations
        self.sensitivities_over_generations = sensitivities_over_generations
        self.constraint_slack_over_generations = constraint_slack_over_generations
        self.save_dir = save_dir
        self.obj = obj
        self.units = units
        self.obj_fn = obj_fn

    def plot_wire_lengths_over_generations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 30
        })
        
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect all data points for scatter plot
        x_values = []  # iteration numbers
        y_values = []  # wire lengths in micrometers
        
        for iteration, wire_length_by_edge in enumerate(self.wire_lengths_over_generations):
            for edge_name, wire_length_m in wire_length_by_edge.items():
                x_values.append(iteration)
                y_values.append(wire_length_m * 1e6)  # Convert from m to μm
        
        # Create scatter plot
        ax.scatter(x_values, y_values, alpha=0.6, s=50, color='blue')
        ax.set_xlabel("Generation", fontweight="bold")
        ax.set_ylabel("Wire Length (μm)", fontweight="bold")
        ax.set_title("Wire Lengths Over Generations", fontweight="bold", pad=20)
        ax.set_yscale("log")
        
        # Set x-axis to show only integer values
        if x_values:  # Check if there are any data points
            min_x = min(x_values)
            max_x = max(x_values)
            ax.set_xticks(range(min_x, max_x + 1))
        
        # Improve styling
        fig.patch.set_facecolor("#f8f9fa")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/wire_lengths_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_params_over_generations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        f = open(f"{self.save_dir}/param_data.json", 'w')
        copied_params = []
        for i in range(len(self.params_over_generations)):
            # Build dict by iterating through param_names to ensure we use the correct Variable objects as keys
            param_dict = {}
            for var_obj, name in self.param_names.items():
                if var_obj in self.params_over_generations[i]:
                    param_dict[name] = self.params_over_generations[i][var_obj]
            copied_params.append(param_dict)
        json.dump(copied_params, f)
        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 30
        })
        plot_param_vals = [{} for _ in range(len(self.params_over_generations))]
        for param in self.plot_list:
            plot_label = self.plot_list_labels[param]
            if plot_label in self.plot_list_exclude:
                continue
            values = []
            logger.info(f"Plotting {self.plot_list_names[self.plot_list_labels[param]]}")
            log_info(f"params over generations: {self.params_over_generations}")
            for i in range(len(self.params_over_generations)):
                values.append(xreplace_safe(param, self.params_over_generations[i]))
                plot_param_vals[i][self.plot_list_labels[param]] = values[i]
            # Create figure with better sizing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with improved styling
            ax.plot(values, linewidth=3, markersize=15, marker="o", color="black")
            ax.set_xlabel("Generation", fontweight="bold")
            ax.set_title(f"{self.plot_list_names[self.plot_list_labels[param]]}", fontweight="bold", pad=20)
            ax.set_yscale("log")

            ax.set_xticks(range(len(values)))
            
            # Improve grid and styling
            #ax.grid(True, alpha=0.3, linestyle='--')
            fig.patch.set_facecolor("#f8f9fa")
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{self.plot_list_labels[param]}_over_iters.png", dpi=300, bbox_inches='tight')
            plt.close()
        f = open(f"{self.save_dir}/plot_param_data.json", 'w')
        json.dump(plot_param_vals, f)

    def plot_wire_delays_over_generations(self):

        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "figure.titlesize": 30
        })
        
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all parameters from the first generation
        params = ["max wire delay", "device delay"]
        
        # Use a color cycle for different parameters
        colors = plt.cm.tab10(range(len(params)))


        max_wire_delay_over_generations = []
        for iteration in self.wire_delays_over_generations:
            max_wire_delay = 0
            for edge_name, wire_delay_ns in iteration.items():
                max_wire_delay = max(max_wire_delay, wire_delay_ns)
            max_wire_delay_over_generations.append(max_wire_delay)

        container = {
            "max wire delay": max_wire_delay_over_generations,
            "device delay": self.device_delays_over_generations
        }
        
        # Plot each parameter
        for idx, param in enumerate(params):
            x_values = []
            values = []
            for i in range(len(container[param])):
                values.append(container[param][i])
                x_values.append(self.params_over_generations[i][self.codesign_module.hw.circuit_model.tech_model.base_params.L]*1e9)
            
            # Skip parameters that are all zero
            if len(values) > 0 and all(v == 0 for v in values):
                continue
            
            # Plot with different colors and markers
            ax.plot(x_values, values, linewidth=2.5, markersize=10, marker="o", 
                    color=colors[idx], label=param, alpha=0.8)
        
        ax.set_xlabel("Gate Length (nm)", fontweight="bold")
        ax.set_ylabel("Delay (ns)", fontweight="bold")
        ax.set_title("Max Wire Delay vs Device Delay Over Generations", fontweight="bold", pad=20)
        ax.set_yscale("log")
        
        ax.legend(loc='best', fontsize=20)
        
        # Improve styling
        fig.patch.set_facecolor("#f8f9fa")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/wire_delays_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_obj_over_generations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        x = [i/2.0 for i in range(len(self.obj_over_generations))]

        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 30
        })
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create alternating red and blue line segments
        for i in range(len(self.obj_over_generations) - 1)[::2]:
            x_start = x[i]
            x_mid = (x_start + 0.5)

            if (i + 2) < len(self.obj_over_generations):
                x_end = x[i + 2]
                # Blue line from x.5 to x+1
                ax.plot([x_mid, x_end], [self.obj_over_generations[i + 1], self.obj_over_generations[i + 2]], 'b-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
            
            # Red line from x to x.5
            ax.plot([x_start, x_mid], [self.obj_over_generations[i], self.obj_over_generations[i + 1]], 'r-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
        

        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_title(f"{self.obj} per iteration ({self.units})", fontweight="bold", pad=20)
        ax.set_yscale("log")
        #ax.grid(True, alpha=0.3, linestyle='--')
        fig.patch.set_facecolor("#f8f9fa")
        ax.legend(["forward pass", "inverse pass"], fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{self.obj_fn}_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lag_factor_over_generations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        x = [i for i in range(len(self.lag_factor_over_generations))]
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with improved styling
        ax.plot(x, self.lag_factor_over_generations, linewidth=3, markersize=15, marker="o", color="black")
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_title("Inverse Pass Lag Factor per iteration", fontweight="bold", pad=20)
        ax.set_yscale("log")
        
        # Improve grid and styling
        #ax.grid(True, alpha=0.3, linestyle='--')
        fig.patch.set_facecolor("#f8f9fa")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/lag_factor_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_sensitivities_single_plot(self):
        exclude_list = ["clk_period"]
        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "figure.titlesize": 30
        })
        
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all parameters from the first generation
        params = list(self.sensitivities_over_generations[0].keys())
        params = [param for param in params if self.param_names[param] not in exclude_list]
        
        # Use a color cycle for different parameters
        colors = plt.cm.tab10(range(len(params)))
        
        # Track how many parameters are actually plotted
        plotted_count = 0
        
        # Plot each parameter
        for idx, param in enumerate(params):
            values = []
            for i in range(len(self.sensitivities_over_generations)):
                values.append(xreplace_safe(param, self.sensitivities_over_generations[i]))
            
            # Skip parameters that are all zero
            if len(values) > 0 and all(v == 0 for v in values):
                continue
            
            # Plot with different colors and markers
            ax.plot(values, linewidth=2.5, markersize=10, marker="o", 
                    color=colors[idx], label=param, alpha=0.8)
            plotted_count += 1
        
        ax.set_xlabel("Generation", fontweight="bold")
        ax.set_ylabel("Sensitivity", fontweight="bold")
        ax.set_title("Sensitivities Over Generations", fontweight="bold", pad=20)
        ax.set_xticks(range(len(self.sensitivities_over_generations)))
        
        # Add legend outside the plot area if there are many parameters
        if plotted_count > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
        else:
            ax.legend(loc='best', fontsize=20)
        
        # Improve styling
        fig.patch.set_facecolor("#f8f9fa")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/sensitivities/all_sensitivities_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sensitivities_over_generations(self):
        if len(self.sensitivities_over_generations) == 0:
            return
        if not os.path.exists(self.save_dir + "/sensitivities"):
            os.makedirs(self.save_dir + "/sensitivities")
        f = open(f"{self.save_dir}/sensitivities_data.json", 'w')
        copied_sensitivities = []
        for i in range(len(self.sensitivities_over_generations)):
            # Build dict by iterating through param_names to ensure we use the correct Variable objects as keys
            sensitivity_dict = {}
            for var_obj, name in self.param_names.items():
                if var_obj in self.sensitivities_over_generations[i]:
                    sensitivity_dict[name] = self.sensitivities_over_generations[i][var_obj]
            copied_sensitivities.append(sensitivity_dict)
        json.dump(copied_sensitivities, f)
        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 30
        })
        for param in self.sensitivities_over_generations[0]:
            values = []
            logger.info(f"Plotting sensitivities for {param}")
            for i in range(len(self.sensitivities_over_generations)):
                values.append(xreplace_safe(param, self.sensitivities_over_generations[i]))
            
            if len(values) > 0 and values[0] == 0: 
                continue
            # Create figure with better sizing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with improved styling
            ax.plot(values, linewidth=3, markersize=15, marker="o", color="black")
            ax.set_xlabel("Generation", fontweight="bold")
            ax.set_title(f"{param}", fontweight="bold", pad=20)

            ax.set_xticks(range(len(values)))
            
            # Improve grid and styling
            #ax.grid(True, alpha=0.3, linestyle='--')
            fig.patch.set_facecolor("#f8f9fa")
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/sensitivities/{param}_sensitivities_over_iters.png", dpi=300, bbox_inches='tight')
            plt.close()
        self.create_sensitivities_single_plot()

    def plot_constraint_slack_over_generations(self):
        if len(self.constraint_slack_over_generations) == 0:
            return
        if not os.path.exists(self.save_dir + "/constraint_slack"):
            os.makedirs(self.save_dir + "/constraint_slack")
        f = open(f"{self.save_dir}/constraint_slack_data.json", 'w')
        json.dump(self.constraint_slack_over_generations, f)
        # Set larger font sizes and better styling
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 30,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 30
        })
        for constraint in self.constraint_slack_over_generations[0]:
            values = []
            logger.info(f"Plotting slack for {constraint}")
            for i in range(len(self.constraint_slack_over_generations)):
                values.append(-1*self.constraint_slack_over_generations[i][constraint])
            
            if len(values) > 0 and values[0] == 0: 
                continue
            # Create figure with better sizing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with improved styling
            ax.plot(values, linewidth=3, markersize=15, marker="o", color="black")
            ax.set_xlabel("Generation", fontweight="bold")
            ax.set_title(f"{constraint}", fontweight="bold", pad=20)
            ax.set_yscale("log")
            
            ax.set_xticks(range(len(values)))
            fig.patch.set_facecolor("#f8f9fa")
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/constraint_slack/{constraint}_constraint_slack_over_iters.png", dpi=300, bbox_inches='tight')
            plt.close()
        create_constraint_slack_single_plot(self.constraint_slack_over_generations, self.save_dir)

    def plot_block_vectors_over_generations(self):
        if not os.path.exists(os.path.join(self.save_dir, "../block_vectors")):
            logger.info(f"block vectors directory does not exist, returning")
            return
        visualize_all_block_vectors(os.path.join(self.save_dir, "../block_vectors"), self.codesign_module.vitis_top_function, os.path.join(self.save_dir, "block_vectors_visualization"))


def create_sensitivities_single_plot(sensitivities_over_generations, save_dir):
    if len(sensitivities_over_generations) == 0:
        return
    if not os.path.exists(save_dir + "/sensitivities"):
        os.makedirs(save_dir + "/sensitivities")
    
    # Set larger font sizes and better styling
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 30,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 20,
        "figure.titlesize": 30
    })
    
    # Create figure with better sizing
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all parameters from the first generation
    params = list(sensitivities_over_generations[0].keys())
    
    # Use a color cycle for different parameters
    colors = plt.cm.tab10(range(len(params)))
    
    # Track how many parameters are actually plotted
    plotted_count = 0
    
    # Plot each parameter
    for idx, param in enumerate(params):
        values = []
        for i in range(len(sensitivities_over_generations)):
            #values.append(xreplace_safe(param, sensitivities_over_generations[i]))
            values.append(sensitivities_over_generations[i][param])
        
        # Skip parameters that are all zero
        if len(values) > 0 and all(v == 0 for v in values):
            continue
        
        # Plot with different colors and markers
        ax.plot(values, linewidth=2.5, markersize=10, marker="o", 
                color=colors[idx], label=param, alpha=0.8)
        plotted_count += 1
    
    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel("Sensitivity", fontweight="bold")
    ax.set_title("Sensitivities Over Generations", fontweight="bold", pad=20)
    ax.set_xticks(range(len(sensitivities_over_generations)))
    
    # Add legend outside the plot area if there are many parameters
    if plotted_count > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    else:
        ax.legend(loc='best', fontsize=20)
    
    # Improve styling
    fig.patch.set_facecolor("#f8f9fa")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sensitivities/all_sensitivities_over_iters.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_constraint_slack_single_plot(constraint_slack_over_generations, save_dir):
    if len(constraint_slack_over_generations) == 0:
        return
    if not os.path.exists(save_dir + "/constraint_slack"):
        os.makedirs(save_dir + "/constraint_slack")
    
    # Set larger font sizes and better styling
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 30,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 20,
        "figure.titlesize": 30
    })
    
    # Create figure with better sizing
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all parameters from the first generation
    params = list(constraint_slack_over_generations[0].keys())

    initial_slacks = {param: abs(constraint_slack_over_generations[0][param]) for param in params}
    
    # Use a color cycle for different parameters
    colors = plt.cm.tab10(range(len(params)))
    
    # Track how many parameters are actually plotted
    plotted_count = 0
    
    series_data = {}
    finite_values = []
    
    # Plot each parameter
    for idx, param in enumerate(params):
        values = []
        for i in range(len(constraint_slack_over_generations)):
            eps = 1e-18
            if constraint_slack_over_generations[i][param] < 0:
                slack_normalized = (
                    constraint_slack_over_generations[i][param] / initial_slacks[param]
                    if initial_slacks[param] != 0
                    else constraint_slack_over_generations[i][param]
                )
                log_barrier = -math.log(-slack_normalized + eps)
                values.append(log_barrier)
                finite_values.append(log_barrier)
            else:
                values.append(math.inf)
        series_data[param] = values
    
    y_min = min(finite_values) if finite_values else -1
    y_max = max(finite_values) if finite_values else 1
    span = y_max - y_min if y_max != y_min else 1
    overflow_value = y_max + 0.15 * span
    
    for idx, param in enumerate(params):
        values = series_data[param]
        plot_values = [v if math.isfinite(v) else overflow_value for v in values]
        # Skip parameters that are all zero
        if len(values) > 0 and all(v == 0 for v in values):
            continue
        
        # Plot with different colors and markers
        ax.plot(plot_values, linewidth=2.5, markersize=10, marker="o", clip_on=False,
                color=colors[idx], label=param, alpha=0.8)
        plotted_count += 1
    
    ax.set_xlabel("Generation", fontweight="bold")
    ax.set_ylabel("Log Barrier Slack", fontweight="bold")
    ax.set_title("Constraint Slack Over Generations", fontweight="bold", pad=20)
    ax.set_xticks(range(len(constraint_slack_over_generations)))
    
    ax.set_ylim(y_min, y_max+0.1*span)
    fig.text(
        0.5,
        0.02,
        "Note: Slack uses -log(-(lhs - rhs)) for constraints of the form lhs <= rhs.",
        ha="center",
        fontsize=18,
    )
    
    # Add legend outside the plot area if there are many parameters
    if plotted_count > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    else:
        ax.legend(loc='best', fontsize=20)
    
    # Improve styling
    fig.patch.set_facecolor("#f8f9fa")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/constraint_slack/all_constraint_slack_over_iters.png", dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    log_dir = get_latest_log_dir()
    #sensitivities_over_generations = json.load(open(os.path.join(log_dir, "figs", "sensitivities_data.json")))
    constraint_slack_over_generations = json.load(open(os.path.join(log_dir, "figs", "constraint_slack_data.json")))
    #create_sensitivities_single_plot(sensitivities_over_generations, os.path.join(log_dir, "figs"))
    create_constraint_slack_single_plot(constraint_slack_over_generations, os.path.join(log_dir, "figs"))