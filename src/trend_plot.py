import matplotlib.pyplot as plt
import os
from src.sim_util import xreplace_safe

import logging
logger = logging.getLogger(__name__)

class TrendPlot:
    def __init__(self, codesign_module, params_over_generations, obj_over_generations, lag_factor_over_generations, wire_lengths_over_generations, sensitivities_over_generations, constraint_slack_over_generations, save_dir, obj="Energy Delay Product", units="nJ*ns", obj_fn="edp"):
        self.codesign_module = codesign_module
        self.params_over_generations = params_over_generations
        self.plot_list = set(self.codesign_module.hw.obj_sub_exprs.values())
        self.plot_list_exclude = set(["execution_time", "passive power", "active power"])
        logger.info(f"plot list: {self.plot_list}")
        self.plot_list_labels = {param: label for label, param in self.codesign_module.hw.obj_sub_exprs.items()}
        self.plot_list_names = self.codesign_module.hw.obj_sub_plot_names
        self.obj_over_generations = obj_over_generations
        self.lag_factor_over_generations = lag_factor_over_generations
        self.wire_lengths_over_generations = wire_lengths_over_generations
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
        f = open(f"{self.save_dir}/param_data.txt", 'w')
        f.write(str(self.params_over_generations))
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
        for param in self.plot_list:
            plot_label = self.plot_list_labels[param]
            if plot_label in self.plot_list_exclude:
                continue
            values = []
            logger.info(f"Plotting {self.plot_list_names[self.plot_list_labels[param]]}")
            logger.info(f"params over generations: {self.params_over_generations}")
            for i in range(len(self.params_over_generations)):
                values.append(xreplace_safe(param, self.params_over_generations[i]))
            
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

    def plot_sensitivities_over_generations(self):
        if len(self.sensitivities_over_generations) == 0:
            return
        if not os.path.exists(self.save_dir + "/sensitivities"):
            os.makedirs(self.save_dir + "/sensitivities")
        f = open(f"{self.save_dir}/sensitivities_data.txt", 'w')
        f.write(str(self.sensitivities_over_generations))
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

    def plot_constraint_slack_over_generations(self):
        if len(self.constraint_slack_over_generations) == 0:
            return
        if not os.path.exists(self.save_dir + "/constraint_slack"):
            os.makedirs(self.save_dir + "/constraint_slack")
        f = open(f"{self.save_dir}/constraint_slack_data.txt", 'w')
        f.write(str(self.constraint_slack_over_generations))
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