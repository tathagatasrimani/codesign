import matplotlib.pyplot as plt
import os
from src.sim_util import xreplace_safe

import logging
logger = logging.getLogger(__name__)

class TrendPlot:
    def __init__(self, codesign_module, params_over_iterations, obj_over_iterations, lag_factor_over_iterations, save_dir, obj="Energy Delay Product", units="nJ*ns", obj_fn="edp"):
        self.codesign_module = codesign_module
        self.params_over_iterations = params_over_iterations
        self.plot_list = set(self.codesign_module.hw.obj_sub_exprs.values())
        self.plot_list_exclude = set(["execution_time", "passive power", "active power"])
        logger.info(f"plot list: {self.plot_list}")
        self.plot_list_labels = {param: label for label, param in self.codesign_module.hw.obj_sub_exprs.items()}
        self.plot_list_names = self.codesign_module.hw.obj_sub_plot_names
        self.obj_over_iterations = obj_over_iterations
        self.lag_factor_over_iterations = lag_factor_over_iterations
        self.save_dir = save_dir
        self.obj = obj
        self.units = units
        self.obj_fn = obj_fn

    def plot_params_over_iterations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        f = open(f"{self.save_dir}/param_data.txt", 'w')
        f.write(str(self.params_over_iterations))
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
            for i in range(len(self.params_over_iterations)):
                values.append(xreplace_safe(param, self.params_over_iterations[i]))
            
            # Create figure with better sizing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with improved styling
            ax.plot(values, linewidth=3, markersize=15, marker="o", color="black")
            ax.set_xlabel("Generation", fontweight="bold")
            ax.set_title(f"{self.plot_list_names[self.plot_list_labels[param]]}", fontweight="bold", pad=20)
            ax.set_yscale("log")
            
            # Improve grid and styling
            #ax.grid(True, alpha=0.3, linestyle='--')
            fig.patch.set_facecolor("#f8f9fa")
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{self.plot_list_labels[param]}_over_iters.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_obj_over_iterations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        x = [i/2.0 for i in range(len(self.obj_over_iterations))]

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
        for i in range(len(self.obj_over_iterations) - 1)[::2]:
            x_start = x[i]
            x_mid = (x_start + 0.5)

            if (i + 2) < len(self.obj_over_iterations):
                x_end = x[i + 2]
                # Blue line from x.5 to x+1
                ax.plot([x_mid, x_end], [self.obj_over_iterations[i + 1], self.obj_over_iterations[i + 2]], 'b-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
            
            # Red line from x to x.5
            ax.plot([x_start, x_mid], [self.obj_over_iterations[i], self.obj_over_iterations[i + 1]], 'r-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
        

        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_title(f"{self.obj} per iteration ({self.units})", fontweight="bold", pad=20)
        ax.set_yscale("log")
        #ax.grid(True, alpha=0.3, linestyle='--')
        fig.patch.set_facecolor("#f8f9fa")
        ax.legend(["forward pass", "inverse pass"], fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{self.obj_fn}_over_iters.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_lag_factor_over_iterations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        x = [i for i in range(len(self.lag_factor_over_iterations))]
        # Create figure with better sizing
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with improved styling
        ax.plot(x, self.lag_factor_over_iterations, linewidth=3, markersize=15, marker="o", color="black")
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