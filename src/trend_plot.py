import matplotlib.pyplot as plt
import os

class TrendPlot:
    def __init__(self, codesign_module, params_over_iterations, edp_over_iterations, lag_factor_over_iterations, save_dir, obj="Energy Delay Product", units="nJ*ns", obj_fn="edp"):
        self.codesign_module = codesign_module
        self.params_over_iterations = params_over_iterations
        self.plot_list = set([
            self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd,
            self.codesign_module.hw.circuit_model.tech_model.V_th_eff,
            self.codesign_module.hw.circuit_model.tech_model.base_params.u_n,
            self.codesign_module.hw.circuit_model.tech_model.base_params.L,
            self.codesign_module.hw.circuit_model.tech_model.base_params.W,
            self.codesign_module.hw.circuit_model.tech_model.base_params.tox,
            self.codesign_module.hw.circuit_model.tech_model.base_params.k_gate,
            self.codesign_module.hw.circuit_model.tech_model.m1_Rsq,
            self.codesign_module.hw.circuit_model.tech_model.m1_Csq,
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_rho,
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_k,
            self.codesign_module.hw.circuit_model.tech_model.base_params.t_1,
            self.codesign_module.hw.circuit_model.tech_model.eot,
        ])
        self.plot_list_labels = {
            self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd: "Vdd",
            self.codesign_module.hw.circuit_model.tech_model.V_th_eff: "Vth",
            self.codesign_module.hw.circuit_model.tech_model.base_params.u_n: "u_n",
            self.codesign_module.hw.circuit_model.tech_model.base_params.L: "L",
            self.codesign_module.hw.circuit_model.tech_model.base_params.W: "W",
            self.codesign_module.hw.circuit_model.tech_model.base_params.tox: "tox",
            self.codesign_module.hw.circuit_model.tech_model.base_params.k_gate: "k_gate",
            self.codesign_module.hw.circuit_model.tech_model.m1_Rsq: "m1_Rsq",
            self.codesign_module.hw.circuit_model.tech_model.m1_Csq: "m1_Csq",
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_rho: "m1_rho",
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_k: "m1_k",
            self.codesign_module.hw.circuit_model.tech_model.base_params.t_1: "t_1",
            self.codesign_module.hw.circuit_model.tech_model.eot: "eot",
        }
        self.plot_list_names = {
            self.codesign_module.hw.circuit_model.tech_model.base_params.V_dd: "Logic Supply Voltage per iteration (V)",
            self.codesign_module.hw.circuit_model.tech_model.V_th_eff: "Transistor Vth per iteration (V)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.u_n: "Transistor u_n per iteration (m²/V·s)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.L: "Gate Length per iteration (m)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.W: "Gate Width per iteration (m)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.tox: "Gate Oxide Thickness per iteration (m)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.k_gate: "Gate Permittivity per iteration (F/m)",
            self.codesign_module.hw.circuit_model.tech_model.m1_Rsq: "metal 1 Rsq per iteration (Ohm*um)",
            self.codesign_module.hw.circuit_model.tech_model.m1_Csq: "metal 1 Csq per iteration (F/um)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_rho: "metal 1 rho per iteration (Ohm*m)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.m1_k: "metal 1 k per iteration (F/m)",
            self.codesign_module.hw.circuit_model.tech_model.base_params.t_1: "physical body thickness per iteration (m)",
            self.codesign_module.hw.circuit_model.tech_model.eot: "electrical oxide thickness per iteration (m)",
        }
        self.edp_over_iterations = edp_over_iterations
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
            values = []
            for i in range(len(self.params_over_iterations)):
                values.append(param.xreplace(self.params_over_iterations[i]))
            
            # Create figure with better sizing
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot with improved styling
            ax.plot(values, linewidth=3, markersize=15, marker="o", color="black")
            ax.set_xlabel("Iteration", fontweight="bold")
            ax.set_title(f"{self.plot_list_names[param]}", fontweight="bold", pad=20)
            ax.set_yscale("log")
            
            # Improve grid and styling
            #ax.grid(True, alpha=0.3, linestyle='--')
            fig.patch.set_facecolor("#f8f9fa")
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/{self.plot_list_labels[param]}_over_iters.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_edp_over_iterations(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        x = [i/2.0 for i in range(len(self.edp_over_iterations))]

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
        for i in range(len(self.edp_over_iterations) - 1)[::2]:
            x_start = x[i]
            x_end = x[i + 2]
            x_mid = (x_start + x_end) / 2

            # Blue line from x.5 to x+1
            ax.plot([x_mid, x_end], [self.edp_over_iterations[i + 1], self.edp_over_iterations[i + 2]], 'b-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
            
            # Red line from x to x.5
            ax.plot([x_start, x_mid], [self.edp_over_iterations[i], self.edp_over_iterations[i + 1]], 'r-', linewidth=3, markersize=10, marker="o", markerfacecolor="black", markeredgecolor="black")
        

        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_title(f"{self.obj} per iteration ({self.units})", fontweight="bold", pad=20)
        ax.set_yscale("log")
        #ax.grid(True, alpha=0.3, linestyle='--')
        fig.patch.set_facecolor("#f8f9fa")
        ax.legend(["inverse pass", "forward pass"], fontsize=18)
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