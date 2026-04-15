"""
Physics-based filamentary RRAM cell model.

State variable: A — conductive filament cross-sectional area [m²]

Equations (from mem_cell_equations/):
  Resistance:     R(A) = R_off * exp(-A / A_ref) + R_on_min
  Joule heating:  T_ss = T0 + I² * R(A) * R_th
  Switching rate: F_i = F_0 * exp[ -E_a*q/(k_B*T) * (1 - F_s/F_c) ]
                  where F_s = |I| / I_c  (normalized current as switching force)
  Filament ODE:   dA/dt = S_set(A) * F_set(T, F_s) - S_reset(A) * F_reset(T, F_s)
                  S_set(A)   = (A_max - A) / A_max
                  S_reset(A) = A / A_max
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

k_B = 1.380649e-23     # Boltzmann constant [J/K]
q   = 1.602176634e-19  # elementary charge [C]
T0  = 300.0            # ambient temperature [K]


class MemCell:
    """
    Filamentary RRAM cell.

    Parameters
    ----------
    R_off      : off-state resistance [Ω]
    R_on_min   : on-state resistance floor (residual) [Ω]
    A_ref      : characteristic filament area scale [m²] — sets slope of R vs A
    A_max      : maximum achievable filament area [m²]
    E_a_set    : SET activation energy [eV]
    E_a_reset  : RESET activation energy [eV]
    F_c_set    : critical normalized current at which SET barrier collapses (F_s/F_c = 1)
    F_c_reset  : critical normalized current at which RESET barrier collapses
    I_c        : normalization current for switching force F_s = |I|/I_c [A]
    F_0        : attempt frequency / kinetic prefactor [m²/s]
    R_th       : filament thermal resistance [K/W]
    R_s        : external series resistance [Ω]
    """

    def __init__(
        self,
        R_off=1e6,
        R_on_min=100.0,
        A_ref=5e-17,
        A_max=5e-16,
        E_a_set=0.8,
        E_a_reset=1.0,
        F_c_set=1.0,
        F_c_reset=1.0,
        I_c=1e-4,
        F_0=1e-16,
        R_th=2e4,
        R_s=0.0,
    ):
        self.R_off     = R_off
        self.R_on_min  = R_on_min
        self.A_ref     = A_ref
        self.A_max     = A_max
        self.E_a_set   = E_a_set
        self.E_a_reset = E_a_reset
        self.F_c_set   = F_c_set
        self.F_c_reset = F_c_reset
        self.I_c       = I_c
        self.F_0       = F_0
        self.R_th      = R_th
        self.R_s       = R_s

    # ------------------------------------------------------------------
    # Resistance model
    # ------------------------------------------------------------------

    def resistance(self, A):
        """
        Cell resistance as a function of filament area.
          R(A) = R_off * exp(-A / A_ref) + R_on_min
        """
        A = np.clip(A, 0.0, self.A_max)
        return self.R_off * np.exp(-A / self.A_ref) + self.R_on_min

    def cell_current(self, V_s, A):
        """Current through cell for source voltage V_s and filament state A."""
        return V_s / (self.R_s + self.resistance(A))

    # ------------------------------------------------------------------
    # Thermal model
    # ------------------------------------------------------------------

    def temperature(self, I, A):
        """
        Steady-state filament temperature from Joule heating.
          T_ss = T0 + I² * R(A) * R_th
        """
        return T0 + I**2 * self.resistance(A) * self.R_th

    # ------------------------------------------------------------------
    # Switching kinetics (field-assisted Arrhenius with barrier lowering)
    # ------------------------------------------------------------------

    def _rate(self, T, F_s, E_a, F_c):
        """
        F_i = F_0 * exp[ -E_a*q/(k_B*T) * (1 - F_s/F_c) ]

        Physical picture: applied field (F_s) lowers the activation barrier
        by a fraction F_s/F_c; at F_s = F_c the barrier vanishes entirely.
        """
        barrier_fraction = np.clip(F_s / F_c, 0.0, 1.0)
        return self.F_0 * np.exp(-E_a * q / (k_B * T) * (1.0 - barrier_fraction))

    def rate_set(self, T, F_s):
        return self._rate(T, F_s, self.E_a_set, self.F_c_set)

    def rate_reset(self, T, F_s):
        return self._rate(T, F_s, self.E_a_reset, self.F_c_reset)

    # ------------------------------------------------------------------
    # Filament ODE
    # ------------------------------------------------------------------

    def dAdt(self, A, V_s):
        """
        dA/dt = S_set(A) * rate_set(T, F_s_fwd) - S_reset(A) * rate_reset(T, F_s_rev)

        S_set(A)   = (A_max - A) / A_max  — fraction of area available to grow
        S_reset(A) = A / A_max            — fraction of area available to dissolve

        Positive V_s drives SET; negative V_s drives RESET (bipolar).
        """
        A = np.clip(A, 0.0, self.A_max)
        I = self.cell_current(V_s, A)
        T = self.temperature(I, A)

        F_s_fwd = max(0.0,  I) / self.I_c   # forward current drives SET
        F_s_rev = max(0.0, -I) / self.I_c   # reverse current drives RESET

        S_set   = (self.A_max - A) / self.A_max
        S_reset = A / self.A_max

        return S_set * self.rate_set(T, F_s_fwd) - S_reset * self.rate_reset(T, F_s_rev)

    # ------------------------------------------------------------------
    # Steady-state solve  (analogous to compute_Id_accurate in MVS model)
    # ------------------------------------------------------------------

    def solve_steady_state(self, V_s, A_init=None):
        """
        Find the steady-state filament area where dA/dt = 0 for a fixed V_s.

        Returns (A_ss, I_ss, R_ss).
        """
        if A_init is None:
            A_init = self.A_max / 2.0

        def equation(A):
            return self.dAdt(A[0], V_s)

        A_sol = fsolve(equation, [A_init], full_output=False)[0]
        A_sol = np.clip(A_sol, 0.0, self.A_max)
        I_sol = self.cell_current(V_s, A_sol)
        R_sol = self.resistance(A_sol)
        return A_sol, I_sol, R_sol

    # ------------------------------------------------------------------
    # Transient simulation
    # ------------------------------------------------------------------

    def simulate(self, t_eval, V_func, A0=0.0):
        """
        Integrate the filament ODE under a time-varying source voltage.

        Parameters
        ----------
        t_eval : 1-D array of time points [s]
        V_func : callable, V_func(t) → source voltage [V]
        A0     : initial filament area [m²] (default 0, fully reset)

        Returns
        -------
        t : time array [s]
        A : filament area vs time [m²]
        R : resistance vs time [Ω]
        I : current vs time [A]
        """
        def ode(t, y):
            return [self.dAdt(y[0], V_func(t))]

        sol = solve_ivp(
            ode,
            t_span=(t_eval[0], t_eval[-1]),
            y0=[A0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-20,
        )

        A_sol = np.clip(sol.y[0], 0.0, self.A_max)
        R_sol = self.resistance(A_sol)
        I_sol = np.array([self.cell_current(V_func(t), A) for t, A in zip(sol.t, A_sol)])
        return sol.t, A_sol, R_sol, I_sol
