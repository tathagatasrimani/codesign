import numpy as np
from src import sim_util


def precompute_pareto_values(tech_model) -> dict:
    """Precompute {delay, E_act_inv, P_pass_inv, area} for every row in tech_model.pareto_df.

    Temporarily iterates through all pareto rows, applying each design point to the
    tech model to extract numeric values, then restores the original design point.

    Returns a dict of numpy arrays (one entry per pareto row), shared across all
    LogicUnitModel instances to avoid redundant computation.
    """
    n = len(tech_model.pareto_df)
    delays      = np.empty(n)
    energies    = np.empty(n)
    powers      = np.empty(n)
    areas       = np.empty(n)

    saved_dp = dict(tech_model.base_params.cur_design_point)

    for i, row in enumerate(tech_model.pareto_df.itertuples(index=False)):
        tech_model.set_params_from_design_point({"logic": row._asdict()})
        tv = tech_model.base_params.tech_values
        delays[i]   = float(sim_util.xreplace_safe(tech_model.delay, tv))
        energies[i] = float(sim_util.xreplace_safe(tech_model.E_act_inv, tv))
        powers[i]   = float(sim_util.xreplace_safe(tech_model.P_pass_inv, tv))
        areas[i]    = float(sim_util.xreplace_safe(tech_model.base_params.area, tv))

    tech_model.set_params_from_design_point({"logic": saved_dp})

    return {"delay": delays, "E_act_inv": energies, "P_pass_inv": powers, "area": areas}


class LogicUnitModel:
    """Per-functional-unit logic technology model.

    Analogous to MemoryModel: one instance per unique FU resource (rsc_name_unique),
    selecting a design point from the shared tech model pareto front.

    All instances share the same precomputed arrays (from precompute_pareto_values),
    so only the current design point index differs per instance.
    """

    def __init__(self, precomputed: dict, name: str, function: str):
        self._precomputed = precomputed  # shared dict of numpy arrays
        self.name = name        # rsc_name_unique
        self.function = function  # e.g. "Add16"
        self._design_point_index = 0
        self._apply_design_point()

    def _apply_design_point(self):
        i = self._design_point_index
        self.delay      = float(self._precomputed["delay"][i])
        self.E_act_inv  = float(self._precomputed["E_act_inv"][i])
        self.P_pass_inv = float(self._precomputed["P_pass_inv"][i])
        self.area       = float(self._precomputed["area"][i])

    @property
    def num_design_points(self):
        return len(self._precomputed["delay"])

    def set_design_point(self, index_or_dict):
        index = index_or_dict["index"] if isinstance(index_or_dict, dict) else int(index_or_dict)
        assert 0 <= index < self.num_design_points
        self._design_point_index = index
        self._apply_design_point()

    def get_design_point_row(self):
        return {
            "index": self._design_point_index,
            "delay": self.delay,
            "E_act_inv": self.E_act_inv,
            "P_pass_inv": self.P_pass_inv,
            "area": self.area,
        }

    def __repr__(self):
        return f"LogicUnitModel(name={self.name}, function={self.function}, dp={self._design_point_index}/{self.num_design_points})"
