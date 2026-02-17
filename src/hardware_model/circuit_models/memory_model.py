import os
import math
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Path to the pareto-pruned CSVs (relative to the codesign repo root)
_PARETO_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "tech_models", "rsg", "destiny_3d_cache",
    "config", "sweep_out", "full_csvs", "pareto",
)

# Columns that are metadata / not real design metrics
_META_COLUMNS = {"_source_file", "_capacity"}


def _bits_to_capacity_label(total_size_bits):
    """Convert a total_size in bits to the capacity label used in pareto filenames.

    Rounds up to the nearest power-of-two KB boundary that has a pareto file.
    Returns e.g. '2KB', '128KB', etc.
    """
    size_kb = total_size_bits / 8 / 1024
    if size_kb <= 0:
        return None

    # Available pareto buckets (must match the filenames in the pareto dir)
    buckets_kb = [2, 4, 8, 16, 32, 64, 128, 256]

    # Round up to the smallest bucket that can hold this size
    for b in buckets_kb:
        if size_kb <= b:
            return f"{b}KB"

    # Larger than all buckets — use the largest available
    return f"{buckets_kb[-1]}KB"


def _load_pareto(capacity_label):
    """Load and return the pareto DataFrame for a given capacity label."""
    path = os.path.join(_PARETO_DIR, f"pareto_{capacity_label}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pareto file for capacity '{capacity_label}': expected {path}"
        )
    df = pd.read_csv(path)
    # Drop any unnamed trailing columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


class MemoryModel:
    """Represents a single memory instance with its Pareto design space.

    Loads the Pareto-pruned exploration results for the appropriate capacity
    and exposes the currently selected design point's metrics as attributes.

    Attributes set dynamically (via setattr) from the pareto CSV columns include:
        cacheAccessMode, cacheArea_mm2, cacheHitLatency_ns, cacheLeakage_mW,
        data_readLatency_ns, tag_bankArea_um2, ... (all columns in the CSV)

    Parameters
    ----------
    memory_info : dict
        Entry from memory_mapping["flattened"], must contain at least
        'total_size' (in bits).
    """

    def __init__(self, memory_info, name=None):
        self.memory_info = memory_info
        self.name = name
        self.total_size_bits = memory_info["total_size"]

        self.capacity_label = _bits_to_capacity_label(self.total_size_bits)

        if self.capacity_label is not None:
            try:
                self.pareto_df = _load_pareto(self.capacity_label)
            except FileNotFoundError:
                logger.warning(
                    f"No pareto data for capacity {self.capacity_label} "
                    f"(total_size={self.total_size_bits} bits), metrics will not be set"
                )
                self.pareto_df = None
        else:
            self.pareto_df = None

        self._design_point_index = 0
        self._metric_columns = []

        if self.pareto_df is not None and not self.pareto_df.empty:
            self._metric_columns = [
                c for c in self.pareto_df.columns if c not in _META_COLUMNS
            ]
            self._apply_design_point()

    def _apply_design_point(self):
        """Set attributes from the current design point row."""
        row = self.pareto_df.iloc[self._design_point_index]
        for col in self._metric_columns:
            setattr(self, col, row[col])

    @property
    def design_point_index(self):
        return self._design_point_index

    @property
    def num_design_points(self):
        if self.pareto_df is None:
            return 0
        return len(self.pareto_df)

    def set_params_from_design_point(self, design_point):
        memory_config = design_point.get("memory", {})
        if self.name in memory_config:
            self.set_design_point(memory_config[self.name])
        else:
            logger.warning(f"No memory config for '{self.name}' in design point")

    def set_design_point(self, index_or_dict):
        """Select a design point by index into the pareto DataFrame.

        Accepts either an integer index or a dict with an "index" key
        (as stored in expanded design points).
        Updates all metric attributes to reflect the new design point.
        """
        if isinstance(index_or_dict, dict):
            index = index_or_dict["index"]
        else:
            index = index_or_dict
        if self.pareto_df is None:
            raise ValueError("No pareto data loaded for this memory")
        if not (0 <= index < len(self.pareto_df)):
            raise IndexError(
                f"Design point index {index} out of range "
                f"[0, {len(self.pareto_df)})"
            )
        self._design_point_index = index
        self._apply_design_point()

    def get_design_point_row(self):
        """Return the full row (as a dict) for the current design point."""
        if self.pareto_df is None:
            return {}
        return self.pareto_df.iloc[self._design_point_index].to_dict()

    def __repr__(self):
        n = self.num_design_points
        idx = self._design_point_index
        cap = self.capacity_label or "?"
        name = self.name or "?"
        return f"MemoryModel(name={name}, capacity={cap}, design_point={idx}/{n})"
