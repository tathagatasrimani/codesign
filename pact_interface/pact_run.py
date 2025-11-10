from enum import verify
import logging
import re
import os
import copy
import shutil
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# from .io_export import write_floorplan_csv, write_ptrace_csv, prepare_lcf_csv


class PACTRun:
    """
    Thin wrapper around PACT binary/script, similar to openroad_interface/openroad_run.py

    Expected PACT CLI (adjust to your environment):
      pact --floorplan <floorplan.csv> --power <ptrace.csv> --lcf <lcf.csv> --out <out_dir>
    If your PACT entrypoint differs, edit the command building in `run()`.
    """

    def __init__(self, cfg, codesign_root_dir):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.pd_dir = os.path.join(self.codesign_root_dir, "src/tmp/pd")
        self.directory = os.path.join(self.codesign_root_dir, "src/tmp/thm")

    def setup(
        self,
        graph,
        test_file: str,
        clock_freq_hz,
        vdd_V,
        name_map,
        default_activity: float = 0.1,
        default_cap_F: float = 1e-15,
    ) -> str:
        """
        Prepare the working directory and export per-instance power data from graph.

        Returns
        -------
        str
            Path to the generated inst_power.csv file.
        """

        # Clean and re-create working directory
        logger.info("Setting up environment for thermal simulation.")
        if os.path.exists(self.directory):
            logger.info(f"Removing existing directory: {self.directory}")
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)
        logger.info(f"Created directory: {self.directory}")

        os.makedirs(self.directory + "/results")
        logger.info(f"Created results directory: {self.directory}/results")

        # Derive parameters from cfg or graph metadata
        f_hz = (
            clock_freq_hz
            or getattr(self.cfg, "clock_freq_hz", None)
            or graph.graph.get("clock_freq_hz", None)
            or 5e8
        )
        vdd = (
            vdd_V
            or getattr(self.cfg, "vdd_V", None)
            or graph.graph.get("vdd_V", None)
            or 0.7
        )
        a_default = default_activity
        c_default = default_cap_F

        # Default name mapper (strip hierarchy and array suffixes)
        def _default_name_map(s: str) -> str:
            t = s.split("/")[-1]
            t = re.sub(r"\[\d+\]$", "", t)
            return t

        name_map = name_map or _default_name_map

    # ============================================================
    # 1) Generate per-instance power CSV
    # ============================================================
        inst_power_csv = os.path.join(self.results_dir, "inst_power.csv")
        out_rows = []

        for n in graph.nodes():
            data = graph.nodes[n]

            # 1) Explicit power
            if "power_W" in data:
                p = float(data["power_W"])
            # 2) Dynamic + leakage
            elif "active_power_W" in data or "leak_power_W" in data:
                p = float(data.get("active_power_W", 0.0)) + float(data.get("leak_power_W", 0.0))
            # 3) Derived from operation rate and energy per operation
            elif "energy_per_op_J" in data and "ops_per_s" in data:
                p = float(data["energy_per_op_J"]) * float(data["ops_per_s"])
            # 4) Estimated using switching formula: a * f * C * V^2 (+ leakage)
            else:
                a = float(data.get("activity", a_default))
                f = float(data.get("clock_freq_hz", f_hz))
                C = float(data.get("cap_F", c_default))
                V = float(data.get("vdd_V", vdd))
                p = a * f * C * (V ** 2)
                p += float(data.get("leak_power_W", 0.0))

            inst_name = name_map(str(n))
            out_rows.append((inst_name, p))

        # Merge duplicates (if multiple graph nodes map to the same DEF instance)
        merged: Dict[str, float] = defaultdict(float)
        for inst, p in out_rows:
            merged[inst] += p

        with open(inst_power_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["inst", "power_W"])
            for inst, p in merged.items():
                w.writerow([inst, p])

        logger.info(f"[PACT] Wrote per-instance power: {inst_power_csv} ({len(merged)} instances)")
        return inst_power_csv

    # ============================================================
    # 2) DEF parsing and instance-to-grid mapping
    # ============================================================
    def _parse_def(self, def_path: str):
        dbu = None
        die = None  # (x0,y0,x1,y1) DBU
        comps = []  # [(inst, x, y, orient)]
        comp_pat = re.compile(r"^\s*-\s+(\S+)\s+\S+(?:\s+\+\s+PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+))?")
        with open(def_path, "r") as f:
            for line in f:
                if "UNITS DISTANCE MICRONS" in line:
                    dbu = int(re.findall(r"MICRONS\s+(\d+)", line)[0])
                elif line.strip().startswith("DIEAREA"):
                    nums = list(map(int, re.findall(r"(-?\d+)", line)))
                    die = (nums[0], nums[1], nums[2], nums[3])
                else:
                    m = comp_pat.match(line)
                    if m and m.group(2):
                        inst, x, y, orient = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                        comps.append((inst, x, y, orient))
        if dbu is None or die is None:
            raise RuntimeError("DEF missing UNITS or DIEAREA")
        return dbu, die, comps

    def _bin_to_grid(self, x_dbu, y_dbu, die, nx, ny):
        x0, y0, x1, y1 = die
        gx = min(nx-1, max(0, int((x_dbu - x0) / max(1, (x1-x0)) * nx)))
        gy = min(ny-1, max(0, int((y_dbu - y0) / max(1, (y1-y0)) * ny)))
        return gx, gy

    def _map_power_to_grid(self, comps, die, inst_power_csv, Nx, Ny) -> Dict[Tuple[int, int], float]:
        """
        Aggregate per-instance power onto a Nx-by-Ny grid using instance anchor coordinates.
        """
        pmap: Dict[str, float] = {}
        with open(inst_power_csv, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            for r in rd:
                if not r:
                    continue
                pmap[r[0]] = float(r[1])

        grid = defaultdict(float)
        missing = 0
        for inst, x, y, _ in comps:
            p = pmap.get(inst, 0.0)
            if p == 0.0 and inst not in pmap:
                missing += 1
            gx, gy = self._bin_to_grid(x, y, die, Nx, Ny)
            grid[(gx, gy)] += p

        if missing:
            logger.warning(f"[PACT] Instances missing power entries: {missing}")
        return grid

    # ============================================================
    # 3) Generate PACT inputs (floorplan, ptrace, lcf)
    # ============================================================
    def _write_ptrace(self, grid: Dict[Tuple[int, int], float], out_csv: str):
        """
        Write ptrace.csv with columns: gx, gy, power_W
        """
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gx", "gy", "power_W"])
            for (gx, gy), pw in grid.items():
                w.writerow([gx, gy, pw])

    def _write_floorplan_from_die(self, die, dbu: int, Nx: int, Ny: int, out_csv: str):
        """
        Write floorplan.csv inferred from DIEAREA and grid resolution.
        Columns: gx, gy, x_um, y_um, cell_w_um, cell_h_um
        """
        x0, y0, x1, y1 = die
        chip_w_um = (x1 - x0) / dbu
        chip_h_um = (y1 - y0) / dbu
        cell_w = chip_w_um / Nx
        cell_h = chip_h_um / Ny
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gx", "gy", "x_um", "y_um", "cell_w_um", "cell_h_um"])
            for gx in range(Nx):
                for gy in range(Ny):
                    x_um = gx * cell_w
                    y_um = gy * cell_h
                    w.writerow([gx, gy, x_um, y_um, cell_w, cell_h])

    def _write_lcf_default(self, out_csv: str):
        """
        Write a minimal lcf.csv describing a simple Si/TIM/heatsink stack.
        Columns: layer, thickness_um, k_W_mK, rhoCp_J_m3K, is_power_layer
        Customize this later for your technology/package.
        """
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "thickness_um", "k_W_mK", "rhoCp_J_m3K", "is_power_layer"])
            w.writerow(["si", 100.0, 130.0, 1.65e6, 1])
            w.writerow(["tim", 20.0, 4.0, 2.0e6, 0])
            w.writerow(["hs", 1000.0, 200.0, 3.0e6, 0])

    def generate_pact_inputs(
        self,
        def_path: str,
        inst_power_csv: str,
        grid_res: Tuple[int, int] = (256, 256),
    ):
        """
        Build floorplan.csv, ptrace.csv, and lcf.csv entirely inside this module.
        """
        dbu, die, comps = self._parse_def(def_path)
        Nx, Ny = grid_res
        grid = self._map_power_to_grid(comps, die, inst_power_csv, Nx, Ny)

        self._write_ptrace(grid, os.path.join(self.results_dir, "ptrace.csv"))
        self._write_floorplan_from_die(die, dbu, Nx, Ny, os.path.join(self.results_dir, "floorplan.csv"))
        self._write_lcf_default(os.path.join(self.results_dir, "lcf.csv"))

        logger.info("[PACT] Generated floorplan.csv, ptrace.csv, lcf.csv")

    # --------------------------- Solver Orchestration --------------------------
    def run(
        self,
        graph,
        test_file: str,
        lef_paths: List[str],
        def_path: str,
        inst_power_csv,
        pact_extra_args,
        grid_res: Tuple[int, int] = (256, 256),
        run_solver: bool = True,
        # Optional metrics
        compute_phi: bool = True,
        T_hot: float = 100.0,
        compute_grad: bool = True,
    ):
        """
        Orchestrate the full flow:
          1) setup() -> export per-instance power if not provided
          2) generate_pact_inputs() -> triplet CSVs
          3) optional solver invocation
          4) read temps.csv and attach metrics to graph.graph
        """
        # 1) Ensure inst_power.csv exists (generate from graph if missing)
        if inst_power_csv is None:
            inst_power_csv = self.setup(graph, test_file)
        else:
            # Still prepare directories if user supplied a CSV
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir, exist_ok=True)

        # 2) Generate triplet inputs wholly within this module
        self.generate_pact_inputs(def_path=def_path, inst_power_csv=inst_power_csv, grid_res=grid_res)

        # 3) Optionally invoke PACT
        if run_solver:
            cmd = [
                self.pact_bin,
                "--floorplan", os.path.join(self.results_dir, "floorplan.csv"),
                "--power",     os.path.join(self.results_dir, "ptrace.csv"),
                "--lcf",       os.path.join(self.results_dir, "lcf.csv"),
                "--out",       self.results_dir,
            ]
            if pact_extra_args:
                cmd.extend(pact_extra_args)

            logger.info("[PACT] Running: " + " ".join(cmd))
            try:
                import subprocess
                subprocess.check_call(cmd)
            except FileNotFoundError:
                logger.warning("[PACT] PACT binary not found; skipping solver run.")
            except subprocess.CalledProcessError as e:
                logger.error(f"[PACT] PACT run failed: {e}")

        # 4) Read temperatures and attach metrics
        temps_csv = Path(self.results_dir) / "temps.csv"
        if temps_csv.exists():
            Tmax, Tavg, N, phi, grad_max = self._read_and_measure_temps(
                temps_csv, grid_res, compute_phi=compute_phi, T_hot=T_hot, compute_grad=compute_grad
            )
            graph.graph["PACT_Tmax_C"] = Tmax
            graph.graph["PACT_Tavg_C"] = Tavg
            graph.graph["PACT_grid_cells"] = N
            if compute_phi:
                graph.graph["PACT_phi_hot_area_frac"] = phi
            if compute_grad:
                graph.graph["PACT_grad_max_C_per_cell"] = grad_max
            logger.info(
                f"[PACT] Tmax={Tmax:.2f}C, Tavg={Tavg:.2f}C"
                + (f", phi={phi:.4f}" if compute_phi else "")
                + (f", grad_max={grad_max:.2f} C/cell" if compute_grad else "")
            )
        else:
            logger.warning("[PACT] temps.csv not found; skipping metrics.")

        return graph

    # --------------------------- Metrics on Temperature Grid --------------------
    def _read_and_measure_temps(
        self,
        temps_csv: Path,
        grid_res: Tuple[int, int],
        compute_phi: bool = True,
        T_hot: float = 100.0,
        compute_grad: bool = True,
    ):
        """
        Read temps.csv with columns: gx, gy, temp_C and compute:
          - Tmax, Tavg
          - Hot area fraction phi = |{T > T_hot}| / N (optional)
          - Max 4-neighbor gradient in C/cell (optional, discrete metric)
        """
        Nx, Ny = grid_res
        # Initialize grid with NaNs and fill with inputs
        T = [[float("nan")] * Ny for _ in range(Nx)]
        vals = []
        with open(temps_csv, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            for r in rd:
                gx = int(r[0]); gy = int(r[1]); tc = float(r[2])
                if 0 <= gx < Nx and 0 <= gy < Ny:
                    T[gx][gy] = tc
                    vals.append(tc)

        if not vals:
            return 0.0, 0.0, Nx * Ny, 0.0, 0.0

        Tmax = max(vals)
        Tavg = sum(vals) / len(vals)
        N = Nx * Ny

        phi = 0.0
        if compute_phi:
            hot = 0
            for gx in range(Nx):
                for gy in range(Ny):
                    if not math.isnan(T[gx][gy]) and T[gx][gy] > T_hot:
                        hot += 1
            phi = hot / N

        grad_max = 0.0
        if compute_grad:
            def upd(gx1, gy1, gx2, gy2):
                nonlocal grad_max
                t1 = T[gx1][gy1]; t2 = T[gx2][gy2]
                if not (math.isnan(t1) or math.isnan(t2)):
                    grad_max = max(grad_max, abs(t1 - t2))  # C per 1-cell step

            for gx in range(Nx):
                for gy in range(Ny):
                    if math.isnan(T[gx][gy]): 
                        continue
                    if gx + 1 < Nx: upd(gx, gy, gx + 1, gy)
                    if gy + 1 < Ny: upd(gx, gy, gx, gy + 1)

        return Tmax, Tavg, N, phi, grad_max