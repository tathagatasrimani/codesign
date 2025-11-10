import csv
import re
import shutil
from pathlib import Path


def parse_lef_macro_sizes(lef_path):
    """
    Parse LEF macros and collect physical sizes (width, height) in microns.

    Supported LEF subset:
      MACRO <name>
        ...
        SIZE <w> BY <h> ;
        ...
      END <name>
    Returns:
      dict[str, tuple[float, float]]: macro_sizes[name] = (width_um, height_um)
    """
    macro_sizes = {}
    current = None
    size_pat = re.compile(r"\bSIZE\s+([\d\.]+)\s+BY\s+([\d\.]+)\s*;")
    macro_pat = re.compile(r"\bMACRO\s+(\S+)")
    end_pat = re.compile(r"\bEND\s+(\S+)")

    with open(lef_path, "r") as f:
        for line in f:
            m = macro_pat.search(line)
            if m:
                current = m.group(1)
                continue
            if current:
                ms = size_pat.search(line)
                if ms:
                    w = float(ms.group(1))
                    h = float(ms.group(2))
                    macro_sizes[current] = (w, h)
                me = end_pat.search(line)
                if me:
                    current = None
    return macro_sizes


def parse_def_units(def_path):
    """
    Parse DEF units: UNITS DISTANCE MICRONS <dbu>.
    Returns:
      int dbu_per_micron
    Raises:
      ValueError if not found.
    """
    units_pat = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)")
    with open(def_path, "r") as f:
        for line in f:
            mu = units_pat.search(line)
            if mu:
                return int(mu.group(1))
    raise ValueError("DEF missing 'UNITS DISTANCE MICRONS <dbu>'")


def parse_def_components(def_path, dbu_per_micron=None):
    """
    Parse DEF components placements into a list of dicts:
      {inst, macro, x_um, y_um, orient}

    Supported DEF subset:
      UNITS DISTANCE MICRONS <dbu>
      COMPONENTS
        - <inst> <macro>
          + PLACED ( <x> <y> ) <orient>
      END COMPONENTS
    """
    if dbu_per_micron is None:
        dbu_per_micron = parse_def_units(def_path)

    comps = []
    in_components = False

    comp_pat = re.compile(r"-\s+(\S+)\s+(\S+)")
    placed_pat = re.compile(r"\+\s*PLACED\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)")
    end_components_pat = re.compile(r"END\s+COMPONENTS")

    with open(def_path, "r") as f:
        for line in f:
            if line.strip().startswith("COMPONENTS"):
                in_components = True
                continue
            if in_components:
                if end_components_pat.search(line):
                    in_components = False
                    continue

                m = comp_pat.search(line)
                if not m:
                    continue

                inst = m.group(1)
                macro = m.group(2)

                # Try to find "+ PLACED" on same or next line
                mp = placed_pat.search(line)
                if not mp:
                    nxt = next(f, "")
                    mp = placed_pat.search(nxt)

                if mp:
                    x_dbu = int(mp.group(1))
                    y_dbu = int(mp.group(2))
                    orient = mp.group(3)
                    x_um = x_dbu / dbu_per_micron
                    y_um = y_dbu / dbu_per_micron
                    comps.append(dict(inst=inst, macro=macro, x_um=x_um, y_um=y_um, orient=orient))
    return comps


def adjust_size_for_orient(w_um, h_um, orient):
    """
    Handle orientation swaps for rotations that flip width/height.
    For simplicity, only swap for 90/270 rotations.
    """
    rot90_like = {"R90", "R270", "MYR90", "MXR90"}
    if orient in rot90_like:
        return h_um, w_um
    return w_um, h_um


def write_floorplan_csv(out_csv, def_path, lef_path):
    """
    Create a PACT-friendly floorplan CSV with SI units (meters):
      name,x,y,w,h
    Coordinates and sizes are converted from microns to meters.
    """
    macro_sizes = parse_lef_macro_sizes(lef_path)
    comps = parse_def_components(def_path)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "x", "y", "w", "h"])
        for c in comps:
            macro = c["macro"]
            if macro not in macro_sizes:
                # Optionally log or raise if needed.
                continue
            w_um, h_um = macro_sizes[macro]
            w_um, h_um = adjust_size_for_orient(w_um, h_um, c["orient"])

            wr.writerow([
                c["inst"],
                c["x_um"] * 1e-6,
                c["y_um"] * 1e-6,
                w_um * 1e-6,
                h_um * 1e-6,
            ])
    return str(out_csv)


def write_ptrace_csv(out_csv, inst_power_w):
    """
    Write a simplest steady-state power trace CSV:
      name,power_W
    inst_power_w: dict[name] = power in Watt
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "power_W"])
        for name, p in inst_power_w.items():
            wr.writerow([name, float(p)])
    return str(out_csv)


def prepare_lcf_csv(out_csv, template_csv):
    """
    Copy LCF (layer coupling/materials) template into output location.
    This is enough for an MVP. You can later generate this programmatically.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template_csv, out_csv)
    return str(out_csv)
