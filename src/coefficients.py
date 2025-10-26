import yaml

tech_params = yaml.load(open("src/yaml/tech_params.yaml", "r"), Loader=yaml.Loader)

latency = tech_params["latency"]
dynamic_power = tech_params["dynamic_power"]
leakage_power = tech_params["leakage_power"]
area = tech_params["area"]

SMALLEST_TECH_NODE = 7
INVERTER_NAME = "Not16"

def create_coefficients(sizes):
    """
    Compute logical-effort-like coefficients (alpha, beta, gamma) for latency, dynamic power, and leakage
    power across technology sizes.

    Args:
        sizes (list of int): List of technology node sizes to compute coefficients for.

    Returns:
        dict: Dictionary with keys 'alpha', 'beta', and 'gamma', each mapping to a sub-dictionary
            of normalized coefficients for each element.
    """
    coeffs = {
        "alpha": {},
        "beta": {},
        "gamma": {},
        "area": {}
    }
    for elem in latency[SMALLEST_TECH_NODE]:
        total = 0
        for size in sizes:
            total += latency[size][elem] / latency[size][INVERTER_NAME]
        coeffs["gamma"][elem] = total / len(sizes) * 100

    for elem in dynamic_power[SMALLEST_TECH_NODE]:
        total = 0
        for size in sizes:
            total += dynamic_power[size][elem] / dynamic_power[size][INVERTER_NAME]
        coeffs["alpha"][elem] = total / len(sizes) * 100

    for elem in leakage_power[SMALLEST_TECH_NODE]:
        total = 0
        for size in sizes:
            total += leakage_power[size][elem] / leakage_power[size][INVERTER_NAME]
        coeffs["beta"][elem] = total / len(sizes) * 100

    for elem in area[SMALLEST_TECH_NODE]:
        total = 0
        for size in sizes:
            total += area[size][elem] / area[size][INVERTER_NAME]
        coeffs["area"][elem] = total / len(sizes)
    return coeffs

def create_and_save_coefficients(sizes):
    """
    Compute coefficients for the given sizes and save them to 'src/yaml/coefficients.yaml'.

    Args:
        sizes (list of int): List of technology node sizes to compute coefficients for.

    Returns:
        None
    """
    coeffs = create_coefficients(sizes)
    with open("src/yaml/coefficients.yaml", 'w') as f:
        f.write(yaml.dump(coeffs))
    return coeffs

def main():
   
    sizes = [7]

    coeffs = create_coefficients(sizes)
    coeffs_individual = {}
    coeffs_individual[7] = create_coefficients([7])
    coeffs_individual[130] = create_coefficients([130])

    with open("yaml/coefficients.yaml", 'w') as f:
        f.write(yaml.dump(coeffs))

if __name__ == "__main__":
    main()