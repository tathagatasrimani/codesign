import yaml

tech_params = yaml.load(open("tech_params.yaml", "r"), Loader=yaml.Loader)

latency = tech_params["latency"]
dynamic_power = tech_params["dynamic_power"]
leakage_power = tech_params["leakage_power"]


def create_coefficients(sizes):
    coeffs = {
        "alpha": {},
        "beta": {},
        "gamma": {}
    }
    for elem in latency[3]:
        total = 0
        for size in sizes:
            total += latency[size][elem] / latency[size]["Invert"]
        coeffs["gamma"][elem] = total / len(sizes)

    for elem in dynamic_power[3]:
        total = 0
        for size in sizes:
            total += dynamic_power[size][elem] / dynamic_power[size]["Invert"]
        coeffs["alpha"][elem] = total / len(sizes)

    for elem in leakage_power[3]:
        total = 0
        for size in sizes:
            total += leakage_power[size][elem] / leakage_power[size]["Invert"]
        coeffs["beta"][elem] = total / len(sizes)
    return coeffs


def main():
   
    sizes = [40]

    coeffs = create_coefficients(sizes)
    coeffs_individual = {}
    coeffs_individual[3] = create_coefficients([3])
    coeffs_individual[5] = create_coefficients([5])
    coeffs_individual[7] = create_coefficients([7])
    coeffs_individual[40] = create_coefficients([40])

    with open("coefficients.yaml", 'w') as f:
        f.write(yaml.dump(coeffs))

    # with open("coefficients_individual.yaml", 'w') as f:
    #     f.write(yaml.dump(coeffs_individual))

if __name__ == "__main__":
    main()