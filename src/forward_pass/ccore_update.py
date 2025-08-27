import yaml
import os
import re
import logging

logger = logging.getLogger(__name__)

def update_ccores(area_data, delay_data):
    """
    Update the area and delay values for ccore hardware units by modifying their header files.

    Args:
        area_data (dict): Dictionary mapping unit names to area values.
        delay_data (dict): Dictionary mapping unit names to delay values.

    Returns:
        None
    """

    # Iterate through the units in the YAML file
    for unit in area_data.keys():
        unit_filename = unit.lower()
        header_file = f"src/tmp/benchmark/src/ccores/{unit_filename}.h"  # Corresponding .h file
        
        if not os.path.exists(header_file):
            logger.info(f"Warning: {header_file} not found. Skipping...")
            continue

        # Read the contents of the header file
        with open(header_file, "r") as file:
            content = file.read()

        # Replace .area(x) and .delay(x) with values from YAML
        if unit in area_data:
            content = re.sub(r"\.area\(\d+\.?\d*\)", f".area({area_data[unit]})", content)

        if unit in delay_data:
            content = re.sub(r"\.delay\(\d+\.?\d*\)", f".delay({delay_data[unit]})", content)

        # Write the modified content back to the header file
        with open(header_file, "w") as file:
            file.write(content)

        logger.info(f"Updated {header_file} with area={area_data.get(unit)} and delay={delay_data.get(unit)}.")

    logger.info("Processing completed.")


if __name__ == "__main__":
    # Load technology parameters from tech_params.yaml
    yaml_file = "tech_params.yaml"
    with open(yaml_file, "r") as file:
        tech_params = yaml.safe_load(file)

    # Define the nm size to use (set this manually or fetch dynamically if needed)
    NM_SIZE = 7  # Change this to the required technology node (e.g., 3, 5, 7, ...)

    # Validate if the given nm size exists in the YAML file
    if NM_SIZE not in tech_params["area"] or NM_SIZE not in tech_params["latency"]:
        print(f"Error: {NM_SIZE}nm data not found in tech_params.yaml")
        exit(1)
    
    update_ccores(tech_params["area"][NM_SIZE], tech_params["latency"][NM_SIZE])
