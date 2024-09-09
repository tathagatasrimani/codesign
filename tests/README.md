# TEST SCRIPTS
Scripts to test functionalities of codesign.

_____________________
# cacti_abs_results.sh

## Cache Simulation and SymPy Expression Generation
This script generates results based on cache configuration, technology parameters, and optional SymPy expressions. Results are stored in `src/cacti_validation/results`.

## Prerequisites
- Ensure all necessary Python dependencies are installed.
- Verify that your configuration files (`.cfg`) and technology parameter files (`.dat`) are located in the appropriate directories.

## How to Use
Run the script with the following command:
./cacti_abs_results.sh -CFG <config_name> -DAT <tech_size> -SYMPY <sympy_file> -gen

### Input Arguments:
- **`-CFG <config_name>`**:  
  *Required*. Path or name of the cache configuration file (without the `.cfg` extension). Defaults to `cache`.

- **`-DAT <tech_size>`**:  
  *Optional*. Technology size (e.g., `90nm`). If not provided, the script will process `45nm`, `90nm`, and `180nm`.

- **`-SYMPY <sympy_file>`**:  
  *Optional*. Path or name of the SymPy file (without the extension) if it's different from the cache configuration name.

- **`-gen`**:  
  *Optional*. Set this flag to generate SymPy expressions from the cache configuration file. Defaults to `false`.

_____________________
# cacti_grad_results.sh

## Cache Configuration and SymPy Expression Generation
This script generates results based on cache configuration, technology parameters, and optional SymPy expressions. Results are stored in:
- `src/cacti_validation/grad_plots`
- `src/cacti_validation/results`

## How to Use
Run the script with the following command:
./<script_name>.sh -CFG <config_name> -DAT <tech_size> -SYMPY <sympy_file> -gen

### Input Arguments:
- **`-CFG <config_name>`**:  
  *Required*. Path or name of the cache configuration file (without the `.cfg` extension). Defaults to `cache`.

- **`-DAT <tech_size>`**:  
  *Optional*. Technology node (e.g., `90nm`). If not provided, the script will run for `45nm`, `90nm`, and `180nm`.

- **`-SYMPY <sympy_file>`**:  
  *Optional*. Path or name of the SymPy file (without the extension) if it's different from the cache configuration name.

- **`-gen`**:  
  *Optional*. Set this flag to generate SymPy expressions from the cache configuration file. Defaults to `false`.

_____________________


