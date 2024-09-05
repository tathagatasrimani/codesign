# Codesign Framework

Application aware technology - architecture co-design framework.


### Setup

You can setup the appropriate environment either by creating a new conda env with:
```
conda env create -f environment.yml
```
or 
```
conda env create --name codesign --file requirements_conda.txt
```
You can also install the dependencies using pip:
```
pip install -r requirements_pip.txt
```

You have to separately install the [staticfg](https://github.com/coetaur0/staticfg) library by folling the instructions in the repository.

#### Inverse Pass
The inverse pass builds symbolic equations using [sympy](https://docs.sympy.org/latest/index.html) and does optimization using [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) using [ipopt](https://github.com/coin-or/Ipopt) as the solver. To install the dependencies appropriately follow the instructions [here](https://pyomo.readthedocs.io/en/stable/installation.html).

If you are running on Apple Silicon, there are issues with the pyomo - ipopt plugin via libblas and liblapack libraries. In order to fix this follow the instructions suggested by user `fasmb24` in [this issue](https://forums.developer.apple.com/forums/thread/693696).


### Execution

#### Forward Pass
Run `./simulate.sh` from the `src` directory with the following arguments:
```
./simulate.sh -n <workload_name> -c <arch_config>
```
use optional flag `-q` to run in quiet mode. This will not print logs in the `src/benchmarks/json_dump` directory. The `workload_name` is the name of the workload to be simulated fronm `models/benchmarks/`. The `arch_config` is the architecture configuration file to be used for the simulation. The architecture configuration includes a `.gml` file in `architectures/` and an entry in `hw_cfgs.ini`. The architecture configuration file is the name of the `.gml` file without the extension.

#### Inverse Pass
The inverse pass is run similarly to the forward pass. You run the `./symbolic_simulate.sh` instead, with the added optional argument `-o` to specify the optimization technique.
```
./symbolic_simulate.sh -n <workload_name> -c <arch_config> -o <opt>
```
where `<opt>` is either `scp`, or `ipopt` (default). 
#### Codesign

The `-s` flag is used to run an architecture search. This will run the architecture search algorithm and output the best architecture found. This flag can be called with the `-a <AREA>` flag to specify the area constraint for the architecture search. This area value is in $\mu m^2$.