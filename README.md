# Codesign Framework

Technology aware application to architecture codesign flow

For AI applications, see [this codesign framework](https://github.com/r09g/ai_codesign)

### Setup

#### Inverse Pass
The inverse pass builds symbolic equations using [sympy](https://docs.sympy.org/latest/index.html) and does optimization using [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) using [ipopt](https://github.com/coin-or/Ipopt) as the solver. To install the dependencies appropriately follow the instructions [here](https://pyomo.readthedocs.io/en/stable/installation.html).

If you are running on Apple Silicon, there are issues with the pyomo - ipopt plugin via libblas and liblapack libraries. In order to fix this follow the instructions suggested by user `fasmb24` in [this issue](https://forums.developer.apple.com/forums/thread/693696).


### Execution
Run `./simulate.sh` from the `src` directory with the following arguments:
```
./simulate.sh -n <workload_name>
```
use optional flag `-q` to run in quiet mode. This will not print logs in the `src/benchmarks/json_dump` directory.

The `-s` flag is used to run an architecture search. This will run the architecture search algorithm and output the best architecture found. This flag can be called with the `-a <AREA>` flag to specify the area constraint for the architecture search. This area value is in $\mu m^2$.