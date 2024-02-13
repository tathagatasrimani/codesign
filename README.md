# Codesign Framework

Technology aware application to architecture codesign flow

For AI applications, see [this codesign framework](https://github.com/r09g/ai_codesign)

### Setup

### Execution
Run `./simulate.sh` from the `src` directory with the following arguments:
```
./simulate.sh -n <workload_name>
```
use optional flag `-q` to run in quiet mode. This will not print logs in the `src/benchmarks/json_dump` directory.

The `-s` flag is used to run an architecture search. This will run the architecture search algorithm and output the best architecture found. This flag can be called with the `-a <AREA>` flag to specify the area constraint for the architecture search. This area value is in $\mu m^2$.