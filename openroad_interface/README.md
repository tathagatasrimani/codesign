# Openroad Interface

Allows codesign to utilize OpenROAD for resistance, capacitance, and length calculations. The main function takes in two parameters, the graph file name and tcl file directory, and will return a pandas DataFrame containing all information about openroad rcl and estimated rcl.

## Installation
1. Build OpenROAD using the documentation that they provide: https://openroad.readthedocs.io/en/latest/user/Build.html
2. Once OpenROAD is built, the command "openroad", which activates the tcl shell, should be able to run everywhere. If this is not the case, just ensure that this command is able to run in the OpenROAD test directory by exporting the route. To do this, go to the root of Codesign then run:
```
export PATH=$PATH:$(pwd)/openroad_interface/OpenROAD/build/src:$PATH
```

## Usage
To run, execute from codesign root:
```
python3 -m openroad_interface.validation
```
exec.py can be changed to run the main validation script for different graph designs and also visual that information through a graph plotter function. 