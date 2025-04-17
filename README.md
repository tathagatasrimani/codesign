# Codesign Framework

Application aware technology - architecture co-design framework.

#### Submodules

##### Cacti
Run make in the cacti directory.

##### OpenROAD
To install and build OpenROAD, follow the instructions on [this page](https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/docs/user/Build.md). Not all OSes are supported. One known to work is Ubuntu 22.04. 

#### Inverse Pass
The inverse pass builds symbolic equations using [sympy](https://docs.sympy.org/latest/index.html) and does optimization using [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) using [ipopt](https://github.com/coin-or/Ipopt) as the solver. To install the dependencies appropriately follow the instructions [here](https://pyomo.readthedocs.io/en/stable/installation.html).

If you are running on Apple Silicon, there are issues with the pyomo - ipopt plugin via libblas and liblapack libraries. In order to fix this follow the instructions suggested by user `fasmb24` in [this issue](https://forums.developer.apple.com/forums/thread/693696).


#### Install instructions: 
1. Make sure you have miniconda installed.
2. Run conda env create -f environment_simplified.yml 
