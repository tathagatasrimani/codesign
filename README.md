# Codesign Framework

Application aware technology - architecture co-design framework.

##### OpenROAD
To install and build OpenROAD, follow the instructions on [this page](https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/docs/user/Build.md). Not all OSes are supported. One known to work is Ubuntu 22.04. 

#### Inverse Pass
The inverse pass builds symbolic equations using [sympy](https://docs.sympy.org/latest/index.html) and does optimization using [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) using [ipopt](https://github.com/coin-or/Ipopt) as the solver. To install the dependencies appropriately follow the instructions [here](https://pyomo.readthedocs.io/en/stable/installation.html).

If you are running on Apple Silicon, there are issues with the pyomo - ipopt plugin via libblas and liblapack libraries. In order to fix this follow the instructions suggested by user `fasmb24` in [this issue](https://forums.developer.apple.com/forums/thread/693696).


#### Install instructions RSG Linux machines: 
1. Make sure you are running bash. You can check by running "echo $0".
2. Then, cd into codesign folder and source full_env_start.sh


#### Install instructions on other machines:
1. Make sure you are running bash. You can check by by running "echo $0". If you're not, you can start by running "bash"
2. Then, cd into codesign folder. 
3. Create a new bash script to source catapult based on your particular installation. 
4. source this script instead of stanford_catapult_env.sh at the end of full_env_start.sh
5. source full_env_start.sh
