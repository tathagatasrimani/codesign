# Codesign Framework

Application aware technology - architecture co-design framework.

##### OpenROAD
To install and build OpenROAD, follow the instructions on [this page](https://github.com/The-OpenROAD-Project/OpenROAD/blob/master/docs/user/Build.md). Not all OSes are supported. One known to work is Ubuntu 22.04. 

#### Inverse Pass
The inverse pass builds symbolic equations using [sympy](https://docs.sympy.org/latest/index.html) and does optimization using [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) using [ipopt](https://github.com/coin-or/Ipopt) as the solver. To install the dependencies appropriately follow the instructions [here](https://pyomo.readthedocs.io/en/stable/installation.html).

If you are running on Apple Silicon, there are issues with the pyomo - ipopt plugin via libblas and liblapack libraries. In order to fix this follow the instructions suggested by user `fasmb24` in [this issue](https://forums.developer.apple.com/forums/thread/693696).

#### Running the flow: 
To run the codesign flow, run the following command from this directory: 
python3 -m src.codesign -b matmult

To run the flow without considering memory as part of the system, run:
python3 -m src.codesign -b matmult --no_memory true

##### Running with checkpoints:
For debug purposes, you may not want to run the entire flow each time. To help, you can save a checkpoint (transfer contents of src/tmp directory to separate save directory) and load it back to src/tmp on your next run. You can do this through a few flags in src/yaml/codesign_cfg.yaml, or on the command line

--checkpoint_load_dir: directory to load snapshot of src/tmp from after previously saving to that directory

--checkpoint_step: Step after which the real execution of the framework will begin. For example, if you specify vitis as the checkpoint step, then in codesign.py, all steps before and including vitis (also scalehls) will be skipped. Instead, we just read out the results from src/tmp for the next step and make sure to still do any other misellaneous initialization before that.

--save_checkpoint: Must set to True in order to have your src/tmp directory copied to a separate save directory when the program exits

--checkpoint_save_dir: The directory that src/tmp gets saved to

--checkpoint_save_step: optionally, you can specify a step after which the program will stop and save src/tmp to a separate directory. So if you specify netlist, then src/tmp will be saved after all netlist related files have been generated.


#### Install instructions RSG Linux machines: 
1. Make sure you are running bash. You can check by running "echo $0".
2. Then, cd into codesign folder and source full_env_start.sh


#### Install instructions on other machines:
1. Make sure you are running bash. You can check by by running "echo $0". If you're not, you can start by running "bash"
2. Then, cd into codesign folder. 
3. Create a new bash script to source catapult based on your particular installation. 
4. source this script instead of stanford_catapult_env.sh at the end of full_env_start.sh
5. source full_env_start.sh


# Acknowledgements
NSF FuSe2 Award 2425218, NSF GRFP, Stanford System X, TSMC
