# Codesign Framework

Application aware technology - architecture co-design framework.

## Install instructions RSG Linux machines or BLUEY machine at CMU: 
cd into codesign root directory (<otherpath>/codesign) if you're not there already. Then run:
./gui_install.py

(the command above runs source full_env_start.sh internally)

## Setting up the environment
Each time you open a new terminal, you must run:
source full_env_start.sh (or .csh if you are using cshell)

This sets environment variables so you can run the codesign flow. 

## Running the flow: 
To run the codesign flow, run the following command from the codesign root directory: 
run_codesign --config <desired_config>

Add configs to a yaml file in test/additional_configs. The codesign framework will search through all yaml files in this directory for a matching config. 

To see more options on how to run the codesign flow, please run:
run_codesign -h

## Running with checkpoints:
For debug purposes, you may not want to run the entire flow each time. To help, you can save a checkpoint (transfer contents of src/tmp directory to separate save directory) and load it back to src/tmp on your next run. You can do this through a few flags in src/yaml/codesign_cfg.yaml, or on the command line

The possible checkpoints in the code are: 
scalehls
vitis
netlist
schedule
pd

NOTE: All saved checkpoints are stored in the test/saved_checkpoints directory

### To resume execution of the flow from a saved checkpoint:
--checkpoint_start_step: Step AFTER which the real execution of the framework will begin. For example, if you specify vitis as the checkpoint step, then in codesign.py, all steps before and including vitis (also scalehls) will be skipped. Instead, we just read out the results from src/tmp for the next step and make sure to still do any other misellaneous initialization before that.

--checkpoint_load_dir: directory to load snapshot of src/tmp from after previously saving to that directory ("none" means no directory is loaded). 

--stop_at_checkpoint: OPTIONAL WHEN RESUMING. The step AFTER which the program will STOP. This is only used when resuming execution if you want it to stop after a subsequent step. If not specified, the   flow will continue to run as normal. 

### To create a new saved checkpoint:
--checkpoint_save_dir: The directory that src/tmp gets saved to. If "none" is specified, no checkpoint is saved upon program exit.

--stop_at_checkpoint: The step after which the program will STOP. The src/tmp directory will be saved upon exit if the save_checkpoint flag is set.

These are added as arguments in the codesign_cfg.yaml file.

### Create a checkpoint after program exit:
If you want to create a checkpoint after the program has already exited (this can be useful if you want to save multiple states for debugging):
From the codesign root directory, run:
create_checkpoint -d <name of checkpoint>

The checkpoint will be created in test/saved_checkpoints. It is not reccommended that you try to resume execution of the flow from one of these checkpoints. 

### Example usage: 
Runs the flow up until vitis completed, then saves a checkpoint
run_codesign --config vitis_resnet_checkpoint_after_vitis

Loads a checkpoint from a successful vitis run, then continues the flow from there and stops after pd
run_codesign --config vitis_resnet_load_checkpoint_after_vitis_stop_after_pd

## Running Regressions:
For running many experiments simultaneously, use the run_regression feature. This must be run from the codesign root directory (i.e. the directory this README is in)

run_regression -h # displays full usage guide.

Regression tests are stored in the test/regressions folder. A regression list is a list of testlists and its filename ends in .list.yaml. A testlist ends in .yaml and has a list of individual tests that are run. i.e.

    some_regression.list.yaml
        one_testlist.yaml
            - application test 1
            - application test 2
            - ...

As an example, the below command runs the all_gemm regression which then runs the lots_of_gemms.yaml testlist which has lots of gemm tests. -m specifies the max number of parallel threads you wish to run the framework at the same time.

run_regression -l auto_tests/all_gemm.list.yaml -m 10

## Cleaning up results from prior runs:
To converse disk space, you may need to clear up results from prior runs. Use these commands: 

clean_checkpoints: removes all saved checkpoints
clean_logs: removes all saved logs
clean_tmp: removes all tmp directories from prior runs
clean_codesign: runs all of the above commands. 


# Acknowledgements
NSF FuSe2 Award 2425218, NSF GRFP, Stanford System X, TSMC
