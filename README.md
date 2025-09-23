# Codesign Framework

Application aware technology - architecture co-design framework.

#### Install instructions RSG Linux machines or BLUEY machine at CMU: 
cd into codesign folder and source full_env_start.sh

#### Running the flow: 
To run the codesign flow, run the following command from this directory: 
python3 -m src.codesign --config <desired config>

The configs are set in src/yaml/codesign_cfg.yaml

##### Running with checkpoints:
For debug purposes, you may not want to run the entire flow each time. To help, you can save a checkpoint (transfer contents of src/tmp directory to separate save directory) and load it back to src/tmp on your next run. You can do this through a few flags in src/yaml/codesign_cfg.yaml, or on the command line

The possible checkpoint points in the code are: 
scalehls
vitis
netlist
schedule
pd

To resume execution of the flow from a saved checkpoint:
--checkpoint_start_step: Step AFTER which the real execution of the framework will begin. For example, if you specify vitis as the checkpoint step, then in codesign.py, all steps before and including vitis (also scalehls) will be skipped. Instead, we just read out the results from src/tmp for the next step and make sure to still do any other misellaneous initialization before that.

--checkpoint_load_dir: directory to load snapshot of src/tmp from after previously saving to that directory

--stop_at_checkpoint: OPTIONAL WHEN RESUMING. The step AFTER which the program will STOP. This is only used when resuming execution if you want it to stop after a subsequent step. If not specified, the   flow will continue to run as normal. 

To create a new saved checkpoint:
--save_checkpoint: Must set to True in order to have your src/tmp directory copied to a separate save directory when the program exits. 

--checkpoint_save_dir: The directory that src/tmp gets saved to if save_checkpoint is True

--stop_at_checkpoint: The step after which the program will STOP. The src/tmp directory will be saved upon exit if the save_checkpoint flag is set.

These are added as arguments in the codesign_cfg.yaml file.

Example usage: 
## Runs the flow up until vitis completed, then saves a checkpoint
python3 -m src.codesign --config resnet_create_checkpoint_after_vitis

## Loads a checkpoint from a successful vitis run, then continues the flow from there and stops after pd
python3 -m src.codesign --config resnet_run_schedule_and_pd_after_vitis


# Acknowledgements
NSF FuSe2 Award 2425218, NSF GRFP, Stanford System X, TSMC
