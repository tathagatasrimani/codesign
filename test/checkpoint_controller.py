import argparse
import os
import shutil

from src import codesign
from src import sim_util

# Determine after which step we resume from
checkpoint_map = {
    "none": 0,
    "setup": 1,
    "vitis_unlimited": 2,
    "scalehls": 3,
    "vitis": 4,
    "netlist": 5,
    "schedule": 6,
    "pd": 7
}

# Class should be used at the beginning of a codesign run.
# based on the input checkpoint, create a codesign object
# with the proper state configured for the next step. 
# We assume that the proper state is saved in some tmp directory or similar.
# This only supports vitis/scalehls, not catapult
class CheckpointController:
    def __init__(self, cfg, codesign_root_dir, tmp_dir):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir
        self.tmp_dir = tmp_dir


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="Create Checkpoint",
            description="Creates a checkpoint from the current tmp directory. This could be useful to save the state of a run that you killed with ctrl-c or if it errored out but you want to keep it around for debugging.",
            epilog="the end",
        )
        parser.add_argument(
            "-d",
            "--directory",
            type=str,
            help="the name of the directory to save the checkpoint in (will be created under test/saved_checkpoints)"
        )

        args = parser.parse_args()

        if args.directory is None:
            print("Error: must specify a directory name with -d/--directory")
            exit(1)


        ## check if we are in the codesign directory
        cwd = os.getcwd()
        if not os.path.exists(os.path.join(cwd, "src")) or not os.path.exists(os.path.join(cwd, "test")):
            print("Error: must be run from the codesign root directory")
            exit(1)
        
        ## copy the tmp directory to the checkpoint directory
        checkpoint_save_dir = os.path.join(cwd, "test/saved_checkpoints", args.directory)
        tmp_dir = os.path.join(cwd, self.tmp_dir)
        if os.path.exists(checkpoint_save_dir):
            print(f"ERROR: checkpoint directory {checkpoint_save_dir} already exists.")
            exit(1)
        shutil.copytree(tmp_dir, checkpoint_save_dir)
        print(f"Checkpoint saved to {checkpoint_save_dir}")

    def create_checkpoint(self):
        ''' copy the tmp directory to the checkpoint directory. This is run on program end '''
        checkpoint_save_dir = os.path.join(self.codesign_root_dir, "test/saved_checkpoints", self.cfg["args"]["checkpoint_save_dir"])
        tmp_dir = os.path.join(self.codesign_root_dir, self.tmp_dir)
        if os.path.exists(checkpoint_save_dir):
            shutil.rmtree(checkpoint_save_dir, ignore_errors=True)

        shutil.copytree(tmp_dir, checkpoint_save_dir)

    def load_checkpoint(self):
        ''' copy the checkpoint directory to tmp, using codesign_root_dir for all paths '''
        checkpoint_load_dir = os.path.join(self.codesign_root_dir, "test/saved_checkpoints", self.cfg["args"]["checkpoint_load_dir"])
        tmp_dir = os.path.join(self.codesign_root_dir, self.tmp_dir)
        
        assert os.path.exists(checkpoint_load_dir), f"Checkpoint directory does not exist: {checkpoint_load_dir}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.copytree(checkpoint_load_dir, tmp_dir)

    def check_checkpoint(self, checkpoint_start_step, iteration_count):
        ''' return True if we should do this step, False if we should skip it '''
        if checkpoint_map[checkpoint_start_step] > checkpoint_map[self.cfg["args"]["checkpoint_start_step"]] or iteration_count != 0:
            return True
        else:
            return False

    def check_end_checkpoint(self, checkpoint_step):
        ''' raise an exception to stop the program if we reached the stopping point '''
        ## check if we reached the stopping point
        print("Checking END CHECKPOINT step: "+checkpoint_step)
        if checkpoint_map[checkpoint_step] == checkpoint_map[self.cfg["args"]["stop_at_checkpoint"]]:
            
            ## if we did, check if we should save a checkpoint
            if self.cfg["args"]["checkpoint_save_dir"] != "none":
                self.create_checkpoint()

            raise Exception("^^CHECKPOINT REACHED: "+checkpoint_step+" . RUN SUCCEEDED^^, which was specified in the config " + self.cfg["args"]["config"] + " as the stopping point for this run.")
        