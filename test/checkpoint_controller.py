import os
import shutil

from src import codesign
from src import sim_util

# Determine after which step we resume from
checkpoint_map = {
    "none": 0,
    "scalehls": 1,
    "vitis": 2,
    "netlist": 3,
    "schedule": 4,
    "pd": 5
}

# Class should be used at the beginning of a codesign run.
# based on the input checkpoint, create a codesign object
# with the proper state configured for the next step. 
# We assume that the proper state is saved in some tmp directory or similar.
# This only supports vitis/scalehls, not catapult
class CheckpointController:
    def __init__(self, cfg, codesign_root_dir):
        self.cfg = cfg
        self.codesign_root_dir = codesign_root_dir

    def create_checkpoint(self):
        ''' copy the tmp directory to the checkpoint directory. This is run on program end '''
        checkpoint_save_dir = os.path.join(self.codesign_root_dir, "test/saved_checkpoints", self.cfg["args"]["checkpoint_save_dir"])
        tmp_dir = os.path.join(self.codesign_root_dir, "src/tmp")
        if os.path.exists(checkpoint_save_dir):
            shutil.rmtree(checkpoint_save_dir, ignore_errors=True)

        shutil.copytree(tmp_dir, checkpoint_save_dir)

    def load_checkpoint(self):
        ''' copy the checkpoint directory to tmp, using codesign_root_dir for all paths '''
        checkpoint_load_dir = os.path.join(self.codesign_root_dir, "test/saved_checkpoints", self.cfg["args"]["checkpoint_load_dir"])
        tmp_dir = os.path.join(self.codesign_root_dir, "src/tmp")
        
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

            raise Exception("Finished "+checkpoint_step+", which was specified in the config " + self.cfg["args"]["config"] + " as the stopping point for this run.")
        