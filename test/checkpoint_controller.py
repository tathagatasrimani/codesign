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
    def __init__(self, cfg):
        self.cfg = cfg

    def create_checkpoint(self):
        if os.path.exists(self.cfg["args"]["checkpoint_save_dir"]):
            shutil.rmtree(self.cfg["args"]["checkpoint_save_dir"])
        shutil.copytree("src/tmp", self.cfg["args"]["checkpoint_save_dir"])

    def load_checkpoint(self, checkpoint_start_step):
        assert os.path.exists(self.cfg["args"]["checkpoint_load_dir"]), "Checkpoint directory does not exist"
        if os.path.exists("src/tmp"):
            shutil.rmtree("src/tmp")
        shutil.copytree(self.cfg["args"]["checkpoint_load_dir"], "src/tmp")
        