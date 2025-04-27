import logging

logger = logging.getLogger(__name__)

from .abstract_simulate import AbstractSimulator

class ConcreteSimulator(AbstractSimulator):
    def __init__(self):
        self.total_passive_energy = 0
        self.total_active_energy = 0
        self.execution_time = 0

    def calculate_passive_energy(self, hw, total_execution_time):
        passive_power = 0
        for node in hw.netlist:
            data = hw.netlist.nodes[node]
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                passive_power += hw.memories[rsc_name]["Standby leakage per bank(mW)"] * 1e6 # convert from mW to nW
            else:
                passive_power += hw.leakage_power[data["function"]]
                logger.info(f"(passive power) {data['function']}: {hw.leakage_power[data['function']]}")
        self.total_passive_energy = passive_power * total_execution_time*1e-9
        
    def calculate_active_energy(self, hw, scheduled_dfg):
        self.total_active_energy = 0
        for node in scheduled_dfg:
            data = scheduled_dfg.nodes[node]
            if node == "end" or data["function"] == "nop": continue
            if data["function"] == "Buf" or data["function"] == "MainMem":
                rsc_name = data["library"][data["library"].find("__")+1:]
                if data["module"].find("wport") != -1:
                    self.total_active_energy += hw.memories[rsc_name]["Dynamic write energy (nJ)"]
                else:
                    self.total_active_energy += hw.memories[rsc_name]["Dynamic read energy (nJ)"]
            else:
                energy = hw.dynamic_power[data["function"]] * hw.latency[data["function"]]*1e-9
                self.total_active_energy += energy
                logger.info(f"(active energy) {data['function']}: {energy}")
    
    def calculate_edp(self, hw, scheduled_dfg):
        self.execution_time = scheduled_dfg.nodes["end"]["start_time"]
        self.calculate_passive_energy(hw, self.execution_time)
        self.calculate_active_energy(hw, scheduled_dfg)
        return self.total_passive_energy * self.execution_time + self.total_active_energy
        