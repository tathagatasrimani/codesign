import json
import graphviz as gv
import re
from collections import deque
import ast
import configparser as cp
import yaml

from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils

HW_CONFIG_FILE = "hw_cfgs.ini"

benchmark = 'simple'
expr_to_node = {}
func_ref = {}

op2sym_map = {
    "And": "and",
    "Or": "or",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "IsNot": "!=",
    "USub": "-",
    "UAdd": "+",
    "Not": "!",
    "Invert": "~",
    "Regs": "Regs"
}

latency_scale = {
    512: 1,
    1024: 2,
    2048: 3,
    4096: 4,
    8192: 5,
    16384: 6,
    32768: 7,
    65536: 8,
    131072: 9,
    262144: 10,
    524288: 11,
    1048576: 12,
    2097152: 13,
    4194304: 14,
    8388608: 15,
    16777216: 16,
    33554432: 17,
    67108864: 18,
    134217728: 19,
    268435456: 20,
    536870912: 21
}

power_scale = {
    512: 1,
    1024: 2,
    2048: 3,
    4096: 4,
    8192: 5,
    16384: 6,
    32768: 7,
    65536: 8,
    131072: 9,
    262144: 10,
    524288: 11,
    1048576: 12,
    2097152: 13,
    4194304: 14,
    8388608: 15,
    16777216: 16,
    33554432: 17,
    67108864: 18,
    134217728: 19,
    268435456: 20,
    536870912: 21
}

class HardwareModel:
	
	def __init__(self, cfg=None, id=None, bandwidth=None, mem_layers=None, pitch=None, transistor_size=None, cache_size=None):
		'''
		Simulates the effect of 2 different constructors. Either supply cfg, or supply the rest of the arguments. 
		In this form for backward compatability. I want to deprecate the manual construction soon.
		'''
		if cfg is None:
			self.set_hw_config_vars(id, bandwidth, mem_layers, pitch, transistor_size, cache_size)
		else:
			config = cp.ConfigParser()
			config.read(HW_CONFIG_FILE)
			self.set_hw_config_vars(config.getint(cfg, "id"), config.getint(cfg, "bandwidth"), 
					 config.getint(cfg, "nummemlayers"), config.getint(cfg, "interconnectpitch"), 
					 config.getint(cfg, "transistorsize"), config.getint(cfg, "cachesize"))
		self.hw_allocated = {}

		self.init_misc_vars()

		self.dynamic_allocation = False
		if cfg is not None:
			self.allocate_hw_from_config(config[cfg])
		
		self.set_technology_parameters()
	
	def set_hw_config_vars(self, id, bandwidth, mem_layers, pitch, transistor_size, cache_size):
		self.id = id
		self.max_bw = bandwidth
		self.bw_avail = bandwidth
		self.mem_layers = mem_layers
		self.pitch = pitch
		self.transistor_size = transistor_size
		self.cache_size = cache_size

	def allocate_hw_from_config(self, config):
		'''
		allocate hardware from a config file
		'''
		self.hw_allocated = dict(config)
		#remove the other config variables from the hardware allocated dict
		# hardware allocated should be refactored to PEs allocated or something like that.
		self.hw_allocated.pop("id")
		self.hw_allocated.pop("bandwidth")
		self.hw_allocated.pop("nummemlayers")
		self.hw_allocated.pop("interconnectpitch")
		self.hw_allocated.pop("transistorsize")
		self.hw_allocated.pop("cachesize")

		# convert lower case to Camel Case
		self.hw_allocated['Add'] = self.hw_allocated.pop('add')
		self.hw_allocated['Regs'] = self.hw_allocated.pop('regs')
		self.hw_allocated['Mult'] = self.hw_allocated.pop('mult')
		self.hw_allocated['Sub'] = self.hw_allocated.pop('sub')
		self.hw_allocated['FloorDiv'] = self.hw_allocated.pop('floordiv')
		self.hw_allocated['Gt'] = self.hw_allocated.pop('gt')
		self.hw_allocated['And'] = self.hw_allocated.pop('and')
		self.hw_allocated['Or'] = self.hw_allocated.pop('or')
		self.hw_allocated['Mod'] = self.hw_allocated.pop('mod')
		self.hw_allocated['LShift'] = self.hw_allocated.pop('lshift')
		self.hw_allocated['RShift'] = self.hw_allocated.pop('rshift')
		self.hw_allocated['BitOr'] = self.hw_allocated.pop('bitor')
		self.hw_allocated['BitXor'] = self.hw_allocated.pop('bitxor')
		self.hw_allocated['BitAnd'] = self.hw_allocated.pop('bitand')
		self.hw_allocated['Eq'] = self.hw_allocated.pop('eq')
		self.hw_allocated['NotEq'] = self.hw_allocated.pop('noteq')
		self.hw_allocated['Lt'] = self.hw_allocated.pop('lt')
		self.hw_allocated['LtE'] = self.hw_allocated.pop('lte')
		self.hw_allocated['GtE'] = self.hw_allocated.pop('gte')
		self.hw_allocated['IsNot'] = self.hw_allocated.pop('isnot')
		self.hw_allocated['USub'] = self.hw_allocated.pop('usub')
		self.hw_allocated['UAdd'] = self.hw_allocated.pop('uadd')
		self.hw_allocated['Not'] = self.hw_allocated.pop('not')
		self.hw_allocated['Invert'] = self.hw_allocated.pop('invert')

		for k, v in self.hw_allocated.items():
			self.hw_allocated[k] = int(v)
		
		tmp = True
		for key, value in self.hw_allocated.items():
			tmp &= (value == -1)
		self.dynamic_allocation = tmp
		# print(f"dynamic_allocation flag: {self.dynamic_allocation}")

	def set_technology_parameters(self):
		tech_params = yaml.load(open('tech_params.yaml', 'r'), Loader=yaml.Loader)

		self.area = tech_params['area'][self.transistor_size]
		self.latency = tech_params['latency'][self.transistor_size]
		self.latency_scale = latency_scale
		self.dynamic_power = tech_params['dynamic_power'][self.transistor_size]
		self.leakage_power = tech_params['leakage_power'][self.transistor_size]
		# print(f"t_size: {self.transistor_size}, cache: {self.cache_size}, mem_layers: {self.mem_layers}, pitch: {self.pitch}")
		# print(f"tech_params[mem_area][t_size][cache_size][mem_layers]: {tech_params['mem_area'][self.transistor_size][self.cache_size][self.mem_layers]}")
		
		# this reg stuff should have its own numbers. Those mem numbers are for SRAM cache
		# self.area["Regs"] = tech_params['mem_area'][self.transistor_size][self.cache_size][self.mem_layers][self.pitch]
		# self.latency["Regs"] = tech_params['mem_latency'][self.cache_size][self.mem_layers][self.pitch]
		# self.dynamic_power["Regs"] = tech_params['mem_dynamic_power'][self.cache_size][self.mem_layers][self.pitch]
		# self.leakage_power["Regs"] = 1e-6*tech_params['mem_leakage_power'][self.cache_size][self.mem_layers][self.pitch]

		self.mem_area = tech_params['mem_area'][self.transistor_size][self.cache_size][self.mem_layers][self.pitch]
		# units of mW
		self.mem_leakage_power = tech_params['mem_leakage_power'][self.cache_size][self.mem_layers][self.pitch]
		# how does mem latency get incorporated? Currently reg latency = mem_latency. Is this why my num clock cycles is so high?
		## DO THIS!!!!
	
	def update_cache_size(self, cache_size):
		pass

	def init_misc_vars(self):
		self.compute_operation_totals = {}

		self.memory_cfgs = {}
		self.mem_state = {}
		for variable in self.memory_cfgs.keys():
			self.mem_state[variable]=False
		self.cycles = 0

		# what is this doing?
		for op in op2sym_map:
			self.compute_operation_totals[op] = 0
		
		self.hw_allocated = {}
		self.hw_allocated["Regs"] = 0

		for key in op2sym_map.keys():
			self.hw_allocated[key] = 0
		
		# I shouldn't have the whole dict as an instance variable, the instance var 
		# should be a single scalar based on some tech/ application parameters.
		self.power_scale = power_scale 

	def set_loop_counts(self, loop_counts):
		self.loop_counts = loop_counts

	def set_var_sizes(self, var_sizes):
		self.var_sizes = var_sizes

	def print_stats(self):
	   s = '''
	   cycles={cycles}
	   allocated={allocated}
	   utilized={utilized}
	   '''.format(cycles=self.cycles, \
			    allocated=str(self.hw_allocated))
	   return s

    
    