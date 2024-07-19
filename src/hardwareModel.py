import json
import graphviz as gv
import re
from collections import deque
import ast

from staticfg.builder import CFGBuilder
from ast_utils import ASTUtils

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

# using 64 bit numbers, 5GHz clock
latency = {
    7: {
        "And": 1.75,
        "Or": 1.789,
        "Add": 16,
        "Sub": 16.175,
        "Mult": 56.24,
        "FloorDiv": 43.5,
        "Mod": 1.935,
        "LShift": 0.027,
        "RShift": 0.027,
        "BitOr": 0.256,
        "BitXor": 0.236,
        "BitAnd": 0.24,
        "Eq": 1.76,
        "NotEq": 1.935,
        "Lt": 1.935,
        "LtE": 1.935,
        "Gt": 2.615,
        "GtE": 2.64,
        "USub": 16.175,
        "UAdd": 16,
        "IsNot": 0.175,
        "Not": 0.175,
        "Invert": 0.175,
        "Regs": 2
    },
    5: {
        "And": 1.545,
        "Or": 1.58,
        "Add": 14.155,
        "Sub": 14.31,
        "Mult": 49.77,
        "FloorDiv": 38.56,
        "Mod": 1.71,
        "LShift": 0.024,
        "RShift": 0.024,
        "BitOr": 0.225,
        "BitXor": 0.208,
        "BitAnd": 0.213,
        "Eq": 1.555,
        "NotEq": 1.71,
        "Lt": 1.71,
        "LtE": 1.71,
        "Gt": 2.315,
        "GtE": 2.335,
        "USub": 14.31,
        "UAdd": 14.155,
        "IsNot": 0.155,
        "Not": 0.155,
        "Invert": 0.155,
        "Regs": 2
    },
    3: {
        "And": 1.395,
        "Or": 1.425,
        "Add": 12.755,
        "Sub": 12.895,
        "Mult": 44.835,
        "FloorDiv": 34.741,
        "Mod": 1.54,
        "LShift": 0.214,
        "RShift": 0.214,
        "BitOr": 0.2,
        "BitXor": 0.188,
        "BitAnd": 0.192,
        "Eq": 1.4,
        "NotEq": 1.54,
        "Lt": 1.54,
        "LtE": 1.54,
        "Gt": 2.085,
        "GtE": 2.105,
        "USub": 12.895,
        "UAdd": 12.755,
        "IsNot": 0.14,
        "Not": 0.14,
        "Invert": 0.14,
        "Regs": 2
    }
}

# in nW
dynamic_power = {
    7: { 
        "And": 81.809,
        "Or": 77.035,
        "Add": 49.728,
        "Sub": 56.896,
        "Mult": 9227.952,
        "FloorDiv": 26128.576,
        "Mod": 116.025,
        "LShift": 78.144,
        "RShift": 78.144,
        "BitOr": 22.72,
        "BitXor": 25.6,
        "BitAnd": 24.128,
        "Eq": 108.857,
        "NotEq": 176.977,
        "Lt": 170.539,
        "LtE": 160.534,
        "Gt": 162.316,
        "GtE": 143.232,
        "USub": 56.896,
        "UAdd": 49.728,
        "IsNot": 7.168,
        "Not": 7.168,
        "Invert": 7.168,
        "Regs": 1
    },
    5: { 
        "And": 64.62911,
        "Or": 60.85765,
        "Add": 39.28512,
        "Sub": 44.94784,
        "Mult": 7290.08208,
        "FloorDiv": 20641.57504,
        "Mod": 91.65975,
        "LShift": 61.73376,
        "RShift": 61.73376,
        "BitOr": 17.9488,
        "BitXor": 20.224,
        "BitAnd": 19.06112,
        "Eq": 85.99703,
        "NotEq": 139.81183,
        "Lt": 134.72581,
        "LtE": 126.82186,
        "Gt": 128.22964,
        "GtE": 113.15328,
        "USub": 44.94784,
        "UAdd": 39.28512,
        "IsNot": 5.66272,
        "Not": 5.66272,
        "Invert": 5.66272,
        "Regs": 1
    },
    3: { 
        "And": 47.1792503,
        "Or": 44.4260845,
        "Add": 28.6781376,
        "Sub": 32.8119232,
        "Mult": 5321.759918,
        "FloorDiv": 15068.34978,
        "Mod": 66.9116175,
        "LShift": 45.0656448,
        "RShift": 45.0656448,
        "BitOr": 13.102624,
        "BitXor": 14.76352,
        "BitAnd": 13.9146176,
        "Eq": 62.7778319,
        "NotEq": 102.0626359,
        "Lt": 98.3498413,
        "LtE": 92.5799578,
        "Gt": 93.6076372,
        "GtE": 82.6018944,
        "USub": 32.8119232,
        "UAdd": 28.6781376,
        "IsNot": 4.1337856,
        "Not": 4.1337856,
        "Invert": 4.1337856,
        "Regs": 1
    }
}

leakage_power = {
     7: { 
        "And": 4340,
        "Or": 4340,
        "Add": 2986.666667,
        "Sub": 3413.333333,
        "Mult": 538320,
        "FloorDiv": 1422080,
        "Mod": 3860,
        "LShift": 4266.666667,
        "RShift": 4266.666667,
        "BitOr": 1280,
        "BitXor": 1706.666667,
        "BitAnd": 1280,
        "Eq": 3433.333333,
        "NotEq": 6046.666667,
        "Lt": 6046.666667,
        "LtE": 4600,
        "Gt": 4600,
        "GtE": 4693.333333,
        "USub": 3413.333333,
        "UAdd": 2986.666667,
        "IsNot": 426.6666667,
        "Not": 426.6666667,
        "Invert": 426.6666667,
        "Regs": 426.6666667
    },
    5: { 
        "And": 3428.6,
        "Or": 3428.6,
        "Add": 2359.466667,
        "Sub": 2696.533333,
        "Mult": 425272.8,
        "FloorDiv": 1123443.2,
        "Mod": 3049.4,
        "LShift": 3370.666667,
        "RShift": 3370.666667,
        "BitOr": 1011.2,
        "BitXor": 1348.266667,
        "BitAnd": 1011.2,
        "Eq": 2712.333333,
        "NotEq": 4776.866667,
        "Lt": 4776.866667,
        "LtE": 3634,
        "Gt": 3634,
        "GtE": 3707.733333,
        "USub": 2696.533333,
        "UAdd": 2359.466667,
        "IsNot": 337.0666667,
        "Not": 337.0666667,
        "Invert": 337.0666667,
        "Regs": 1
    },
    3: { 
        "And": 2502.878,
        "Or": 2502.878,
        "Add": 1722.410667,
        "Sub": 1968.469333,
        "Mult": 310449.144,
        "FloorDiv": 820113.536,
        "Mod": 2226.062,
        "LShift": 2460.586667,
        "RShift": 2460.586667,
        "BitOr": 738.176,
        "BitXor": 984.2346667,
        "BitAnd": 738.176,
        "Eq": 1980.003333,
        "NotEq": 3487.112667,
        "Lt": 3487.112667,
        "LtE": 2652.82,
        "Gt": 2652.82,
        "GtE": 2706.645333,
        "USub": 1968.469333,
        "UAdd": 1722.410667,
        "IsNot": 246.0586667,
        "Not": 246.0586667,
        "Invert": 246.0586667,
        "Regs": 1
    }
}

area = {
    7: { 
        "And": 459.3456,
        "Or": 374.976,
        "Add": 282.9312,
        "Sub": 379.6992,
        "Mult": 51264.3168,
        "FloorDiv": 155840.7168,
        "Mod": 400.248,
        "LShift": 478.3104,
        "RShift": 478.3104,
        "BitOr": 110.592,
        "BitXor": 1147.456,
        "BitAnd": 135.4752,
        "Eq": 303.48,
        "NotEq": 606.8016,
        "Lt": 923.11872,
        "LtE": 950.62788,
        "Gt": 953.48997,
        "GtE": 831.07392,
        "USub": 282.9312,
        "UAdd": 379.6992,
        "IsNot": 96.768,
        "Not": 96.768,
        "Invert": 96.768,
        "Regs": 1
    },
    5: { 
        "And": 252.64008,
        "Or": 206.2368,
        "Add": 155.61216,
        "Sub": 208.83456,
        "Mult": 28195.37424,
        "FloorDiv": 85712.39424,
        "Mod": 220.1364,
        "LShift": 263.07072,
        "RShift": 263.07072,
        "BitOr": 60.8256,
        "BitXor": 81.1008,
        "BitAnd": 74.51136,
        "Eq": 166.914,
        "NotEq": 333.74088,
        "Lt": 507.715296,
        "LtE": 522.845334,
        "Gt": 524.4194835,
        "GtE": 457.090656,
        "USub": 155.61216,
        "UAdd": 208.83456,
        "IsNot": 53.2224,
        "Not": 53.2224,
        "Invert": 53.2224,
        "Regs": 1
    },
    3: { 
        "And": 149.0576472,
        "Or": 121.679712,
        "Add": 91.8111744,
        "Sub": 123.2123904,
        "Mult": 16635.2708,
        "FloorDiv": 50570.3126,
        "Mod": 129.880476,
        "LShift": 155.2117248,
        "RShift": 155.2117248,
        "BitOr": 35.887104,
        "BitXor": 47.849472,
        "BitAnd": 43.9617024,
        "Eq": 98.47926,
        "NotEq": 196.9071192,
        "Lt": 299.5520246,
        "LtE": 308.4787471,
        "Gt": 309.4074953,
        "GtE": 269.683487,
        "USub": 91.8111744,
        "UAdd": 123.2123904,
        "IsNot": 31.401216,
        "Not": 31.401216,
        "Invert": 31.401216,
        "Regs": 1
    }
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

mem_area_7_5 = {
     1: {
          1: {
               100: 8.26
          },
          2: {
               100: 21.38,
               10: 8.419, 
               1: 5.373, 
               0.1: 7.34
          }, 
          4: {
               100: 23.02,
               10: 7.499,
               1: 4.025,
               0.1: 2.535
          }
     }, 
     2: {
          1: {
               100: 20.83
          },
          2: {
               100: 27.952,
               10: 13.249,  
               1: 9.822, 
               0.1: 8.061
          }, 
          4: {
               100: 21.33,
               10: 9.657,
               1: 8.05,
               0.1: 7.334
          }
     }, 
     4: {
          1: {
               100: 42.1
          },
          2: {
               100: 43.182,
               10: 26.661, 
               1: 22.363, 
               0.1: 25.8
          }, 
          4: {
               100: 27.83,
               10: 14.089,
               1: 11.77,
               0.1: 25.796
          }
     }, 
     8: {
          1: {
               100: 73.12
          },
          2: {
               100: 65.894,
               10: 43.287, 
               1: 41.146, 
               0.1: 39.63
          }, 
          4: {
               100: 43,
               10: 28.191,
               1: 22.36,
               0.1: 25.796
          }
     }, 
     16: {
          1: {
               100: 137.7
          },
          2: {
               100: 111.754,
               10: 75.741, 
               1: 76.972, 
               0.1: 74.62
          }, 
          4: {
               100: 69.6,
               10: 43.72,
               1: 40.73,
               0.1: 39.219
          }
     }
}

mem_area_3 = {
     1: {
          1: {
               100: 0.34
          },
          2: {
               100: 0.878,
               10: 0.346, 
               1: 0.221, 
               0.1: 0.301
          }, 
          4: {
               100: 0.946,
               10: 0.308,
               1: 0.165,
               0.1: 0.104
          }
     }, 
     2: {
          1: {
               100: 0.856
          },
          2: {
               100: 1.15,
               10: 0.544,  
               1: 0.403, 
               0.1: 0.331
          }, 
          4: {
               100: 0.876,
               10: 0.397,
               1: 0.331,
               0.1: 0.301
          }
     }, 
     4: {
          1: {
               100: 1.73
          },
          2: {
               100: 1.774,
               10: 1.095, 
               1: 0.919, 
               0.1: 1.06
          }, 
          4: {
               100: 1.144,
               10: 0.579,
               1: 0.484,
               0.1: 0.38
          }
     }, 
     8: {
          1: {
               100: 3.004
          },
          2: {
               100: 2.707,
               10: 1.779, 
               1: 1.691, 
               0.1: 1.628
          }, 
          4: {
               100: 1.767,
               10: 1.159,
               1: 0.919,
               0.1: 1.06
          }
     }, 
     16: {
          1: {
               100: 5.658
          },
          2: {
               100: 4.592,
               10: 3.112, 
               1: 3.162, 
               0.1: 3.066
          }, 
          4: {
               100: 2.86,
               10: 1.796,
               1: 1.673,
               0.1: 1.611
          }
     }
}

mem_latency = {
     1: {
          1: {
               100: 21.325
          },
          2: {
               100: 5.125,
               10: 3.045, 
               1: 14.445, 
               0.1: 2.14
          }, 
          4: {
               100: 4.625,
               10: 2.405,
               1: 1.895,
               0.1: 10.755
          }
     }, 
     2: {
          1: {
               100: 19.51
          },
          2: {
               100: 6.44,
               10: 4.375, 
               1: 21.35, 
               0.1: 21.35
          }, 
          4: {
               100: 5.185,
               10: 3.075,
               1: 2.38,
               0.1: 2.14
          }
     }, 
     4: {
          1: {
               100: 7.54
          },
          2: {
               100: 8.745,
               10: 5.615, 
               1: 4.8225, 
               0.1: 4.8
          }, 
          4: {
               100: 6.475,
               10: 4.285,
               1: 3.385,
               0.1: 21.35
          }
     }, 
     8: {
          1: {
               100: 12.825
          },
          2: {
               100: 11.685,
               10: 8.35, 
               1: 7.175, 
               0.1: 6.98
          }, 
          4: {
               100: 8.825,
               10: 5.59,
               1: 4.825,
               0.1: 4.825
          }
     }, 
     16: {
          1: {
               100: 20.63
          },
          2: {
               100: 18.575,
               10: 13.56, 
               1: 12.095, 
               0.1: 11.85
          }, 
          4: {
               100: 12.145,
               10: 8.36,
               1: 7.13,
               0.1: 6.935
          }
     }
}

# in nW
mem_dynamic_power = {
     1: {
          1: {
               100: 0.003
          },
          2: {
               100: 0.132,
               10: 0.026, 
               1: 0.004, 
               0.1: 0.035
          }, 
          4: {
               100: 1.914,
               10: 0.042,
               1: 0.031,
               0.1: 0.003
          }
     }, 
     2: {
          1: {
               100: 0.007
          },
          2: {
               100: 0.109,
               10: 0.022, 
               1: 0.007, 
               0.1: 0.006
          }, 
          4: {
               100: 0.925,
               10: 0.036,
               1: 0.033,
               0.1: 0.035
          }
     }, 
     4: {
          1: {
               100: 0.022
          },
          2: {
               100: 0.086,
               10: 0.025, 
               1: 0.025, 
               0.1: 0.027
          }, 
          4: {
               100: 1.333,
               10: 0.03,
               1: 0.028,
               0.1: 0.004
          }
     }, 
     8: {
          1: {
               100: 0.017
          },
          2: {
               100: 0.07,
               10: 0.021, 
               1: 0.023, 
               0.1: 0.023
          }, 
          4: {
               100: 1.069,
               10: 0.03,
               1: 0.023,
               0.1: 0.027
          }
     }, 
     16: {
          1: {
               100: 0.014
          },
          2: {
               100: 0.047,
               10: 0.017, 
               1: 0.019, 
               0.1: 0.019
          }, 
          4: {
               100: 0.786,
               10: 0.024,
               1: 0.024,
               0.1: 0.023
          }
     }
}

# in nW: table measurements were in mW so multiplying by 10^-6
mem_leakage_power = {
     1: {
          1: {
               100: 84.488e-6
          },
          2: {
               100: 134.778e-6,
               10: 120.24e-6, 
               1: 113.61e-6, 
               0.1: 163.32e-6
          }, 
          4: {
               100: 332e-6,
               10: 147.956e-6,
               1: 160.134e-6,
               0.1: 125.092e-6
          }
     }, 
     2: {
          1: {
               100: 203.537e-6
          },
          2: {
               100: 253.749e-6,
               10: 240.276e-6, 
               1: 201.01e-6, 
               0.1: 167.29e-6
          }, 
          4: {
               100: 367.021e-6,
               10: 239.929e-6,
               1: 297.911e-6,
               0.1: 326.641e-6
          }
     }, 
     4: {
          1: {
               100: 430.711e-6
          },
          2: {
               100: 449.317e-6,
               10: 446.145e-6, 
               1: 449.08e-6, 
               0.1: 523.43e-6
          }, 
          4: {
               100: 502.546e-6,
               10: 476.738e-6,
               1: 470.928e-6,
               0.1: 393.097e-6
          }
     }, 
     8: {
          1: {
               100: 771.303e-6
          },
          2: {
               100: 797.864e-6,
               10: 830.644e-6, 
               1: 872.39e-6, 
               0.1: 872.39e-6
          }, 
          4: {
               100: 889.741e-6,
               10: 872.388e-6,
               1: 898.161e-6,
               0.1: 1046.85e-6
          }
     }, 
     16: {
          1: {
               100: 1496.1e-6
          },
          2: {
               100: 1626.42e-6,
               10: 1532.73e-6, 
               1: 1621.4e-6, 
               0.1: 1621.4e-6
          }, 
          4: {
               100: 1688.8e-6,
               10: 1653.34e-6,
               1: 1728.534e-6,
               0.1: 1728.53e-6
          }
     }
}

noc_area_3 = {
     16: {
          1: {
               100: 5.658 # 100% of memory
          }
	}
}

noc_latency = {
     16: {
          1: {
               100: 2.06 # 10% of memory
          }
     }
}

# in nW
noc_dynamic_power = {
     16: {
          1: {
               100: 0.0105 # 75% of memory
          }
     }
}

# in nW: table measurements were in mW so multiplying by 10^-6
noc_leakage_power = {
     16: {
          1: {
               100: 1496.1e-6 # 100% of memory
          }
     }
}


class HardwareModel:

    def __init__(self,id,bandwidth,mem_layers,pitch,transistor_size,cache_size,loop_counts={},var_sizes={}):
        self.max_bw = bandwidth
        self.bw_avail = bandwidth

        self.loop_counts = loop_counts

        self.memory_cfgs = {}
        self.mem_state = {}
        for variable in self.memory_cfgs.keys():
            self.mem_state[variable]=False
        
        self.compute_operation_totals = {}
        for op in op2sym_map:
             self.compute_operation_totals[op] = 0
        # number of non-memory elements allocated
        self.transistor_size = transistor_size
        self.mem_layers = mem_layers
        self.pitch = pitch
        self.hw_allocated = {}
        self.hw_allocated["Regs"] = 0
        self.hw_allocated["NoCs"] = 0
        self.loop_variables = loop_counts
        self.var_sizes = var_sizes
        self.id = id
        self.cache_size = cache_size
        self.area = area[transistor_size]
        self.latency = latency[transistor_size]
        self.latency_scale = latency_scale
        self.dynamic_power = dynamic_power[transistor_size]
        self.leakage_power = leakage_power[transistor_size]
        self.power_scale = power_scale
        if transistor_size == 3:
            self.area["Regs"] = mem_area_3[cache_size][mem_layers][pitch]
            self.area["NoCs"] = noc_area_3[cache_size][mem_layers][pitch]
        else:
            self.area["Regs"] = mem_area_7_5[cache_size][mem_layers][pitch]
            self.area["NoCs"] = mem_area_7_5[cache_size][mem_layers][pitch]
        self.latency["Regs"] = mem_latency[cache_size][mem_layers][pitch]
        self.dynamic_power["Regs"] = mem_dynamic_power[cache_size][mem_layers][pitch]
        self.leakage_power["Regs"] = mem_leakage_power[cache_size][mem_layers][pitch]
        self.latency["NoCs"] = noc_latency[cache_size][mem_layers][pitch]
        self.dynamic_power["NoCs"] = noc_dynamic_power[cache_size][mem_layers][pitch]
        self.leakage_power["NoCs"] = noc_leakage_power[cache_size][mem_layers][pitch]
        for key in op2sym_map.keys():
                self.hw_allocated[key] = 0

        self.cycles = 0


    def print_stats(self):
        s = '''
        cycles={cycles}
        allocated={allocated}
        utilized={utilized}
        '''.format(cycles=self.cycles, \
                   allocated=str(self.hw_allocated))
        return s

    
    
