# This file contains code from NVSim, (c) 2012-2013,  Pennsylvania State University
#and Hewlett-Packard Company. See LICENSE_NVSim file in the top-level directory.
#No part of DESTINY Project, including this file, may be copied,
#modified, propagated, or distributed except according to the terms
#contained in the LICENSE file.

# Global variables that will be initialized at runtime
inputParameter = None
tech = None
devtech = None
cell = None
gtech = None
localWire = None     # The wire type of local interconnects (for example, wire in mat)
globalWire = None    # The wire type of global interconnects (for example, the ones that connect mats)
sweepCells = None

invalid_value = 1e41
infinite_ramp = 1e41
