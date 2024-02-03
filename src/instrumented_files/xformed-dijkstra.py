import sys
from instrument_lib import *
import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop
import numpy as np


class CS161Vertex:

    def __init__(self, v):
        print('enter scope 1')
        print(1, 12)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        v_1 = instrument_read(v, 'v')
        write_instrument_read(v_1, 'v_1')
        if type(v_1) == np.ndarray:
            print('malloc', sys.getsizeof(v_1), 'v_1', v_1.shape)
        elif type(v_1) == list:
            dims = []
            tmp = v_1
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_1), 'v_1', dims)
        elif type(v_1) == tuple:
            print('malloc', sys.getsizeof(v_1), 'v_1', [len(v_1)])
        else:
            print('malloc', sys.getsizeof(v_1), 'v_1')
        instrument_read(self, 'self').inNeighbors = []
        instrument_read(self, 'self').outNeighbors = []
        instrument_read(self, 'self').value = instrument_read(v_1, 'v_1')
        instrument_read(self, 'self').inTime = None
        instrument_read(self, 'self').outTime = None
        instrument_read(self, 'self').status = 'unvisited'
        instrument_read(self, 'self').parent = None
        instrument_read(self, 'self').estD = instrument_read(math, 'math').inf
        print('exit scope 1')

    def hasOutNeighbor(self, v):
        print('enter scope 2')
        print(1, 24)
        print(6, 25)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(6, 26)
        v_2 = instrument_read(v, 'v')
        write_instrument_read(v_2, 'v_2')
        if type(v_2) == np.ndarray:
            print('malloc', sys.getsizeof(v_2), 'v_2', v_2.shape)
        elif type(v_2) == list:
            dims = []
            tmp = v_2
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_2), 'v_2', dims)
        elif type(v_2) == tuple:
            print('malloc', sys.getsizeof(v_2), 'v_2', [len(v_2)])
        else:
            print('malloc', sys.getsizeof(v_2), 'v_2')
        if instrument_read(v_2, 'v_2') in instrument_read(self, 'self'
            ).getOutNeighbors():
            print('exit scope 2')
            return True
        print('exit scope 2')
        return False
        print('exit scope 2')

    def hasInNeighbor(self, v):
        print('enter scope 3')
        print(1, 31)
        print(13, 32)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(13, 33)
        v_3 = instrument_read(v, 'v')
        write_instrument_read(v_3, 'v_3')
        if type(v_3) == np.ndarray:
            print('malloc', sys.getsizeof(v_3), 'v_3', v_3.shape)
        elif type(v_3) == list:
            dims = []
            tmp = v_3
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_3), 'v_3', dims)
        elif type(v_3) == tuple:
            print('malloc', sys.getsizeof(v_3), 'v_3', [len(v_3)])
        else:
            print('malloc', sys.getsizeof(v_3), 'v_3')
        if instrument_read(v_3, 'v_3') in instrument_read(self, 'self'
            ).getInNeighbors():
            print('exit scope 3')
            return True
        print('exit scope 3')
        return False
        print('exit scope 3')

    def hasNeighbor(self, v):
        print('enter scope 4')
        print(1, 38)
        print(20, 39)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(20, 40)
        v_4 = instrument_read(v, 'v')
        write_instrument_read(v_4, 'v_4')
        if type(v_4) == np.ndarray:
            print('malloc', sys.getsizeof(v_4), 'v_4', v_4.shape)
        elif type(v_4) == list:
            dims = []
            tmp = v_4
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_4), 'v_4', dims)
        elif type(v_4) == tuple:
            print('malloc', sys.getsizeof(v_4), 'v_4', [len(v_4)])
        else:
            print('malloc', sys.getsizeof(v_4), 'v_4')
        if instrument_read(v_4, 'v_4') in instrument_read(self, 'self'
            ).getInNeighbors() or instrument_read(v_4, 'v_4'
            ) in instrument_read(self, 'self').getOutNeighbors():
            print('exit scope 4')
            return True
        print('exit scope 4')
        return False
        print('exit scope 4')

    def getOutNeighbors(self):
        print('enter scope 5')
        print(1, 45)
        print(27, 46)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 5')
        return [instrument_read_sub(instrument_read(v_5, 'v_5'), 'v_5', 0,
            None, None, False) for v_5 in instrument_read(self, 'self').
            outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 49)
        print(31, 50)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 6')
        return [instrument_read_sub(instrument_read(v_6, 'v_6'), 'v_6', 0,
            None, None, False) for v_6 in instrument_read(self, 'self').
            inNeighbors]
        print('exit scope 6')

    def getOutNeighborsWithWeights(self):
        print('enter scope 7')
        print(1, 53)
        print(35, 54)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 7')
        return instrument_read(self, 'self').outNeighbors
        print('exit scope 7')

    def getInNeighborsWithWeights(self):
        print('enter scope 8')
        print(1, 57)
        print(39, 58)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 8')
        return instrument_read(self, 'self').inNeighbors
        print('exit scope 8')

    def addOutNeighbor(self, v, wt):
        print('enter scope 9')
        print(1, 61)
        print(43, 62)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(43, 63)
        v_9 = instrument_read(v, 'v')
        write_instrument_read(v_9, 'v_9')
        if type(v_9) == np.ndarray:
            print('malloc', sys.getsizeof(v_9), 'v_9', v_9.shape)
        elif type(v_9) == list:
            dims = []
            tmp = v_9
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_9), 'v_9', dims)
        elif type(v_9) == tuple:
            print('malloc', sys.getsizeof(v_9), 'v_9', [len(v_9)])
        else:
            print('malloc', sys.getsizeof(v_9), 'v_9')
        print(43, 64)
        wt_9 = instrument_read(wt, 'wt')
        write_instrument_read(wt_9, 'wt_9')
        if type(wt_9) == np.ndarray:
            print('malloc', sys.getsizeof(wt_9), 'wt_9', wt_9.shape)
        elif type(wt_9) == list:
            dims = []
            tmp = wt_9
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_9), 'wt_9', dims)
        elif type(wt_9) == tuple:
            print('malloc', sys.getsizeof(wt_9), 'wt_9', [len(wt_9)])
        else:
            print('malloc', sys.getsizeof(wt_9), 'wt_9')
        instrument_read(self, 'self').outNeighbors.append((instrument_read(
            v_9, 'v_9'), instrument_read(wt_9, 'wt_9')))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 67)
        print(46, 68)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(46, 69)
        v_10 = instrument_read(v, 'v')
        write_instrument_read(v_10, 'v_10')
        if type(v_10) == np.ndarray:
            print('malloc', sys.getsizeof(v_10), 'v_10', v_10.shape)
        elif type(v_10) == list:
            dims = []
            tmp = v_10
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_10), 'v_10', dims)
        elif type(v_10) == tuple:
            print('malloc', sys.getsizeof(v_10), 'v_10', [len(v_10)])
        else:
            print('malloc', sys.getsizeof(v_10), 'v_10')
        print(46, 70)
        wt_10 = instrument_read(wt, 'wt')
        write_instrument_read(wt_10, 'wt_10')
        if type(wt_10) == np.ndarray:
            print('malloc', sys.getsizeof(wt_10), 'wt_10', wt_10.shape)
        elif type(wt_10) == list:
            dims = []
            tmp = wt_10
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_10), 'wt_10', dims)
        elif type(wt_10) == tuple:
            print('malloc', sys.getsizeof(wt_10), 'wt_10', [len(wt_10)])
        else:
            print('malloc', sys.getsizeof(wt_10), 'wt_10')
        instrument_read(self, 'self').inNeighbors.append((instrument_read(
            v_10, 'v_10'), instrument_read(wt_10, 'wt_10')))
        print('exit scope 10')


class CS161Graph:

    def __init__(self):
        print('enter scope 11')
        print(1, 76)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        instrument_read(self, 'self').vertices = []
        print('exit scope 11')

    def addVertex(self, n):
        print('enter scope 12')
        print(1, 80)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        n_12 = instrument_read(n, 'n')
        write_instrument_read(n_12, 'n_12')
        if type(n_12) == np.ndarray:
            print('malloc', sys.getsizeof(n_12), 'n_12', n_12.shape)
        elif type(n_12) == list:
            dims = []
            tmp = n_12
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(n_12), 'n_12', dims)
        elif type(n_12) == tuple:
            print('malloc', sys.getsizeof(n_12), 'n_12', [len(n_12)])
        else:
            print('malloc', sys.getsizeof(n_12), 'n_12')
        instrument_read(self, 'self').vertices.append(instrument_read(n_12,
            'n_12'))
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 85)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        u_13 = instrument_read(u, 'u')
        write_instrument_read(u_13, 'u_13')
        if type(u_13) == np.ndarray:
            print('malloc', sys.getsizeof(u_13), 'u_13', u_13.shape)
        elif type(u_13) == list:
            dims = []
            tmp = u_13
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_13), 'u_13', dims)
        elif type(u_13) == tuple:
            print('malloc', sys.getsizeof(u_13), 'u_13', [len(u_13)])
        else:
            print('malloc', sys.getsizeof(u_13), 'u_13')
        v_13 = instrument_read(v, 'v')
        write_instrument_read(v_13, 'v_13')
        if type(v_13) == np.ndarray:
            print('malloc', sys.getsizeof(v_13), 'v_13', v_13.shape)
        elif type(v_13) == list:
            dims = []
            tmp = v_13
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_13), 'v_13', dims)
        elif type(v_13) == tuple:
            print('malloc', sys.getsizeof(v_13), 'v_13', [len(v_13)])
        else:
            print('malloc', sys.getsizeof(v_13), 'v_13')
        wt_13 = instrument_read(wt, 'wt')
        write_instrument_read(wt_13, 'wt_13')
        if type(wt_13) == np.ndarray:
            print('malloc', sys.getsizeof(wt_13), 'wt_13', wt_13.shape)
        elif type(wt_13) == list:
            dims = []
            tmp = wt_13
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_13), 'wt_13', dims)
        elif type(wt_13) == tuple:
            print('malloc', sys.getsizeof(wt_13), 'wt_13', [len(wt_13)])
        else:
            print('malloc', sys.getsizeof(wt_13), 'wt_13')
        instrument_read(u_13, 'u_13').addOutNeighbor(instrument_read(v_13,
            'v_13'), wt=wt_13)
        instrument_read(v_13, 'v_13').addInNeighbor(instrument_read(u_13,
            'u_13'), wt=wt_13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 93)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        u_14 = instrument_read(u, 'u')
        write_instrument_read(u_14, 'u_14')
        if type(u_14) == np.ndarray:
            print('malloc', sys.getsizeof(u_14), 'u_14', u_14.shape)
        elif type(u_14) == list:
            dims = []
            tmp = u_14
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_14), 'u_14', dims)
        elif type(u_14) == tuple:
            print('malloc', sys.getsizeof(u_14), 'u_14', [len(u_14)])
        else:
            print('malloc', sys.getsizeof(u_14), 'u_14')
        v_14 = instrument_read(v, 'v')
        write_instrument_read(v_14, 'v_14')
        if type(v_14) == np.ndarray:
            print('malloc', sys.getsizeof(v_14), 'v_14', v_14.shape)
        elif type(v_14) == list:
            dims = []
            tmp = v_14
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_14), 'v_14', dims)
        elif type(v_14) == tuple:
            print('malloc', sys.getsizeof(v_14), 'v_14', [len(v_14)])
        else:
            print('malloc', sys.getsizeof(v_14), 'v_14')
        wt_14 = instrument_read(wt, 'wt')
        write_instrument_read(wt_14, 'wt_14')
        if type(wt_14) == np.ndarray:
            print('malloc', sys.getsizeof(wt_14), 'wt_14', wt_14.shape)
        elif type(wt_14) == list:
            dims = []
            tmp = wt_14
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_14), 'wt_14', dims)
        elif type(wt_14) == tuple:
            print('malloc', sys.getsizeof(wt_14), 'wt_14', [len(wt_14)])
        else:
            print('malloc', sys.getsizeof(wt_14), 'wt_14')
        instrument_read(self, 'self').addDiEdge(instrument_read(u_14,
            'u_14'), instrument_read(v_14, 'v_14'), wt=wt_14)
        instrument_read(self, 'self').addDiEdge(instrument_read(v_14,
            'v_14'), instrument_read(u_14, 'u_14'), wt=wt_14)
        print('exit scope 14')

    def getDirEdges(self):
        print('enter scope 15')
        print(1, 101)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        ret_15 = []
        write_instrument_read(ret_15, 'ret_15')
        if type(ret_15) == np.ndarray:
            print('malloc', sys.getsizeof(ret_15), 'ret_15', ret_15.shape)
        elif type(ret_15) == list:
            dims = []
            tmp = ret_15
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(ret_15), 'ret_15', dims)
        elif type(ret_15) == tuple:
            print('malloc', sys.getsizeof(ret_15), 'ret_15', [len(ret_15)])
        else:
            print('malloc', sys.getsizeof(ret_15), 'ret_15')
        for v_15 in instrument_read(self, 'self').vertices:
            for u_15, wt_15 in instrument_read(v_15, 'v_15'
                ).getOutNeighborsWithWeights():
                instrument_read(ret_15, 'ret_15').append([instrument_read(
                    v_15, 'v_15'), instrument_read(u_15, 'u_15'),
                    instrument_read(wt_15, 'wt_15')])
        print('exit scope 15')
        return instrument_read(ret_15, 'ret_15')
        print('exit scope 15')


class CS161Graph:

    def __init__(self):
        print('enter scope 16')
        print(1, 112)
        print(70, 113)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(70, 114)
        instrument_read(self, 'self').vertices = []
        print('exit scope 16')

    def addVertex(self, n):
        print('enter scope 17')
        print(1, 116)
        print(73, 117)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(73, 118)
        n_17 = instrument_read(n, 'n')
        write_instrument_read(n_17, 'n_17')
        if type(n_17) == np.ndarray:
            print('malloc', sys.getsizeof(n_17), 'n_17', n_17.shape)
        elif type(n_17) == list:
            dims = []
            tmp = n_17
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(n_17), 'n_17', dims)
        elif type(n_17) == tuple:
            print('malloc', sys.getsizeof(n_17), 'n_17', [len(n_17)])
        else:
            print('malloc', sys.getsizeof(n_17), 'n_17')
        instrument_read(self, 'self').vertices.append(instrument_read(n_17,
            'n_17'))
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 121)
        print(76, 122)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(76, 123)
        u_18 = instrument_read(u, 'u')
        write_instrument_read(u_18, 'u_18')
        if type(u_18) == np.ndarray:
            print('malloc', sys.getsizeof(u_18), 'u_18', u_18.shape)
        elif type(u_18) == list:
            dims = []
            tmp = u_18
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_18), 'u_18', dims)
        elif type(u_18) == tuple:
            print('malloc', sys.getsizeof(u_18), 'u_18', [len(u_18)])
        else:
            print('malloc', sys.getsizeof(u_18), 'u_18')
        print(76, 124)
        v_18 = instrument_read(v, 'v')
        write_instrument_read(v_18, 'v_18')
        if type(v_18) == np.ndarray:
            print('malloc', sys.getsizeof(v_18), 'v_18', v_18.shape)
        elif type(v_18) == list:
            dims = []
            tmp = v_18
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_18), 'v_18', dims)
        elif type(v_18) == tuple:
            print('malloc', sys.getsizeof(v_18), 'v_18', [len(v_18)])
        else:
            print('malloc', sys.getsizeof(v_18), 'v_18')
        print(76, 125)
        wt_18 = instrument_read(wt, 'wt')
        write_instrument_read(wt_18, 'wt_18')
        if type(wt_18) == np.ndarray:
            print('malloc', sys.getsizeof(wt_18), 'wt_18', wt_18.shape)
        elif type(wt_18) == list:
            dims = []
            tmp = wt_18
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_18), 'wt_18', dims)
        elif type(wt_18) == tuple:
            print('malloc', sys.getsizeof(wt_18), 'wt_18', [len(wt_18)])
        else:
            print('malloc', sys.getsizeof(wt_18), 'wt_18')
        instrument_read(u_18, 'u_18').addOutNeighbor(instrument_read(v_18,
            'v_18'), wt=wt_18)
        instrument_read(v_18, 'v_18').addInNeighbor(instrument_read(u_18,
            'u_18'), wt=wt_18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 129)
        print(79, 130)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(79, 131)
        u_19 = instrument_read(u, 'u')
        write_instrument_read(u_19, 'u_19')
        if type(u_19) == np.ndarray:
            print('malloc', sys.getsizeof(u_19), 'u_19', u_19.shape)
        elif type(u_19) == list:
            dims = []
            tmp = u_19
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_19), 'u_19', dims)
        elif type(u_19) == tuple:
            print('malloc', sys.getsizeof(u_19), 'u_19', [len(u_19)])
        else:
            print('malloc', sys.getsizeof(u_19), 'u_19')
        print(79, 132)
        v_19 = instrument_read(v, 'v')
        write_instrument_read(v_19, 'v_19')
        if type(v_19) == np.ndarray:
            print('malloc', sys.getsizeof(v_19), 'v_19', v_19.shape)
        elif type(v_19) == list:
            dims = []
            tmp = v_19
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(v_19), 'v_19', dims)
        elif type(v_19) == tuple:
            print('malloc', sys.getsizeof(v_19), 'v_19', [len(v_19)])
        else:
            print('malloc', sys.getsizeof(v_19), 'v_19')
        print(79, 133)
        wt_19 = instrument_read(wt, 'wt')
        write_instrument_read(wt_19, 'wt_19')
        if type(wt_19) == np.ndarray:
            print('malloc', sys.getsizeof(wt_19), 'wt_19', wt_19.shape)
        elif type(wt_19) == list:
            dims = []
            tmp = wt_19
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(wt_19), 'wt_19', dims)
        elif type(wt_19) == tuple:
            print('malloc', sys.getsizeof(wt_19), 'wt_19', [len(wt_19)])
        else:
            print('malloc', sys.getsizeof(wt_19), 'wt_19')
        instrument_read(self, 'self').addDiEdge(instrument_read(u_19,
            'u_19'), instrument_read(v_19, 'v_19'), wt=wt_19)
        instrument_read(self, 'self').addDiEdge(instrument_read(v_19,
            'v_19'), instrument_read(u_19, 'u_19'), wt=wt_19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 137)
        print(82, 138)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(82, 139)
        ret_20 = []
        write_instrument_read(ret_20, 'ret_20')
        if type(ret_20) == np.ndarray:
            print('malloc', sys.getsizeof(ret_20), 'ret_20', ret_20.shape)
        elif type(ret_20) == list:
            dims = []
            tmp = ret_20
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(ret_20), 'ret_20', dims)
        elif type(ret_20) == tuple:
            print('malloc', sys.getsizeof(ret_20), 'ret_20', [len(ret_20)])
        else:
            print('malloc', sys.getsizeof(ret_20), 'ret_20')
        for v_20 in instrument_read(self, 'self').vertices:
            for u_20, wt_20 in instrument_read(v_20, 'v_20'
                ).getOutNeighborsWithWeights():
                instrument_read(ret_20, 'ret_20').append([instrument_read(
                    v_20, 'v_20'), instrument_read(u_20, 'u_20'),
                    instrument_read(wt_20, 'wt_20')])
        print('exit scope 20')
        return instrument_read(ret_20, 'ret_20')
        print('exit scope 20')


def randomGraph(n, p, wts=[1]):
    print('enter scope 21')
    print(1, 146)
    print(91, 147)
    n_21 = instrument_read(n, 'n')
    write_instrument_read(n_21, 'n_21')
    if type(n_21) == np.ndarray:
        print('malloc', sys.getsizeof(n_21), 'n_21', n_21.shape)
    elif type(n_21) == list:
        dims = []
        tmp = n_21
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(n_21), 'n_21', dims)
    elif type(n_21) == tuple:
        print('malloc', sys.getsizeof(n_21), 'n_21', [len(n_21)])
    else:
        print('malloc', sys.getsizeof(n_21), 'n_21')
    print(91, 148)
    p_21 = instrument_read(p, 'p')
    write_instrument_read(p_21, 'p_21')
    if type(p_21) == np.ndarray:
        print('malloc', sys.getsizeof(p_21), 'p_21', p_21.shape)
    elif type(p_21) == list:
        dims = []
        tmp = p_21
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(p_21), 'p_21', dims)
    elif type(p_21) == tuple:
        print('malloc', sys.getsizeof(p_21), 'p_21', [len(p_21)])
    else:
        print('malloc', sys.getsizeof(p_21), 'p_21')
    print(91, 149)
    wts_21 = instrument_read(wts, 'wts')
    write_instrument_read(wts_21, 'wts_21')
    if type(wts_21) == np.ndarray:
        print('malloc', sys.getsizeof(wts_21), 'wts_21', wts_21.shape)
    elif type(wts_21) == list:
        dims = []
        tmp = wts_21
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(wts_21), 'wts_21', dims)
    elif type(wts_21) == tuple:
        print('malloc', sys.getsizeof(wts_21), 'wts_21', [len(wts_21)])
    else:
        print('malloc', sys.getsizeof(wts_21), 'wts_21')
    print(91, 150)
    G_21 = CS161Graph()
    write_instrument_read(G_21, 'G_21')
    if type(G_21) == np.ndarray:
        print('malloc', sys.getsizeof(G_21), 'G_21', G_21.shape)
    elif type(G_21) == list:
        dims = []
        tmp = G_21
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_21), 'G_21', dims)
    elif type(G_21) == tuple:
        print('malloc', sys.getsizeof(G_21), 'G_21', [len(G_21)])
    else:
        print('malloc', sys.getsizeof(G_21), 'G_21')
    print(91, 151)
    V_21 = [CS161Vertex(instrument_read(x_21, 'x_21')) for x_21 in range(
        instrument_read(n_21, 'n_21'))]
    write_instrument_read(V_21, 'V_21')
    if type(V_21) == np.ndarray:
        print('malloc', sys.getsizeof(V_21), 'V_21', V_21.shape)
    elif type(V_21) == list:
        dims = []
        tmp = V_21
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(V_21), 'V_21', dims)
    elif type(V_21) == tuple:
        print('malloc', sys.getsizeof(V_21), 'V_21', [len(V_21)])
    else:
        print('malloc', sys.getsizeof(V_21), 'V_21')
    for v_21 in instrument_read(V_21, 'V_21'):
        instrument_read(G_21, 'G_21').addVertex(instrument_read(v_21, 'v_21'))
    for v_21 in instrument_read(V_21, 'V_21'):
        print(95, 155)
        i_21 = 0
        write_instrument_read(i_21, 'i_21')
        if type(i_21) == np.ndarray:
            print('malloc', sys.getsizeof(i_21), 'i_21', i_21.shape)
        elif type(i_21) == list:
            dims = []
            tmp = i_21
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(i_21), 'i_21', dims)
        elif type(i_21) == tuple:
            print('malloc', sys.getsizeof(i_21), 'i_21', [len(i_21)])
        else:
            print('malloc', sys.getsizeof(i_21), 'i_21')
        for w_21 in instrument_read(V_21, 'V_21'):
            if instrument_read(v_21, 'v_21') != instrument_read(w_21, 'w_21'):
                if random() < instrument_read(p_21, 'p_21'):
                    instrument_read(G_21, 'G_21').addDiEdge(instrument_read
                        (v_21, 'v_21'), instrument_read(w_21, 'w_21'), wt=
                        choice(wts_21))
                    print(102, 160)
                    i_21 += 1
                    write_instrument_read(i_21, 'i_21')
            if instrument_read(i_21, 'i_21') > 15:
                break
    print('exit scope 21')
    return instrument_read(G_21, 'G_21')
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 166)
    print(109, 167)
    w_22 = instrument_read(w, 'w')
    write_instrument_read(w_22, 'w_22')
    if type(w_22) == np.ndarray:
        print('malloc', sys.getsizeof(w_22), 'w_22', w_22.shape)
    elif type(w_22) == list:
        dims = []
        tmp = w_22
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_22), 'w_22', dims)
    elif type(w_22) == tuple:
        print('malloc', sys.getsizeof(w_22), 'w_22', [len(w_22)])
    else:
        print('malloc', sys.getsizeof(w_22), 'w_22')
    print(109, 168)
    G_22 = instrument_read(G, 'G')
    write_instrument_read(G_22, 'G_22')
    if type(G_22) == np.ndarray:
        print('malloc', sys.getsizeof(G_22), 'G_22', G_22.shape)
    elif type(G_22) == list:
        dims = []
        tmp = G_22
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_22), 'G_22', dims)
    elif type(G_22) == tuple:
        print('malloc', sys.getsizeof(G_22), 'G_22', [len(G_22)])
    else:
        print('malloc', sys.getsizeof(G_22), 'G_22')
    for v_22 in instrument_read(G_22, 'G_22').vertices:
        print(111, 170)
        instrument_read(v_22, 'v_22').status = 'unvisited'
    print(112, 171)
    n_22 = len(instrument_read(G_22, 'G_22').vertices)
    write_instrument_read(n_22, 'n_22')
    if type(n_22) == np.ndarray:
        print('malloc', sys.getsizeof(n_22), 'n_22', n_22.shape)
    elif type(n_22) == list:
        dims = []
        tmp = n_22
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(n_22), 'n_22', dims)
    elif type(n_22) == tuple:
        print('malloc', sys.getsizeof(n_22), 'n_22', [len(n_22)])
    else:
        print('malloc', sys.getsizeof(n_22), 'n_22')
    print(112, 172)
    Ls_22 = [[] for i_22 in range(instrument_read(n_22, 'n_22'))]
    write_instrument_read(Ls_22, 'Ls_22')
    if type(Ls_22) == np.ndarray:
        print('malloc', sys.getsizeof(Ls_22), 'Ls_22', Ls_22.shape)
    elif type(Ls_22) == list:
        dims = []
        tmp = Ls_22
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(Ls_22), 'Ls_22', dims)
    elif type(Ls_22) == tuple:
        print('malloc', sys.getsizeof(Ls_22), 'Ls_22', [len(Ls_22)])
    else:
        print('malloc', sys.getsizeof(Ls_22), 'Ls_22')
    print(112, 173)
    Ls_22[0] = [instrument_read(w_22, 'w_22')]
    write_instrument_read_sub(Ls_22, 'Ls_22', 0, None, None, False)
    print(112, 174)
    instrument_read(w_22, 'w_22').status = 'visited'
    for i_22 in range(instrument_read(n_22, 'n_22')):
        for u_22 in instrument_read_sub(instrument_read(Ls_22, 'Ls_22'),
            'Ls_22', instrument_read(i_22, 'i_22'), None, None, False):
            for v_22 in instrument_read(u_22, 'u_22').getOutNeighbors():
                if instrument_read(v_22, 'v_22').status == 'unvisited':
                    print(120, 179)
                    instrument_read(v_22, 'v_22').status = 'visited'
                    print(120, 180)
                    instrument_read(v_22, 'v_22').parent = instrument_read(u_22
                        , 'u_22')
                    instrument_read_sub(instrument_read(Ls_22, 'Ls_22'),
                        'Ls_22', instrument_read(i_22, 'i_22') + 1, None,
                        None, False).append(instrument_read(v_22, 'v_22'))
    print('exit scope 22')
    return instrument_read(Ls_22, 'Ls_22')
    print('exit scope 22')


def BFS_shortestPaths(w, G):
    print('enter scope 23')
    print(1, 185)
    print(125, 186)
    w_23 = instrument_read(w, 'w')
    write_instrument_read(w_23, 'w_23')
    if type(w_23) == np.ndarray:
        print('malloc', sys.getsizeof(w_23), 'w_23', w_23.shape)
    elif type(w_23) == list:
        dims = []
        tmp = w_23
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_23), 'w_23', dims)
    elif type(w_23) == tuple:
        print('malloc', sys.getsizeof(w_23), 'w_23', [len(w_23)])
    else:
        print('malloc', sys.getsizeof(w_23), 'w_23')
    print(125, 187)
    G_23 = instrument_read(G, 'G')
    write_instrument_read(G_23, 'G_23')
    if type(G_23) == np.ndarray:
        print('malloc', sys.getsizeof(G_23), 'G_23', G_23.shape)
    elif type(G_23) == list:
        dims = []
        tmp = G_23
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_23), 'G_23', dims)
    elif type(G_23) == tuple:
        print('malloc', sys.getsizeof(G_23), 'G_23', [len(G_23)])
    else:
        print('malloc', sys.getsizeof(G_23), 'G_23')
    print(125, 188)
    Ls_23 = BFS(instrument_read(w_23, 'w_23'), instrument_read(G_23, 'G_23'))
    write_instrument_read(Ls_23, 'Ls_23')
    if type(Ls_23) == np.ndarray:
        print('malloc', sys.getsizeof(Ls_23), 'Ls_23', Ls_23.shape)
    elif type(Ls_23) == list:
        dims = []
        tmp = Ls_23
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(Ls_23), 'Ls_23', dims)
    elif type(Ls_23) == tuple:
        print('malloc', sys.getsizeof(Ls_23), 'Ls_23', [len(Ls_23)])
    else:
        print('malloc', sys.getsizeof(Ls_23), 'Ls_23')
    for i_23 in range(len(instrument_read(Ls_23, 'Ls_23'))):
        for w_23 in instrument_read_sub(instrument_read(Ls_23, 'Ls_23'),
            'Ls_23', instrument_read(i_23, 'i_23'), None, None, False):
            print(129, 191)
            path_23 = []
            write_instrument_read(path_23, 'path_23')
            if type(path_23) == np.ndarray:
                print('malloc', sys.getsizeof(path_23), 'path_23', path_23.
                    shape)
            elif type(path_23) == list:
                dims = []
                tmp = path_23
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(path_23), 'path_23', dims)
            elif type(path_23) == tuple:
                print('malloc', sys.getsizeof(path_23), 'path_23', [len(
                    path_23)])
            else:
                print('malloc', sys.getsizeof(path_23), 'path_23')
            print(129, 192)
            current_23 = instrument_read(w_23, 'w_23')
            write_instrument_read(current_23, 'current_23')
            if type(current_23) == np.ndarray:
                print('malloc', sys.getsizeof(current_23), 'current_23',
                    current_23.shape)
            elif type(current_23) == list:
                dims = []
                tmp = current_23
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(current_23), 'current_23', dims)
            elif type(current_23) == tuple:
                print('malloc', sys.getsizeof(current_23), 'current_23', [
                    len(current_23)])
            else:
                print('malloc', sys.getsizeof(current_23), 'current_23')
            for j_23 in range(instrument_read(i_23, 'i_23')):
                instrument_read(path_23, 'path_23').append(instrument_read(
                    current_23, 'current_23'))
                print(132, 195)
                current_23 = instrument_read(current_23, 'current_23').parent
                write_instrument_read(current_23, 'current_23')
                if type(current_23) == np.ndarray:
                    print('malloc', sys.getsizeof(current_23), 'current_23',
                        current_23.shape)
                elif type(current_23) == list:
                    dims = []
                    tmp = current_23
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(current_23), 'current_23',
                        dims)
                elif type(current_23) == tuple:
                    print('malloc', sys.getsizeof(current_23), 'current_23',
                        [len(current_23)])
                else:
                    print('malloc', sys.getsizeof(current_23), 'current_23')
            instrument_read(path_23, 'path_23').append(instrument_read(
                current_23, 'current_23'))
            instrument_read(path_23, 'path_23').reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 200)
    print(136, 201)
    w_24 = instrument_read(w, 'w')
    write_instrument_read(w_24, 'w_24')
    if type(w_24) == np.ndarray:
        print('malloc', sys.getsizeof(w_24), 'w_24', w_24.shape)
    elif type(w_24) == list:
        dims = []
        tmp = w_24
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_24), 'w_24', dims)
    elif type(w_24) == tuple:
        print('malloc', sys.getsizeof(w_24), 'w_24', [len(w_24)])
    else:
        print('malloc', sys.getsizeof(w_24), 'w_24')
    print(136, 202)
    G_24 = instrument_read(G, 'G')
    write_instrument_read(G_24, 'G_24')
    if type(G_24) == np.ndarray:
        print('malloc', sys.getsizeof(G_24), 'G_24', G_24.shape)
    elif type(G_24) == list:
        dims = []
        tmp = G_24
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_24), 'G_24', dims)
    elif type(G_24) == tuple:
        print('malloc', sys.getsizeof(G_24), 'G_24', [len(G_24)])
    else:
        print('malloc', sys.getsizeof(G_24), 'G_24')
    for v_24 in instrument_read(G_24, 'G_24').vertices:
        print(138, 204)
        instrument_read(v_24, 'v_24').estD = instrument_read(math, 'math').inf
    print(139, 205)
    instrument_read(w_24, 'w_24').estD = 0
    print(139, 206)
    unsureVertices_24 = instrument_read_sub(instrument_read(G_24, 'G_24').
        vertices, 'G_24.vertices', None, None, None, True)
    write_instrument_read(unsureVertices_24, 'unsureVertices_24')
    if type(unsureVertices_24) == np.ndarray:
        print('malloc', sys.getsizeof(unsureVertices_24),
            'unsureVertices_24', unsureVertices_24.shape)
    elif type(unsureVertices_24) == list:
        dims = []
        tmp = unsureVertices_24
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(unsureVertices_24),
            'unsureVertices_24', dims)
    elif type(unsureVertices_24) == tuple:
        print('malloc', sys.getsizeof(unsureVertices_24),
            'unsureVertices_24', [len(unsureVertices_24)])
    else:
        print('malloc', sys.getsizeof(unsureVertices_24), 'unsureVertices_24')
    while len(instrument_read(unsureVertices_24, 'unsureVertices_24')) > 0:
        print(141, 208)
        u_24 = None
        write_instrument_read(u_24, 'u_24')
        if type(u_24) == np.ndarray:
            print('malloc', sys.getsizeof(u_24), 'u_24', u_24.shape)
        elif type(u_24) == list:
            dims = []
            tmp = u_24
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_24), 'u_24', dims)
        elif type(u_24) == tuple:
            print('malloc', sys.getsizeof(u_24), 'u_24', [len(u_24)])
        else:
            print('malloc', sys.getsizeof(u_24), 'u_24')
        print(141, 209)
        minD_24 = instrument_read(math, 'math').inf
        write_instrument_read(minD_24, 'minD_24')
        if type(minD_24) == np.ndarray:
            print('malloc', sys.getsizeof(minD_24), 'minD_24', minD_24.shape)
        elif type(minD_24) == list:
            dims = []
            tmp = minD_24
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(minD_24), 'minD_24', dims)
        elif type(minD_24) == tuple:
            print('malloc', sys.getsizeof(minD_24), 'minD_24', [len(minD_24)])
        else:
            print('malloc', sys.getsizeof(minD_24), 'minD_24')
        for x_24 in instrument_read(unsureVertices_24, 'unsureVertices_24'):
            if instrument_read(x_24, 'x_24').estD < instrument_read(minD_24,
                'minD_24'):
                print(146, 212)
                minD_24 = instrument_read(x_24, 'x_24').estD
                write_instrument_read(minD_24, 'minD_24')
                if type(minD_24) == np.ndarray:
                    print('malloc', sys.getsizeof(minD_24), 'minD_24',
                        minD_24.shape)
                elif type(minD_24) == list:
                    dims = []
                    tmp = minD_24
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(minD_24), 'minD_24', dims)
                elif type(minD_24) == tuple:
                    print('malloc', sys.getsizeof(minD_24), 'minD_24', [len
                        (minD_24)])
                else:
                    print('malloc', sys.getsizeof(minD_24), 'minD_24')
                print(146, 213)
                u_24 = instrument_read(x_24, 'x_24')
                write_instrument_read(u_24, 'u_24')
                if type(u_24) == np.ndarray:
                    print('malloc', sys.getsizeof(u_24), 'u_24', u_24.shape)
                elif type(u_24) == list:
                    dims = []
                    tmp = u_24
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(u_24), 'u_24', dims)
                elif type(u_24) == tuple:
                    print('malloc', sys.getsizeof(u_24), 'u_24', [len(u_24)])
                else:
                    print('malloc', sys.getsizeof(u_24), 'u_24')
        if instrument_read(u_24, 'u_24') == None:
            print('exit scope 24')
            return
        for v_24, wt_24 in instrument_read(u_24, 'u_24'
            ).getOutNeighborsWithWeights():
            if instrument_read(u_24, 'u_24').estD + instrument_read(wt_24,
                'wt_24') < instrument_read(v_24, 'v_24').estD:
                print(153, 218)
                instrument_read(v_24, 'v_24').estD = instrument_read(u_24,
                    'u_24').estD + instrument_read(wt_24, 'wt_24')
                print(153, 219)
                instrument_read(v_24, 'v_24').parent = instrument_read(u_24,
                    'u_24')
        instrument_read(unsureVertices_24, 'unsureVertices_24').remove(
            instrument_read(u_24, 'u_24'))
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 223)
    print(157, 224)
    w_25 = instrument_read(w, 'w')
    write_instrument_read(w_25, 'w_25')
    if type(w_25) == np.ndarray:
        print('malloc', sys.getsizeof(w_25), 'w_25', w_25.shape)
    elif type(w_25) == list:
        dims = []
        tmp = w_25
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_25), 'w_25', dims)
    elif type(w_25) == tuple:
        print('malloc', sys.getsizeof(w_25), 'w_25', [len(w_25)])
    else:
        print('malloc', sys.getsizeof(w_25), 'w_25')
    print(157, 225)
    G_25 = instrument_read(G, 'G')
    write_instrument_read(G_25, 'G_25')
    if type(G_25) == np.ndarray:
        print('malloc', sys.getsizeof(G_25), 'G_25', G_25.shape)
    elif type(G_25) == list:
        dims = []
        tmp = G_25
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_25), 'G_25', dims)
    elif type(G_25) == tuple:
        print('malloc', sys.getsizeof(G_25), 'G_25', [len(G_25)])
    else:
        print('malloc', sys.getsizeof(G_25), 'G_25')
    dijkstraDumb(instrument_read(w_25, 'w_25'), instrument_read(G_25, 'G_25'))
    for v_25 in instrument_read(G_25, 'G_25').vertices:
        if instrument_read(v_25, 'v_25').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(162, 230)
        path_25 = []
        write_instrument_read(path_25, 'path_25')
        if type(path_25) == np.ndarray:
            print('malloc', sys.getsizeof(path_25), 'path_25', path_25.shape)
        elif type(path_25) == list:
            dims = []
            tmp = path_25
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(path_25), 'path_25', dims)
        elif type(path_25) == tuple:
            print('malloc', sys.getsizeof(path_25), 'path_25', [len(path_25)])
        else:
            print('malloc', sys.getsizeof(path_25), 'path_25')
        print(162, 231)
        current_25 = instrument_read(v_25, 'v_25')
        write_instrument_read(current_25, 'current_25')
        if type(current_25) == np.ndarray:
            print('malloc', sys.getsizeof(current_25), 'current_25',
                current_25.shape)
        elif type(current_25) == list:
            dims = []
            tmp = current_25
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(current_25), 'current_25', dims)
        elif type(current_25) == tuple:
            print('malloc', sys.getsizeof(current_25), 'current_25', [len(
                current_25)])
        else:
            print('malloc', sys.getsizeof(current_25), 'current_25')
        while instrument_read(current_25, 'current_25') != instrument_read(w_25
            , 'w_25'):
            instrument_read(path_25, 'path_25').append(instrument_read(
                current_25, 'current_25'))
            print(164, 234)
            current_25 = instrument_read(current_25, 'current_25').parent
            write_instrument_read(current_25, 'current_25')
            if type(current_25) == np.ndarray:
                print('malloc', sys.getsizeof(current_25), 'current_25',
                    current_25.shape)
            elif type(current_25) == list:
                dims = []
                tmp = current_25
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(current_25), 'current_25', dims)
            elif type(current_25) == tuple:
                print('malloc', sys.getsizeof(current_25), 'current_25', [
                    len(current_25)])
            else:
                print('malloc', sys.getsizeof(current_25), 'current_25')
        instrument_read(path_25, 'path_25').append(instrument_read(
            current_25, 'current_25'))
        instrument_read(path_25, 'path_25').reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 239)
    print(168, 240)
    w_26 = instrument_read(w, 'w')
    write_instrument_read(w_26, 'w_26')
    if type(w_26) == np.ndarray:
        print('malloc', sys.getsizeof(w_26), 'w_26', w_26.shape)
    elif type(w_26) == list:
        dims = []
        tmp = w_26
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_26), 'w_26', dims)
    elif type(w_26) == tuple:
        print('malloc', sys.getsizeof(w_26), 'w_26', [len(w_26)])
    else:
        print('malloc', sys.getsizeof(w_26), 'w_26')
    print(168, 241)
    G_26 = instrument_read(G, 'G')
    write_instrument_read(G_26, 'G_26')
    if type(G_26) == np.ndarray:
        print('malloc', sys.getsizeof(G_26), 'G_26', G_26.shape)
    elif type(G_26) == list:
        dims = []
        tmp = G_26
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_26), 'G_26', dims)
    elif type(G_26) == tuple:
        print('malloc', sys.getsizeof(G_26), 'G_26', [len(G_26)])
    else:
        print('malloc', sys.getsizeof(G_26), 'G_26')
    for v_26 in instrument_read(G_26, 'G_26').vertices:
        print(170, 243)
        instrument_read(v_26, 'v_26').estD = instrument_read(math, 'math').inf
    print(171, 244)
    instrument_read(w_26, 'w_26').estD = 0
    print(171, 245)
    unsureVertices_26 = instrument_read(heapdict, 'heapdict').heapdict()
    write_instrument_read(unsureVertices_26, 'unsureVertices_26')
    if type(unsureVertices_26) == np.ndarray:
        print('malloc', sys.getsizeof(unsureVertices_26),
            'unsureVertices_26', unsureVertices_26.shape)
    elif type(unsureVertices_26) == list:
        dims = []
        tmp = unsureVertices_26
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(unsureVertices_26),
            'unsureVertices_26', dims)
    elif type(unsureVertices_26) == tuple:
        print('malloc', sys.getsizeof(unsureVertices_26),
            'unsureVertices_26', [len(unsureVertices_26)])
    else:
        print('malloc', sys.getsizeof(unsureVertices_26), 'unsureVertices_26')
    for v_26 in instrument_read(G_26, 'G_26').vertices:
        print(173, 247)
        unsureVertices_26[instrument_read(instrument_read(v_26, 'v_26'),
            'v_26')] = instrument_read(v_26, 'v_26').estD
        write_instrument_read_sub(unsureVertices_26, 'unsureVertices_26',
            instrument_read(instrument_read(v_26, 'v_26'), 'v_26'), None,
            None, False)
    while len(instrument_read(unsureVertices_26, 'unsureVertices_26')) > 0:
        print(175, 249)
        u_26, dist_26 = instrument_read(unsureVertices_26, 'unsureVertices_26'
            ).popitem()
        write_instrument_read(dist_26, 'dist_26')
        if type(dist_26) == np.ndarray:
            print('malloc', sys.getsizeof(dist_26), 'dist_26', dist_26.shape)
        elif type(dist_26) == list:
            dims = []
            tmp = dist_26
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(dist_26), 'dist_26', dims)
        elif type(dist_26) == tuple:
            print('malloc', sys.getsizeof(dist_26), 'dist_26', [len(dist_26)])
        else:
            print('malloc', sys.getsizeof(dist_26), 'dist_26')
        if instrument_read(u_26, 'u_26').estD == instrument_read(math, 'math'
            ).inf:
            print('exit scope 26')
            return
        for v_26, wt_26 in instrument_read(u_26, 'u_26'
            ).getOutNeighborsWithWeights():
            if instrument_read(u_26, 'u_26').estD + instrument_read(wt_26,
                'wt_26') < instrument_read(v_26, 'v_26').estD:
                print(182, 254)
                instrument_read(v_26, 'v_26').estD = instrument_read(u_26,
                    'u_26').estD + instrument_read(wt_26, 'wt_26')
                print(182, 255)
                unsureVertices_26[instrument_read(instrument_read(v_26,
                    'v_26'), 'v_26')] = instrument_read(u_26, 'u_26'
                    ).estD + instrument_read(wt_26, 'wt_26')
                write_instrument_read_sub(unsureVertices_26,
                    'unsureVertices_26', instrument_read(instrument_read(
                    v_26, 'v_26'), 'v_26'), None, None, False)
                print(182, 256)
                instrument_read(v_26, 'v_26').parent = instrument_read(u_26,
                    'u_26')
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 259)
    print(186, 260)
    w_27 = instrument_read(w, 'w')
    write_instrument_read(w_27, 'w_27')
    if type(w_27) == np.ndarray:
        print('malloc', sys.getsizeof(w_27), 'w_27', w_27.shape)
    elif type(w_27) == list:
        dims = []
        tmp = w_27
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(w_27), 'w_27', dims)
    elif type(w_27) == tuple:
        print('malloc', sys.getsizeof(w_27), 'w_27', [len(w_27)])
    else:
        print('malloc', sys.getsizeof(w_27), 'w_27')
    print(186, 261)
    G_27 = instrument_read(G, 'G')
    write_instrument_read(G_27, 'G_27')
    if type(G_27) == np.ndarray:
        print('malloc', sys.getsizeof(G_27), 'G_27', G_27.shape)
    elif type(G_27) == list:
        dims = []
        tmp = G_27
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(G_27), 'G_27', dims)
    elif type(G_27) == tuple:
        print('malloc', sys.getsizeof(G_27), 'G_27', [len(G_27)])
    else:
        print('malloc', sys.getsizeof(G_27), 'G_27')
    dijkstra(instrument_read(w_27, 'w_27'), instrument_read(G_27, 'G_27'))
    for v_27 in instrument_read(G_27, 'G_27').vertices:
        if instrument_read(v_27, 'v_27').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(191, 266)
        path_27 = []
        write_instrument_read(path_27, 'path_27')
        if type(path_27) == np.ndarray:
            print('malloc', sys.getsizeof(path_27), 'path_27', path_27.shape)
        elif type(path_27) == list:
            dims = []
            tmp = path_27
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(path_27), 'path_27', dims)
        elif type(path_27) == tuple:
            print('malloc', sys.getsizeof(path_27), 'path_27', [len(path_27)])
        else:
            print('malloc', sys.getsizeof(path_27), 'path_27')
        print(191, 267)
        current_27 = instrument_read(v_27, 'v_27')
        write_instrument_read(current_27, 'current_27')
        if type(current_27) == np.ndarray:
            print('malloc', sys.getsizeof(current_27), 'current_27',
                current_27.shape)
        elif type(current_27) == list:
            dims = []
            tmp = current_27
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(current_27), 'current_27', dims)
        elif type(current_27) == tuple:
            print('malloc', sys.getsizeof(current_27), 'current_27', [len(
                current_27)])
        else:
            print('malloc', sys.getsizeof(current_27), 'current_27')
        while instrument_read(current_27, 'current_27') != instrument_read(w_27
            , 'w_27'):
            instrument_read(path_27, 'path_27').append(instrument_read(
                current_27, 'current_27'))
            print(193, 270)
            current_27 = instrument_read(current_27, 'current_27').parent
            write_instrument_read(current_27, 'current_27')
            if type(current_27) == np.ndarray:
                print('malloc', sys.getsizeof(current_27), 'current_27',
                    current_27.shape)
            elif type(current_27) == list:
                dims = []
                tmp = current_27
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(current_27), 'current_27', dims)
            elif type(current_27) == tuple:
                print('malloc', sys.getsizeof(current_27), 'current_27', [
                    len(current_27)])
            else:
                print('malloc', sys.getsizeof(current_27), 'current_27')
        instrument_read(path_27, 'path_27').append(instrument_read(
            current_27, 'current_27'))
        instrument_read(path_27, 'path_27').reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=1):
    print('enter scope 28')
    print(1, 275)
    print(197, 276)
    myFn_28 = instrument_read(myFn, 'myFn')
    write_instrument_read(myFn_28, 'myFn_28')
    if type(myFn_28) == np.ndarray:
        print('malloc', sys.getsizeof(myFn_28), 'myFn_28', myFn_28.shape)
    elif type(myFn_28) == list:
        dims = []
        tmp = myFn_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(myFn_28), 'myFn_28', dims)
    elif type(myFn_28) == tuple:
        print('malloc', sys.getsizeof(myFn_28), 'myFn_28', [len(myFn_28)])
    else:
        print('malloc', sys.getsizeof(myFn_28), 'myFn_28')
    print(197, 277)
    nVals_28 = instrument_read(nVals, 'nVals')
    write_instrument_read(nVals_28, 'nVals_28')
    if type(nVals_28) == np.ndarray:
        print('malloc', sys.getsizeof(nVals_28), 'nVals_28', nVals_28.shape)
    elif type(nVals_28) == list:
        dims = []
        tmp = nVals_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(nVals_28), 'nVals_28', dims)
    elif type(nVals_28) == tuple:
        print('malloc', sys.getsizeof(nVals_28), 'nVals_28', [len(nVals_28)])
    else:
        print('malloc', sys.getsizeof(nVals_28), 'nVals_28')
    print(197, 278)
    pFn_28 = instrument_read(pFn, 'pFn')
    write_instrument_read(pFn_28, 'pFn_28')
    if type(pFn_28) == np.ndarray:
        print('malloc', sys.getsizeof(pFn_28), 'pFn_28', pFn_28.shape)
    elif type(pFn_28) == list:
        dims = []
        tmp = pFn_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(pFn_28), 'pFn_28', dims)
    elif type(pFn_28) == tuple:
        print('malloc', sys.getsizeof(pFn_28), 'pFn_28', [len(pFn_28)])
    else:
        print('malloc', sys.getsizeof(pFn_28), 'pFn_28')
    print(197, 279)
    numTrials_28 = instrument_read(numTrials, 'numTrials')
    write_instrument_read(numTrials_28, 'numTrials_28')
    if type(numTrials_28) == np.ndarray:
        print('malloc', sys.getsizeof(numTrials_28), 'numTrials_28',
            numTrials_28.shape)
    elif type(numTrials_28) == list:
        dims = []
        tmp = numTrials_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(numTrials_28), 'numTrials_28', dims)
    elif type(numTrials_28) == tuple:
        print('malloc', sys.getsizeof(numTrials_28), 'numTrials_28', [len(
            numTrials_28)])
    else:
        print('malloc', sys.getsizeof(numTrials_28), 'numTrials_28')
    print(197, 280)
    nValues_28 = []
    write_instrument_read(nValues_28, 'nValues_28')
    if type(nValues_28) == np.ndarray:
        print('malloc', sys.getsizeof(nValues_28), 'nValues_28', nValues_28
            .shape)
    elif type(nValues_28) == list:
        dims = []
        tmp = nValues_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(nValues_28), 'nValues_28', dims)
    elif type(nValues_28) == tuple:
        print('malloc', sys.getsizeof(nValues_28), 'nValues_28', [len(
            nValues_28)])
    else:
        print('malloc', sys.getsizeof(nValues_28), 'nValues_28')
    print(197, 281)
    tValues_28 = []
    write_instrument_read(tValues_28, 'tValues_28')
    if type(tValues_28) == np.ndarray:
        print('malloc', sys.getsizeof(tValues_28), 'tValues_28', tValues_28
            .shape)
    elif type(tValues_28) == list:
        dims = []
        tmp = tValues_28
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(tValues_28), 'tValues_28', dims)
    elif type(tValues_28) == tuple:
        print('malloc', sys.getsizeof(tValues_28), 'tValues_28', [len(
            tValues_28)])
    else:
        print('malloc', sys.getsizeof(tValues_28), 'tValues_28')
    for n_28 in instrument_read(nVals_28, 'nVals_28'):
        print(199, 283)
        runtime_28 = 0
        write_instrument_read(runtime_28, 'runtime_28')
        if type(runtime_28) == np.ndarray:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28',
                runtime_28.shape)
        elif type(runtime_28) == list:
            dims = []
            tmp = runtime_28
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28', dims)
        elif type(runtime_28) == tuple:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28', [len(
                runtime_28)])
        else:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28')
        for t_28 in range(instrument_read(numTrials_28, 'numTrials_28')):
            print(202, 285)
            G_28 = randomGraph(instrument_read(n_28, 'n_28') * 10000, pFn(
                instrument_read(n_28, 'n_28')))
            write_instrument_read(G_28, 'G_28')
            if type(G_28) == np.ndarray:
                print('malloc', sys.getsizeof(G_28), 'G_28', G_28.shape)
            elif type(G_28) == list:
                dims = []
                tmp = G_28
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(G_28), 'G_28', dims)
            elif type(G_28) == tuple:
                print('malloc', sys.getsizeof(G_28), 'G_28', [len(G_28)])
            else:
                print('malloc', sys.getsizeof(G_28), 'G_28')
            print(202, 286)
            start_28 = instrument_read(time, 'time').time()
            write_instrument_read(start_28, 'start_28')
            if type(start_28) == np.ndarray:
                print('malloc', sys.getsizeof(start_28), 'start_28',
                    start_28.shape)
            elif type(start_28) == list:
                dims = []
                tmp = start_28
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(start_28), 'start_28', dims)
            elif type(start_28) == tuple:
                print('malloc', sys.getsizeof(start_28), 'start_28', [len(
                    start_28)])
            else:
                print('malloc', sys.getsizeof(start_28), 'start_28')
            myFn(instrument_read_sub(instrument_read(G_28, 'G_28').vertices,
                'G_28.vertices', 0, None, None, False), instrument_read(
                G_28, 'G_28'))
            print(202, 288)
            end_28 = instrument_read(time, 'time').time()
            write_instrument_read(end_28, 'end_28')
            if type(end_28) == np.ndarray:
                print('malloc', sys.getsizeof(end_28), 'end_28', end_28.shape)
            elif type(end_28) == list:
                dims = []
                tmp = end_28
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(end_28), 'end_28', dims)
            elif type(end_28) == tuple:
                print('malloc', sys.getsizeof(end_28), 'end_28', [len(end_28)])
            else:
                print('malloc', sys.getsizeof(end_28), 'end_28')
            print(202, 289)
            runtime_28 += (instrument_read(end_28, 'end_28') -
                instrument_read(start_28, 'start_28')) * 1000
            write_instrument_read(runtime_28, 'runtime_28')
        print(203, 290)
        runtime_28 = instrument_read(runtime_28, 'runtime_28'
            ) / instrument_read(numTrials_28, 'numTrials_28')
        write_instrument_read(runtime_28, 'runtime_28')
        if type(runtime_28) == np.ndarray:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28',
                runtime_28.shape)
        elif type(runtime_28) == list:
            dims = []
            tmp = runtime_28
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28', dims)
        elif type(runtime_28) == tuple:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28', [len(
                runtime_28)])
        else:
            print('malloc', sys.getsizeof(runtime_28), 'runtime_28')
        instrument_read(nValues_28, 'nValues_28').append(instrument_read(
            n_28, 'n_28'))
        instrument_read(tValues_28, 'tValues_28').append(instrument_read(
            runtime_28, 'runtime_28'))
    print('exit scope 28')
    return instrument_read(nValues_28, 'nValues_28'), instrument_read(
        tValues_28, 'tValues_28')
    print('exit scope 28')


def smallFrac(n):
    print('enter scope 29')
    print(1, 296)
    print(207, 297)
    n_29 = instrument_read(n, 'n')
    write_instrument_read(n_29, 'n_29')
    if type(n_29) == np.ndarray:
        print('malloc', sys.getsizeof(n_29), 'n_29', n_29.shape)
    elif type(n_29) == list:
        dims = []
        tmp = n_29
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(n_29), 'n_29', dims)
    elif type(n_29) == tuple:
        print('malloc', sys.getsizeof(n_29), 'n_29', [len(n_29)])
    else:
        print('malloc', sys.getsizeof(n_29), 'n_29')
    print('exit scope 29')
    return float(5 / instrument_read(n_29, 'n_29'))
    print('exit scope 29')


def read_random_graph_from_file(n, p, wts=[1]):
    print('enter scope 30')
    print(1, 301)
    print(211, 302)
    n_30 = n
    print(211, 303)
    p_30 = p
    print(211, 304)
    wts_30 = wts
    print('exit scope 30')
    return randomGraph(n_30, p_30, wts_30)
    print('exit scope 30')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    print(214, 310)
    G_NVM = read_random_graph_from_file(5, 0.2)
    BFS_shortestPaths(instrument_read_sub(instrument_read(G_NVM, 'G_NVM').
        vertices, 'G_NVM.vertices', 0, None, None, False), instrument_read(
        G_NVM, 'G_NVM'))
    dijkstraDumb_shortestPaths(instrument_read_sub(instrument_read(G_NVM,
        'G_NVM').vertices, 'G_NVM.vertices', 0, None, None, False),
        instrument_read(G_NVM, 'G_NVM'))
    print(214, 313)
    G_NVM = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    dijkstra_shortestPaths(instrument_read_sub(instrument_read(G_NVM,
        'G_NVM').vertices, 'G_NVM.vertices', 0, None, None, False),
        instrument_read(G_NVM, 'G_NVM'))
    print(214, 315)
    nValues_0 = [10]
    write_instrument_read(nValues_0, 'nValues_0')
    if type(nValues_0) == np.ndarray:
        print('malloc', sys.getsizeof(nValues_0), 'nValues_0', nValues_0.shape)
    elif type(nValues_0) == list:
        dims = []
        tmp = nValues_0
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(nValues_0), 'nValues_0', dims)
    elif type(nValues_0) == tuple:
        print('malloc', sys.getsizeof(nValues_0), 'nValues_0', [len(nValues_0)]
            )
    else:
        print('malloc', sys.getsizeof(nValues_0), 'nValues_0')
    print(214, 316)
    nDijkstra_0, tDijkstra_0 = runTrials(instrument_read(BFS, 'BFS'),
        instrument_read(nValues_0, 'nValues_0'), instrument_read(smallFrac,
        'smallFrac'))
    write_instrument_read(tDijkstra_0, 'tDijkstra_0')
    if type(tDijkstra_0) == np.ndarray:
        print('malloc', sys.getsizeof(tDijkstra_0), 'tDijkstra_0',
            tDijkstra_0.shape)
    elif type(tDijkstra_0) == list:
        dims = []
        tmp = tDijkstra_0
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(tDijkstra_0), 'tDijkstra_0', dims)
    elif type(tDijkstra_0) == tuple:
        print('malloc', sys.getsizeof(tDijkstra_0), 'tDijkstra_0', [len(
            tDijkstra_0)])
    else:
        print('malloc', sys.getsizeof(tDijkstra_0), 'tDijkstra_0')
