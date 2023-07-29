import sys
from instrument_lib import *
import sys
from instrument_lib import *
import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop


class CS161Vertex:

    def __init__(self, v):
        print('enter scope 1')
        print(1, 10)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_1 = instrument_read(v, 'v')
        write_instrument_read(v_1, 'v_1')
        print('malloc', sys.getsizeof(v_1), 'v_1')
        instrument_read(self, 'self').inNeighbors = []
        instrument_read(self, 'self').outNeighbors = []
        instrument_read(self, 'self').value = instrument_read(v, 'v')
        instrument_read(self, 'self').inTime = None
        instrument_read(self, 'self').outTime = None
        instrument_read(self, 'self').status = 'unvisited'
        instrument_read(self, 'self').parent = None
        instrument_read(self, 'self').estD = instrument_read(math, 'math').inf
        print('exit scope 1')

    def hasOutNeighbor(self, v):
        print('enter scope 2')
        print(1, 21)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_2 = instrument_read(v, 'v')
        write_instrument_read(v_2, 'v_2')
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
        print(1, 26)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_3 = instrument_read(v, 'v')
        write_instrument_read(v_3, 'v_3')
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
        print(1, 31)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_4 = instrument_read(v, 'v')
        write_instrument_read(v_4, 'v_4')
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
        print(1, 36)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 5')
        return [instrument_read_sub(instrument_read(v_5, 'v_5'), 'v_5', 0,
            None, None, False) for v_5 in instrument_read(self, 'self').
            outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 39)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 6')
        return [instrument_read_sub(instrument_read(v_6, 'v_6'), 'v_6', 0,
            None, None, False) for v_6 in instrument_read(self, 'self').
            inNeighbors]
        print('exit scope 6')

    def getOutNeighborsWithWeights(self):
        print('enter scope 7')
        print(1, 42)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 7')
        return instrument_read(self, 'self').outNeighbors
        print('exit scope 7')

    def getInNeighborsWithWeights(self):
        print('enter scope 8')
        print(1, 45)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 8')
        return instrument_read(self, 'self').inNeighbors
        print('exit scope 8')

    def addOutNeighbor(self, v, wt):
        print('enter scope 9')
        print(1, 48)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_9 = instrument_read(v, 'v')
        write_instrument_read(v_9, 'v_9')
        print('malloc', sys.getsizeof(v_9), 'v_9')
        wt_9 = instrument_read(wt, 'wt')
        write_instrument_read(wt_9, 'wt_9')
        print('malloc', sys.getsizeof(wt_9), 'wt_9')
        instrument_read(self, 'self').outNeighbors.append((instrument_read(
            v_9, 'v_9'), instrument_read(wt_9, 'wt_9')))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 51)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v_10 = instrument_read(v, 'v')
        write_instrument_read(v_10, 'v_10')
        print('malloc', sys.getsizeof(v_10), 'v_10')
        wt_10 = instrument_read(wt, 'wt')
        write_instrument_read(wt_10, 'wt_10')
        print('malloc', sys.getsizeof(wt_10), 'wt_10')
        instrument_read(self, 'self').inNeighbors.append((instrument_read(
            v_10, 'v_10'), instrument_read(wt_10, 'wt_10')))
        print('exit scope 10')


class CS161Graph:

    def __init__(self):
        print('enter scope 11')
        print(1, 57)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        instrument_read(self, 'self').vertices = []
        print('exit scope 11')

    def addVertex(self, n):
        print('enter scope 12')
        print(1, 60)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        n_12 = instrument_read(n, 'n')
        write_instrument_read(n_12, 'n_12')
        print('malloc', sys.getsizeof(n_12), 'n_12')
        instrument_read(self, 'self').vertices.append(instrument_read(n_12,
            'n_12'))
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 64)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u_13 = instrument_read(u, 'u')
        write_instrument_read(u_13, 'u_13')
        print('malloc', sys.getsizeof(u_13), 'u_13')
        v_13 = instrument_read(v, 'v')
        write_instrument_read(v_13, 'v_13')
        print('malloc', sys.getsizeof(v_13), 'v_13')
        wt_13 = instrument_read(wt, 'wt')
        write_instrument_read(wt_13, 'wt_13')
        print('malloc', sys.getsizeof(wt_13), 'wt_13')
        instrument_read(u_13, 'u_13').addOutNeighbor(instrument_read(v_13,
            'v_13'), wt=wt_13)
        instrument_read(v_13, 'v_13').addInNeighbor(instrument_read(u_13,
            'u_13'), wt=wt_13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 69)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u_14 = instrument_read(u, 'u')
        write_instrument_read(u_14, 'u_14')
        print('malloc', sys.getsizeof(u_14), 'u_14')
        v_14 = instrument_read(v, 'v')
        write_instrument_read(v_14, 'v_14')
        print('malloc', sys.getsizeof(v_14), 'v_14')
        wt_14 = instrument_read(wt, 'wt')
        write_instrument_read(wt_14, 'wt_14')
        print('malloc', sys.getsizeof(wt_14), 'wt_14')
        instrument_read(self, 'self').addDiEdge(instrument_read(u_14,
            'u_14'), instrument_read(v_14, 'v_14'), wt=wt_14)
        instrument_read(self, 'self').addDiEdge(instrument_read(v_14,
            'v_14'), instrument_read(u_14, 'u_14'), wt=wt_14)
        print('exit scope 14')

    def getDirEdges(self):
        print('enter scope 15')
        print(1, 75)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        ret = []
        write_instrument_read(ret, 'ret')
        print('malloc', sys.getsizeof(ret), 'ret')
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
        print(1, 83)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print(70, 84)
        instrument_read(self, 'self').vertices = []
        print('exit scope 16')

    def addVertex(self, n):
        print('enter scope 17')
        print(1, 86)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        n_17 = instrument_read(n, 'n')
        write_instrument_read(n_17, 'n_17')
        print('malloc', sys.getsizeof(n_17), 'n_17')
        instrument_read(self, 'self').vertices.append(instrument_read(n_17,
            'n_17'))
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 90)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u_18 = instrument_read(u, 'u')
        write_instrument_read(u_18, 'u_18')
        print('malloc', sys.getsizeof(u_18), 'u_18')
        v_18 = instrument_read(v, 'v')
        write_instrument_read(v_18, 'v_18')
        print('malloc', sys.getsizeof(v_18), 'v_18')
        wt_18 = instrument_read(wt, 'wt')
        write_instrument_read(wt_18, 'wt_18')
        print('malloc', sys.getsizeof(wt_18), 'wt_18')
        instrument_read(u_18, 'u_18').addOutNeighbor(instrument_read(v_18,
            'v_18'), wt=wt_18)
        instrument_read(v_18, 'v_18').addInNeighbor(instrument_read(u_18,
            'u_18'), wt=wt_18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 95)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u_19 = instrument_read(u, 'u')
        write_instrument_read(u_19, 'u_19')
        print('malloc', sys.getsizeof(u_19), 'u_19')
        v_19 = instrument_read(v, 'v')
        write_instrument_read(v_19, 'v_19')
        print('malloc', sys.getsizeof(v_19), 'v_19')
        wt_19 = instrument_read(wt, 'wt')
        write_instrument_read(wt_19, 'wt_19')
        print('malloc', sys.getsizeof(wt_19), 'wt_19')
        instrument_read(self, 'self').addDiEdge(instrument_read(u_19,
            'u_19'), instrument_read(v_19, 'v_19'), wt=wt_19)
        instrument_read(self, 'self').addDiEdge(instrument_read(v_19,
            'v_19'), instrument_read(u_19, 'u_19'), wt=wt_19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 101)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print(82, 102)
        ret_20 = []
        write_instrument_read(ret_20, 'ret_20')
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
    print(1, 112)
    n_21 = instrument_read(n, 'n')
    write_instrument_read(n_21, 'n_21')
    print('malloc', sys.getsizeof(n_21), 'n_21')
    p_21 = instrument_read(p, 'p')
    write_instrument_read(p_21, 'p_21')
    print('malloc', sys.getsizeof(p_21), 'p_21')
    wts_21 = instrument_read(wts, 'wts')
    write_instrument_read(wts_21, 'wts_21')
    print('malloc', sys.getsizeof(wts_21), 'wts_21')
    print(91, 113)
    G_21 = CS161Graph()
    write_instrument_read(G_21, 'G_21')
    print('malloc', sys.getsizeof(G_21), 'G_21')
    print(91, 114)
    V_21 = [CS161Vertex(instrument_read(x_21, 'x_21')) for x_21 in range(
        instrument_read(n_21, 'n_21'))]
    write_instrument_read(V_21, 'V_21')
    print('malloc', sys.getsizeof(V_21), 'V_21')
    for v_21 in instrument_read(V_21, 'V_21'):
        instrument_read(G_21, 'G_21').addVertex(instrument_read(v_21, 'v_21'))
    for v_21 in instrument_read(V_21, 'V_21'):
        print(95, 118)
        i_21 = 0
        write_instrument_read(i_21, 'i_21')
        print('malloc', sys.getsizeof(i_21), 'i_21')
        for w_21 in instrument_read(V_21, 'V_21'):
            if instrument_read(v_21, 'v_21') != instrument_read(w_21, 'w_21'):
                if random() < instrument_read(p_21, 'p_21'):
                    instrument_read(G_21, 'G_21').addDiEdge(instrument_read
                        (v_21, 'v_21'), instrument_read(w_21, 'w_21'), wt=
                        choice(wts_21))
                    print(102, 123)
                    i_21 += 1
                    write_instrument_read(i_21, 'i_21')
            if instrument_read(i_21, 'i_21') > 15:
                break
    print('exit scope 21')
    return instrument_read(G_21, 'G_21')
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 127)
    w_22 = instrument_read(w, 'w')
    write_instrument_read(w_22, 'w_22')
    print('malloc', sys.getsizeof(w_22), 'w_22')
    G_22 = instrument_read(G, 'G')
    write_instrument_read(G_22, 'G_22')
    print('malloc', sys.getsizeof(G_22), 'G_22')
    for v_22 in instrument_read(G_22, 'G_22').vertices:
        print(110, 129)
        instrument_read(v_22, 'v_22').status = 'unvisited'
    print(111, 130)
    n_22 = len(instrument_read(G_22, 'G_22').vertices)
    write_instrument_read(n_22, 'n_22')
    print('malloc', sys.getsizeof(n_22), 'n_22')
    print(111, 131)
    Ls_22 = [[] for i_22 in range(instrument_read(n_22, 'n_22'))]
    write_instrument_read(Ls_22, 'Ls_22')
    print('malloc', sys.getsizeof(Ls_22), 'Ls_22')
    print(111, 132)
    Ls_22[0] = [instrument_read(w_22, 'w_22')]
    write_instrument_read_sub(Ls_22, 'Ls_22', 0, None, None, False)
    print(111, 133)
    instrument_read(w_22, 'w_22').status = 'visited'
    for i_22 in range(instrument_read(n_22, 'n_22')):
        for u_22 in instrument_read_sub(instrument_read(Ls_22, 'Ls_22'),
            'Ls_22', instrument_read(i_22, 'i_22'), None, None, False):
            for v_22 in instrument_read(u_22, 'u_22').getOutNeighbors():
                if instrument_read(v_22, 'v_22').status == 'unvisited':
                    print(119, 138)
                    instrument_read(v_22, 'v_22').status = 'visited'
                    print(119, 139)
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
    print(1, 143)
    w_23 = instrument_read(w, 'w')
    write_instrument_read(w_23, 'w_23')
    print('malloc', sys.getsizeof(w_23), 'w_23')
    G_23 = instrument_read(G, 'G')
    write_instrument_read(G_23, 'G_23')
    print('malloc', sys.getsizeof(G_23), 'G_23')
    print(124, 144)
    Ls_23 = BFS(instrument_read(w_23, 'w_23'), instrument_read(G_23, 'G_23'))
    write_instrument_read(Ls_23, 'Ls_23')
    print('malloc', sys.getsizeof(Ls_23), 'Ls_23')
    for i_23 in range(len(instrument_read(Ls_23, 'Ls_23'))):
        for w_23 in instrument_read_sub(instrument_read(Ls_23, 'Ls_23'),
            'Ls_23', instrument_read(i_23, 'i_23'), None, None, False):
            print(128, 148)
            path_23 = []
            write_instrument_read(path_23, 'path_23')
            print('malloc', sys.getsizeof(path_23), 'path_23')
            print(128, 149)
            current_23 = instrument_read(w_23, 'w_23')
            write_instrument_read(current_23, 'current_23')
            print('malloc', sys.getsizeof(current_23), 'current_23')
            for j_23 in range(instrument_read(i_23, 'i_23')):
                instrument_read(path_23, 'path_23').append(instrument_read(
                    current_23, 'current_23'))
                print(131, 152)
                current_23 = instrument_read(current_23, 'current_23').parent
                write_instrument_read(current_23, 'current_23')
                print('malloc', sys.getsizeof(current_23), 'current_23')
            instrument_read(path_23, 'path_23').append(instrument_read(
                current_23, 'current_23'))
            instrument_read(path_23, 'path_23').reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 157)
    w_24 = instrument_read(w, 'w')
    write_instrument_read(w_24, 'w_24')
    print('malloc', sys.getsizeof(w_24), 'w_24')
    G_24 = instrument_read(G, 'G')
    write_instrument_read(G_24, 'G_24')
    print('malloc', sys.getsizeof(G_24), 'G_24')
    for v_24 in instrument_read(G_24, 'G_24').vertices:
        print(136, 159)
        instrument_read(v_24, 'v_24').estD = instrument_read(math, 'math').inf
    print(137, 160)
    instrument_read(w_24, 'w_24').estD = 0
    print(137, 161)
    unsureVertices_24 = instrument_read_sub(instrument_read(G_24, 'G_24').
        vertices, 'G_24.vertices', None, None, None, True)
    write_instrument_read(unsureVertices_24, 'unsureVertices_24')
    print('malloc', sys.getsizeof(unsureVertices_24), 'unsureVertices_24')
    while len(instrument_read(unsureVertices_24, 'unsureVertices_24')) > 0:
        print(139, 164)
        u_24 = None
        write_instrument_read(u_24, 'u_24')
        print('malloc', sys.getsizeof(u_24), 'u_24')
        print(139, 165)
        minD_24 = instrument_read(math, 'math').inf
        write_instrument_read(minD_24, 'minD_24')
        print('malloc', sys.getsizeof(minD_24), 'minD_24')
        for x_24 in instrument_read(unsureVertices_24, 'unsureVertices_24'):
            if instrument_read(x_24, 'x_24').estD < instrument_read(minD_24,
                'minD_24'):
                print(144, 168)
                minD_24 = instrument_read(x_24, 'x_24').estD
                write_instrument_read(minD_24, 'minD_24')
                print('malloc', sys.getsizeof(minD_24), 'minD_24')
                print(144, 169)
                u_24 = instrument_read(x_24, 'x_24')
                write_instrument_read(u_24, 'u_24')
                print('malloc', sys.getsizeof(u_24), 'u_24')
        if instrument_read(u_24, 'u_24') == None:
            print('exit scope 24')
            return
        for v_24, wt_24 in instrument_read(u_24, 'u_24'
            ).getOutNeighborsWithWeights():
            if instrument_read(u_24, 'u_24').estD + instrument_read(wt_24,
                'wt_24') < instrument_read(v_24, 'v_24').estD:
                print(151, 176)
                instrument_read(v_24, 'v_24').estD = instrument_read(u_24,
                    'u_24').estD + instrument_read(wt_24, 'wt_24')
                print(151, 177)
                instrument_read(v_24, 'v_24').parent = instrument_read(u_24,
                    'u_24')
        instrument_read(unsureVertices_24, 'unsureVertices_24').remove(
            instrument_read(u_24, 'u_24'))
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 181)
    w_25 = instrument_read(w, 'w')
    write_instrument_read(w_25, 'w_25')
    print('malloc', sys.getsizeof(w_25), 'w_25')
    G_25 = instrument_read(G, 'G')
    write_instrument_read(G_25, 'G_25')
    print('malloc', sys.getsizeof(G_25), 'G_25')
    dijkstraDumb(instrument_read(w_25, 'w_25'), instrument_read(G_25, 'G_25'))
    for v_25 in instrument_read(G_25, 'G_25').vertices:
        if instrument_read(v_25, 'v_25').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(160, 187)
        path_25 = []
        write_instrument_read(path_25, 'path_25')
        print('malloc', sys.getsizeof(path_25), 'path_25')
        print(160, 188)
        current_25 = instrument_read(v_25, 'v_25')
        write_instrument_read(current_25, 'current_25')
        print('malloc', sys.getsizeof(current_25), 'current_25')
        while instrument_read(current_25, 'current_25') != instrument_read(w_25
            , 'w_25'):
            instrument_read(path_25, 'path_25').append(instrument_read(
                current_25, 'current_25'))
            print(162, 191)
            current_25 = instrument_read(current_25, 'current_25').parent
            write_instrument_read(current_25, 'current_25')
            print('malloc', sys.getsizeof(current_25), 'current_25')
        instrument_read(path_25, 'path_25').append(instrument_read(
            current_25, 'current_25'))
        instrument_read(path_25, 'path_25').reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 196)
    w_26 = instrument_read(w, 'w')
    write_instrument_read(w_26, 'w_26')
    print('malloc', sys.getsizeof(w_26), 'w_26')
    G_26 = instrument_read(G, 'G')
    write_instrument_read(G_26, 'G_26')
    print('malloc', sys.getsizeof(G_26), 'G_26')
    for v_26 in instrument_read(G_26, 'G_26').vertices:
        print(167, 198)
        instrument_read(v_26, 'v_26').estD = instrument_read(math, 'math').inf
    print(168, 199)
    instrument_read(w_26, 'w_26').estD = 0
    print(168, 200)
    unsureVertices_26 = instrument_read(heapdict, 'heapdict').heapdict()
    write_instrument_read(unsureVertices_26, 'unsureVertices_26')
    print('malloc', sys.getsizeof(unsureVertices_26), 'unsureVertices_26')
    for v_26 in instrument_read(G_26, 'G_26').vertices:
        print(170, 202)
        unsureVertices_26[instrument_read(instrument_read(v_26, 'v_26'),
            'v_26')] = instrument_read(v_26, 'v_26').estD
        write_instrument_read_sub(unsureVertices_26, 'unsureVertices_26',
            instrument_read(instrument_read(v_26, 'v_26'), 'v_26'), None,
            None, False)
    while len(instrument_read(unsureVertices_26, 'unsureVertices_26')) > 0:
        print(172, 205)
        u_26, dist_26 = instrument_read(unsureVertices_26, 'unsureVertices_26'
            ).popitem()
        write_instrument_read(dist_26, 'dist_26')
        print('malloc', sys.getsizeof(dist_26), 'dist_26')
        if instrument_read(u_26, 'u_26').estD == instrument_read(math, 'math'
            ).inf:
            print('exit scope 26')
            return
        for v_26, wt_26 in instrument_read(u_26, 'u_26'
            ).getOutNeighborsWithWeights():
            if instrument_read(u_26, 'u_26').estD + instrument_read(wt_26,
                'wt_26') < instrument_read(v_26, 'v_26').estD:
                print(179, 212)
                instrument_read(v_26, 'v_26').estD = instrument_read(u_26,
                    'u_26').estD + instrument_read(wt_26, 'wt_26')
                print(179, 213)
                unsureVertices_26[instrument_read(instrument_read(v_26,
                    'v_26'), 'v_26')] = instrument_read(u_26, 'u_26'
                    ).estD + instrument_read(wt_26, 'wt_26')
                write_instrument_read_sub(unsureVertices_26,
                    'unsureVertices_26', instrument_read(instrument_read(
                    v_26, 'v_26'), 'v_26'), None, None, False)
                print(179, 214)
                instrument_read(v_26, 'v_26').parent = instrument_read(u_26,
                    'u_26')
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 217)
    w_27 = instrument_read(w, 'w')
    write_instrument_read(w_27, 'w_27')
    print('malloc', sys.getsizeof(w_27), 'w_27')
    G_27 = instrument_read(G, 'G')
    write_instrument_read(G_27, 'G_27')
    print('malloc', sys.getsizeof(G_27), 'G_27')
    dijkstra(instrument_read(w_27, 'w_27'), instrument_read(G_27, 'G_27'))
    for v_27 in instrument_read(G_27, 'G_27').vertices:
        if instrument_read(v_27, 'v_27').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(188, 223)
        path_27 = []
        write_instrument_read(path_27, 'path_27')
        print('malloc', sys.getsizeof(path_27), 'path_27')
        print(188, 224)
        current_27 = instrument_read(v_27, 'v_27')
        write_instrument_read(current_27, 'current_27')
        print('malloc', sys.getsizeof(current_27), 'current_27')
        while instrument_read(current_27, 'current_27') != instrument_read(w_27
            , 'w_27'):
            instrument_read(path_27, 'path_27').append(instrument_read(
                current_27, 'current_27'))
            print(190, 227)
            current_27 = instrument_read(current_27, 'current_27').parent
            write_instrument_read(current_27, 'current_27')
            print('malloc', sys.getsizeof(current_27), 'current_27')
        instrument_read(path_27, 'path_27').append(instrument_read(
            current_27, 'current_27'))
        instrument_read(path_27, 'path_27').reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=1):
    print('enter scope 28')
    print(1, 232)
    myFn_28 = instrument_read(myFn, 'myFn')
    write_instrument_read(myFn_28, 'myFn_28')
    print('malloc', sys.getsizeof(myFn_28), 'myFn_28')
    nVals_28 = instrument_read(nVals, 'nVals')
    write_instrument_read(nVals_28, 'nVals_28')
    print('malloc', sys.getsizeof(nVals_28), 'nVals_28')
    pFn_28 = instrument_read(pFn, 'pFn')
    write_instrument_read(pFn_28, 'pFn_28')
    print('malloc', sys.getsizeof(pFn_28), 'pFn_28')
    numTrials_28 = instrument_read(numTrials, 'numTrials')
    write_instrument_read(numTrials_28, 'numTrials_28')
    print('malloc', sys.getsizeof(numTrials_28), 'numTrials_28')
    print(194, 233)
    nValues_28 = []
    write_instrument_read(nValues_28, 'nValues_28')
    print('malloc', sys.getsizeof(nValues_28), 'nValues_28')
    print(194, 234)
    tValues_28 = []
    write_instrument_read(tValues_28, 'tValues_28')
    print('malloc', sys.getsizeof(tValues_28), 'tValues_28')
    for n_28 in instrument_read(nVals_28, 'nVals_28'):
        print(196, 237)
        runtime_28 = 0
        write_instrument_read(runtime_28, 'runtime_28')
        print('malloc', sys.getsizeof(runtime_28), 'runtime_28')
        for t_28 in range(instrument_read(numTrials_28, 'numTrials_28')):
            print(199, 239)
            G_28 = randomGraph(instrument_read(n_28, 'n_28') * 30000, pFn(
                instrument_read(n_28, 'n_28')))
            write_instrument_read(G_28, 'G_28')
            print('malloc', sys.getsizeof(G_28), 'G_28')
            print(199, 240)
            start_28 = instrument_read(time, 'time').time()
            write_instrument_read(start_28, 'start_28')
            print('malloc', sys.getsizeof(start_28), 'start_28')
            myFn(instrument_read_sub(instrument_read(G_28, 'G_28').vertices,
                'G_28.vertices', 0, None, None, False), instrument_read(
                G_28, 'G_28'))
            print(199, 242)
            end_28 = instrument_read(time, 'time').time()
            write_instrument_read(end_28, 'end_28')
            print('malloc', sys.getsizeof(end_28), 'end_28')
            print(199, 243)
            runtime_28 += (instrument_read(end_28, 'end_28') -
                instrument_read(start_28, 'start_28')) * 1000
            write_instrument_read(runtime_28, 'runtime_28')
        print(200, 244)
        runtime_28 = instrument_read(runtime_28, 'runtime_28'
            ) / instrument_read(numTrials_28, 'numTrials_28')
        write_instrument_read(runtime_28, 'runtime_28')
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
    print(1, 249)
    n_29 = instrument_read(n, 'n')
    write_instrument_read(n_29, 'n_29')
    print('malloc', sys.getsizeof(n_29), 'n_29')
    print('exit scope 29')
    return float(5 / instrument_read(n_29, 'n_29'))
    print('exit scope 29')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    print(207, 255)
    G_0 = randomGraph(5, 0.2)
    write_instrument_read(G_0, 'G_0')
    print('malloc', sys.getsizeof(G_0), 'G_0')
    BFS_shortestPaths(instrument_read_sub(instrument_read(G_0, 'G_0').
        vertices, 'G_0.vertices', 0, None, None, False), instrument_read(
        G_0, 'G_0'))
    dijkstraDumb_shortestPaths(instrument_read_sub(instrument_read(G_0,
        'G_0').vertices, 'G_0.vertices', 0, None, None, False),
        instrument_read(G_0, 'G_0'))
    print(207, 258)
    G_0 = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    write_instrument_read(G_0, 'G_0')
    print('malloc', sys.getsizeof(G_0), 'G_0')
    dijkstra_shortestPaths(instrument_read_sub(instrument_read(G_0, 'G_0').
        vertices, 'G_0.vertices', 0, None, None, False), instrument_read(
        G_0, 'G_0'))
    print(207, 260)
    nValues_0 = [10]
    write_instrument_read(nValues_0, 'nValues_0')
    print('malloc', sys.getsizeof(nValues_0), 'nValues_0')
    print(207, 261)
    nDijkstra_0, tDijkstra_0 = runTrials(instrument_read(BFS, 'BFS'),
        instrument_read(nValues_0, 'nValues_0'), instrument_read(smallFrac,
        'smallFrac'))
    write_instrument_read(tDijkstra_0, 'tDijkstra_0')
    print('malloc', sys.getsizeof(tDijkstra_0), 'tDijkstra_0')
