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
        self = self
        v_1 = v
        self.inNeighbors = []
        self.outNeighbors = []
        self.value = v
        self.inTime = None
        self.outTime = None
        self.status = 'unvisited'
        self.parent = None
        self.estD = math.inf
        print('exit scope 1')

    def hasOutNeighbor(self, v):
        print('enter scope 2')
        print(1, 21)
        self = self
        v_2 = v
        if v_2 in self.getOutNeighbors():
            print('exit scope 2')
            return True
        print('exit scope 2')
        return False
        print('exit scope 2')

    def hasInNeighbor(self, v):
        print('enter scope 3')
        print(1, 26)
        self = self
        v_3 = v
        if v_3 in self.getInNeighbors():
            print('exit scope 3')
            return True
        print('exit scope 3')
        return False
        print('exit scope 3')

    def hasNeighbor(self, v):
        print('enter scope 4')
        print(1, 31)
        self = self
        v_4 = v
        if v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors():
            print('exit scope 4')
            return True
        print('exit scope 4')
        return False
        print('exit scope 4')

    def getOutNeighbors(self):
        print('enter scope 5')
        print(1, 36)
        self = self
        print('exit scope 5')
        return [v_5[0] for v_5 in self.outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 39)
        self = self
        print('exit scope 6')
        return [v_6[0] for v_6 in self.inNeighbors]
        print('exit scope 6')

    def getOutNeighborsWithWeights(self):
        print('enter scope 7')
        print(1, 42)
        self = self
        print('exit scope 7')
        return self.outNeighbors
        print('exit scope 7')

    def getInNeighborsWithWeights(self):
        print('enter scope 8')
        print(1, 45)
        self = self
        print('exit scope 8')
        return self.inNeighbors
        print('exit scope 8')

    def addOutNeighbor(self, v, wt):
        print('enter scope 9')
        print(1, 48)
        self = self
        v_9 = v
        wt_9 = wt
        self.outNeighbors.append((v_9, wt_9))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 51)
        self = self
        v_10 = v
        wt_10 = wt
        self.inNeighbors.append((v_10, wt_10))
        print('exit scope 10')


class CS161Graph:

    def __init__(self):
        print('enter scope 11')
        print(1, 57)
        self = self
        self.vertices = []
        print('exit scope 11')

    def addVertex(self, n):
        print('enter scope 12')
        print(1, 60)
        self = self
        n_12 = n
        self.vertices.append(n_12)
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 64)
        self = self
        u_13 = u
        v_13 = v
        wt_13 = wt
        u_13.addOutNeighbor(v_13, wt=wt_13)
        v_13.addInNeighbor(u_13, wt=wt_13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 69)
        self = self
        u_14 = u
        v_14 = v
        wt_14 = wt
        self.addDiEdge(u_14, v_14, wt=wt_14)
        self.addDiEdge(v_14, u_14, wt=wt_14)
        print('exit scope 14')

    def getDirEdges(self):
        print('enter scope 15')
        print(1, 75)
        self = self
        ret = []
        for v_15 in self.vertices:
            for u_15, wt_15 in v_15.getOutNeighborsWithWeights():
                ret_15.append([v_15, u_15, wt_15])
        print('exit scope 15')
        return ret_15
        print('exit scope 15')


class CS161Graph:

    def __init__(self):
        print('enter scope 16')
        print(1, 83)
        self = self
        print(70, 84)
        self.vertices = []
        print('exit scope 16')

    def addVertex(self, n):
        print('enter scope 17')
        print(1, 86)
        self = self
        n_17 = n
        self.vertices.append(n_17)
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 90)
        self = self
        u_18 = u
        v_18 = v
        wt_18 = wt
        u_18.addOutNeighbor(v_18, wt=wt_18)
        v_18.addInNeighbor(u_18, wt=wt_18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 95)
        self = self
        u_19 = u
        v_19 = v
        wt_19 = wt
        self.addDiEdge(u_19, v_19, wt=wt_19)
        self.addDiEdge(v_19, u_19, wt=wt_19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 101)
        self = self
        print(82, 102)
        ret_20 = []
        for v_20 in self.vertices:
            for u_20, wt_20 in v_20.getOutNeighborsWithWeights():
                ret_20.append([v_20, u_20, wt_20])
        print('exit scope 20')
        return ret_20
        print('exit scope 20')


def randomGraph(n, p, wts=[1]):
    print('enter scope 21')
    print(1, 112)
    n_21 = n
    p_21 = p
    wts_21 = wts
    print(91, 113)
    G_21 = CS161Graph()
    print(91, 114)
    V_21 = [CS161Vertex(x_21) for x_21 in range(n_21)]
    for v_21 in V_21:
        G_21.addVertex(v_21)
    for v_21 in V_21:
        print(95, 118)
        i_21 = 0
        for w_21 in V_21:
            if v_21 != w_21:
                if random() < p_21:
                    G_21.addDiEdge(v_21, w_21, wt=choice(wts_21))
                    print(102, 123)
                    i_21 += 1
            if i_21 > 15:
                break
    print('exit scope 21')
    return G_21
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 127)
    w_22 = w
    G_22 = G
    for v_22 in G_22.vertices:
        print(110, 129)
        v_22.status = 'unvisited'
    print(111, 130)
    n_22 = len(G_22.vertices)
    print(111, 131)
    Ls_22 = [[] for i_22 in range(n_22)]
    print(111, 132)
    Ls_22[0] = [w_22]
    print(111, 133)
    w_22.status = 'visited'
    for i_22 in range(n_22):
        for u_22 in Ls_22[i_22]:
            for v_22 in u_22.getOutNeighbors():
                if v_22.status == 'unvisited':
                    print(119, 138)
                    v_22.status = 'visited'
                    print(119, 139)
                    v_22.parent = u_22
                    Ls_22[i_22 + 1].append(v_22)
    print('exit scope 22')
    return Ls_22
    print('exit scope 22')


def BFS_shortestPaths(w, G):
    print('enter scope 23')
    print(1, 143)
    w_23 = w
    G_23 = G
    print(124, 144)
    Ls_23 = BFS(w_23, G_23)
    for i_23 in range(len(Ls_23)):
        for w_23 in Ls_23[i_23]:
            print(128, 148)
            path_23 = []
            print(128, 149)
            current_23 = w_23
            for j_23 in range(i_23):
                path_23.append(current_23)
                print(131, 152)
                current_23 = current_23.parent
            path_23.append(current_23)
            path_23.reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 157)
    w_24 = w
    G_24 = G
    for v_24 in G_24.vertices:
        print(136, 159)
        v_24.estD = math.inf
    print(137, 160)
    w_24.estD = 0
    print(137, 161)
    unsureVertices_24 = G_24.vertices[:]
    while len(unsureVertices_24) > 0:
        print(139, 164)
        u_24 = None
        print(139, 165)
        minD_24 = math.inf
        for x_24 in unsureVertices_24:
            if x_24.estD < minD_24:
                print(144, 168)
                minD_24 = x_24.estD
                print(144, 169)
                u_24 = x_24
        if u_24 == None:
            print('exit scope 24')
            return
        for v_24, wt_24 in u_24.getOutNeighborsWithWeights():
            if u_24.estD + wt_24 < v_24.estD:
                print(151, 176)
                v_24.estD = u_24.estD + wt_24
                print(151, 177)
                v_24.parent = u_24
        unsureVertices_24.remove(u_24)
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 181)
    w_25 = w
    G_25 = G
    dijkstraDumb(w_25, G_25)
    for v_25 in G_25.vertices:
        if v_25.estD == math.inf:
            continue
        print(160, 187)
        path_25 = []
        print(160, 188)
        current_25 = v_25
        while current_25 != w_25:
            path_25.append(current_25)
            print(162, 191)
            current_25 = current_25.parent
        path_25.append(current_25)
        path_25.reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 196)
    w_26 = w
    G_26 = G
    for v_26 in G_26.vertices:
        print(167, 198)
        v_26.estD = math.inf
    print(168, 199)
    w_26.estD = 0
    print(168, 200)
    unsureVertices_26 = heapdict.heapdict()
    for v_26 in G_26.vertices:
        print(170, 202)
        unsureVertices_26[v_26] = v_26.estD
    while len(unsureVertices_26) > 0:
        print(172, 205)
        u_26, dist_26 = unsureVertices_26.popitem()
        if u_26.estD == math.inf:
            print('exit scope 26')
            return
        for v_26, wt_26 in u_26.getOutNeighborsWithWeights():
            if u_26.estD + wt_26 < v_26.estD:
                print(179, 212)
                v_26.estD = u_26.estD + wt_26
                print(179, 213)
                unsureVertices_26[v_26] = u_26.estD + wt_26
                print(179, 214)
                v_26.parent = u_26
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 217)
    w_27 = w
    G_27 = G
    dijkstra(w_27, G_27)
    for v_27 in G_27.vertices:
        if v_27.estD == math.inf:
            continue
        print(188, 223)
        path_27 = []
        print(188, 224)
        current_27 = v_27
        while current_27 != w_27:
            path_27.append(current_27)
            print(190, 227)
            current_27 = current_27.parent
        path_27.append(current_27)
        path_27.reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=1):
    print('enter scope 28')
    print(1, 232)
    myFn_28 = myFn
    nVals_28 = nVals
    pFn_28 = pFn
    numTrials_28 = numTrials
    print(194, 233)
    nValues_28 = []
    print(194, 234)
    tValues_28 = []
    for n_28 in nVals_28:
        print(196, 237)
        runtime_28 = 0
        for t_28 in range(numTrials_28):
            print(199, 239)
            G_28 = randomGraph(n_28 * 30000, pFn(n_28))
            print(199, 240)
            start_28 = time.time()
            myFn(G_28.vertices[0], G_28)
            print(199, 242)
            end_28 = time.time()
            print(199, 243)
            runtime_28 += (end_28 - start_28) * 1000
        print(200, 244)
        runtime_28 = runtime_28 / numTrials_28
        nValues_28.append(n_28)
        tValues_28.append(runtime_28)
    print('exit scope 28')
    return nValues_28, tValues_28
    print('exit scope 28')


def smallFrac(n):
    print('enter scope 29')
    print(1, 249)
    n_29 = n
    print('exit scope 29')
    return float(5 / n_29)
    print('exit scope 29')


if __name__ == '__main__':
    loop.start_unroll
    print(207, 255)
    G_0 = randomGraph(5, 0.2)
    BFS_shortestPaths(G_0.vertices[0], G_0)
    dijkstraDumb_shortestPaths(G_0.vertices[0], G_0)
    print(207, 258)
    G_0 = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    dijkstra_shortestPaths(G_0.vertices[0], G_0)
    print(207, 260)
    nValues_0 = [10]
    print(207, 261)
    nDijkstra_0, tDijkstra_0 = runTrials(BFS, nValues_0, smallFrac)
