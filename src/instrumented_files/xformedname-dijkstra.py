import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop
import numpy as np


class CS161Vertex:

    def __init__(self, v):
        self = self
        v_1 = v
        self.inNeighbors = []
        self.outNeighbors = []
        self.value = v_1
        self.inTime = None
        self.outTime = None
        self.status = 'unvisited'
        self.parent = None
        self.estD = math.inf

    def hasOutNeighbor(self, v):
        self = self
        v_2 = v
        if v_2 in self.getOutNeighbors():
            return True
        return False

    def hasInNeighbor(self, v):
        self = self
        v_3 = v
        if v_3 in self.getInNeighbors():
            return True
        return False

    def hasNeighbor(self, v):
        self = self
        v_4 = v
        if v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors():
            return True
        return False

    def getOutNeighbors(self):
        self = self
        return [v_5[0] for v_5 in self.outNeighbors]

    def getInNeighbors(self):
        self = self
        return [v_6[0] for v_6 in self.inNeighbors]

    def getOutNeighborsWithWeights(self):
        self = self
        return self.outNeighbors

    def getInNeighborsWithWeights(self):
        self = self
        return self.inNeighbors

    def addOutNeighbor(self, v, wt):
        self = self
        v_9 = v
        wt_9 = wt
        self.outNeighbors.append((v_9, wt_9))

    def addInNeighbor(self, v, wt):
        self = self
        v_10 = v
        wt_10 = wt
        self.inNeighbors.append((v_10, wt_10))


class CS161Graph:

    def __init__(self):
        self = self
        self.vertices = []

    def addVertex(self, n):
        self = self
        n_12 = n
        self.vertices.append(n_12)

    def addDiEdge(self, u, v, wt=1):
        self = self
        u_13 = u
        v_13 = v
        wt_13 = wt
        u_13.addOutNeighbor(v_13, wt=wt_13)
        v_13.addInNeighbor(u_13, wt=wt_13)

    def addBiEdge(self, u, v, wt=1):
        self = self
        u_14 = u
        v_14 = v
        wt_14 = wt
        self.addDiEdge(u_14, v_14, wt=wt_14)
        self.addDiEdge(v_14, u_14, wt=wt_14)

    def getDirEdges(self):
        self = self
        ret_15 = []
        for v_15 in self.vertices:
            for u_15, wt_15 in v_15.getOutNeighborsWithWeights():
                ret_15.append([v_15, u_15, wt_15])
        return ret_15


class CS161Graph:

    def __init__(self):
        self = self
        self.vertices = []

    def addVertex(self, n):
        self = self
        n_17 = n
        self.vertices.append(n_17)

    def addDiEdge(self, u, v, wt=1):
        self = self
        u_18 = u
        v_18 = v
        wt_18 = wt
        u_18.addOutNeighbor(v_18, wt=wt_18)
        v_18.addInNeighbor(u_18, wt=wt_18)

    def addBiEdge(self, u, v, wt=1):
        self = self
        u_19 = u
        v_19 = v
        wt_19 = wt
        self.addDiEdge(u_19, v_19, wt=wt_19)
        self.addDiEdge(v_19, u_19, wt=wt_19)

    def getDirEdges(self):
        self = self
        ret_20 = []
        for v_20 in self.vertices:
            for u_20, wt_20 in v_20.getOutNeighborsWithWeights():
                ret_20.append([v_20, u_20, wt_20])
        return ret_20


def randomGraph(n, p, wts=[1]):
    n_21 = n
    p_21 = p
    wts_21 = wts
    G_21 = CS161Graph()
    V_21 = [CS161Vertex(x_21) for x_21 in range(n_21)]
    for v_21 in V_21:
        G_21.addVertex(v_21)
    for v_21 in V_21:
        i_21 = 0
        for w_21 in V_21:
            if v_21 != w_21:
                if random() < p_21:
                    G_21.addDiEdge(v_21, w_21, wt=choice(wts_21))
                    i_21 += 1
            if i_21 > 15:
                break
    return G_21


def BFS(w, G):
    w_22 = w
    G_22 = G
    for v_22 in G_22.vertices:
        v_22.status = 'unvisited'
    n_22 = len(G_22.vertices)
    Ls_22 = [[] for i_22 in range(n_22)]
    Ls_22[0] = [w_22]
    w_22.status = 'visited'
    for i_22 in range(n_22):
        for u_22 in Ls_22[i_22]:
            for v_22 in u_22.getOutNeighbors():
                if v_22.status == 'unvisited':
                    v_22.status = 'visited'
                    v_22.parent = u_22
                    Ls_22[i_22 + 1].append(v_22)
    return Ls_22


def BFS_shortestPaths(w, G):
    w_23 = w
    G_23 = G
    Ls_23 = BFS(w_23, G_23)
    for i_23 in range(len(Ls_23)):
        for w_23 in Ls_23[i_23]:
            path_23 = []
            current_23 = w_23
            for j_23 in range(i_23):
                path_23.append(current_23)
                current_23 = current_23.parent
            path_23.append(current_23)
            path_23.reverse()


def dijkstraDumb(w, G):
    w_24 = w
    G_24 = G
    for v_24 in G_24.vertices:
        v_24.estD = math.inf
    w_24.estD = 0
    unsureVertices_24 = G_24.vertices[:]
    while len(unsureVertices_24) > 0:
        u_24 = None
        minD_24 = math.inf
        for x_24 in unsureVertices_24:
            if x_24.estD < minD_24:
                minD_24 = x_24.estD
                u_24 = x_24
        if u_24 == None:
            return
        for v_24, wt_24 in u_24.getOutNeighborsWithWeights():
            if u_24.estD + wt_24 < v_24.estD:
                v_24.estD = u_24.estD + wt_24
                v_24.parent = u_24
        unsureVertices_24.remove(u_24)


def dijkstraDumb_shortestPaths(w, G):
    w_25 = w
    G_25 = G
    dijkstraDumb(w_25, G_25)
    for v_25 in G_25.vertices:
        if v_25.estD == math.inf:
            continue
        path_25 = []
        current_25 = v_25
        while current_25 != w_25:
            path_25.append(current_25)
            current_25 = current_25.parent
        path_25.append(current_25)
        path_25.reverse()


def dijkstra(w, G):
    w_26 = w
    G_26 = G
    for v_26 in G_26.vertices:
        v_26.estD = math.inf
    w_26.estD = 0
    unsureVertices_26 = heapdict.heapdict()
    for v_26 in G_26.vertices:
        unsureVertices_26[v_26] = v_26.estD
    while len(unsureVertices_26) > 0:
        u_26, dist_26 = unsureVertices_26.popitem()
        if u_26.estD == math.inf:
            return
        for v_26, wt_26 in u_26.getOutNeighborsWithWeights():
            if u_26.estD + wt_26 < v_26.estD:
                v_26.estD = u_26.estD + wt_26
                unsureVertices_26[v_26] = u_26.estD + wt_26
                v_26.parent = u_26


def dijkstra_shortestPaths(w, G):
    w_27 = w
    G_27 = G
    dijkstra(w_27, G_27)
    for v_27 in G_27.vertices:
        if v_27.estD == math.inf:
            continue
        path_27 = []
        current_27 = v_27
        while current_27 != w_27:
            path_27.append(current_27)
            current_27 = current_27.parent
        path_27.append(current_27)
        path_27.reverse()


def runTrials(myFn, nVals, pFn, numTrials=1):
    myFn_28 = myFn
    nVals_28 = nVals
    pFn_28 = pFn
    numTrials_28 = numTrials
    nValues_28 = []
    tValues_28 = []
    for n_28 in nVals_28:
        runtime_28 = 0
        for t_28 in range(numTrials_28):
            G_28 = randomGraph(n_28 * 10000, pFn(n_28))
            start_28 = time.time()
            myFn(G_28.vertices[0], G_28)
            end_28 = time.time()
            runtime_28 += (end_28 - start_28) * 1000
        runtime_28 = runtime_28 / numTrials_28
        nValues_28.append(n_28)
        tValues_28.append(runtime_28)
    return nValues_28, tValues_28


def smallFrac(n):
    n_29 = n
    return float(5 / n_29)


def read_random_graph_from_file(n, p, wts=[1]):
    n_30 = n
    p_30 = p
    wts_30 = wts
    return randomGraph(n_30, p_30, wts_30)


def main():
    G_31 = randomGraph(7, 0.2)
    dijkstra_shortestPaths(G_31.vertices[0], G_31)


if __name__ == '__main__':
    main()
