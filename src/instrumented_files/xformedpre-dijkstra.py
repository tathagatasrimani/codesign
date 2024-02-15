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
        print('exit scope 1')

    def hasOutNeighbor(self, v):
        print('enter scope 2')
        print(1, 24)
        print(6, 25)
        self = self
        print(6, 26)
        v_2 = v
        if v_2 in self.getOutNeighbors():
            print('exit scope 2')
            return True
        print('exit scope 2')
        return False
        print('exit scope 2')

    def hasInNeighbor(self, v):
        print('enter scope 3')
        print(1, 31)
        print(13, 32)
        self = self
        print(13, 33)
        v_3 = v
        if v_3 in self.getInNeighbors():
            print('exit scope 3')
            return True
        print('exit scope 3')
        return False
        print('exit scope 3')

    def hasNeighbor(self, v):
        print('enter scope 4')
        print(1, 38)
        print(20, 39)
        self = self
        print(20, 40)
        v_4 = v
        if v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors():
            print('exit scope 4')
            return True
        print('exit scope 4')
        return False
        print('exit scope 4')

    def getOutNeighbors(self):
        print('enter scope 5')
        print(1, 45)
        print(27, 46)
        self = self
        print('exit scope 5')
        return [v_5[0] for v_5 in self.outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 49)
        print(31, 50)
        self = self
        print('exit scope 6')
        return [v_6[0] for v_6 in self.inNeighbors]
        print('exit scope 6')

    def getOutNeighborsWithWeights(self):
        print('enter scope 7')
        print(1, 53)
        print(35, 54)
        self = self
        print('exit scope 7')
        return self.outNeighbors
        print('exit scope 7')

    def getInNeighborsWithWeights(self):
        print('enter scope 8')
        print(1, 57)
        print(39, 58)
        self = self
        print('exit scope 8')
        return self.inNeighbors
        print('exit scope 8')

    def addOutNeighbor(self, v, wt):
        print('enter scope 9')
        print(1, 61)
        print(43, 62)
        self = self
        print(43, 63)
        v_9 = v
        print(43, 64)
        wt_9 = wt
        self.outNeighbors.append((v_9, wt_9))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 67)
        print(46, 68)
        self = self
        print(46, 69)
        v_10 = v
        print(46, 70)
        wt_10 = wt
        self.inNeighbors.append((v_10, wt_10))
        print('exit scope 10')


class CS161Graph:

    def __init__(self):
        print('enter scope 11')
        print(1, 76)
        self = self
        self.vertices = []
        print('exit scope 11')

    def addVertex(self, n):
        print('enter scope 12')
        print(1, 80)
        self = self
        n_12 = n
        self.vertices.append(n_12)
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 85)
        self = self
        u_13 = u
        v_13 = v
        wt_13 = wt
        u_13.addOutNeighbor(v_13, wt=wt_13)
        v_13.addInNeighbor(u_13, wt=wt_13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 93)
        self = self
        u_14 = u
        v_14 = v
        wt_14 = wt
        self.addDiEdge(u_14, v_14, wt=wt_14)
        self.addDiEdge(v_14, u_14, wt=wt_14)
        print('exit scope 14')

    def getDirEdges(self):
        print('enter scope 15')
        print(1, 101)
        self = self
        ret_15 = []
        for v_15 in self.vertices:
            for u_15, wt_15 in v_15.getOutNeighborsWithWeights():
                ret_15.append([v_15, u_15, wt_15])
        print('exit scope 15')
        return ret_15
        print('exit scope 15')


class CS161Graph:

    def __init__(self):
        print('enter scope 16')
        print(1, 112)
        print(70, 113)
        self = self
        print(70, 114)
        self.vertices = []
        print('exit scope 16')

    def addVertex(self, n):
        print('enter scope 17')
        print(1, 116)
        print(73, 117)
        self = self
        print(73, 118)
        n_17 = n
        self.vertices.append(n_17)
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 121)
        print(76, 122)
        self = self
        print(76, 123)
        u_18 = u
        print(76, 124)
        v_18 = v
        print(76, 125)
        wt_18 = wt
        u_18.addOutNeighbor(v_18, wt=wt_18)
        v_18.addInNeighbor(u_18, wt=wt_18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 129)
        print(79, 130)
        self = self
        print(79, 131)
        u_19 = u
        print(79, 132)
        v_19 = v
        print(79, 133)
        wt_19 = wt
        self.addDiEdge(u_19, v_19, wt=wt_19)
        self.addDiEdge(v_19, u_19, wt=wt_19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 137)
        print(82, 138)
        self = self
        print(82, 139)
        ret_20 = []
        for v_20 in self.vertices:
            for u_20, wt_20 in v_20.getOutNeighborsWithWeights():
                ret_20.append([v_20, u_20, wt_20])
        print('exit scope 20')
        return ret_20
        print('exit scope 20')


def randomGraph(n, p, wts=[1]):
    print('enter scope 21')
    print(1, 146)
    print(91, 147)
    n_21 = n
    print(91, 148)
    p_21 = p
    print(91, 149)
    wts_21 = wts
    print(91, 150)
    G_21 = CS161Graph()
    print(91, 151)
    V_21 = [CS161Vertex(x_21) for x_21 in range(n_21)]
    for v_21 in V_21:
        G_21.addVertex(v_21)
    for v_21 in V_21:
        print(95, 155)
        i_21 = 0
        for w_21 in V_21:
            if v_21 != w_21:
                if random() < p_21:
                    G_21.addDiEdge(v_21, w_21, wt=choice(wts_21))
                    print(102, 160)
                    i_21 += 1
            if i_21 > 15:
                break
    print('exit scope 21')
    return G_21
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 166)
    print(109, 167)
    w_22 = w
    print(109, 168)
    G_22 = G
    for v_22 in G_22.vertices:
        print(111, 170)
        v_22.status = 'unvisited'
    print(112, 171)
    n_22 = len(G_22.vertices)
    print(112, 172)
    Ls_22 = [[] for i_22 in range(n_22)]
    print(112, 173)
    Ls_22[0] = [w_22]
    print(112, 174)
    w_22.status = 'visited'
    for i_22 in range(n_22):
        for u_22 in Ls_22[i_22]:
            for v_22 in u_22.getOutNeighbors():
                if v_22.status == 'unvisited':
                    print(120, 179)
                    v_22.status = 'visited'
                    print(120, 180)
                    v_22.parent = u_22
                    Ls_22[i_22 + 1].append(v_22)
    print('exit scope 22')
    return Ls_22
    print('exit scope 22')


def BFS_shortestPaths(w, G):
    print('enter scope 23')
    print(1, 185)
    print(125, 186)
    w_23 = w
    print(125, 187)
    G_23 = G
    print(125, 188)
    Ls_23 = BFS(w_23, G_23)
    for i_23 in range(len(Ls_23)):
        for w_23 in Ls_23[i_23]:
            print(129, 191)
            path_23 = []
            print(129, 192)
            current_23 = w_23
            for j_23 in range(i_23):
                path_23.append(current_23)
                print(132, 195)
                current_23 = current_23.parent
            path_23.append(current_23)
            path_23.reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 200)
    print(136, 201)
    w_24 = w
    print(136, 202)
    G_24 = G
    for v_24 in G_24.vertices:
        print(138, 204)
        v_24.estD = math.inf
    print(139, 205)
    w_24.estD = 0
    print(139, 206)
    unsureVertices_24 = G_24.vertices[:]
    while len(unsureVertices_24) > 0:
        print(141, 208)
        u_24 = None
        print(141, 209)
        minD_24 = math.inf
        for x_24 in unsureVertices_24:
            if x_24.estD < minD_24:
                print(146, 212)
                minD_24 = x_24.estD
                print(146, 213)
                u_24 = x_24
        if u_24 == None:
            print('exit scope 24')
            return
        for v_24, wt_24 in u_24.getOutNeighborsWithWeights():
            if u_24.estD + wt_24 < v_24.estD:
                print(153, 218)
                v_24.estD = u_24.estD + wt_24
                print(153, 219)
                v_24.parent = u_24
        unsureVertices_24.remove(u_24)
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 223)
    print(157, 224)
    w_25 = w
    print(157, 225)
    G_25 = G
    dijkstraDumb(w_25, G_25)
    for v_25 in G_25.vertices:
        if v_25.estD == math.inf:
            continue
        print(162, 230)
        path_25 = []
        print(162, 231)
        current_25 = v_25
        while current_25 != w_25:
            path_25.append(current_25)
            print(164, 234)
            current_25 = current_25.parent
        path_25.append(current_25)
        path_25.reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 239)
    print(168, 240)
    w_26 = w
    print(168, 241)
    G_26 = G
    for v_26 in G_26.vertices:
        print(170, 243)
        v_26.estD = math.inf
    print(171, 244)
    w_26.estD = 0
    print(171, 245)
    unsureVertices_26 = heapdict.heapdict()
    for v_26 in G_26.vertices:
        print(173, 247)
        unsureVertices_26[v_26] = v_26.estD
    while len(unsureVertices_26) > 0:
        print(175, 249)
        u_26, dist_26 = unsureVertices_26.popitem()
        if u_26.estD == math.inf:
            print('exit scope 26')
            return
        for v_26, wt_26 in u_26.getOutNeighborsWithWeights():
            if u_26.estD + wt_26 < v_26.estD:
                print(182, 254)
                v_26.estD = u_26.estD + wt_26
                print(182, 255)
                unsureVertices_26[v_26] = u_26.estD + wt_26
                print(182, 256)
                v_26.parent = u_26
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 259)
    print(186, 260)
    w_27 = w
    print(186, 261)
    G_27 = G
    dijkstra(w_27, G_27)
    for v_27 in G_27.vertices:
        if v_27.estD == math.inf:
            continue
        print(191, 266)
        path_27 = []
        print(191, 267)
        current_27 = v_27
        while current_27 != w_27:
            path_27.append(current_27)
            print(193, 270)
            current_27 = current_27.parent
        path_27.append(current_27)
        path_27.reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=1):
    print('enter scope 28')
    print(1, 275)
    print(197, 276)
    myFn_28 = myFn
    print(197, 277)
    nVals_28 = nVals
    print(197, 278)
    pFn_28 = pFn
    print(197, 279)
    numTrials_28 = numTrials
    print(197, 280)
    nValues_28 = []
    print(197, 281)
    tValues_28 = []
    for n_28 in nVals_28:
        print(199, 283)
        runtime_28 = 0
        for t_28 in range(numTrials_28):
            print(202, 285)
            G_28 = randomGraph(n_28 * 10000, pFn(n_28))
            print(202, 286)
            start_28 = time.time()
            myFn(G_28.vertices[0], G_28)
            print(202, 288)
            end_28 = time.time()
            print(202, 289)
            runtime_28 += (end_28 - start_28) * 1000
        print(203, 290)
        runtime_28 = runtime_28 / numTrials_28
        nValues_28.append(n_28)
        tValues_28.append(runtime_28)
    print('exit scope 28')
    return nValues_28, tValues_28
    print('exit scope 28')


def smallFrac(n):
    print('enter scope 29')
    print(1, 296)
    print(207, 297)
    n_29 = n
    print('exit scope 29')
    return float(5 / n_29)
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


def main():
    print('enter scope 31')
    print(1, 308)
    print(215, 309)
    G_31 = randomGraph(7, 0.2)
    dijkstra_shortestPaths(G_31.vertices[0], G_31)
    print('exit scope 31')


if __name__ == '__main__':
    main()
