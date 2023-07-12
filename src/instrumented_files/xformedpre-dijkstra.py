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
        v__1 = v
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
        v__2 = v
        if v__2 in self.getOutNeighbors():
            print('exit scope 2')
            return True
        print('exit scope 2')
        return False
        print('exit scope 2')

    def hasInNeighbor(self, v):
        print('enter scope 3')
        print(1, 26)
        self = self
        v__3 = v
        if v__3 in self.getInNeighbors():
            print('exit scope 3')
            return True
        print('exit scope 3')
        return False
        print('exit scope 3')

    def hasNeighbor(self, v):
        print('enter scope 4')
        print(1, 31)
        self = self
        v__4 = v
        if v__4 in self.getInNeighbors() or v__4 in self.getOutNeighbors():
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
        return [v__5[0] for v__5 in self.outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 39)
        self = self
        print('exit scope 6')
        return [v__6[0] for v__6 in self.inNeighbors]
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
        v__9 = v
        wt__9 = wt
        self.outNeighbors.append((v__9, wt__9))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 51)
        self = self
        v__10 = v
        wt__10 = wt
        self.inNeighbors.append((v__10, wt__10))
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
        n__12 = n
        self.vertices.append(n__12)
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 64)
        self = self
        u__13 = u
        v__13 = v
        wt__13 = wt
        u__13.addOutNeighbor(v__13, wt=wt__13)
        v__13.addInNeighbor(u__13, wt=wt__13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 69)
        self = self
        u__14 = u
        v__14 = v
        wt__14 = wt
        self.addDiEdge(u__14, v__14, wt=wt__14)
        self.addDiEdge(v__14, u__14, wt=wt__14)
        print('exit scope 14')

    def getDirEdges(self):
        print('enter scope 15')
        print(1, 75)
        self = self
        ret = []
        for v__15 in self.vertices:
            for u__15, wt__15 in v__15.getOutNeighborsWithWeights():
                ret__15.append([v__15, u__15, wt__15])
        print('exit scope 15')
        return ret__15
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
        n__17 = n
        self.vertices.append(n__17)
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 90)
        self = self
        u__18 = u
        v__18 = v
        wt__18 = wt
        u__18.addOutNeighbor(v__18, wt=wt__18)
        v__18.addInNeighbor(u__18, wt=wt__18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 95)
        self = self
        u__19 = u
        v__19 = v
        wt__19 = wt
        self.addDiEdge(u__19, v__19, wt=wt__19)
        self.addDiEdge(v__19, u__19, wt=wt__19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 101)
        self = self
        print(82, 102)
        ret__20 = []
        for v__20 in self.vertices:
            for u__20, wt__20 in v__20.getOutNeighborsWithWeights():
                ret__20.append([v__20, u__20, wt__20])
        print('exit scope 20')
        return ret__20
        print('exit scope 20')


def randomGraph(n, p, wts=[1]):
    print('enter scope 21')
    print(1, 112)
    n__21 = n
    p__21 = p
    wts__21 = wts
    print(91, 113)
    G__21 = CS161Graph()
    print(91, 114)
    V__21 = [CS161Vertex(x__21) for x__21 in range(n__21)]
    for v__21 in V__21:
        G__21.addVertex(v__21)
    for v__21 in V__21:
        for w__21 in V__21:
            if v__21 != w__21:
                if random() < p__21:
                    G__21.addDiEdge(v__21, w__21, wt=choice(wts__21))
    print('exit scope 21')
    return G__21
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 124)
    w__22 = w
    G__22 = G
    for v__22 in G__22.vertices:
        print(107, 126)
        v__22.status = 'unvisited'
    print(108, 127)
    n__22 = len(G__22.vertices)
    print(108, 128)
    Ls__22 = [[] for i__22 in range(n__22)]
    print(108, 129)
    Ls__22[0] = [w__22]
    print(108, 130)
    w__22.status = 'visited'
    for i__22 in range(n__22):
        for u__22 in Ls__22[i__22]:
            for v__22 in u__22.getOutNeighbors():
                if v__22.status == 'unvisited':
                    print(116, 135)
                    v__22.status = 'visited'
                    print(116, 136)
                    v__22.parent = u__22
                    Ls__22[i__22 + 1].append(v__22)
    print('exit scope 22')
    return Ls__22
    print('exit scope 22')


def BFS_shortestPaths(w, G):
    print('enter scope 23')
    print(1, 140)
    w__23 = w
    G__23 = G
    print(121, 141)
    Ls__23 = BFS(w__23, G__23)
    for i__23 in range(len(Ls__23)):
        for w__23 in Ls__23[i__23]:
            print(125, 145)
            path__23 = []
            print(125, 146)
            current__23 = w__23
            for j__23 in range(i__23):
                path__23.append(current__23)
                print(128, 149)
                current__23 = current__23.parent
            path__23.append(current__23)
            path__23.reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 154)
    w__24 = w
    G__24 = G
    for v__24 in G__24.vertices:
        print(133, 156)
        v__24.estD = math.inf
    print(134, 157)
    w__24.estD = 0
    print(134, 158)
    unsureVertices__24 = G__24.vertices[:]
    while len(unsureVertices__24) > 0:
        print(136, 161)
        u__24 = None
        print(136, 162)
        minD__24 = math.inf
        for x__24 in unsureVertices__24:
            if x__24.estD < minD__24:
                print(141, 165)
                minD__24 = x__24.estD
                print(141, 166)
                u__24 = x__24
        if u__24 == None:
            print('exit scope 24')
            return
        for v__24, wt__24 in u__24.getOutNeighborsWithWeights():
            if u__24.estD + wt__24 < v__24.estD:
                print(148, 173)
                v__24.estD = u__24.estD + wt__24
                print(148, 174)
                v__24.parent = u__24
        unsureVertices__24.remove(u__24)
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 178)
    w__25 = w
    G__25 = G
    dijkstraDumb(w__25, G__25)
    for v__25 in G__25.vertices:
        if v__25.estD == math.inf:
            continue
        print(157, 184)
        path__25 = []
        print(157, 185)
        current__25 = v__25
        while current__25 != w__25:
            path__25.append(current__25)
            print(159, 188)
            current__25 = current__25.parent
        path__25.append(current__25)
        path__25.reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 193)
    w__26 = w
    G__26 = G
    for v__26 in G__26.vertices:
        print(164, 195)
        v__26.estD = math.inf
    print(165, 196)
    w__26.estD = 0
    print(165, 197)
    unsureVertices__26 = heapdict.heapdict()
    for v__26 in G__26.vertices:
        print(167, 199)
        unsureVertices__26[v__26] = v__26.estD
    while len(unsureVertices__26) > 0:
        print(169, 202)
        u__26, dist__26 = unsureVertices__26.popitem()
        if u__26.estD == math.inf:
            print('exit scope 26')
            return
        for v__26, wt__26 in u__26.getOutNeighborsWithWeights():
            if u__26.estD + wt__26 < v__26.estD:
                print(176, 209)
                v__26.estD = u__26.estD + wt__26
                print(176, 210)
                unsureVertices__26[v__26] = u__26.estD + wt__26
                print(176, 211)
                v__26.parent = u__26
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 214)
    w__27 = w
    G__27 = G
    dijkstra(w__27, G__27)
    for v__27 in G__27.vertices:
        if v__27.estD == math.inf:
            continue
        print(185, 220)
        path__27 = []
        print(185, 221)
        current__27 = v__27
        while current__27 != w__27:
            path__27.append(current__27)
            print(187, 224)
            current__27 = current__27.parent
        path__27.append(current__27)
        path__27.reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=25):
    print('enter scope 28')
    print(1, 229)
    myFn__28 = myFn
    nVals__28 = nVals
    pFn__28 = pFn
    numTrials__28 = numTrials
    print(191, 230)
    nValues__28 = []
    print(191, 231)
    tValues__28 = []
    for n__28 in nVals__28:
        print(193, 234)
        runtime__28 = 0
        for t__28 in range(numTrials__28):
            print(196, 236)
            G__28 = randomGraph(n__28, pFn(n__28))
            print(196, 237)
            start__28 = time.time()
            myFn(G__28.vertices[0], G__28)
            print(196, 239)
            end__28 = time.time()
            print(196, 240)
            runtime__28 += (end__28 - start__28) * 1000
        print(197, 241)
        runtime__28 = runtime__28 / numTrials__28
        nValues__28.append(n__28)
        tValues__28.append(runtime__28)
    print('exit scope 28')
    return nValues__28, tValues__28
    print('exit scope 28')


def smallFrac(n):
    print('enter scope 29')
    print(1, 246)
    n__29 = n
    print('exit scope 29')
    return float(5 / n__29)
    print('exit scope 29')


if __name__ == '__main__':
    loop.start_unroll
    print(204, 252)
    G__0 = randomGraph(5, 0.2)
    BFS_shortestPaths(G__0.vertices[0], G__0)
    dijkstraDumb_shortestPaths(G__0.vertices[0], G__0)
    print(204, 255)
    G__0 = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    dijkstra_shortestPaths(G__0.vertices[0], G__0)
    print(204, 257)
    nValues__0 = [10]
    print(204, 258)
    nDijkstra__0, tDijkstra__0 = runTrials(dijkstra, nValues__0, smallFrac)
