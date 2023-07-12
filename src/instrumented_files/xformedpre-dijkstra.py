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
            print('enter scope 16')
            for u__16, wt__16 in v__15.getOutNeighborsWithWeights():
                print('enter scope 17')
                ret__17.append([v__15, u__16, wt__16])
                print('exit scope 17')
            print('exit scope 16')
        print('exit scope 15')
        return ret__15
        print('exit scope 15')


class CS161Graph:

    def __init__(self):
        print('enter scope 18')
        print(1, 83)
        self = self
        print(70, 84)
        self.vertices = []
        print('exit scope 18')

    def addVertex(self, n):
        print('enter scope 19')
        print(1, 86)
        self = self
        n__19 = n
        self.vertices.append(n__19)
        print('exit scope 19')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 20')
        print(1, 90)
        self = self
        u__20 = u
        v__20 = v
        wt__20 = wt
        u__20.addOutNeighbor(v__20, wt=wt__20)
        v__20.addInNeighbor(u__20, wt=wt__20)
        print('exit scope 20')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 21')
        print(1, 95)
        self = self
        u__21 = u
        v__21 = v
        wt__21 = wt
        self.addDiEdge(u__21, v__21, wt=wt__21)
        self.addDiEdge(v__21, u__21, wt=wt__21)
        print('exit scope 21')

    def getDirEdges(self):
        print('enter scope 22')
        print(1, 101)
        self = self
        print(82, 102)
        ret__22 = []
        for v__22 in self.vertices:
            print('enter scope 23')
            for u__23, wt__23 in v__22.getOutNeighborsWithWeights():
                print('enter scope 24')
                ret__22.append([v__22, u__23, wt__23])
                print('exit scope 24')
            print('exit scope 23')
        print('exit scope 22')
        return ret__22
        print('exit scope 22')


def randomGraph(n, p, wts=[1]):
    print('enter scope 25')
    print(1, 112)
    n__25 = n
    p__25 = p
    wts__25 = wts
    print(91, 113)
    G__25 = CS161Graph()
    print(91, 114)
    V__25 = [CS161Vertex(x__25) for x__25 in range(n__25)]
    for v__25 in V__25:
        print('enter scope 26')
        G__25.addVertex(v__25)
        print('exit scope 26')
    for v__25 in V__25:
        print('enter scope 27')
        for w__27 in V__25:
            print('enter scope 28')
            if v__25 != w__27:
                if random() < p__25:
                    G__25.addDiEdge(v__25, w__27, wt=choice(wts__25))
            print('exit scope 28')
        print('exit scope 27')
    print('exit scope 25')
    return G__25
    print('exit scope 25')


def BFS(w, G):
    print('enter scope 29')
    print(1, 124)
    w__29 = w
    G__29 = G
    for v__29 in G__29.vertices:
        print('enter scope 30')
        print(107, 126)
        v__29.status = 'unvisited'
        print('exit scope 30')
    print(108, 127)
    n__29 = len(G__29.vertices)
    print(108, 128)
    Ls__29 = [[] for i__29 in range(n__29)]
    print(108, 129)
    Ls__29[0] = [w__29]
    print(108, 130)
    w__29.status = 'visited'
    for i__29 in range(n__29):
        print('enter scope 31')
        for u__31 in Ls__29[i__29]:
            print('enter scope 32')
            for v__29 in u__31.getOutNeighbors():
                print('enter scope 33')
                if v__29.status == 'unvisited':
                    print(116, 135)
                    v__29.status = 'visited'
                    print(116, 136)
                    v__29.parent = u__31
                    Ls__29[i__29 + 1].append(v__29)
                print('exit scope 33')
            print('exit scope 32')
        print('exit scope 31')
    print('exit scope 29')
    return Ls__29
    print('exit scope 29')


def BFS_shortestPaths(w, G):
    print('enter scope 34')
    print(1, 140)
    w__34 = w
    G__34 = G
    print(121, 141)
    Ls__34 = BFS(w__34, G__34)
    for i__34 in range(len(Ls__34)):
        print('enter scope 35')
        for w__34 in Ls__34[i__34]:
            print('enter scope 36')
            print(125, 145)
            path__36 = []
            print(125, 146)
            current__36 = w__34
            for j__36 in range(i__34):
                print('enter scope 37')
                path__36.append(current__36)
                print(128, 149)
                current__36 = current__36.parent
                print('exit scope 37')
            path__36.append(current__36)
            path__36.reverse()
            print('exit scope 36')
        print('exit scope 35')
    print('exit scope 34')


def dijkstraDumb(w, G):
    print('enter scope 38')
    print(1, 154)
    w__38 = w
    G__38 = G
    for v__38 in G__38.vertices:
        print('enter scope 39')
        print(133, 156)
        v__38.estD = math.inf
        print('exit scope 39')
    print(134, 157)
    w__38.estD = 0
    print(134, 158)
    unsureVertices__38 = G__38.vertices[:]
    while len(unsureVertices__38) > 0:
        print(136, 161)
        u__38 = None
        print(136, 162)
        minD__38 = math.inf
        for x__38 in unsureVertices__38:
            print('enter scope 40')
            if x__38.estD < minD__38:
                print(141, 165)
                minD__38 = x__38.estD
                print(141, 166)
                u__38 = x__38
            print('exit scope 40')
        if u__38 == None:
            print('exit scope 38')
            return
        for v__38, wt__38 in u__38.getOutNeighborsWithWeights():
            print('enter scope 41')
            if u__38.estD + wt__38 < v__38.estD:
                print(148, 173)
                v__38.estD = u__38.estD + wt__38
                print(148, 174)
                v__38.parent = u__38
            print('exit scope 41')
        unsureVertices__38.remove(u__38)
    print('exit scope 38')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 42')
    print(1, 178)
    w__42 = w
    G__42 = G
    dijkstraDumb(w__42, G__42)
    for v__42 in G__42.vertices:
        print('enter scope 43')
        if v__42.estD == math.inf:
            continue
        print(157, 184)
        path__43 = []
        print(157, 185)
        current__43 = v__42
        while current__43 != w__42:
            path__43.append(current__43)
            print(159, 188)
            current__43 = current__43.parent
        path__43.append(current__43)
        path__43.reverse()
        print('exit scope 43')
    print('exit scope 42')


def dijkstra(w, G):
    print('enter scope 44')
    print(1, 193)
    w__44 = w
    G__44 = G
    for v__44 in G__44.vertices:
        print('enter scope 45')
        print(164, 195)
        v__44.estD = math.inf
        print('exit scope 45')
    print(165, 196)
    w__44.estD = 0
    print(165, 197)
    unsureVertices__44 = heapdict.heapdict()
    for v__44 in G__44.vertices:
        print('enter scope 46')
        print(167, 199)
        unsureVertices__44[v__44] = v__44.estD
        print('exit scope 46')
    while len(unsureVertices__44) > 0:
        print(169, 202)
        u__44, dist__44 = unsureVertices__44.popitem()
        if u__44.estD == math.inf:
            print('exit scope 44')
            return
        for v__44, wt__44 in u__44.getOutNeighborsWithWeights():
            print('enter scope 47')
            if u__44.estD + wt__44 < v__44.estD:
                print(176, 209)
                v__44.estD = u__44.estD + wt__44
                print(176, 210)
                unsureVertices__44[v__44] = u__44.estD + wt__44
                print(176, 211)
                v__44.parent = u__44
            print('exit scope 47')
    print('exit scope 44')


def dijkstra_shortestPaths(w, G):
    print('enter scope 48')
    print(1, 214)
    w__48 = w
    G__48 = G
    dijkstra(w__48, G__48)
    for v__48 in G__48.vertices:
        print('enter scope 49')
        if v__48.estD == math.inf:
            continue
        print(185, 220)
        path__49 = []
        print(185, 221)
        current__49 = v__48
        while current__49 != w__48:
            path__49.append(current__49)
            print(187, 224)
            current__49 = current__49.parent
        path__49.append(current__49)
        path__49.reverse()
        print('exit scope 49')
    print('exit scope 48')


def runTrials(myFn, nVals, pFn, numTrials=25):
    print('enter scope 50')
    print(1, 229)
    myFn__50 = myFn
    nVals__50 = nVals
    pFn__50 = pFn
    numTrials__50 = numTrials
    print(191, 230)
    nValues__50 = []
    print(191, 231)
    tValues__50 = []
    for n__50 in nVals__50:
        print('enter scope 51')
        print(193, 234)
        runtime__51 = 0
        for t__51 in range(numTrials__50):
            print('enter scope 52')
            print(196, 236)
            G__52 = randomGraph(n__50, pFn(n__50))
            print(196, 237)
            start__52 = time.time()
            myFn(G__52.vertices[0], G__52)
            print(196, 239)
            end__52 = time.time()
            print(196, 240)
            runtime__51 += (end__52 - start__52) * 1000
            print('exit scope 52')
        print(197, 241)
        runtime__51 = runtime__51 / numTrials__50
        nValues__50.append(n__50)
        tValues__50.append(runtime__51)
        print('exit scope 51')
    print('exit scope 50')
    return nValues__50, tValues__50
    print('exit scope 50')


def smallFrac(n):
    print('enter scope 53')
    print(1, 246)
    n__53 = n
    print('exit scope 53')
    return float(5 / n__53)
    print('exit scope 53')


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
