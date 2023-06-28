import sys
import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop


class CS161Vertex:

    def __init__(self, v):
        print(1, 10)
        self.inNeighbors = []
        self.outNeighbors = []
        self.value = v
        self.inTime = None
        self.outTime = None
        self.status = 'unvisited'
        self.parent = None
        self.estD = math.inf

    def hasOutNeighbor(self, v):
        print(1, 21)
        if v in self.getOutNeighbors():
            print(6, 22)
            return True
        else:
            print(6, 22)
        return False

    def hasInNeighbor(self, v):
        print(1, 26)
        if v in self.getInNeighbors():
            print(13, 27)
            return True
        else:
            print(13, 27)
        return False

    def hasNeighbor(self, v):
        print(1, 31)
        if v in self.getInNeighbors() or v in self.getOutNeighbors():
            print(20, 32)
            return True
        else:
            print(20, 32)
        return False

    def getOutNeighbors(self):
        print(1, 36)
        return [v[0] for v in self.outNeighbors]

    def getInNeighbors(self):
        print(1, 39)
        return [v[0] for v in self.inNeighbors]

    def getOutNeighborsWithWeights(self):
        print(1, 42)
        return self.outNeighbors

    def getInNeighborsWithWeights(self):
        print(1, 45)
        return self.inNeighbors

    def addOutNeighbor(self, v, wt):
        print(1, 48)
        self.outNeighbors.append((v, wt))

    def addInNeighbor(self, v, wt):
        print(1, 51)
        self.inNeighbors.append((v, wt))

    def __str__(self):
        print(1, 54)
        return str(self.value)


class CS161Graph:

    def __init__(self):
        print(1, 60)
        self.vertices = []

    def addVertex(self, n):
        print(1, 63)
        self.vertices.append(n)

    def addDiEdge(self, u, v, wt=1):
        print(1, 67)
        u.addOutNeighbor(v, wt=wt)
        v.addInNeighbor(u, wt=wt)

    def addBiEdge(self, u, v, wt=1):
        print(1, 72)
        self.addDiEdge(u, v, wt=wt)
        self.addDiEdge(v, u, wt=wt)

    def getDirEdges(self):
        print(1, 78)
        ret = []
        for v in self.vertices:
            for u, wt in v.getOutNeighborsWithWeights():
                ret.append([v, u, wt])
        return ret

    def __str__(self):
        print(1, 85)
        ret = 'CS161Graph with:\n'
        ret += '\t Vertices:\n\t'
        for v in self.vertices:
            ret += str(v) + ','
        ret += '\n'
        ret += '\t Edges:\n\t'
        for a, b, wt in self.getDirEdges():
            ret += '(' + str(a) + ',' + str(b) + '; wt:' + str(wt) + ') '
        ret += '\n'
        return ret


class CS161Graph:

    def __init__(self):
        print(1, 98)
        print(84, 99)
        self.vertices = []

    def addVertex(self, n):
        print(1, 101)
        self.vertices.append(n)

    def addDiEdge(self, u, v, wt=1):
        print(1, 105)
        u.addOutNeighbor(v, wt=wt)
        v.addInNeighbor(u, wt=wt)

    def addBiEdge(self, u, v, wt=1):
        print(1, 110)
        self.addDiEdge(u, v, wt=wt)
        self.addDiEdge(v, u, wt=wt)

    def getDirEdges(self):
        print(1, 116)
        print(96, 117)
        ret = []
        for v in self.vertices:
            for u, wt in v.getOutNeighborsWithWeights():
                ret.append([v, u, wt])
        return ret

    def __str__(self):
        print(1, 123)
        print(105, 124)
        ret = 'CS161Graph with:\n'
        print(105, 125)
        ret += '\t Vertices:\n\t'
        for v in self.vertices:
            print(107, 127)
            ret += str(v) + ','
        print(108, 128)
        ret += '\n'
        print(108, 129)
        ret += '\t Edges:\n\t'
        for a, b, wt in self.getDirEdges():
            print(110, 131)
            ret += '(' + str(a) + ',' + str(b) + '; wt:' + str(wt) + ') '
        print(111, 132)
        ret += '\n'
        return ret


def randomGraph(n, p, wts=[1]):
    print(1, 139)
    print(115, 140)
    G = CS161Graph()
    print(115, 141)
    V = [CS161Vertex(x) for x in range(n)]
    for v in V:
        G.addVertex(v)
    for v in V:
        for w in V:
            if v != w:
                print(121, 146)
                if random() < p:
                    print(123, 147)
                    G.addDiEdge(v, w, wt=choice(wts))
                else:
                    print(123, 147)
            else:
                print(121, 146)
    return G


def BFS(w, G):
    print(1, 151)
    for v in G.vertices:
        print(131, 153)
        v.status = 'unvisited'
    print(132, 154)
    n = len(G.vertices)
    print(132, 155)
    Ls = [[] for i in range(n)]
    print(132, 156)
    Ls[0] = [w]
    print(132, 157)
    w.status = 'visited'
    for i in range(n):
        for u in Ls[i]:
            for v in u.getOutNeighbors():
                if v.status == 'unvisited':
                    print(138, 161)
                    print(140, 162)
                    v.status = 'visited'
                    print(140, 163)
                    v.parent = u
                    Ls[i + 1].append(v)
                else:
                    print(138, 161)
    return Ls


def BFS_shortestPaths(w, G):
    print(1, 167)
    print(145, 168)
    Ls = BFS(w, G)
    for i in range(len(Ls)):
        for w in Ls[i]:
            print(149, 172)
            path = []
            print(149, 173)
            current = w
            for j in range(i):
                path.append(current)
                print(152, 176)
                current = current.parent
            path.append(current)
            path.reverse()


def dijkstraDumb(w, G):
    print(1, 181)
    for v in G.vertices:
        print(157, 183)
        v.estD = math.inf
    print(158, 184)
    w.estD = 0
    print(158, 185)
    unsureVertices = G.vertices[:]
    while len(unsureVertices) > 0:
        print(160, 188)
        u = None
        print(160, 189)
        minD = math.inf
        for x in unsureVertices:
            if x.estD < minD:
                print(163, 191)
                print(165, 192)
                minD = x.estD
                print(165, 193)
                u = x
            else:
                print(163, 191)
        if u == None:
            print(164, 194)
            return
        else:
            print(164, 194)
        for v, wt in u.getOutNeighborsWithWeights():
            if u.estD + wt < v.estD:
                print(170, 199)
                print(172, 200)
                v.estD = u.estD + wt
                print(172, 201)
                v.parent = u
            else:
                print(170, 199)
        unsureVertices.remove(u)


def dijkstraDumb_shortestPaths(w, G):
    print(1, 205)
    dijkstraDumb(w, G)
    for v in G.vertices:
        if v.estD == math.inf:
            print(178, 209)
            continue
        else:
            print(178, 209)
        print(181, 211)
        path = []
        print(181, 212)
        current = v
        while current != w:
            path.append(current)
            print(183, 215)
            current = current.parent
        path.append(current)
        path.reverse()


def dijkstra(w, G):
    print(1, 220)
    for v in G.vertices:
        print(188, 222)
        v.estD = math.inf
    print(189, 223)
    w.estD = 0
    print(189, 224)
    unsureVertices = heapdict.heapdict()
    for v in G.vertices:
        print(191, 226)
        unsureVertices[v] = v.estD
    while len(unsureVertices) > 0:
        print(193, 229)
        u, dist = unsureVertices.popitem()
        if u.estD == math.inf:
            print(193, 230)
            return
        else:
            print(193, 230)
        for v, wt in u.getOutNeighborsWithWeights():
            if u.estD + wt < v.estD:
                print(198, 235)
                print(200, 236)
                v.estD = u.estD + wt
                print(200, 237)
                unsureVertices[v] = u.estD + wt
                print(200, 238)
                v.parent = u
            else:
                print(198, 235)


def dijkstra_shortestPaths(w, G):
    print(1, 241)
    dijkstra(w, G)
    for v in G.vertices:
        if v.estD == math.inf:
            print(206, 245)
            continue
        else:
            print(206, 245)
        print(209, 247)
        path = []
        print(209, 248)
        current = v
        while current != w:
            path.append(current)
            print(211, 251)
            current = current.parent
        path.append(current)
        path.reverse()


def runTrials(myFn, nVals, pFn, numTrials=25):
    print(1, 256)
    print(215, 257)
    nValues = []
    print(215, 258)
    tValues = []
    for n in nVals:
        print(217, 261)
        runtime = 0
        for t in range(numTrials):
            print(220, 263)
            G = randomGraph(n, pFn(n))
            print(220, 264)
            start = time.time()
            myFn(G.vertices[0], G)
            print(220, 266)
            end = time.time()
            print(220, 267)
            runtime += (end - start) * 1000
        print(221, 268)
        runtime = runtime / numTrials
        nValues.append(n)
        tValues.append(runtime)
    return nValues, tValues


def smallFrac(n):
    print(1, 273)
    return float(5 / n)


if __name__ == '__main__':
    print(1, 276)
    loop.start_unroll
    print(228, 279)
    G = randomGraph(5, 0.2)
    BFS_shortestPaths(G.vertices[0], G)
    dijkstraDumb_shortestPaths(G.vertices[0], G)
    print(228, 282)
    G = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    dijkstra_shortestPaths(G.vertices[0], G)
    print(228, 284)
    nValues = [10]
    print(228, 285)
    nDijkstra, tDijkstra = runTrials(dijkstra, nValues, smallFrac)
else:
    print(1, 276)
