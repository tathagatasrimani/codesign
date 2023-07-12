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
        v__1 = instrument_read(v, 'v')
        write_instrument_read(v__1, 'v__1')
        print('malloc', sys.getsizeof(v__1), 'v__1')
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
        v__2 = instrument_read(v, 'v')
        write_instrument_read(v__2, 'v__2')
        print('malloc', sys.getsizeof(v__2), 'v__2')
        if instrument_read(v__2, 'v__2') in instrument_read(self, 'self'
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
        v__3 = instrument_read(v, 'v')
        write_instrument_read(v__3, 'v__3')
        print('malloc', sys.getsizeof(v__3), 'v__3')
        if instrument_read(v__3, 'v__3') in instrument_read(self, 'self'
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
        v__4 = instrument_read(v, 'v')
        write_instrument_read(v__4, 'v__4')
        print('malloc', sys.getsizeof(v__4), 'v__4')
        if instrument_read(v__4, 'v__4') in instrument_read(self, 'self'
            ).getInNeighbors() or instrument_read(v__4, 'v__4'
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
        return [instrument_read_sub(instrument_read(v__5, 'v__5'), 'v__5', 
            0, None, None, False) for v__5 in instrument_read(self, 'self')
            .outNeighbors]
        print('exit scope 5')

    def getInNeighbors(self):
        print('enter scope 6')
        print(1, 39)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print('exit scope 6')
        return [instrument_read_sub(instrument_read(v__6, 'v__6'), 'v__6', 
            0, None, None, False) for v__6 in instrument_read(self, 'self')
            .inNeighbors]
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
        v__9 = instrument_read(v, 'v')
        write_instrument_read(v__9, 'v__9')
        print('malloc', sys.getsizeof(v__9), 'v__9')
        wt__9 = instrument_read(wt, 'wt')
        write_instrument_read(wt__9, 'wt__9')
        print('malloc', sys.getsizeof(wt__9), 'wt__9')
        instrument_read(self, 'self').outNeighbors.append((instrument_read(
            v__9, 'v__9'), instrument_read(wt__9, 'wt__9')))
        print('exit scope 9')

    def addInNeighbor(self, v, wt):
        print('enter scope 10')
        print(1, 51)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        v__10 = instrument_read(v, 'v')
        write_instrument_read(v__10, 'v__10')
        print('malloc', sys.getsizeof(v__10), 'v__10')
        wt__10 = instrument_read(wt, 'wt')
        write_instrument_read(wt__10, 'wt__10')
        print('malloc', sys.getsizeof(wt__10), 'wt__10')
        instrument_read(self, 'self').inNeighbors.append((instrument_read(
            v__10, 'v__10'), instrument_read(wt__10, 'wt__10')))
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
        n__12 = instrument_read(n, 'n')
        write_instrument_read(n__12, 'n__12')
        print('malloc', sys.getsizeof(n__12), 'n__12')
        instrument_read(self, 'self').vertices.append(instrument_read(n__12,
            'n__12'))
        print('exit scope 12')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 13')
        print(1, 64)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__13 = instrument_read(u, 'u')
        write_instrument_read(u__13, 'u__13')
        print('malloc', sys.getsizeof(u__13), 'u__13')
        v__13 = instrument_read(v, 'v')
        write_instrument_read(v__13, 'v__13')
        print('malloc', sys.getsizeof(v__13), 'v__13')
        wt__13 = instrument_read(wt, 'wt')
        write_instrument_read(wt__13, 'wt__13')
        print('malloc', sys.getsizeof(wt__13), 'wt__13')
        instrument_read(u__13, 'u__13').addOutNeighbor(instrument_read(
            v__13, 'v__13'), wt=wt__13)
        instrument_read(v__13, 'v__13').addInNeighbor(instrument_read(u__13,
            'u__13'), wt=wt__13)
        print('exit scope 13')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 14')
        print(1, 69)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__14 = instrument_read(u, 'u')
        write_instrument_read(u__14, 'u__14')
        print('malloc', sys.getsizeof(u__14), 'u__14')
        v__14 = instrument_read(v, 'v')
        write_instrument_read(v__14, 'v__14')
        print('malloc', sys.getsizeof(v__14), 'v__14')
        wt__14 = instrument_read(wt, 'wt')
        write_instrument_read(wt__14, 'wt__14')
        print('malloc', sys.getsizeof(wt__14), 'wt__14')
        instrument_read(self, 'self').addDiEdge(instrument_read(u__14,
            'u__14'), instrument_read(v__14, 'v__14'), wt=wt__14)
        instrument_read(self, 'self').addDiEdge(instrument_read(v__14,
            'v__14'), instrument_read(u__14, 'u__14'), wt=wt__14)
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
        for v__15 in instrument_read(self, 'self').vertices:
            for u__15, wt__15 in instrument_read(v__15, 'v__15'
                ).getOutNeighborsWithWeights():
                instrument_read(ret__15, 'ret__15').append([instrument_read
                    (v__15, 'v__15'), instrument_read(u__15, 'u__15'),
                    instrument_read(wt__15, 'wt__15')])
        print('exit scope 15')
        return instrument_read(ret__15, 'ret__15')
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
        n__17 = instrument_read(n, 'n')
        write_instrument_read(n__17, 'n__17')
        print('malloc', sys.getsizeof(n__17), 'n__17')
        instrument_read(self, 'self').vertices.append(instrument_read(n__17,
            'n__17'))
        print('exit scope 17')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 18')
        print(1, 90)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__18 = instrument_read(u, 'u')
        write_instrument_read(u__18, 'u__18')
        print('malloc', sys.getsizeof(u__18), 'u__18')
        v__18 = instrument_read(v, 'v')
        write_instrument_read(v__18, 'v__18')
        print('malloc', sys.getsizeof(v__18), 'v__18')
        wt__18 = instrument_read(wt, 'wt')
        write_instrument_read(wt__18, 'wt__18')
        print('malloc', sys.getsizeof(wt__18), 'wt__18')
        instrument_read(u__18, 'u__18').addOutNeighbor(instrument_read(
            v__18, 'v__18'), wt=wt__18)
        instrument_read(v__18, 'v__18').addInNeighbor(instrument_read(u__18,
            'u__18'), wt=wt__18)
        print('exit scope 18')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 19')
        print(1, 95)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__19 = instrument_read(u, 'u')
        write_instrument_read(u__19, 'u__19')
        print('malloc', sys.getsizeof(u__19), 'u__19')
        v__19 = instrument_read(v, 'v')
        write_instrument_read(v__19, 'v__19')
        print('malloc', sys.getsizeof(v__19), 'v__19')
        wt__19 = instrument_read(wt, 'wt')
        write_instrument_read(wt__19, 'wt__19')
        print('malloc', sys.getsizeof(wt__19), 'wt__19')
        instrument_read(self, 'self').addDiEdge(instrument_read(u__19,
            'u__19'), instrument_read(v__19, 'v__19'), wt=wt__19)
        instrument_read(self, 'self').addDiEdge(instrument_read(v__19,
            'v__19'), instrument_read(u__19, 'u__19'), wt=wt__19)
        print('exit scope 19')

    def getDirEdges(self):
        print('enter scope 20')
        print(1, 101)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print(82, 102)
        ret__20 = []
        write_instrument_read(ret__20, 'ret__20')
        print('malloc', sys.getsizeof(ret__20), 'ret__20')
        for v__20 in instrument_read(self, 'self').vertices:
            for u__20, wt__20 in instrument_read(v__20, 'v__20'
                ).getOutNeighborsWithWeights():
                instrument_read(ret__20, 'ret__20').append([instrument_read
                    (v__20, 'v__20'), instrument_read(u__20, 'u__20'),
                    instrument_read(wt__20, 'wt__20')])
        print('exit scope 20')
        return instrument_read(ret__20, 'ret__20')
        print('exit scope 20')


def randomGraph(n, p, wts=[1]):
    print('enter scope 21')
    print(1, 112)
    n__21 = instrument_read(n, 'n')
    write_instrument_read(n__21, 'n__21')
    print('malloc', sys.getsizeof(n__21), 'n__21')
    p__21 = instrument_read(p, 'p')
    write_instrument_read(p__21, 'p__21')
    print('malloc', sys.getsizeof(p__21), 'p__21')
    wts__21 = instrument_read(wts, 'wts')
    write_instrument_read(wts__21, 'wts__21')
    print('malloc', sys.getsizeof(wts__21), 'wts__21')
    print(91, 113)
    G__21 = CS161Graph()
    write_instrument_read(G__21, 'G__21')
    print('malloc', sys.getsizeof(G__21), 'G__21')
    print(91, 114)
    V__21 = [CS161Vertex(instrument_read(x__21, 'x__21')) for x__21 in
        range(instrument_read(n__21, 'n__21'))]
    write_instrument_read(V__21, 'V__21')
    print('malloc', sys.getsizeof(V__21), 'V__21')
    for v__21 in instrument_read(V__21, 'V__21'):
        instrument_read(G__21, 'G__21').addVertex(instrument_read(v__21,
            'v__21'))
    for v__21 in instrument_read(V__21, 'V__21'):
        for w__21 in instrument_read(V__21, 'V__21'):
            if instrument_read(v__21, 'v__21') != instrument_read(w__21,
                'w__21'):
                if random() < instrument_read(p__21, 'p__21'):
                    instrument_read(G__21, 'G__21').addDiEdge(instrument_read
                        (v__21, 'v__21'), instrument_read(w__21, 'w__21'),
                        wt=choice(wts__21))
    print('exit scope 21')
    return instrument_read(G__21, 'G__21')
    print('exit scope 21')


def BFS(w, G):
    print('enter scope 22')
    print(1, 124)
    w__22 = instrument_read(w, 'w')
    write_instrument_read(w__22, 'w__22')
    print('malloc', sys.getsizeof(w__22), 'w__22')
    G__22 = instrument_read(G, 'G')
    write_instrument_read(G__22, 'G__22')
    print('malloc', sys.getsizeof(G__22), 'G__22')
    for v__22 in instrument_read(G__22, 'G__22').vertices:
        print(107, 126)
        instrument_read(v__22, 'v__22').status = 'unvisited'
    print(108, 127)
    n__22 = len(instrument_read(G__22, 'G__22').vertices)
    write_instrument_read(n__22, 'n__22')
    print('malloc', sys.getsizeof(n__22), 'n__22')
    print(108, 128)
    Ls__22 = [[] for i__22 in range(instrument_read(n__22, 'n__22'))]
    write_instrument_read(Ls__22, 'Ls__22')
    print('malloc', sys.getsizeof(Ls__22), 'Ls__22')
    print(108, 129)
    Ls__22[0] = [instrument_read(w__22, 'w__22')]
    write_instrument_read_sub(Ls__22, 'Ls__22', 0, None, None, False)
    print(108, 130)
    instrument_read(w__22, 'w__22').status = 'visited'
    for i__22 in range(instrument_read(n__22, 'n__22')):
        for u__22 in instrument_read_sub(instrument_read(Ls__22, 'Ls__22'),
            'Ls__22', instrument_read(i__22, 'i__22'), None, None, False):
            for v__22 in instrument_read(u__22, 'u__22').getOutNeighbors():
                if instrument_read(v__22, 'v__22').status == 'unvisited':
                    print(116, 135)
                    instrument_read(v__22, 'v__22').status = 'visited'
                    print(116, 136)
                    instrument_read(v__22, 'v__22').parent = instrument_read(
                        u__22, 'u__22')
                    instrument_read_sub(instrument_read(Ls__22, 'Ls__22'),
                        'Ls__22', instrument_read(i__22, 'i__22') + 1, None,
                        None, False).append(instrument_read(v__22, 'v__22'))
    print('exit scope 22')
    return instrument_read(Ls__22, 'Ls__22')
    print('exit scope 22')


def BFS_shortestPaths(w, G):
    print('enter scope 23')
    print(1, 140)
    w__23 = instrument_read(w, 'w')
    write_instrument_read(w__23, 'w__23')
    print('malloc', sys.getsizeof(w__23), 'w__23')
    G__23 = instrument_read(G, 'G')
    write_instrument_read(G__23, 'G__23')
    print('malloc', sys.getsizeof(G__23), 'G__23')
    print(121, 141)
    Ls__23 = BFS(instrument_read(w__23, 'w__23'), instrument_read(G__23,
        'G__23'))
    write_instrument_read(Ls__23, 'Ls__23')
    print('malloc', sys.getsizeof(Ls__23), 'Ls__23')
    for i__23 in range(len(instrument_read(Ls__23, 'Ls__23'))):
        for w__23 in instrument_read_sub(instrument_read(Ls__23, 'Ls__23'),
            'Ls__23', instrument_read(i__23, 'i__23'), None, None, False):
            print(125, 145)
            path__23 = []
            write_instrument_read(path__23, 'path__23')
            print('malloc', sys.getsizeof(path__23), 'path__23')
            print(125, 146)
            current__23 = instrument_read(w__23, 'w__23')
            write_instrument_read(current__23, 'current__23')
            print('malloc', sys.getsizeof(current__23), 'current__23')
            for j__23 in range(instrument_read(i__23, 'i__23')):
                instrument_read(path__23, 'path__23').append(instrument_read
                    (current__23, 'current__23'))
                print(128, 149)
                current__23 = instrument_read(current__23, 'current__23'
                    ).parent
                write_instrument_read(current__23, 'current__23')
                print('malloc', sys.getsizeof(current__23), 'current__23')
            instrument_read(path__23, 'path__23').append(instrument_read(
                current__23, 'current__23'))
            instrument_read(path__23, 'path__23').reverse()
    print('exit scope 23')


def dijkstraDumb(w, G):
    print('enter scope 24')
    print(1, 154)
    w__24 = instrument_read(w, 'w')
    write_instrument_read(w__24, 'w__24')
    print('malloc', sys.getsizeof(w__24), 'w__24')
    G__24 = instrument_read(G, 'G')
    write_instrument_read(G__24, 'G__24')
    print('malloc', sys.getsizeof(G__24), 'G__24')
    for v__24 in instrument_read(G__24, 'G__24').vertices:
        print(133, 156)
        instrument_read(v__24, 'v__24').estD = instrument_read(math, 'math'
            ).inf
    print(134, 157)
    instrument_read(w__24, 'w__24').estD = 0
    print(134, 158)
    unsureVertices__24 = instrument_read_sub(instrument_read(G__24, 'G__24'
        ).vertices, 'G__24.vertices', None, None, None, True)
    write_instrument_read(unsureVertices__24, 'unsureVertices__24')
    print('malloc', sys.getsizeof(unsureVertices__24), 'unsureVertices__24')
    while len(instrument_read(unsureVertices__24, 'unsureVertices__24')) > 0:
        print(136, 161)
        u__24 = None
        write_instrument_read(u__24, 'u__24')
        print('malloc', sys.getsizeof(u__24), 'u__24')
        print(136, 162)
        minD__24 = instrument_read(math, 'math').inf
        write_instrument_read(minD__24, 'minD__24')
        print('malloc', sys.getsizeof(minD__24), 'minD__24')
        for x__24 in instrument_read(unsureVertices__24, 'unsureVertices__24'):
            if instrument_read(x__24, 'x__24').estD < instrument_read(minD__24,
                'minD__24'):
                print(141, 165)
                minD__24 = instrument_read(x__24, 'x__24').estD
                write_instrument_read(minD__24, 'minD__24')
                print('malloc', sys.getsizeof(minD__24), 'minD__24')
                print(141, 166)
                u__24 = instrument_read(x__24, 'x__24')
                write_instrument_read(u__24, 'u__24')
                print('malloc', sys.getsizeof(u__24), 'u__24')
        if instrument_read(u__24, 'u__24') == None:
            print('exit scope 24')
            return
        for v__24, wt__24 in instrument_read(u__24, 'u__24'
            ).getOutNeighborsWithWeights():
            if instrument_read(u__24, 'u__24').estD + instrument_read(wt__24,
                'wt__24') < instrument_read(v__24, 'v__24').estD:
                print(148, 173)
                instrument_read(v__24, 'v__24').estD = instrument_read(u__24,
                    'u__24').estD + instrument_read(wt__24, 'wt__24')
                print(148, 174)
                instrument_read(v__24, 'v__24').parent = instrument_read(u__24,
                    'u__24')
        instrument_read(unsureVertices__24, 'unsureVertices__24').remove(
            instrument_read(u__24, 'u__24'))
    print('exit scope 24')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 25')
    print(1, 178)
    w__25 = instrument_read(w, 'w')
    write_instrument_read(w__25, 'w__25')
    print('malloc', sys.getsizeof(w__25), 'w__25')
    G__25 = instrument_read(G, 'G')
    write_instrument_read(G__25, 'G__25')
    print('malloc', sys.getsizeof(G__25), 'G__25')
    dijkstraDumb(instrument_read(w__25, 'w__25'), instrument_read(G__25,
        'G__25'))
    for v__25 in instrument_read(G__25, 'G__25').vertices:
        if instrument_read(v__25, 'v__25').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(157, 184)
        path__25 = []
        write_instrument_read(path__25, 'path__25')
        print('malloc', sys.getsizeof(path__25), 'path__25')
        print(157, 185)
        current__25 = instrument_read(v__25, 'v__25')
        write_instrument_read(current__25, 'current__25')
        print('malloc', sys.getsizeof(current__25), 'current__25')
        while instrument_read(current__25, 'current__25') != instrument_read(
            w__25, 'w__25'):
            instrument_read(path__25, 'path__25').append(instrument_read(
                current__25, 'current__25'))
            print(159, 188)
            current__25 = instrument_read(current__25, 'current__25').parent
            write_instrument_read(current__25, 'current__25')
            print('malloc', sys.getsizeof(current__25), 'current__25')
        instrument_read(path__25, 'path__25').append(instrument_read(
            current__25, 'current__25'))
        instrument_read(path__25, 'path__25').reverse()
    print('exit scope 25')


def dijkstra(w, G):
    print('enter scope 26')
    print(1, 193)
    w__26 = instrument_read(w, 'w')
    write_instrument_read(w__26, 'w__26')
    print('malloc', sys.getsizeof(w__26), 'w__26')
    G__26 = instrument_read(G, 'G')
    write_instrument_read(G__26, 'G__26')
    print('malloc', sys.getsizeof(G__26), 'G__26')
    for v__26 in instrument_read(G__26, 'G__26').vertices:
        print(164, 195)
        instrument_read(v__26, 'v__26').estD = instrument_read(math, 'math'
            ).inf
    print(165, 196)
    instrument_read(w__26, 'w__26').estD = 0
    print(165, 197)
    unsureVertices__26 = instrument_read(heapdict, 'heapdict').heapdict()
    write_instrument_read(unsureVertices__26, 'unsureVertices__26')
    print('malloc', sys.getsizeof(unsureVertices__26), 'unsureVertices__26')
    for v__26 in instrument_read(G__26, 'G__26').vertices:
        print(167, 199)
        unsureVertices__26[instrument_read(instrument_read(v__26, 'v__26'),
            'v__26')] = instrument_read(v__26, 'v__26').estD
        write_instrument_read_sub(unsureVertices__26, 'unsureVertices__26',
            instrument_read(instrument_read(v__26, 'v__26'), 'v__26'), None,
            None, False)
    while len(instrument_read(unsureVertices__26, 'unsureVertices__26')) > 0:
        print(169, 202)
        u__26, dist__26 = instrument_read(unsureVertices__26,
            'unsureVertices__26').popitem()
        write_instrument_read(dist__26, 'dist__26')
        print('malloc', sys.getsizeof(dist__26), 'dist__26')
        if instrument_read(u__26, 'u__26').estD == instrument_read(math, 'math'
            ).inf:
            print('exit scope 26')
            return
        for v__26, wt__26 in instrument_read(u__26, 'u__26'
            ).getOutNeighborsWithWeights():
            if instrument_read(u__26, 'u__26').estD + instrument_read(wt__26,
                'wt__26') < instrument_read(v__26, 'v__26').estD:
                print(176, 209)
                instrument_read(v__26, 'v__26').estD = instrument_read(u__26,
                    'u__26').estD + instrument_read(wt__26, 'wt__26')
                print(176, 210)
                unsureVertices__26[instrument_read(instrument_read(v__26,
                    'v__26'), 'v__26')] = instrument_read(u__26, 'u__26'
                    ).estD + instrument_read(wt__26, 'wt__26')
                write_instrument_read_sub(unsureVertices__26,
                    'unsureVertices__26', instrument_read(instrument_read(
                    v__26, 'v__26'), 'v__26'), None, None, False)
                print(176, 211)
                instrument_read(v__26, 'v__26').parent = instrument_read(u__26,
                    'u__26')
    print('exit scope 26')


def dijkstra_shortestPaths(w, G):
    print('enter scope 27')
    print(1, 214)
    w__27 = instrument_read(w, 'w')
    write_instrument_read(w__27, 'w__27')
    print('malloc', sys.getsizeof(w__27), 'w__27')
    G__27 = instrument_read(G, 'G')
    write_instrument_read(G__27, 'G__27')
    print('malloc', sys.getsizeof(G__27), 'G__27')
    dijkstra(instrument_read(w__27, 'w__27'), instrument_read(G__27, 'G__27'))
    for v__27 in instrument_read(G__27, 'G__27').vertices:
        if instrument_read(v__27, 'v__27').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(185, 220)
        path__27 = []
        write_instrument_read(path__27, 'path__27')
        print('malloc', sys.getsizeof(path__27), 'path__27')
        print(185, 221)
        current__27 = instrument_read(v__27, 'v__27')
        write_instrument_read(current__27, 'current__27')
        print('malloc', sys.getsizeof(current__27), 'current__27')
        while instrument_read(current__27, 'current__27') != instrument_read(
            w__27, 'w__27'):
            instrument_read(path__27, 'path__27').append(instrument_read(
                current__27, 'current__27'))
            print(187, 224)
            current__27 = instrument_read(current__27, 'current__27').parent
            write_instrument_read(current__27, 'current__27')
            print('malloc', sys.getsizeof(current__27), 'current__27')
        instrument_read(path__27, 'path__27').append(instrument_read(
            current__27, 'current__27'))
        instrument_read(path__27, 'path__27').reverse()
    print('exit scope 27')


def runTrials(myFn, nVals, pFn, numTrials=25):
    print('enter scope 28')
    print(1, 229)
    myFn__28 = instrument_read(myFn, 'myFn')
    write_instrument_read(myFn__28, 'myFn__28')
    print('malloc', sys.getsizeof(myFn__28), 'myFn__28')
    nVals__28 = instrument_read(nVals, 'nVals')
    write_instrument_read(nVals__28, 'nVals__28')
    print('malloc', sys.getsizeof(nVals__28), 'nVals__28')
    pFn__28 = instrument_read(pFn, 'pFn')
    write_instrument_read(pFn__28, 'pFn__28')
    print('malloc', sys.getsizeof(pFn__28), 'pFn__28')
    numTrials__28 = instrument_read(numTrials, 'numTrials')
    write_instrument_read(numTrials__28, 'numTrials__28')
    print('malloc', sys.getsizeof(numTrials__28), 'numTrials__28')
    print(191, 230)
    nValues__28 = []
    write_instrument_read(nValues__28, 'nValues__28')
    print('malloc', sys.getsizeof(nValues__28), 'nValues__28')
    print(191, 231)
    tValues__28 = []
    write_instrument_read(tValues__28, 'tValues__28')
    print('malloc', sys.getsizeof(tValues__28), 'tValues__28')
    for n__28 in instrument_read(nVals__28, 'nVals__28'):
        print(193, 234)
        runtime__28 = 0
        write_instrument_read(runtime__28, 'runtime__28')
        print('malloc', sys.getsizeof(runtime__28), 'runtime__28')
        for t__28 in range(instrument_read(numTrials__28, 'numTrials__28')):
            print(196, 236)
            G__28 = randomGraph(instrument_read(n__28, 'n__28'), pFn(
                instrument_read(n__28, 'n__28')))
            write_instrument_read(G__28, 'G__28')
            print('malloc', sys.getsizeof(G__28), 'G__28')
            print(196, 237)
            start__28 = instrument_read(time, 'time').time()
            write_instrument_read(start__28, 'start__28')
            print('malloc', sys.getsizeof(start__28), 'start__28')
            myFn(instrument_read_sub(instrument_read(G__28, 'G__28').
                vertices, 'G__28.vertices', 0, None, None, False),
                instrument_read(G__28, 'G__28'))
            print(196, 239)
            end__28 = instrument_read(time, 'time').time()
            write_instrument_read(end__28, 'end__28')
            print('malloc', sys.getsizeof(end__28), 'end__28')
            print(196, 240)
            runtime__28 += (instrument_read(end__28, 'end__28') -
                instrument_read(start__28, 'start__28')) * 1000
            write_instrument_read(runtime__28, 'runtime__28')
        print(197, 241)
        runtime__28 = instrument_read(runtime__28, 'runtime__28'
            ) / instrument_read(numTrials__28, 'numTrials__28')
        write_instrument_read(runtime__28, 'runtime__28')
        print('malloc', sys.getsizeof(runtime__28), 'runtime__28')
        instrument_read(nValues__28, 'nValues__28').append(instrument_read(
            n__28, 'n__28'))
        instrument_read(tValues__28, 'tValues__28').append(instrument_read(
            runtime__28, 'runtime__28'))
    print('exit scope 28')
    return instrument_read(nValues__28, 'nValues__28'), instrument_read(
        tValues__28, 'tValues__28')
    print('exit scope 28')


def smallFrac(n):
    print('enter scope 29')
    print(1, 246)
    n__29 = instrument_read(n, 'n')
    write_instrument_read(n__29, 'n__29')
    print('malloc', sys.getsizeof(n__29), 'n__29')
    print('exit scope 29')
    return float(5 / instrument_read(n__29, 'n__29'))
    print('exit scope 29')


if instrument_read(__name__, '__name__') == '__main__':
    instrument_read(loop, 'loop').start_unroll
    print(204, 252)
    G__0 = randomGraph(5, 0.2)
    write_instrument_read(G__0, 'G__0')
    print('malloc', sys.getsizeof(G__0), 'G__0')
    BFS_shortestPaths(instrument_read_sub(instrument_read(G__0, 'G__0').
        vertices, 'G__0.vertices', 0, None, None, False), instrument_read(
        G__0, 'G__0'))
    dijkstraDumb_shortestPaths(instrument_read_sub(instrument_read(G__0,
        'G__0').vertices, 'G__0.vertices', 0, None, None, False),
        instrument_read(G__0, 'G__0'))
    print(204, 255)
    G__0 = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
    write_instrument_read(G__0, 'G__0')
    print('malloc', sys.getsizeof(G__0), 'G__0')
    dijkstra_shortestPaths(instrument_read_sub(instrument_read(G__0, 'G__0'
        ).vertices, 'G__0.vertices', 0, None, None, False), instrument_read
        (G__0, 'G__0'))
    print(204, 257)
    nValues__0 = [10]
    write_instrument_read(nValues__0, 'nValues__0')
    print('malloc', sys.getsizeof(nValues__0), 'nValues__0')
    print(204, 258)
    nDijkstra__0, tDijkstra__0 = runTrials(instrument_read(dijkstra,
        'dijkstra'), instrument_read(nValues__0, 'nValues__0'),
        instrument_read(smallFrac, 'smallFrac'))
    write_instrument_read(tDijkstra__0, 'tDijkstra__0')
    print('malloc', sys.getsizeof(tDijkstra__0), 'tDijkstra__0')
