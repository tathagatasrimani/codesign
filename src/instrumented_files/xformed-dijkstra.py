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
            print('enter scope 16')
            for u__16, wt__16 in instrument_read(v__15, 'v__15'
                ).getOutNeighborsWithWeights():
                print('enter scope 17')
                instrument_read(ret__17, 'ret__17').append([instrument_read
                    (v__15, 'v__15'), instrument_read(u__16, 'u__16'),
                    instrument_read(wt__16, 'wt__16')])
                print('exit scope 17')
            print('exit scope 16')
        print('exit scope 15')
        return instrument_read(ret__15, 'ret__15')
        print('exit scope 15')


class CS161Graph:

    def __init__(self):
        print('enter scope 18')
        print(1, 83)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print(70, 84)
        instrument_read(self, 'self').vertices = []
        print('exit scope 18')

    def addVertex(self, n):
        print('enter scope 19')
        print(1, 86)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        n__19 = instrument_read(n, 'n')
        write_instrument_read(n__19, 'n__19')
        print('malloc', sys.getsizeof(n__19), 'n__19')
        instrument_read(self, 'self').vertices.append(instrument_read(n__19,
            'n__19'))
        print('exit scope 19')

    def addDiEdge(self, u, v, wt=1):
        print('enter scope 20')
        print(1, 90)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__20 = instrument_read(u, 'u')
        write_instrument_read(u__20, 'u__20')
        print('malloc', sys.getsizeof(u__20), 'u__20')
        v__20 = instrument_read(v, 'v')
        write_instrument_read(v__20, 'v__20')
        print('malloc', sys.getsizeof(v__20), 'v__20')
        wt__20 = instrument_read(wt, 'wt')
        write_instrument_read(wt__20, 'wt__20')
        print('malloc', sys.getsizeof(wt__20), 'wt__20')
        instrument_read(u__20, 'u__20').addOutNeighbor(instrument_read(
            v__20, 'v__20'), wt=wt__20)
        instrument_read(v__20, 'v__20').addInNeighbor(instrument_read(u__20,
            'u__20'), wt=wt__20)
        print('exit scope 20')

    def addBiEdge(self, u, v, wt=1):
        print('enter scope 21')
        print(1, 95)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        u__21 = instrument_read(u, 'u')
        write_instrument_read(u__21, 'u__21')
        print('malloc', sys.getsizeof(u__21), 'u__21')
        v__21 = instrument_read(v, 'v')
        write_instrument_read(v__21, 'v__21')
        print('malloc', sys.getsizeof(v__21), 'v__21')
        wt__21 = instrument_read(wt, 'wt')
        write_instrument_read(wt__21, 'wt__21')
        print('malloc', sys.getsizeof(wt__21), 'wt__21')
        instrument_read(self, 'self').addDiEdge(instrument_read(u__21,
            'u__21'), instrument_read(v__21, 'v__21'), wt=wt__21)
        instrument_read(self, 'self').addDiEdge(instrument_read(v__21,
            'v__21'), instrument_read(u__21, 'u__21'), wt=wt__21)
        print('exit scope 21')

    def getDirEdges(self):
        print('enter scope 22')
        print(1, 101)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        print(82, 102)
        ret__22 = []
        write_instrument_read(ret__22, 'ret__22')
        print('malloc', sys.getsizeof(ret__22), 'ret__22')
        for v__22 in instrument_read(self, 'self').vertices:
            print('enter scope 23')
            for u__23, wt__23 in instrument_read(v__22, 'v__22'
                ).getOutNeighborsWithWeights():
                print('enter scope 24')
                instrument_read(ret__22, 'ret__22').append([instrument_read
                    (v__22, 'v__22'), instrument_read(u__23, 'u__23'),
                    instrument_read(wt__23, 'wt__23')])
                print('exit scope 24')
            print('exit scope 23')
        print('exit scope 22')
        return instrument_read(ret__22, 'ret__22')
        print('exit scope 22')


def randomGraph(n, p, wts=[1]):
    print('enter scope 25')
    print(1, 112)
    n__25 = instrument_read(n, 'n')
    write_instrument_read(n__25, 'n__25')
    print('malloc', sys.getsizeof(n__25), 'n__25')
    p__25 = instrument_read(p, 'p')
    write_instrument_read(p__25, 'p__25')
    print('malloc', sys.getsizeof(p__25), 'p__25')
    wts__25 = instrument_read(wts, 'wts')
    write_instrument_read(wts__25, 'wts__25')
    print('malloc', sys.getsizeof(wts__25), 'wts__25')
    print(91, 113)
    G__25 = CS161Graph()
    write_instrument_read(G__25, 'G__25')
    print('malloc', sys.getsizeof(G__25), 'G__25')
    print(91, 114)
    V__25 = [CS161Vertex(instrument_read(x__25, 'x__25')) for x__25 in
        range(instrument_read(n__25, 'n__25'))]
    write_instrument_read(V__25, 'V__25')
    print('malloc', sys.getsizeof(V__25), 'V__25')
    for v__25 in instrument_read(V__25, 'V__25'):
        print('enter scope 26')
        instrument_read(G__25, 'G__25').addVertex(instrument_read(v__25,
            'v__25'))
        print('exit scope 26')
    for v__25 in instrument_read(V__25, 'V__25'):
        print('enter scope 27')
        for w__27 in instrument_read(V__25, 'V__25'):
            print('enter scope 28')
            if instrument_read(v__25, 'v__25') != instrument_read(w__27,
                'w__27'):
                if random() < instrument_read(p__25, 'p__25'):
                    instrument_read(G__25, 'G__25').addDiEdge(instrument_read
                        (v__25, 'v__25'), instrument_read(w__27, 'w__27'),
                        wt=choice(wts__25))
            print('exit scope 28')
        print('exit scope 27')
    print('exit scope 25')
    return instrument_read(G__25, 'G__25')
    print('exit scope 25')


def BFS(w, G):
    print('enter scope 29')
    print(1, 124)
    w__29 = instrument_read(w, 'w')
    write_instrument_read(w__29, 'w__29')
    print('malloc', sys.getsizeof(w__29), 'w__29')
    G__29 = instrument_read(G, 'G')
    write_instrument_read(G__29, 'G__29')
    print('malloc', sys.getsizeof(G__29), 'G__29')
    for v__29 in instrument_read(G__29, 'G__29').vertices:
        print('enter scope 30')
        print(107, 126)
        instrument_read(v__29, 'v__29').status = 'unvisited'
        print('exit scope 30')
    print(108, 127)
    n__29 = len(instrument_read(G__29, 'G__29').vertices)
    write_instrument_read(n__29, 'n__29')
    print('malloc', sys.getsizeof(n__29), 'n__29')
    print(108, 128)
    Ls__29 = [[] for i__29 in range(instrument_read(n__29, 'n__29'))]
    write_instrument_read(Ls__29, 'Ls__29')
    print('malloc', sys.getsizeof(Ls__29), 'Ls__29')
    print(108, 129)
    Ls__29[0] = [instrument_read(w__29, 'w__29')]
    write_instrument_read_sub(Ls__29, 'Ls__29', 0, None, None, False)
    print(108, 130)
    instrument_read(w__29, 'w__29').status = 'visited'
    for i__29 in range(instrument_read(n__29, 'n__29')):
        print('enter scope 31')
        for u__31 in instrument_read_sub(instrument_read(Ls__29, 'Ls__29'),
            'Ls__29', instrument_read(i__29, 'i__29'), None, None, False):
            print('enter scope 32')
            for v__29 in instrument_read(u__31, 'u__31').getOutNeighbors():
                print('enter scope 33')
                if instrument_read(v__29, 'v__29').status == 'unvisited':
                    print(116, 135)
                    instrument_read(v__29, 'v__29').status = 'visited'
                    print(116, 136)
                    instrument_read(v__29, 'v__29').parent = instrument_read(
                        u__31, 'u__31')
                    instrument_read_sub(instrument_read(Ls__29, 'Ls__29'),
                        'Ls__29', instrument_read(i__29, 'i__29') + 1, None,
                        None, False).append(instrument_read(v__29, 'v__29'))
                print('exit scope 33')
            print('exit scope 32')
        print('exit scope 31')
    print('exit scope 29')
    return instrument_read(Ls__29, 'Ls__29')
    print('exit scope 29')


def BFS_shortestPaths(w, G):
    print('enter scope 34')
    print(1, 140)
    w__34 = instrument_read(w, 'w')
    write_instrument_read(w__34, 'w__34')
    print('malloc', sys.getsizeof(w__34), 'w__34')
    G__34 = instrument_read(G, 'G')
    write_instrument_read(G__34, 'G__34')
    print('malloc', sys.getsizeof(G__34), 'G__34')
    print(121, 141)
    Ls__34 = BFS(instrument_read(w__34, 'w__34'), instrument_read(G__34,
        'G__34'))
    write_instrument_read(Ls__34, 'Ls__34')
    print('malloc', sys.getsizeof(Ls__34), 'Ls__34')
    for i__34 in range(len(instrument_read(Ls__34, 'Ls__34'))):
        print('enter scope 35')
        for w__34 in instrument_read_sub(instrument_read(Ls__34, 'Ls__34'),
            'Ls__34', instrument_read(i__34, 'i__34'), None, None, False):
            print('enter scope 36')
            print(125, 145)
            path__36 = []
            write_instrument_read(path__36, 'path__36')
            print('malloc', sys.getsizeof(path__36), 'path__36')
            print(125, 146)
            current__36 = instrument_read(w__34, 'w__34')
            write_instrument_read(current__36, 'current__36')
            print('malloc', sys.getsizeof(current__36), 'current__36')
            for j__36 in range(instrument_read(i__34, 'i__34')):
                print('enter scope 37')
                instrument_read(path__36, 'path__36').append(instrument_read
                    (current__36, 'current__36'))
                print(128, 149)
                current__36 = instrument_read(current__36, 'current__36'
                    ).parent
                write_instrument_read(current__36, 'current__36')
                print('malloc', sys.getsizeof(current__36), 'current__36')
                print('exit scope 37')
            instrument_read(path__36, 'path__36').append(instrument_read(
                current__36, 'current__36'))
            instrument_read(path__36, 'path__36').reverse()
            print('exit scope 36')
        print('exit scope 35')
    print('exit scope 34')


def dijkstraDumb(w, G):
    print('enter scope 38')
    print(1, 154)
    w__38 = instrument_read(w, 'w')
    write_instrument_read(w__38, 'w__38')
    print('malloc', sys.getsizeof(w__38), 'w__38')
    G__38 = instrument_read(G, 'G')
    write_instrument_read(G__38, 'G__38')
    print('malloc', sys.getsizeof(G__38), 'G__38')
    for v__38 in instrument_read(G__38, 'G__38').vertices:
        print('enter scope 39')
        print(133, 156)
        instrument_read(v__38, 'v__38').estD = instrument_read(math, 'math'
            ).inf
        print('exit scope 39')
    print(134, 157)
    instrument_read(w__38, 'w__38').estD = 0
    print(134, 158)
    unsureVertices__38 = instrument_read_sub(instrument_read(G__38, 'G__38'
        ).vertices, 'G__38.vertices', None, None, None, True)
    write_instrument_read(unsureVertices__38, 'unsureVertices__38')
    print('malloc', sys.getsizeof(unsureVertices__38), 'unsureVertices__38')
    while len(instrument_read(unsureVertices__38, 'unsureVertices__38')) > 0:
        print(136, 161)
        u__38 = None
        write_instrument_read(u__38, 'u__38')
        print('malloc', sys.getsizeof(u__38), 'u__38')
        print(136, 162)
        minD__38 = instrument_read(math, 'math').inf
        write_instrument_read(minD__38, 'minD__38')
        print('malloc', sys.getsizeof(minD__38), 'minD__38')
        for x__38 in instrument_read(unsureVertices__38, 'unsureVertices__38'):
            print('enter scope 40')
            if instrument_read(x__38, 'x__38').estD < instrument_read(minD__38,
                'minD__38'):
                print(141, 165)
                minD__38 = instrument_read(x__38, 'x__38').estD
                write_instrument_read(minD__38, 'minD__38')
                print('malloc', sys.getsizeof(minD__38), 'minD__38')
                print(141, 166)
                u__38 = instrument_read(x__38, 'x__38')
                write_instrument_read(u__38, 'u__38')
                print('malloc', sys.getsizeof(u__38), 'u__38')
            print('exit scope 40')
        if instrument_read(u__38, 'u__38') == None:
            print('exit scope 38')
            return
        for v__38, wt__38 in instrument_read(u__38, 'u__38'
            ).getOutNeighborsWithWeights():
            print('enter scope 41')
            if instrument_read(u__38, 'u__38').estD + instrument_read(wt__38,
                'wt__38') < instrument_read(v__38, 'v__38').estD:
                print(148, 173)
                instrument_read(v__38, 'v__38').estD = instrument_read(u__38,
                    'u__38').estD + instrument_read(wt__38, 'wt__38')
                print(148, 174)
                instrument_read(v__38, 'v__38').parent = instrument_read(u__38,
                    'u__38')
            print('exit scope 41')
        instrument_read(unsureVertices__38, 'unsureVertices__38').remove(
            instrument_read(u__38, 'u__38'))
    print('exit scope 38')


def dijkstraDumb_shortestPaths(w, G):
    print('enter scope 42')
    print(1, 178)
    w__42 = instrument_read(w, 'w')
    write_instrument_read(w__42, 'w__42')
    print('malloc', sys.getsizeof(w__42), 'w__42')
    G__42 = instrument_read(G, 'G')
    write_instrument_read(G__42, 'G__42')
    print('malloc', sys.getsizeof(G__42), 'G__42')
    dijkstraDumb(instrument_read(w__42, 'w__42'), instrument_read(G__42,
        'G__42'))
    for v__42 in instrument_read(G__42, 'G__42').vertices:
        print('enter scope 43')
        if instrument_read(v__42, 'v__42').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(157, 184)
        path__43 = []
        write_instrument_read(path__43, 'path__43')
        print('malloc', sys.getsizeof(path__43), 'path__43')
        print(157, 185)
        current__43 = instrument_read(v__42, 'v__42')
        write_instrument_read(current__43, 'current__43')
        print('malloc', sys.getsizeof(current__43), 'current__43')
        while instrument_read(current__43, 'current__43') != instrument_read(
            w__42, 'w__42'):
            instrument_read(path__43, 'path__43').append(instrument_read(
                current__43, 'current__43'))
            print(159, 188)
            current__43 = instrument_read(current__43, 'current__43').parent
            write_instrument_read(current__43, 'current__43')
            print('malloc', sys.getsizeof(current__43), 'current__43')
        instrument_read(path__43, 'path__43').append(instrument_read(
            current__43, 'current__43'))
        instrument_read(path__43, 'path__43').reverse()
        print('exit scope 43')
    print('exit scope 42')


def dijkstra(w, G):
    print('enter scope 44')
    print(1, 193)
    w__44 = instrument_read(w, 'w')
    write_instrument_read(w__44, 'w__44')
    print('malloc', sys.getsizeof(w__44), 'w__44')
    G__44 = instrument_read(G, 'G')
    write_instrument_read(G__44, 'G__44')
    print('malloc', sys.getsizeof(G__44), 'G__44')
    for v__44 in instrument_read(G__44, 'G__44').vertices:
        print('enter scope 45')
        print(164, 195)
        instrument_read(v__44, 'v__44').estD = instrument_read(math, 'math'
            ).inf
        print('exit scope 45')
    print(165, 196)
    instrument_read(w__44, 'w__44').estD = 0
    print(165, 197)
    unsureVertices__44 = instrument_read(heapdict, 'heapdict').heapdict()
    write_instrument_read(unsureVertices__44, 'unsureVertices__44')
    print('malloc', sys.getsizeof(unsureVertices__44), 'unsureVertices__44')
    for v__44 in instrument_read(G__44, 'G__44').vertices:
        print('enter scope 46')
        print(167, 199)
        unsureVertices__44[instrument_read(instrument_read(v__44, 'v__44'),
            'v__44')] = instrument_read(v__44, 'v__44').estD
        write_instrument_read_sub(unsureVertices__44, 'unsureVertices__44',
            instrument_read(instrument_read(v__44, 'v__44'), 'v__44'), None,
            None, False)
        print('exit scope 46')
    while len(instrument_read(unsureVertices__44, 'unsureVertices__44')) > 0:
        print(169, 202)
        u__44, dist__44 = instrument_read(unsureVertices__44,
            'unsureVertices__44').popitem()
        write_instrument_read(dist__44, 'dist__44')
        print('malloc', sys.getsizeof(dist__44), 'dist__44')
        if instrument_read(u__44, 'u__44').estD == instrument_read(math, 'math'
            ).inf:
            print('exit scope 44')
            return
        for v__44, wt__44 in instrument_read(u__44, 'u__44'
            ).getOutNeighborsWithWeights():
            print('enter scope 47')
            if instrument_read(u__44, 'u__44').estD + instrument_read(wt__44,
                'wt__44') < instrument_read(v__44, 'v__44').estD:
                print(176, 209)
                instrument_read(v__44, 'v__44').estD = instrument_read(u__44,
                    'u__44').estD + instrument_read(wt__44, 'wt__44')
                print(176, 210)
                unsureVertices__44[instrument_read(instrument_read(v__44,
                    'v__44'), 'v__44')] = instrument_read(u__44, 'u__44'
                    ).estD + instrument_read(wt__44, 'wt__44')
                write_instrument_read_sub(unsureVertices__44,
                    'unsureVertices__44', instrument_read(instrument_read(
                    v__44, 'v__44'), 'v__44'), None, None, False)
                print(176, 211)
                instrument_read(v__44, 'v__44').parent = instrument_read(u__44,
                    'u__44')
            print('exit scope 47')
    print('exit scope 44')


def dijkstra_shortestPaths(w, G):
    print('enter scope 48')
    print(1, 214)
    w__48 = instrument_read(w, 'w')
    write_instrument_read(w__48, 'w__48')
    print('malloc', sys.getsizeof(w__48), 'w__48')
    G__48 = instrument_read(G, 'G')
    write_instrument_read(G__48, 'G__48')
    print('malloc', sys.getsizeof(G__48), 'G__48')
    dijkstra(instrument_read(w__48, 'w__48'), instrument_read(G__48, 'G__48'))
    for v__48 in instrument_read(G__48, 'G__48').vertices:
        print('enter scope 49')
        if instrument_read(v__48, 'v__48').estD == instrument_read(math, 'math'
            ).inf:
            continue
        print(185, 220)
        path__49 = []
        write_instrument_read(path__49, 'path__49')
        print('malloc', sys.getsizeof(path__49), 'path__49')
        print(185, 221)
        current__49 = instrument_read(v__48, 'v__48')
        write_instrument_read(current__49, 'current__49')
        print('malloc', sys.getsizeof(current__49), 'current__49')
        while instrument_read(current__49, 'current__49') != instrument_read(
            w__48, 'w__48'):
            instrument_read(path__49, 'path__49').append(instrument_read(
                current__49, 'current__49'))
            print(187, 224)
            current__49 = instrument_read(current__49, 'current__49').parent
            write_instrument_read(current__49, 'current__49')
            print('malloc', sys.getsizeof(current__49), 'current__49')
        instrument_read(path__49, 'path__49').append(instrument_read(
            current__49, 'current__49'))
        instrument_read(path__49, 'path__49').reverse()
        print('exit scope 49')
    print('exit scope 48')


def runTrials(myFn, nVals, pFn, numTrials=25):
    print('enter scope 50')
    print(1, 229)
    myFn__50 = instrument_read(myFn, 'myFn')
    write_instrument_read(myFn__50, 'myFn__50')
    print('malloc', sys.getsizeof(myFn__50), 'myFn__50')
    nVals__50 = instrument_read(nVals, 'nVals')
    write_instrument_read(nVals__50, 'nVals__50')
    print('malloc', sys.getsizeof(nVals__50), 'nVals__50')
    pFn__50 = instrument_read(pFn, 'pFn')
    write_instrument_read(pFn__50, 'pFn__50')
    print('malloc', sys.getsizeof(pFn__50), 'pFn__50')
    numTrials__50 = instrument_read(numTrials, 'numTrials')
    write_instrument_read(numTrials__50, 'numTrials__50')
    print('malloc', sys.getsizeof(numTrials__50), 'numTrials__50')
    print(191, 230)
    nValues__50 = []
    write_instrument_read(nValues__50, 'nValues__50')
    print('malloc', sys.getsizeof(nValues__50), 'nValues__50')
    print(191, 231)
    tValues__50 = []
    write_instrument_read(tValues__50, 'tValues__50')
    print('malloc', sys.getsizeof(tValues__50), 'tValues__50')
    for n__50 in instrument_read(nVals__50, 'nVals__50'):
        print('enter scope 51')
        print(193, 234)
        runtime__51 = 0
        write_instrument_read(runtime__51, 'runtime__51')
        print('malloc', sys.getsizeof(runtime__51), 'runtime__51')
        for t__51 in range(instrument_read(numTrials__50, 'numTrials__50')):
            print('enter scope 52')
            print(196, 236)
            G__52 = randomGraph(instrument_read(n__50, 'n__50'), pFn(
                instrument_read(n__50, 'n__50')))
            write_instrument_read(G__52, 'G__52')
            print('malloc', sys.getsizeof(G__52), 'G__52')
            print(196, 237)
            start__52 = instrument_read(time, 'time').time()
            write_instrument_read(start__52, 'start__52')
            print('malloc', sys.getsizeof(start__52), 'start__52')
            myFn(instrument_read_sub(instrument_read(G__52, 'G__52').
                vertices, 'G__52.vertices', 0, None, None, False),
                instrument_read(G__52, 'G__52'))
            print(196, 239)
            end__52 = instrument_read(time, 'time').time()
            write_instrument_read(end__52, 'end__52')
            print('malloc', sys.getsizeof(end__52), 'end__52')
            print(196, 240)
            runtime__51 += (instrument_read(end__52, 'end__52') -
                instrument_read(start__52, 'start__52')) * 1000
            write_instrument_read(runtime__51, 'runtime__51')
            print('exit scope 52')
        print(197, 241)
        runtime__51 = instrument_read(runtime__51, 'runtime__51'
            ) / instrument_read(numTrials__50, 'numTrials__50')
        write_instrument_read(runtime__51, 'runtime__51')
        print('malloc', sys.getsizeof(runtime__51), 'runtime__51')
        instrument_read(nValues__50, 'nValues__50').append(instrument_read(
            n__50, 'n__50'))
        instrument_read(tValues__50, 'tValues__50').append(instrument_read(
            runtime__51, 'runtime__51'))
        print('exit scope 51')
    print('exit scope 50')
    return instrument_read(nValues__50, 'nValues__50'), instrument_read(
        tValues__50, 'tValues__50')
    print('exit scope 50')


def smallFrac(n):
    print('enter scope 53')
    print(1, 246)
    n__53 = instrument_read(n, 'n')
    write_instrument_read(n__53, 'n__53')
    print('malloc', sys.getsizeof(n__53), 'n__53')
    print('exit scope 53')
    return float(5 / instrument_read(n__53, 'n__53'))
    print('exit scope 53')


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
