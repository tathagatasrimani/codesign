import heapdict as heapdict # you will need to install the heapdict package to use this
import math
from random import random
from random import choice
import time
from loop import loop
import numpy as np
# Implementation of directed graphs with weighted edges

class CS161Vertex:
    def __init__(self, v):
        self.inNeighbors = [] # list of pairs (nbr, wt), where nbr is a CS161Vertex and wt is a weight
        self.outNeighbors = [] # same as above
        self.value = v
        # useful for DFS/BFS/Dijkstra/Bellman-Ford
        self.inTime = None
        self.outTime = None
        self.status = "unvisited"
        self.parent = None
        self.estD = math.inf
        
    def hasOutNeighbor(self,v):
        if v in self.getOutNeighbors():
            return True
        return False
        
    def hasInNeighbor(self,v):
        if v in self.getInNeighbors():
            return True
        return False
    
    def hasNeighbor(self,v):
        if v in self.getInNeighbors() or v in self.getOutNeighbors():
            return True
        return False
    
    def getOutNeighbors(self):
        return [ v[0] for v in self.outNeighbors ]
    
    def getInNeighbors(self):
        return [ v[0] for v in self.inNeighbors ]
        
    def getOutNeighborsWithWeights(self):
        return self.outNeighbors
    
    def getInNeighborsWithWeights(self):
        return self.inNeighbors
        
    def addOutNeighbor(self,v,wt):
        self.outNeighbors.append((v,wt))
    
    def addInNeighbor(self,v,wt):
        self.inNeighbors.append((v,wt))

# This is a directed graph class for use in CS161.
# It can also be used as an undirected graph by adding edges in both directions.
class CS161Graph:
    def __init__(self):
        self.vertices = []
        
    def addVertex(self,n):
        self.vertices.append(n)
        
    # add a directed edge from CS161Node u to CS161Node v
    def addDiEdge(self,u,v,wt=1):
        u.addOutNeighbor(v,wt=wt)
        v.addInNeighbor(u,wt=wt)
    
    # add edges in both directions between u and v
    def addBiEdge(self,u,v,wt=1):
        self.addDiEdge(u,v,wt=wt)
        self.addDiEdge(v,u,wt=wt)

    # get a list of all the directed edges
    # directed edges are a list of two vertices and a weight
    def getDirEdges(self):
        ret = []
        for v in self.vertices:
            for u, wt in v.getOutNeighborsWithWeights():
                ret.append( [v,u,wt] )
        return ret

class CS161Graph:
    def __init__(self):
        self.vertices = []
        
    def addVertex(self,n):
        self.vertices.append(n)
        
    # add a directed edge from CS161Node u to CS161Node v
    def addDiEdge(self,u,v,wt=1):
        u.addOutNeighbor(v,wt=wt)
        v.addInNeighbor(u,wt=wt)
    
    # add edges in both directions between u and v
    def addBiEdge(self,u,v,wt=1):
        self.addDiEdge(u,v,wt=wt)
        self.addDiEdge(v,u,wt=wt)

    # get a list of all the directed edges
    # directed edges are a list of two vertices and a weight
    def getDirEdges(self):
        ret = []
        for v in self.vertices:
            for u, wt in v.getOutNeighborsWithWeights():
                ret.append( [v,u,wt] )
        return ret

# make a random graph
# This is G(n,p), where we have n vertices and each (directed) edge is present with probability p.
# if you pass in a set of weights, then the weights are chosen uniformly from that set.
# otherwise all weights are 1
def randomGraph(n,p,wts=[1]):
    G = CS161Graph()
    V = [ CS161Vertex(x) for x in range(n) ]
    for v in V:
        G.addVertex(v)
    for v in V:
        i = 0
        for w in V:
            if v != w:
                if random() < p:
                    G.addDiEdge(v,w,wt=choice(wts))
                    i += 1
            if i > 15: break
    return G

def BFS(w, G):
    for v in G.vertices:
        v.status = "unvisited"
    n = len(G.vertices)
    Ls = [ [] for i in range(n) ]
    Ls[0] = [w]
    w.status = "visited"
    for i in range(n):
        for u in Ls[i]:
            for v in u.getOutNeighbors():
                if v.status == "unvisited":
                    v.status = "visited"
                    v.parent = u 
                    Ls[i+1].append(v)
    return Ls

def BFS_shortestPaths(w,G):
    Ls = BFS(w,G)
    # okay, now what are all the shortest paths?
    for i in range(len(Ls)):
        for w in Ls[i]:
            path = []
            current = w
            for j in range(i):
                path.append(current)
                current = current.parent
            path.append(current)
            path.reverse()

# first let's implement this with an array.
def dijkstraDumb(w,G):
    for v in G.vertices:
        v.estD = math.inf
    w.estD = 0
    unsureVertices = G.vertices[:]
    while len(unsureVertices) > 0:
        # find the u with the minimum estD in the dumbest way possible
        u = None
        minD = math.inf
        for x in unsureVertices:
            if x.estD < minD:
                minD = x.estD
                u = x
        if u == None:
            # then there is nothing more that I can reach
            return
        # update u's neighbors
        for v,wt in u.getOutNeighborsWithWeights():
            if u.estD + wt < v.estD:
                v.estD = u.estD + wt
                v.parent = u
        unsureVertices.remove(u)
    # that's it!  Now each vertex holds estD which is its distance from w

def dijkstraDumb_shortestPaths(w,G):
    dijkstraDumb(w,G)
    # okay, now what are all the shortest paths?
    for v in G.vertices:
        if v.estD == math.inf:
            continue
        path = []
        current = v
        while current != w:
            path.append(current)
            current = current.parent
        path.append(current)
        path.reverse()

# now let's try this with a heap
def dijkstra(w,G):
    for v in G.vertices:
        v.estD = math.inf
    w.estD = 0
    unsureVertices = heapdict.heapdict()
    for v in G.vertices:
        unsureVertices[v] = v.estD
    while len(unsureVertices) > 0:
        # find the u with the minimum estD, using the heap
        u, dist = unsureVertices.popitem() 
        if u.estD == math.inf:
            # then there is nothing more that I can reach
            return
        # update u's neighbors
        for v,wt in u.getOutNeighborsWithWeights():
            if u.estD + wt < v.estD:
                v.estD = u.estD + wt
                unsureVertices[v] = u.estD + wt #update the key in the heapdict
                v.parent = u
    # that's it!  Now each vertex holds estD which is its distance from w

def dijkstra_shortestPaths(w,G):
    dijkstra(w,G)
    # okay, now what are all the shortest paths?
    for v in G.vertices:
        if v.estD == math.inf:
            continue
        path = []
        current = v
        while current != w:
            path.append(current)
            current = current.parent
        path.append(current)
        path.reverse()

# generate a bunch of random graphs and run an alg to compute shortest paths (implicitly)
def runTrials(myFn, nVals, pFn, numTrials=1):
    nValues = []
    tValues = []
    for n in nVals:
        # run myFn several times and average to get a decent idea.
        runtime = 0
        for t in range(numTrials):
            G = randomGraph(n*10000, pFn(n))  #Random graph on n vertices with about pn^2 edges
            start = time.time()
            myFn( G.vertices[0], G ) 
            end = time.time()
            runtime += (end - start) * 1000 # measure in milliseconds
        runtime = runtime/numTrials
        nValues.append(n)
        tValues.append(runtime)
    return nValues, tValues

def smallFrac(n):
    return float(5/n)

def read_random_graph_from_file(n, p, wts=[1]):
    return randomGraph(n, p, wts)

def main():
    # test on a random graph
    G = randomGraph(20, 0.3)
    # BFS_shortestPaths(G.vertices[0],G)
    # dijkstraDumb_shortestPaths(G.vertices[0], G)
    # G = randomGraph(5,.4,[1,2,3,4,5])
    dijkstra_shortestPaths(G.vertices[0], G)
    # nValues = [10]
    # nDijkstra, tDijkstra = runTrials(BFS, nValues,smallFrac)

if __name__ == "__main__":
    main()
