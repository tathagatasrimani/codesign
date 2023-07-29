digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop
def __init__(self, v):...
def hasOutNeighbor(self, v):...
def hasInNeighbor(self, v):...
def hasNeighbor(self, v):...
def getOutNeighbors(self):...
def getInNeighbors(self):...
def getOutNeighborsWithWeights(self):...
def getInNeighborsWithWeights(self):...
def addOutNeighbor(self, v, wt):...
def addInNeighbor(self, v, wt):...
def __init__(self):...
def addVertex(self, n):...
def addDiEdge(self, u, v, wt=1):...
def addBiEdge(self, u, v, wt=1):...
def getDirEdges(self):...
def __init__(self):...
def addVertex(self, n):...
def addDiEdge(self, u, v, wt=1):...
def addBiEdge(self, u, v, wt=1):...
def getDirEdges(self):...
def randomGraph(n, p, wts=[1]):...
def BFS(w, G):...
def BFS_shortestPaths(w, G):...
def dijkstraDumb(w, G):...
def dijkstraDumb_shortestPaths(w, G):...
def dijkstra(w, G):...
def dijkstra_shortestPaths(w, G):...
def runTrials(myFn, nVals, pFn, numTrials=1):...
def smallFrac(n):...
if __name__ == '__main__':
"]
	207 [label="loop.start_unroll
G = randomGraph(5, 0.2)
BFS_shortestPaths(G.vertices[0], G)
dijkstraDumb_shortestPaths(G.vertices[0], G)
G = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
dijkstra_shortestPaths(G.vertices[0], G)
nValues = [10]
nDijkstra, tDijkstra = runTrials(BFS, nValues, smallFrac)
"]
	"207_calls" [label="randomGraph
BFS_shortestPaths
dijkstraDumb_shortestPaths
randomGraph
dijkstra_shortestPaths
runTrials" shape=box]
	207 -> "207_calls" [label=calls style=dashed]
	1 -> 207 [label="__name__ == '__main__'"]
	subgraph cluster__init__ {
		graph [label=__init__]
		70 [label="self.vertices = []
"]
	}
	subgraph clusterhasOutNeighbor {
		graph [label=hasOutNeighbor]
		6 [label="if v in self.getOutNeighbors():
"]
		7 [label="return True
"]
		6 -> 7 [label="v in self.getOutNeighbors()"]
		8 [label="return False
"]
		6 -> 8 [label="(v not in self.getOutNeighbors())"]
	}
	subgraph clusterhasInNeighbor {
		graph [label=hasInNeighbor]
		13 [label="if v in self.getInNeighbors():
"]
		14 [label="return True
"]
		13 -> 14 [label="v in self.getInNeighbors()"]
		15 [label="return False
"]
		13 -> 15 [label="(v not in self.getInNeighbors())"]
	}
	subgraph clusterhasNeighbor {
		graph [label=hasNeighbor]
		20 [label="if v in self.getInNeighbors() or v in self.getOutNeighbors():
"]
		21 [label="return True
"]
		20 -> 21 [label="v in self.getInNeighbors() or v in self.getOutNeighbors()"]
		22 [label="return False
"]
		20 -> 22 [label="(not (v in self.getInNeighbors() or v in self.getOutNeighbors()))"]
	}
	subgraph clustergetOutNeighbors {
		graph [label=getOutNeighbors]
		27 [label="return [v[0] for v in self.outNeighbors]
"]
	}
	subgraph clustergetInNeighbors {
		graph [label=getInNeighbors]
		31 [label="return [v[0] for v in self.inNeighbors]
"]
	}
	subgraph clustergetOutNeighborsWithWeights {
		graph [label=getOutNeighborsWithWeights]
		35 [label="return self.outNeighbors
"]
	}
	subgraph clustergetInNeighborsWithWeights {
		graph [label=getInNeighborsWithWeights]
		39 [label="return self.inNeighbors
"]
	}
	subgraph clusteraddOutNeighbor {
		graph [label=addOutNeighbor]
		43 [label="self.outNeighbors.append((v, wt))
"]
		"43_calls" [label="self.outNeighbors.append" shape=box]
		43 -> "43_calls" [label=calls style=dashed]
	}
	subgraph clusteraddInNeighbor {
		graph [label=addInNeighbor]
		46 [label="self.inNeighbors.append((v, wt))
"]
		"46_calls" [label="self.inNeighbors.append" shape=box]
		46 -> "46_calls" [label=calls style=dashed]
	}
	subgraph clusteraddVertex {
		graph [label=addVertex]
		73 [label="self.vertices.append(n)
"]
		"73_calls" [label="self.vertices.append" shape=box]
		73 -> "73_calls" [label=calls style=dashed]
	}
	subgraph clusteraddDiEdge {
		graph [label=addDiEdge]
		76 [label="u.addOutNeighbor(v, wt=wt)
v.addInNeighbor(u, wt=wt)
"]
		"76_calls" [label="u.addOutNeighbor
v.addInNeighbor" shape=box]
		76 -> "76_calls" [label=calls style=dashed]
	}
	subgraph clusteraddBiEdge {
		graph [label=addBiEdge]
		79 [label="self.addDiEdge(u, v, wt=wt)
self.addDiEdge(v, u, wt=wt)
"]
		"79_calls" [label="self.addDiEdge
self.addDiEdge" shape=box]
		79 -> "79_calls" [label=calls style=dashed]
	}
	subgraph clustergetDirEdges {
		graph [label=getDirEdges]
		82 [label="ret = []
"]
		83 [label="for v in self.vertices:
"]
		84 [label="for u, wt in v.getOutNeighborsWithWeights():
"]
		86 [label="ret.append([v, u, wt])
"]
		"86_calls" [label="ret.append" shape=box]
		86 -> "86_calls" [label=calls style=dashed]
		86 -> 84 [label=""]
		84 -> 86 [label="v.getOutNeighborsWithWeights()"]
		84 -> 83 [label=""]
		83 -> 84 [label="self.vertices"]
		85 [label="return ret
"]
		83 -> 85 [label=""]
		82 -> 83 [label=""]
	}
	subgraph clusterrandomGraph {
		graph [label=randomGraph]
		91 [label="G = CS161Graph()
V = [CS161Vertex(x) for x in range(n)]
"]
		"91_calls" [label="CS161Graph
CS161Vertex
range" shape=box]
		91 -> "91_calls" [label=calls style=dashed]
		92 [label="for v in V:
"]
		93 [label="G.addVertex(v)
"]
		"93_calls" [label="G.addVertex" shape=box]
		93 -> "93_calls" [label=calls style=dashed]
		93 -> 92 [label=""]
		92 -> 93 [label=V]
		94 [label="for v in V:
"]
		95 [label="i = 0
"]
		97 [label="for w in V:
"]
		98 [label="if v != w:
"]
		100 [label="if random() < p:
"]
		102 [label="G.addDiEdge(v, w, wt=choice(wts))
i += 1
"]
		"102_calls" [label="G.addDiEdge" shape=box]
		102 -> "102_calls" [label=calls style=dashed]
		101 [label="if i > 15:
"]
		101 -> 94 [label="i > 15"]
		101 -> 97 [label="(i <= 15)"]
		102 -> 101 [label=""]
		100 -> 102 [label="random() < p"]
		100 -> 101 [label="(random() >= p)"]
		98 -> 100 [label="v != w"]
		98 -> 101 [label="(v == w)"]
		97 -> 98 [label=V]
		97 -> 94 [label=""]
		95 -> 97 [label=""]
		94 -> 95 [label=V]
		96 [label="return G
"]
		94 -> 96 [label=""]
		92 -> 94 [label=""]
		91 -> 92 [label=""]
	}
	subgraph clusterBFS {
		graph [label=BFS]
		109 [label="for v in G.vertices:
"]
		110 [label="v.status = 'unvisited'
"]
		110 -> 109 [label=""]
		109 -> 110 [label="G.vertices"]
		111 [label="n = len(G.vertices)
Ls = [[] for i in range(n)]
Ls[0] = [w]
w.status = 'visited'
"]
		"111_calls" [label="len
range" shape=box]
		111 -> "111_calls" [label=calls style=dashed]
		112 [label="for i in range(n):
"]
		113 [label="for u in Ls[i]:
"]
		115 [label="for v in u.getOutNeighbors():
"]
		117 [label="if v.status == 'unvisited':
"]
		119 [label="v.status = 'visited'
v.parent = u
Ls[i + 1].append(v)
"]
		"119_calls" [label="Ls.append" shape=box]
		119 -> "119_calls" [label=calls style=dashed]
		119 -> 115 [label=""]
		117 -> 119 [label="v.status == 'unvisited'"]
		117 -> 115 [label="(v.status != 'unvisited')"]
		115 -> 117 [label="u.getOutNeighbors()"]
		115 -> 113 [label=""]
		113 -> 115 [label="Ls[i]"]
		113 -> 112 [label=""]
		112 -> 113 [label="range(n)"]
		114 [label="return Ls
"]
		112 -> 114 [label=""]
		111 -> 112 [label=""]
		109 -> 111 [label=""]
	}
	subgraph clusterBFS_shortestPaths {
		graph [label=BFS_shortestPaths]
		124 [label="Ls = BFS(w, G)
"]
		"124_calls" [label=BFS shape=box]
		124 -> "124_calls" [label=calls style=dashed]
		125 [label="for i in range(len(Ls)):
"]
		126 [label="for w in Ls[i]:
"]
		128 [label="path = []
current = w
"]
		130 [label="for j in range(i):
"]
		131 [label="path.append(current)
current = current.parent
"]
		"131_calls" [label="path.append" shape=box]
		131 -> "131_calls" [label=calls style=dashed]
		131 -> 130 [label=""]
		130 -> 131 [label="range(i)"]
		132 [label="path.append(current)
path.reverse()
"]
		"132_calls" [label="path.append
path.reverse" shape=box]
		132 -> "132_calls" [label=calls style=dashed]
		132 -> 126 [label=""]
		130 -> 132 [label=""]
		128 -> 130 [label=""]
		126 -> 128 [label="Ls[i]"]
		126 -> 125 [label=""]
		125 -> 126 [label="range(len(Ls))"]
		124 -> 125 [label=""]
	}
	subgraph clusterdijkstraDumb {
		graph [label=dijkstraDumb]
		135 [label="for v in G.vertices:
"]
		136 [label="v.estD = math.inf
"]
		136 -> 135 [label=""]
		135 -> 136 [label="G.vertices"]
		137 [label="w.estD = 0
unsureVertices = G.vertices[:]
"]
		138 [label="while len(unsureVertices) > 0:
"]
		139 [label="u = None
minD = math.inf
"]
		141 [label="for x in unsureVertices:
"]
		142 [label="if x.estD < minD:
"]
		144 [label="minD = x.estD
u = x
"]
		144 -> 141 [label=""]
		142 -> 144 [label="x.estD < minD"]
		142 -> 141 [label="(x.estD >= minD)"]
		141 -> 142 [label=unsureVertices]
		143 [label="if u == None:
"]
		146 [label=return
]
		143 -> 146 [label="u == None"]
		147 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		149 [label="if u.estD + wt < v.estD:
"]
		151 [label="v.estD = u.estD + wt
v.parent = u
"]
		151 -> 147 [label=""]
		149 -> 151 [label="u.estD + wt < v.estD"]
		149 -> 147 [label="(u.estD + wt >= v.estD)"]
		147 -> 149 [label="u.getOutNeighborsWithWeights()"]
		150 [label="unsureVertices.remove(u)
"]
		"150_calls" [label="unsureVertices.remove" shape=box]
		150 -> "150_calls" [label=calls style=dashed]
		150 -> 138 [label=""]
		147 -> 150 [label=""]
		143 -> 147 [label="(u != None)"]
		141 -> 143 [label=""]
		139 -> 141 [label=""]
		138 -> 139 [label="len(unsureVertices) > 0"]
		137 -> 138 [label=""]
		135 -> 137 [label=""]
	}
	subgraph clusterdijkstraDumb_shortestPaths {
		graph [label=dijkstraDumb_shortestPaths]
		155 [label="dijkstraDumb(w, G)
"]
		"155_calls" [label=dijkstraDumb shape=box]
		155 -> "155_calls" [label=calls style=dashed]
		156 [label="for v in G.vertices:
"]
		157 [label="if v.estD == math.inf:
"]
		160 [label="path = []
current = v
"]
		161 [label="while current != w:
"]
		162 [label="path.append(current)
current = current.parent
"]
		"162_calls" [label="path.append" shape=box]
		162 -> "162_calls" [label=calls style=dashed]
		162 -> 161 [label=""]
		161 -> 162 [label="current != w"]
		163 [label="path.append(current)
path.reverse()
"]
		"163_calls" [label="path.append
path.reverse" shape=box]
		163 -> "163_calls" [label=calls style=dashed]
		163 -> 156 [label=""]
		161 -> 163 [label="(current == w)"]
		160 -> 161 [label=""]
		157 -> 160 [label="(v.estD != math.inf)"]
		157 -> 156 [label="v.estD == math.inf"]
		156 -> 157 [label="G.vertices"]
		155 -> 156 [label=""]
	}
	subgraph clusterdijkstra {
		graph [label=dijkstra]
		166 [label="for v in G.vertices:
"]
		167 [label="v.estD = math.inf
"]
		167 -> 166 [label=""]
		166 -> 167 [label="G.vertices"]
		168 [label="w.estD = 0
unsureVertices = heapdict.heapdict()
"]
		"168_calls" [label="heapdict.heapdict" shape=box]
		168 -> "168_calls" [label=calls style=dashed]
		169 [label="for v in G.vertices:
"]
		170 [label="unsureVertices[v] = v.estD
"]
		170 -> 169 [label=""]
		169 -> 170 [label="G.vertices"]
		171 [label="while len(unsureVertices) > 0:
"]
		172 [label="u, dist = unsureVertices.popitem()
if u.estD == math.inf:
"]
		"172_calls" [label="unsureVertices.popitem" shape=box]
		172 -> "172_calls" [label=calls style=dashed]
		174 [label=return
]
		172 -> 174 [label="u.estD == math.inf"]
		175 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		177 [label="if u.estD + wt < v.estD:
"]
		179 [label="v.estD = u.estD + wt
unsureVertices[v] = u.estD + wt
v.parent = u
"]
		179 -> 175 [label=""]
		177 -> 179 [label="u.estD + wt < v.estD"]
		177 -> 175 [label="(u.estD + wt >= v.estD)"]
		175 -> 177 [label="u.getOutNeighborsWithWeights()"]
		175 -> 171 [label=""]
		172 -> 175 [label="(u.estD != math.inf)"]
		171 -> 172 [label="len(unsureVertices) > 0"]
		169 -> 171 [label=""]
		168 -> 169 [label=""]
		166 -> 168 [label=""]
	}
	subgraph clusterdijkstra_shortestPaths {
		graph [label=dijkstra_shortestPaths]
		183 [label="dijkstra(w, G)
"]
		"183_calls" [label=dijkstra shape=box]
		183 -> "183_calls" [label=calls style=dashed]
		184 [label="for v in G.vertices:
"]
		185 [label="if v.estD == math.inf:
"]
		188 [label="path = []
current = v
"]
		189 [label="while current != w:
"]
		190 [label="path.append(current)
current = current.parent
"]
		"190_calls" [label="path.append" shape=box]
		190 -> "190_calls" [label=calls style=dashed]
		190 -> 189 [label=""]
		189 -> 190 [label="current != w"]
		191 [label="path.append(current)
path.reverse()
"]
		"191_calls" [label="path.append
path.reverse" shape=box]
		191 -> "191_calls" [label=calls style=dashed]
		191 -> 184 [label=""]
		189 -> 191 [label="(current == w)"]
		188 -> 189 [label=""]
		185 -> 188 [label="(v.estD != math.inf)"]
		185 -> 184 [label="v.estD == math.inf"]
		184 -> 185 [label="G.vertices"]
		183 -> 184 [label=""]
	}
	subgraph clusterrunTrials {
		graph [label=runTrials]
		194 [label="nValues = []
tValues = []
"]
		195 [label="for n in nVals:
"]
		196 [label="runtime = 0
"]
		198 [label="for t in range(numTrials):
"]
		199 [label="G = randomGraph(n * 10000, pFn(n))
start = time.time()
myFn(G.vertices[0], G)
end = time.time()
runtime += (end - start) * 1000
"]
		"199_calls" [label="randomGraph
time.time
myFn
time.time" shape=box]
		199 -> "199_calls" [label=calls style=dashed]
		199 -> 198 [label=""]
		198 -> 199 [label="range(numTrials)"]
		200 [label="runtime = runtime / numTrials
nValues.append(n)
tValues.append(runtime)
"]
		"200_calls" [label="nValues.append
tValues.append" shape=box]
		200 -> "200_calls" [label=calls style=dashed]
		200 -> 195 [label=""]
		198 -> 200 [label=""]
		196 -> 198 [label=""]
		195 -> 196 [label=nVals]
		197 [label="return nValues, tValues
"]
		195 -> 197 [label=""]
		194 -> 195 [label=""]
	}
	subgraph clustersmallFrac {
		graph [label=smallFrac]
		204 [label="return float(5 / n)
"]
	}
}
