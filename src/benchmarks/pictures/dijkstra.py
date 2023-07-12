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
def runTrials(myFn, nVals, pFn, numTrials=25):...
def smallFrac(n):...
if __name__ == '__main__':
"]
	204 [label="loop.start_unroll
G = randomGraph(5, 0.2)
BFS_shortestPaths(G.vertices[0], G)
dijkstraDumb_shortestPaths(G.vertices[0], G)
G = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
dijkstra_shortestPaths(G.vertices[0], G)
nValues = [10]
nDijkstra, tDijkstra = runTrials(dijkstra, nValues, smallFrac)
"]
	"204_calls" [label="randomGraph
BFS_shortestPaths
dijkstraDumb_shortestPaths
randomGraph
dijkstra_shortestPaths
runTrials" shape=box]
	204 -> "204_calls" [label=calls style=dashed]
	1 -> 204 [label="__name__ == '__main__'"]
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
		95 [label="for w in V:
"]
		97 [label="if v != w:
"]
		99 [label="if random() < p:
"]
		101 [label="G.addDiEdge(v, w, wt=choice(wts))
"]
		"101_calls" [label="G.addDiEdge" shape=box]
		101 -> "101_calls" [label=calls style=dashed]
		101 -> 95 [label=""]
		99 -> 101 [label="random() < p"]
		99 -> 95 [label="(random() >= p)"]
		97 -> 99 [label="v != w"]
		97 -> 95 [label="(v == w)"]
		95 -> 97 [label=V]
		95 -> 94 [label=""]
		94 -> 95 [label=V]
		96 [label="return G
"]
		94 -> 96 [label=""]
		92 -> 94 [label=""]
		91 -> 92 [label=""]
	}
	subgraph clusterBFS {
		graph [label=BFS]
		106 [label="for v in G.vertices:
"]
		107 [label="v.status = 'unvisited'
"]
		107 -> 106 [label=""]
		106 -> 107 [label="G.vertices"]
		108 [label="n = len(G.vertices)
Ls = [[] for i in range(n)]
Ls[0] = [w]
w.status = 'visited'
"]
		"108_calls" [label="len
range" shape=box]
		108 -> "108_calls" [label=calls style=dashed]
		109 [label="for i in range(n):
"]
		110 [label="for u in Ls[i]:
"]
		112 [label="for v in u.getOutNeighbors():
"]
		114 [label="if v.status == 'unvisited':
"]
		116 [label="v.status = 'visited'
v.parent = u
Ls[i + 1].append(v)
"]
		"116_calls" [label="Ls.append" shape=box]
		116 -> "116_calls" [label=calls style=dashed]
		116 -> 112 [label=""]
		114 -> 116 [label="v.status == 'unvisited'"]
		114 -> 112 [label="(v.status != 'unvisited')"]
		112 -> 114 [label="u.getOutNeighbors()"]
		112 -> 110 [label=""]
		110 -> 112 [label="Ls[i]"]
		110 -> 109 [label=""]
		109 -> 110 [label="range(n)"]
		111 [label="return Ls
"]
		109 -> 111 [label=""]
		108 -> 109 [label=""]
		106 -> 108 [label=""]
	}
	subgraph clusterBFS_shortestPaths {
		graph [label=BFS_shortestPaths]
		121 [label="Ls = BFS(w, G)
"]
		"121_calls" [label=BFS shape=box]
		121 -> "121_calls" [label=calls style=dashed]
		122 [label="for i in range(len(Ls)):
"]
		123 [label="for w in Ls[i]:
"]
		125 [label="path = []
current = w
"]
		127 [label="for j in range(i):
"]
		128 [label="path.append(current)
current = current.parent
"]
		"128_calls" [label="path.append" shape=box]
		128 -> "128_calls" [label=calls style=dashed]
		128 -> 127 [label=""]
		127 -> 128 [label="range(i)"]
		129 [label="path.append(current)
path.reverse()
"]
		"129_calls" [label="path.append
path.reverse" shape=box]
		129 -> "129_calls" [label=calls style=dashed]
		129 -> 123 [label=""]
		127 -> 129 [label=""]
		125 -> 127 [label=""]
		123 -> 125 [label="Ls[i]"]
		123 -> 122 [label=""]
		122 -> 123 [label="range(len(Ls))"]
		121 -> 122 [label=""]
	}
	subgraph clusterdijkstraDumb {
		graph [label=dijkstraDumb]
		132 [label="for v in G.vertices:
"]
		133 [label="v.estD = math.inf
"]
		133 -> 132 [label=""]
		132 -> 133 [label="G.vertices"]
		134 [label="w.estD = 0
unsureVertices = G.vertices[:]
"]
		135 [label="while len(unsureVertices) > 0:
"]
		136 [label="u = None
minD = math.inf
"]
		138 [label="for x in unsureVertices:
"]
		139 [label="if x.estD < minD:
"]
		141 [label="minD = x.estD
u = x
"]
		141 -> 138 [label=""]
		139 -> 141 [label="x.estD < minD"]
		139 -> 138 [label="(x.estD >= minD)"]
		138 -> 139 [label=unsureVertices]
		140 [label="if u == None:
"]
		143 [label=return
]
		140 -> 143 [label="u == None"]
		144 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		146 [label="if u.estD + wt < v.estD:
"]
		148 [label="v.estD = u.estD + wt
v.parent = u
"]
		148 -> 144 [label=""]
		146 -> 148 [label="u.estD + wt < v.estD"]
		146 -> 144 [label="(u.estD + wt >= v.estD)"]
		144 -> 146 [label="u.getOutNeighborsWithWeights()"]
		147 [label="unsureVertices.remove(u)
"]
		"147_calls" [label="unsureVertices.remove" shape=box]
		147 -> "147_calls" [label=calls style=dashed]
		147 -> 135 [label=""]
		144 -> 147 [label=""]
		140 -> 144 [label="(u != None)"]
		138 -> 140 [label=""]
		136 -> 138 [label=""]
		135 -> 136 [label="len(unsureVertices) > 0"]
		134 -> 135 [label=""]
		132 -> 134 [label=""]
	}
	subgraph clusterdijkstraDumb_shortestPaths {
		graph [label=dijkstraDumb_shortestPaths]
		152 [label="dijkstraDumb(w, G)
"]
		"152_calls" [label=dijkstraDumb shape=box]
		152 -> "152_calls" [label=calls style=dashed]
		153 [label="for v in G.vertices:
"]
		154 [label="if v.estD == math.inf:
"]
		157 [label="path = []
current = v
"]
		158 [label="while current != w:
"]
		159 [label="path.append(current)
current = current.parent
"]
		"159_calls" [label="path.append" shape=box]
		159 -> "159_calls" [label=calls style=dashed]
		159 -> 158 [label=""]
		158 -> 159 [label="current != w"]
		160 [label="path.append(current)
path.reverse()
"]
		"160_calls" [label="path.append
path.reverse" shape=box]
		160 -> "160_calls" [label=calls style=dashed]
		160 -> 153 [label=""]
		158 -> 160 [label="(current == w)"]
		157 -> 158 [label=""]
		154 -> 157 [label="(v.estD != math.inf)"]
		154 -> 153 [label="v.estD == math.inf"]
		153 -> 154 [label="G.vertices"]
		152 -> 153 [label=""]
	}
	subgraph clusterdijkstra {
		graph [label=dijkstra]
		163 [label="for v in G.vertices:
"]
		164 [label="v.estD = math.inf
"]
		164 -> 163 [label=""]
		163 -> 164 [label="G.vertices"]
		165 [label="w.estD = 0
unsureVertices = heapdict.heapdict()
"]
		"165_calls" [label="heapdict.heapdict" shape=box]
		165 -> "165_calls" [label=calls style=dashed]
		166 [label="for v in G.vertices:
"]
		167 [label="unsureVertices[v] = v.estD
"]
		167 -> 166 [label=""]
		166 -> 167 [label="G.vertices"]
		168 [label="while len(unsureVertices) > 0:
"]
		169 [label="u, dist = unsureVertices.popitem()
if u.estD == math.inf:
"]
		"169_calls" [label="unsureVertices.popitem" shape=box]
		169 -> "169_calls" [label=calls style=dashed]
		171 [label=return
]
		169 -> 171 [label="u.estD == math.inf"]
		172 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		174 [label="if u.estD + wt < v.estD:
"]
		176 [label="v.estD = u.estD + wt
unsureVertices[v] = u.estD + wt
v.parent = u
"]
		176 -> 172 [label=""]
		174 -> 176 [label="u.estD + wt < v.estD"]
		174 -> 172 [label="(u.estD + wt >= v.estD)"]
		172 -> 174 [label="u.getOutNeighborsWithWeights()"]
		172 -> 168 [label=""]
		169 -> 172 [label="(u.estD != math.inf)"]
		168 -> 169 [label="len(unsureVertices) > 0"]
		166 -> 168 [label=""]
		165 -> 166 [label=""]
		163 -> 165 [label=""]
	}
	subgraph clusterdijkstra_shortestPaths {
		graph [label=dijkstra_shortestPaths]
		180 [label="dijkstra(w, G)
"]
		"180_calls" [label=dijkstra shape=box]
		180 -> "180_calls" [label=calls style=dashed]
		181 [label="for v in G.vertices:
"]
		182 [label="if v.estD == math.inf:
"]
		185 [label="path = []
current = v
"]
		186 [label="while current != w:
"]
		187 [label="path.append(current)
current = current.parent
"]
		"187_calls" [label="path.append" shape=box]
		187 -> "187_calls" [label=calls style=dashed]
		187 -> 186 [label=""]
		186 -> 187 [label="current != w"]
		188 [label="path.append(current)
path.reverse()
"]
		"188_calls" [label="path.append
path.reverse" shape=box]
		188 -> "188_calls" [label=calls style=dashed]
		188 -> 181 [label=""]
		186 -> 188 [label="(current == w)"]
		185 -> 186 [label=""]
		182 -> 185 [label="(v.estD != math.inf)"]
		182 -> 181 [label="v.estD == math.inf"]
		181 -> 182 [label="G.vertices"]
		180 -> 181 [label=""]
	}
	subgraph clusterrunTrials {
		graph [label=runTrials]
		191 [label="nValues = []
tValues = []
"]
		192 [label="for n in nVals:
"]
		193 [label="runtime = 0
"]
		195 [label="for t in range(numTrials):
"]
		196 [label="G = randomGraph(n, pFn(n))
start = time.time()
myFn(G.vertices[0], G)
end = time.time()
runtime += (end - start) * 1000
"]
		"196_calls" [label="randomGraph
time.time
myFn
time.time" shape=box]
		196 -> "196_calls" [label=calls style=dashed]
		196 -> 195 [label=""]
		195 -> 196 [label="range(numTrials)"]
		197 [label="runtime = runtime / numTrials
nValues.append(n)
tValues.append(runtime)
"]
		"197_calls" [label="nValues.append
tValues.append" shape=box]
		197 -> "197_calls" [label=calls style=dashed]
		197 -> 192 [label=""]
		195 -> 197 [label=""]
		193 -> 195 [label=""]
		192 -> 193 [label=nVals]
		194 [label="return nValues, tValues
"]
		192 -> 194 [label=""]
		191 -> 192 [label=""]
	}
	subgraph clustersmallFrac {
		graph [label=smallFrac]
		201 [label="return float(5 / n)
"]
	}
}
