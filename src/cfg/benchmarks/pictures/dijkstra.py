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
def __str__(self):...
def __init__(self):...
def addVertex(self, n):...
def addDiEdge(self, u, v, wt=1):...
def addBiEdge(self, u, v, wt=1):...
def getDirEdges(self):...
def __str__(self):...
def __init__(self):...
def addVertex(self, n):...
def addDiEdge(self, u, v, wt=1):...
def addBiEdge(self, u, v, wt=1):...
def getDirEdges(self):...
def __str__(self):...
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
	228 [label="loop.start_unroll
G = randomGraph(5, 0.2)
BFS_shortestPaths(G.vertices[0], G)
dijkstraDumb_shortestPaths(G.vertices[0], G)
G = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
dijkstra_shortestPaths(G.vertices[0], G)
nValues = [10]
nDijkstra, tDijkstra = runTrials(dijkstra, nValues, smallFrac)
"]
	"228_calls" [label="randomGraph
BFS_shortestPaths
dijkstraDumb_shortestPaths
randomGraph
dijkstra_shortestPaths
runTrials" shape=box]
	228 -> "228_calls" [label=calls style=dashed]
	1 -> 228 [label="__name__ == '__main__'"]
	subgraph cluster__init__ {
		graph [label=__init__]
		84 [label="self.vertices = []
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
	subgraph cluster__str__ {
		graph [label=__str__]
		105 [label="ret = 'CS161Graph with:\n'
ret += '\t Vertices:\n\t'
"]
		106 [label="for v in self.vertices:
"]
		107 [label="ret += str(v) + ','
"]
		"107_calls" [label=str shape=box]
		107 -> "107_calls" [label=calls style=dashed]
		107 -> 106 [label=""]
		106 -> 107 [label="self.vertices"]
		108 [label="ret += '\n'
ret += '\t Edges:\n\t'
"]
		109 [label="for a, b, wt in self.getDirEdges():
"]
		110 [label="ret += '(' + str(a) + ',' + str(b) + '; wt:' + str(wt) + ') '
"]
		"110_calls" [label="str
str
str" shape=box]
		110 -> "110_calls" [label=calls style=dashed]
		110 -> 109 [label=""]
		109 -> 110 [label="self.getDirEdges()"]
		111 [label="ret += '\n'
return ret
"]
		109 -> 111 [label=""]
		108 -> 109 [label=""]
		106 -> 108 [label=""]
		105 -> 106 [label=""]
	}
	subgraph clusteraddVertex {
		graph [label=addVertex]
		87 [label="self.vertices.append(n)
"]
		"87_calls" [label="self.vertices.append" shape=box]
		87 -> "87_calls" [label=calls style=dashed]
	}
	subgraph clusteraddDiEdge {
		graph [label=addDiEdge]
		90 [label="u.addOutNeighbor(v, wt=wt)
v.addInNeighbor(u, wt=wt)
"]
		"90_calls" [label="u.addOutNeighbor
v.addInNeighbor" shape=box]
		90 -> "90_calls" [label=calls style=dashed]
	}
	subgraph clusteraddBiEdge {
		graph [label=addBiEdge]
		93 [label="self.addDiEdge(u, v, wt=wt)
self.addDiEdge(v, u, wt=wt)
"]
		"93_calls" [label="self.addDiEdge
self.addDiEdge" shape=box]
		93 -> "93_calls" [label=calls style=dashed]
	}
	subgraph clustergetDirEdges {
		graph [label=getDirEdges]
		96 [label="ret = []
"]
		97 [label="for v in self.vertices:
"]
		98 [label="for u, wt in v.getOutNeighborsWithWeights():
"]
		100 [label="ret.append([v, u, wt])
"]
		"100_calls" [label="ret.append" shape=box]
		100 -> "100_calls" [label=calls style=dashed]
		100 -> 98 [label=""]
		98 -> 100 [label="v.getOutNeighborsWithWeights()"]
		98 -> 97 [label=""]
		97 -> 98 [label="self.vertices"]
		99 [label="return ret
"]
		97 -> 99 [label=""]
		96 -> 97 [label=""]
	}
	subgraph clusterrandomGraph {
		graph [label=randomGraph]
		115 [label="G = CS161Graph()
V = [CS161Vertex(x) for x in range(n)]
"]
		"115_calls" [label="CS161Graph
CS161Vertex
range" shape=box]
		115 -> "115_calls" [label=calls style=dashed]
		116 [label="for v in V:
"]
		117 [label="G.addVertex(v)
"]
		"117_calls" [label="G.addVertex" shape=box]
		117 -> "117_calls" [label=calls style=dashed]
		117 -> 116 [label=""]
		116 -> 117 [label=V]
		118 [label="for v in V:
"]
		119 [label="for w in V:
"]
		121 [label="if v != w:
"]
		123 [label="if random() < p:
"]
		125 [label="G.addDiEdge(v, w, wt=choice(wts))
"]
		"125_calls" [label="G.addDiEdge" shape=box]
		125 -> "125_calls" [label=calls style=dashed]
		125 -> 119 [label=""]
		123 -> 125 [label="random() < p"]
		123 -> 119 [label="(random() >= p)"]
		121 -> 123 [label="v != w"]
		121 -> 119 [label="(v == w)"]
		119 -> 121 [label=V]
		119 -> 118 [label=""]
		118 -> 119 [label=V]
		120 [label="return G
"]
		118 -> 120 [label=""]
		116 -> 118 [label=""]
		115 -> 116 [label=""]
	}
	subgraph clusterBFS {
		graph [label=BFS]
		130 [label="for v in G.vertices:
"]
		131 [label="v.status = 'unvisited'
"]
		131 -> 130 [label=""]
		130 -> 131 [label="G.vertices"]
		132 [label="n = len(G.vertices)
Ls = [[] for i in range(n)]
Ls[0] = [w]
w.status = 'visited'
"]
		"132_calls" [label="len
range" shape=box]
		132 -> "132_calls" [label=calls style=dashed]
		133 [label="for i in range(n):
"]
		134 [label="for u in Ls[i]:
"]
		136 [label="for v in u.getOutNeighbors():
"]
		138 [label="if v.status == 'unvisited':
"]
		140 [label="v.status = 'visited'
v.parent = u
Ls[i + 1].append(v)
"]
		"140_calls" [label="Ls.append" shape=box]
		140 -> "140_calls" [label=calls style=dashed]
		140 -> 136 [label=""]
		138 -> 140 [label="v.status == 'unvisited'"]
		138 -> 136 [label="(v.status != 'unvisited')"]
		136 -> 138 [label="u.getOutNeighbors()"]
		136 -> 134 [label=""]
		134 -> 136 [label="Ls[i]"]
		134 -> 133 [label=""]
		133 -> 134 [label="range(n)"]
		135 [label="return Ls
"]
		133 -> 135 [label=""]
		132 -> 133 [label=""]
		130 -> 132 [label=""]
	}
	subgraph clusterBFS_shortestPaths {
		graph [label=BFS_shortestPaths]
		145 [label="Ls = BFS(w, G)
"]
		"145_calls" [label=BFS shape=box]
		145 -> "145_calls" [label=calls style=dashed]
		146 [label="for i in range(len(Ls)):
"]
		147 [label="for w in Ls[i]:
"]
		149 [label="path = []
current = w
"]
		151 [label="for j in range(i):
"]
		152 [label="path.append(current)
current = current.parent
"]
		"152_calls" [label="path.append" shape=box]
		152 -> "152_calls" [label=calls style=dashed]
		152 -> 151 [label=""]
		151 -> 152 [label="range(i)"]
		153 [label="path.append(current)
path.reverse()
"]
		"153_calls" [label="path.append
path.reverse" shape=box]
		153 -> "153_calls" [label=calls style=dashed]
		153 -> 147 [label=""]
		151 -> 153 [label=""]
		149 -> 151 [label=""]
		147 -> 149 [label="Ls[i]"]
		147 -> 146 [label=""]
		146 -> 147 [label="range(len(Ls))"]
		145 -> 146 [label=""]
	}
	subgraph clusterdijkstraDumb {
		graph [label=dijkstraDumb]
		156 [label="for v in G.vertices:
"]
		157 [label="v.estD = math.inf
"]
		157 -> 156 [label=""]
		156 -> 157 [label="G.vertices"]
		158 [label="w.estD = 0
unsureVertices = G.vertices[:]
"]
		159 [label="while len(unsureVertices) > 0:
"]
		160 [label="u = None
minD = math.inf
"]
		162 [label="for x in unsureVertices:
"]
		163 [label="if x.estD < minD:
"]
		165 [label="minD = x.estD
u = x
"]
		165 -> 162 [label=""]
		163 -> 165 [label="x.estD < minD"]
		163 -> 162 [label="(x.estD >= minD)"]
		162 -> 163 [label=unsureVertices]
		164 [label="if u == None:
"]
		167 [label=return
]
		164 -> 167 [label="u == None"]
		168 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		170 [label="if u.estD + wt < v.estD:
"]
		172 [label="v.estD = u.estD + wt
v.parent = u
"]
		172 -> 168 [label=""]
		170 -> 172 [label="u.estD + wt < v.estD"]
		170 -> 168 [label="(u.estD + wt >= v.estD)"]
		168 -> 170 [label="u.getOutNeighborsWithWeights()"]
		171 [label="unsureVertices.remove(u)
"]
		"171_calls" [label="unsureVertices.remove" shape=box]
		171 -> "171_calls" [label=calls style=dashed]
		171 -> 159 [label=""]
		168 -> 171 [label=""]
		164 -> 168 [label="(u != None)"]
		162 -> 164 [label=""]
		160 -> 162 [label=""]
		159 -> 160 [label="len(unsureVertices) > 0"]
		158 -> 159 [label=""]
		156 -> 158 [label=""]
	}
	subgraph clusterdijkstraDumb_shortestPaths {
		graph [label=dijkstraDumb_shortestPaths]
		176 [label="dijkstraDumb(w, G)
"]
		"176_calls" [label=dijkstraDumb shape=box]
		176 -> "176_calls" [label=calls style=dashed]
		177 [label="for v in G.vertices:
"]
		178 [label="if v.estD == math.inf:
"]
		181 [label="path = []
current = v
"]
		182 [label="while current != w:
"]
		183 [label="path.append(current)
current = current.parent
"]
		"183_calls" [label="path.append" shape=box]
		183 -> "183_calls" [label=calls style=dashed]
		183 -> 182 [label=""]
		182 -> 183 [label="current != w"]
		184 [label="path.append(current)
path.reverse()
"]
		"184_calls" [label="path.append
path.reverse" shape=box]
		184 -> "184_calls" [label=calls style=dashed]
		184 -> 177 [label=""]
		182 -> 184 [label="(current == w)"]
		181 -> 182 [label=""]
		178 -> 181 [label="(v.estD != math.inf)"]
		178 -> 177 [label="v.estD == math.inf"]
		177 -> 178 [label="G.vertices"]
		176 -> 177 [label=""]
	}
	subgraph clusterdijkstra {
		graph [label=dijkstra]
		187 [label="for v in G.vertices:
"]
		188 [label="v.estD = math.inf
"]
		188 -> 187 [label=""]
		187 -> 188 [label="G.vertices"]
		189 [label="w.estD = 0
unsureVertices = heapdict.heapdict()
"]
		"189_calls" [label="heapdict.heapdict" shape=box]
		189 -> "189_calls" [label=calls style=dashed]
		190 [label="for v in G.vertices:
"]
		191 [label="unsureVertices[v] = v.estD
"]
		191 -> 190 [label=""]
		190 -> 191 [label="G.vertices"]
		192 [label="while len(unsureVertices) > 0:
"]
		193 [label="u, dist = unsureVertices.popitem()
if u.estD == math.inf:
"]
		"193_calls" [label="unsureVertices.popitem" shape=box]
		193 -> "193_calls" [label=calls style=dashed]
		195 [label=return
]
		193 -> 195 [label="u.estD == math.inf"]
		196 [label="for v, wt in u.getOutNeighborsWithWeights():
"]
		198 [label="if u.estD + wt < v.estD:
"]
		200 [label="v.estD = u.estD + wt
unsureVertices[v] = u.estD + wt
v.parent = u
"]
		200 -> 196 [label=""]
		198 -> 200 [label="u.estD + wt < v.estD"]
		198 -> 196 [label="(u.estD + wt >= v.estD)"]
		196 -> 198 [label="u.getOutNeighborsWithWeights()"]
		196 -> 192 [label=""]
		193 -> 196 [label="(u.estD != math.inf)"]
		192 -> 193 [label="len(unsureVertices) > 0"]
		190 -> 192 [label=""]
		189 -> 190 [label=""]
		187 -> 189 [label=""]
	}
	subgraph clusterdijkstra_shortestPaths {
		graph [label=dijkstra_shortestPaths]
		204 [label="dijkstra(w, G)
"]
		"204_calls" [label=dijkstra shape=box]
		204 -> "204_calls" [label=calls style=dashed]
		205 [label="for v in G.vertices:
"]
		206 [label="if v.estD == math.inf:
"]
		209 [label="path = []
current = v
"]
		210 [label="while current != w:
"]
		211 [label="path.append(current)
current = current.parent
"]
		"211_calls" [label="path.append" shape=box]
		211 -> "211_calls" [label=calls style=dashed]
		211 -> 210 [label=""]
		210 -> 211 [label="current != w"]
		212 [label="path.append(current)
path.reverse()
"]
		"212_calls" [label="path.append
path.reverse" shape=box]
		212 -> "212_calls" [label=calls style=dashed]
		212 -> 205 [label=""]
		210 -> 212 [label="(current == w)"]
		209 -> 210 [label=""]
		206 -> 209 [label="(v.estD != math.inf)"]
		206 -> 205 [label="v.estD == math.inf"]
		205 -> 206 [label="G.vertices"]
		204 -> 205 [label=""]
	}
	subgraph clusterrunTrials {
		graph [label=runTrials]
		215 [label="nValues = []
tValues = []
"]
		216 [label="for n in nVals:
"]
		217 [label="runtime = 0
"]
		219 [label="for t in range(numTrials):
"]
		220 [label="G = randomGraph(n, pFn(n))
start = time.time()
myFn(G.vertices[0], G)
end = time.time()
runtime += (end - start) * 1000
"]
		"220_calls" [label="randomGraph
time.time
myFn
time.time" shape=box]
		220 -> "220_calls" [label=calls style=dashed]
		220 -> 219 [label=""]
		219 -> 220 [label="range(numTrials)"]
		221 [label="runtime = runtime / numTrials
nValues.append(n)
tValues.append(runtime)
"]
		"221_calls" [label="nValues.append
tValues.append" shape=box]
		221 -> "221_calls" [label=calls style=dashed]
		221 -> 216 [label=""]
		219 -> 221 [label=""]
		217 -> 219 [label=""]
		216 -> 217 [label=nVals]
		218 [label="return nValues, tValues
"]
		216 -> 218 [label=""]
		215 -> 216 [label=""]
	}
	subgraph clustersmallFrac {
		graph [label=smallFrac]
		225 [label="return float(5 / n)
"]
	}
}
