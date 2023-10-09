digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import heapdict as heapdict
import math
from random import random
from random import choice
import time
from loop import loop
import numpy as np
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
	210 [label="loop.start_unroll
G_0 = randomGraph(5, 0.2)
BFS_shortestPaths(G_0.vertices[0], G_0)
dijkstraDumb_shortestPaths(G_0.vertices[0], G_0)
G_0 = randomGraph(5, 0.4, [1, 2, 3, 4, 5])
dijkstra_shortestPaths(G_0.vertices[0], G_0)
nValues_0 = [10]
nDijkstra_0, tDijkstra_0 = runTrials(BFS, nValues_0, smallFrac)
"]
	"210_calls" [label="randomGraph
BFS_shortestPaths
dijkstraDumb_shortestPaths
randomGraph
dijkstra_shortestPaths
runTrials" shape=box]
	210 -> "210_calls" [label=calls style=dashed]
	1 -> 210 [label="__name__ == '__main__'"]
	subgraph cluster__init__ {
		graph [label=__init__]
		70 [label="self = self
self.vertices = []
"]
	}
	subgraph clusterhasOutNeighbor {
		graph [label=hasOutNeighbor]
		6 [label="self = self
v_2 = v
if v_2 in self.getOutNeighbors():
"]
		7 [label="return True
"]
		6 -> 7 [label="v_2 in self.getOutNeighbors()"]
		8 [label="return False
"]
		6 -> 8 [label="(v_2 not in self.getOutNeighbors())"]
	}
	subgraph clusterhasInNeighbor {
		graph [label=hasInNeighbor]
		13 [label="self = self
v_3 = v
if v_3 in self.getInNeighbors():
"]
		14 [label="return True
"]
		13 -> 14 [label="v_3 in self.getInNeighbors()"]
		15 [label="return False
"]
		13 -> 15 [label="(v_3 not in self.getInNeighbors())"]
	}
	subgraph clusterhasNeighbor {
		graph [label=hasNeighbor]
		20 [label="self = self
v_4 = v
if v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors():
"]
		21 [label="return True
"]
		20 -> 21 [label="v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors()"]
		22 [label="return False
"]
		20 -> 22 [label="(not (v_4 in self.getInNeighbors() or v_4 in self.getOutNeighbors()))"]
	}
	subgraph clustergetOutNeighbors {
		graph [label=getOutNeighbors]
		27 [label="self = self
return [v_5[0] for v_5 in self.outNeighbors]
"]
	}
	subgraph clustergetInNeighbors {
		graph [label=getInNeighbors]
		31 [label="self = self
return [v_6[0] for v_6 in self.inNeighbors]
"]
	}
	subgraph clustergetOutNeighborsWithWeights {
		graph [label=getOutNeighborsWithWeights]
		35 [label="self = self
return self.outNeighbors
"]
	}
	subgraph clustergetInNeighborsWithWeights {
		graph [label=getInNeighborsWithWeights]
		39 [label="self = self
return self.inNeighbors
"]
	}
	subgraph clusteraddOutNeighbor {
		graph [label=addOutNeighbor]
		43 [label="self = self
v_9 = v
wt_9 = wt
self.outNeighbors.append((v_9, wt_9))
"]
		"43_calls" [label="self.outNeighbors.append" shape=box]
		43 -> "43_calls" [label=calls style=dashed]
	}
	subgraph clusteraddInNeighbor {
		graph [label=addInNeighbor]
		46 [label="self = self
v_10 = v
wt_10 = wt
self.inNeighbors.append((v_10, wt_10))
"]
		"46_calls" [label="self.inNeighbors.append" shape=box]
		46 -> "46_calls" [label=calls style=dashed]
	}
	subgraph clusteraddVertex {
		graph [label=addVertex]
		73 [label="self = self
n_17 = n
self.vertices.append(n_17)
"]
		"73_calls" [label="self.vertices.append" shape=box]
		73 -> "73_calls" [label=calls style=dashed]
	}
	subgraph clusteraddDiEdge {
		graph [label=addDiEdge]
		76 [label="self = self
u_18 = u
v_18 = v
wt_18 = wt
u_18.addOutNeighbor(v_18, wt=wt_18)
v_18.addInNeighbor(u_18, wt=wt_18)
"]
		"76_calls" [label="u_18.addOutNeighbor
v_18.addInNeighbor" shape=box]
		76 -> "76_calls" [label=calls style=dashed]
	}
	subgraph clusteraddBiEdge {
		graph [label=addBiEdge]
		79 [label="self = self
u_19 = u
v_19 = v
wt_19 = wt
self.addDiEdge(u_19, v_19, wt=wt_19)
self.addDiEdge(v_19, u_19, wt=wt_19)
"]
		"79_calls" [label="self.addDiEdge
self.addDiEdge" shape=box]
		79 -> "79_calls" [label=calls style=dashed]
	}
	subgraph clustergetDirEdges {
		graph [label=getDirEdges]
		82 [label="self = self
ret_20 = []
"]
		83 [label="for v_20 in self.vertices:
"]
		84 [label="for u_20, wt_20 in v_20.getOutNeighborsWithWeights():
"]
		86 [label="ret_20.append([v_20, u_20, wt_20])
"]
		"86_calls" [label="ret_20.append" shape=box]
		86 -> "86_calls" [label=calls style=dashed]
		86 -> 84 [label=""]
		84 -> 86 [label="v_20.getOutNeighborsWithWeights()"]
		84 -> 83 [label=""]
		83 -> 84 [label="self.vertices"]
		85 [label="return ret_20
"]
		83 -> 85 [label=""]
		82 -> 83 [label=""]
	}
	subgraph clusterrandomGraph {
		graph [label=randomGraph]
		91 [label="n_21 = n
p_21 = p
wts_21 = wts
G_21 = CS161Graph()
V_21 = [CS161Vertex(x_21) for x_21 in range(n_21)]
"]
		"91_calls" [label="CS161Graph
CS161Vertex
range" shape=box]
		91 -> "91_calls" [label=calls style=dashed]
		92 [label="for v_21 in V_21:
"]
		93 [label="G_21.addVertex(v_21)
"]
		"93_calls" [label="G_21.addVertex" shape=box]
		93 -> "93_calls" [label=calls style=dashed]
		93 -> 92 [label=""]
		92 -> 93 [label=V_21]
		94 [label="for v_21 in V_21:
"]
		95 [label="i_21 = 0
"]
		97 [label="for w_21 in V_21:
"]
		98 [label="if v_21 != w_21:
"]
		100 [label="if random() < p_21:
"]
		102 [label="G_21.addDiEdge(v_21, w_21, wt=choice(wts_21))
i_21 += 1
"]
		"102_calls" [label="G_21.addDiEdge" shape=box]
		102 -> "102_calls" [label=calls style=dashed]
		101 [label="if i_21 > 15:
"]
		101 -> 94 [label="i_21 > 15"]
		101 -> 97 [label="(i_21 <= 15)"]
		102 -> 101 [label=""]
		100 -> 102 [label="random() < p_21"]
		100 -> 101 [label="(random() >= p_21)"]
		98 -> 100 [label="v_21 != w_21"]
		98 -> 101 [label="(v_21 == w_21)"]
		97 -> 98 [label=V_21]
		97 -> 94 [label=""]
		95 -> 97 [label=""]
		94 -> 95 [label=V_21]
		96 [label="return G_21
"]
		94 -> 96 [label=""]
		92 -> 94 [label=""]
		91 -> 92 [label=""]
	}
	subgraph clusterBFS {
		graph [label=BFS]
		109 [label="w_22 = w
G_22 = G
"]
		110 [label="for v_22 in G_22.vertices:
"]
		111 [label="v_22.status = 'unvisited'
"]
		111 -> 110 [label=""]
		110 -> 111 [label="G_22.vertices"]
		112 [label="n_22 = len(G_22.vertices)
Ls_22 = [[] for i_22 in range(n_22)]
Ls_22[0] = [w_22]
w_22.status = 'visited'
"]
		"112_calls" [label="len
range" shape=box]
		112 -> "112_calls" [label=calls style=dashed]
		113 [label="for i_22 in range(n_22):
"]
		114 [label="for u_22 in Ls_22[i_22]:
"]
		116 [label="for v_22 in u_22.getOutNeighbors():
"]
		118 [label="if v_22.status == 'unvisited':
"]
		120 [label="v_22.status = 'visited'
v_22.parent = u_22
Ls_22[i_22 + 1].append(v_22)
"]
		"120_calls" [label="Ls_22.append" shape=box]
		120 -> "120_calls" [label=calls style=dashed]
		120 -> 116 [label=""]
		118 -> 120 [label="v_22.status == 'unvisited'"]
		118 -> 116 [label="(v_22.status != 'unvisited')"]
		116 -> 118 [label="u_22.getOutNeighbors()"]
		116 -> 114 [label=""]
		114 -> 116 [label="Ls_22[i_22]"]
		114 -> 113 [label=""]
		113 -> 114 [label="range(n_22)"]
		115 [label="return Ls_22
"]
		113 -> 115 [label=""]
		112 -> 113 [label=""]
		110 -> 112 [label=""]
		109 -> 110 [label=""]
	}
	subgraph clusterBFS_shortestPaths {
		graph [label=BFS_shortestPaths]
		125 [label="w_23 = w
G_23 = G
Ls_23 = BFS(w_23, G_23)
"]
		"125_calls" [label=BFS shape=box]
		125 -> "125_calls" [label=calls style=dashed]
		126 [label="for i_23 in range(len(Ls_23)):
"]
		127 [label="for w_23 in Ls_23[i_23]:
"]
		129 [label="path_23 = []
current_23 = w_23
"]
		131 [label="for j_23 in range(i_23):
"]
		132 [label="path_23.append(current_23)
current_23 = current_23.parent
"]
		"132_calls" [label="path_23.append" shape=box]
		132 -> "132_calls" [label=calls style=dashed]
		132 -> 131 [label=""]
		131 -> 132 [label="range(i_23)"]
		133 [label="path_23.append(current_23)
path_23.reverse()
"]
		"133_calls" [label="path_23.append
path_23.reverse" shape=box]
		133 -> "133_calls" [label=calls style=dashed]
		133 -> 127 [label=""]
		131 -> 133 [label=""]
		129 -> 131 [label=""]
		127 -> 129 [label="Ls_23[i_23]"]
		127 -> 126 [label=""]
		126 -> 127 [label="range(len(Ls_23))"]
		125 -> 126 [label=""]
	}
	subgraph clusterdijkstraDumb {
		graph [label=dijkstraDumb]
		136 [label="w_24 = w
G_24 = G
"]
		137 [label="for v_24 in G_24.vertices:
"]
		138 [label="v_24.estD = math.inf
"]
		138 -> 137 [label=""]
		137 -> 138 [label="G_24.vertices"]
		139 [label="w_24.estD = 0
unsureVertices_24 = G_24.vertices[:]
"]
		140 [label="while len(unsureVertices_24) > 0:
"]
		141 [label="u_24 = None
minD_24 = math.inf
"]
		143 [label="for x_24 in unsureVertices_24:
"]
		144 [label="if x_24.estD < minD_24:
"]
		146 [label="minD_24 = x_24.estD
u_24 = x_24
"]
		146 -> 143 [label=""]
		144 -> 146 [label="x_24.estD < minD_24"]
		144 -> 143 [label="(x_24.estD >= minD_24)"]
		143 -> 144 [label=unsureVertices_24]
		145 [label="if u_24 == None:
"]
		148 [label=return
]
		145 -> 148 [label="u_24 == None"]
		149 [label="for v_24, wt_24 in u_24.getOutNeighborsWithWeights():
"]
		151 [label="if u_24.estD + wt_24 < v_24.estD:
"]
		153 [label="v_24.estD = u_24.estD + wt_24
v_24.parent = u_24
"]
		153 -> 149 [label=""]
		151 -> 153 [label="u_24.estD + wt_24 < v_24.estD"]
		151 -> 149 [label="(u_24.estD + wt_24 >= v_24.estD)"]
		149 -> 151 [label="u_24.getOutNeighborsWithWeights()"]
		152 [label="unsureVertices_24.remove(u_24)
"]
		"152_calls" [label="unsureVertices_24.remove" shape=box]
		152 -> "152_calls" [label=calls style=dashed]
		152 -> 140 [label=""]
		149 -> 152 [label=""]
		145 -> 149 [label="(u_24 != None)"]
		143 -> 145 [label=""]
		141 -> 143 [label=""]
		140 -> 141 [label="len(unsureVertices_24) > 0"]
		139 -> 140 [label=""]
		137 -> 139 [label=""]
		136 -> 137 [label=""]
	}
	subgraph clusterdijkstraDumb_shortestPaths {
		graph [label=dijkstraDumb_shortestPaths]
		157 [label="w_25 = w
G_25 = G
dijkstraDumb(w_25, G_25)
"]
		"157_calls" [label=dijkstraDumb shape=box]
		157 -> "157_calls" [label=calls style=dashed]
		158 [label="for v_25 in G_25.vertices:
"]
		159 [label="if v_25.estD == math.inf:
"]
		162 [label="path_25 = []
current_25 = v_25
"]
		163 [label="while current_25 != w_25:
"]
		164 [label="path_25.append(current_25)
current_25 = current_25.parent
"]
		"164_calls" [label="path_25.append" shape=box]
		164 -> "164_calls" [label=calls style=dashed]
		164 -> 163 [label=""]
		163 -> 164 [label="current_25 != w_25"]
		165 [label="path_25.append(current_25)
path_25.reverse()
"]
		"165_calls" [label="path_25.append
path_25.reverse" shape=box]
		165 -> "165_calls" [label=calls style=dashed]
		165 -> 158 [label=""]
		163 -> 165 [label="(current_25 == w_25)"]
		162 -> 163 [label=""]
		159 -> 162 [label="(v_25.estD != math.inf)"]
		159 -> 158 [label="v_25.estD == math.inf"]
		158 -> 159 [label="G_25.vertices"]
		157 -> 158 [label=""]
	}
	subgraph clusterdijkstra {
		graph [label=dijkstra]
		168 [label="w_26 = w
G_26 = G
"]
		169 [label="for v_26 in G_26.vertices:
"]
		170 [label="v_26.estD = math.inf
"]
		170 -> 169 [label=""]
		169 -> 170 [label="G_26.vertices"]
		171 [label="w_26.estD = 0
unsureVertices_26 = heapdict.heapdict()
"]
		"171_calls" [label="heapdict.heapdict" shape=box]
		171 -> "171_calls" [label=calls style=dashed]
		172 [label="for v_26 in G_26.vertices:
"]
		173 [label="unsureVertices_26[v_26] = v_26.estD
"]
		173 -> 172 [label=""]
		172 -> 173 [label="G_26.vertices"]
		174 [label="while len(unsureVertices_26) > 0:
"]
		175 [label="u_26, dist_26 = unsureVertices_26.popitem()
if u_26.estD == math.inf:
"]
		"175_calls" [label="unsureVertices_26.popitem" shape=box]
		175 -> "175_calls" [label=calls style=dashed]
		177 [label=return
]
		175 -> 177 [label="u_26.estD == math.inf"]
		178 [label="for v_26, wt_26 in u_26.getOutNeighborsWithWeights():
"]
		180 [label="if u_26.estD + wt_26 < v_26.estD:
"]
		182 [label="v_26.estD = u_26.estD + wt_26
unsureVertices_26[v_26] = u_26.estD + wt_26
v_26.parent = u_26
"]
		182 -> 178 [label=""]
		180 -> 182 [label="u_26.estD + wt_26 < v_26.estD"]
		180 -> 178 [label="(u_26.estD + wt_26 >= v_26.estD)"]
		178 -> 180 [label="u_26.getOutNeighborsWithWeights()"]
		178 -> 174 [label=""]
		175 -> 178 [label="(u_26.estD != math.inf)"]
		174 -> 175 [label="len(unsureVertices_26) > 0"]
		172 -> 174 [label=""]
		171 -> 172 [label=""]
		169 -> 171 [label=""]
		168 -> 169 [label=""]
	}
	subgraph clusterdijkstra_shortestPaths {
		graph [label=dijkstra_shortestPaths]
		186 [label="w_27 = w
G_27 = G
dijkstra(w_27, G_27)
"]
		"186_calls" [label=dijkstra shape=box]
		186 -> "186_calls" [label=calls style=dashed]
		187 [label="for v_27 in G_27.vertices:
"]
		188 [label="if v_27.estD == math.inf:
"]
		191 [label="path_27 = []
current_27 = v_27
"]
		192 [label="while current_27 != w_27:
"]
		193 [label="path_27.append(current_27)
current_27 = current_27.parent
"]
		"193_calls" [label="path_27.append" shape=box]
		193 -> "193_calls" [label=calls style=dashed]
		193 -> 192 [label=""]
		192 -> 193 [label="current_27 != w_27"]
		194 [label="path_27.append(current_27)
path_27.reverse()
"]
		"194_calls" [label="path_27.append
path_27.reverse" shape=box]
		194 -> "194_calls" [label=calls style=dashed]
		194 -> 187 [label=""]
		192 -> 194 [label="(current_27 == w_27)"]
		191 -> 192 [label=""]
		188 -> 191 [label="(v_27.estD != math.inf)"]
		188 -> 187 [label="v_27.estD == math.inf"]
		187 -> 188 [label="G_27.vertices"]
		186 -> 187 [label=""]
	}
	subgraph clusterrunTrials {
		graph [label=runTrials]
		197 [label="myFn_28 = myFn
nVals_28 = nVals
pFn_28 = pFn
numTrials_28 = numTrials
nValues_28 = []
tValues_28 = []
"]
		198 [label="for n_28 in nVals_28:
"]
		199 [label="runtime_28 = 0
"]
		201 [label="for t_28 in range(numTrials_28):
"]
		202 [label="G_28 = randomGraph(n_28 * 10000, pFn(n_28))
start_28 = time.time()
myFn(G_28.vertices[0], G_28)
end_28 = time.time()
runtime_28 += (end_28 - start_28) * 1000
"]
		"202_calls" [label="randomGraph
time.time
myFn
time.time" shape=box]
		202 -> "202_calls" [label=calls style=dashed]
		202 -> 201 [label=""]
		201 -> 202 [label="range(numTrials_28)"]
		203 [label="runtime_28 = runtime_28 / numTrials_28
nValues_28.append(n_28)
tValues_28.append(runtime_28)
"]
		"203_calls" [label="nValues_28.append
tValues_28.append" shape=box]
		203 -> "203_calls" [label=calls style=dashed]
		203 -> 198 [label=""]
		201 -> 203 [label=""]
		199 -> 201 [label=""]
		198 -> 199 [label=nVals_28]
		200 [label="return nValues_28, tValues_28
"]
		198 -> 200 [label=""]
		197 -> 198 [label=""]
	}
	subgraph clustersmallFrac {
		graph [label=smallFrac]
		207 [label="n_29 = n
return float(5 / n_29)
"]
	}
}
