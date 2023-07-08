digraph "clustermain.c" {
	graph [label="main.c"]
	1 [label="import os
import sys
import math
import numpy
import pandas
def __extract_nodes(matrix):...
def __make_square(matrix, keys, default=0.0):...
def __ensure_rows_positive(matrix):...
def __normalize_rows(matrix):...
def __euclidean_norm(series):...
def __start_state(nodes):...
def __integrate_random_surfer(nodes, transition_probabilities, rsp):...
def power_iteration(transition_weights, rsp=0.15, epsilon=1e-05,...
if __name__ == '__main__':
"]
	59 [label="power_iteration([[1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23,
    4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 
    3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4,
    6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7,
    8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 
    5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7
    ], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 
    9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7]])
"]
	"59_calls" [label=power_iteration shape=box]
	59 -> "59_calls" [label=calls style=dashed]
	1 -> 59 [label="__name__ == '__main__'"]
	subgraph cluster__extract_nodes {
		graph [label=__extract_nodes]
		3 [label="nodes = set()
"]
		"3_calls" [label=set shape=box]
		3 -> "3_calls" [label=calls style=dashed]
		4 [label="for col_key in matrix:
"]
		5 [label="nodes.add(col_key)
"]
		"5_calls" [label="nodes.add" shape=box]
		5 -> "5_calls" [label=calls style=dashed]
		5 -> 4 [label=""]
		4 -> 5 [label=matrix]
		6 [label="for row_key in matrix.T:
"]
		7 [label="nodes.add(row_key)
"]
		"7_calls" [label="nodes.add" shape=box]
		7 -> "7_calls" [label=calls style=dashed]
		7 -> 6 [label=""]
		6 -> 7 [label="matrix.T"]
		8 [label="return nodes
"]
		6 -> 8 [label=""]
		4 -> 6 [label=""]
		3 -> 4 [label=""]
	}
	subgraph cluster__make_square {
		graph [label=__make_square]
		12 [label="matrix = matrix.copy()
def insert_missing_columns(matrix):...
matrix = insert_missing_columns(matrix)
matrix = insert_missing_columns(matrix.T).T
return matrix.fillna(default)
"]
		"12_calls" [label="matrix.copy
insert_missing_columns
insert_missing_columns" shape=box]
		12 -> "12_calls" [label=calls style=dashed]
		subgraph clusterinsert_missing_columns {
			graph [label=insert_missing_columns]
			14 [label="for key in keys:
"]
			15 [label="if not key in matrix:
"]
			17 [label="matrix[key] = pandas.Series(default, index=matrix.index)
"]
			"17_calls" [label="pandas.Series" shape=box]
			17 -> "17_calls" [label=calls style=dashed]
			17 -> 14 [label=""]
			15 -> 17 [label="not key in matrix"]
			15 -> 14 [label="(not not key in matrix)"]
			14 -> 15 [label=keys]
			16 [label="return matrix
"]
			14 -> 16 [label=""]
		}
	}
	subgraph cluster__ensure_rows_positive {
		graph [label=__ensure_rows_positive]
		24 [label="matrix = matrix.T
"]
		25 [label="for col_key in matrix:
"]
		26 [label="if matrix[col_key].sum() == 0.0:
"]
		28 [label="matrix[col_key] = pandas.Series(numpy.ones(len(matrix[col_key])), index=
    matrix.index)
"]
		"28_calls" [label="pandas.Series" shape=box]
		28 -> "28_calls" [label=calls style=dashed]
		28 -> 25 [label=""]
		26 -> 28 [label="matrix[col_key].sum() == 0.0"]
		26 -> 25 [label="(matrix[col_key].sum() != 0.0)"]
		25 -> 26 [label=matrix]
		27 [label="return matrix.T
"]
		25 -> 27 [label=""]
		24 -> 25 [label=""]
	}
	subgraph cluster__normalize_rows {
		graph [label=__normalize_rows]
		33 [label="return matrix.div(matrix.sum(axis=1), axis=0)
"]
	}
	subgraph cluster__euclidean_norm {
		graph [label=__euclidean_norm]
		37 [label="return math.sqrt(series.dot(series))
"]
	}
	subgraph cluster__start_state {
		graph [label=__start_state]
		41 [label="if len(nodes) == 0:
"]
		43 [label="start_prob = 1.0 / float(len(nodes))
return pandas.Series({node: start_prob for node in nodes})
"]
		"43_calls" [label=float shape=box]
		43 -> "43_calls" [label=calls style=dashed]
		41 -> 43 [label="(len(nodes) != 0)"]
		41 -> 43 [label="len(nodes) == 0"]
	}
	subgraph cluster__integrate_random_surfer {
		graph [label=__integrate_random_surfer]
		47 [label="alpha = 1.0 / float(len(nodes)) * rsp
return transition_probabilities.copy().multiply(1.0 - rsp) + alpha
"]
		"47_calls" [label=float shape=box]
		47 -> "47_calls" [label=calls style=dashed]
	}
	subgraph clusterpower_iteration {
		graph [label=power_iteration]
		51 [label="transition_weights = pandas.DataFrame(transition_weights)
nodes = __extract_nodes(transition_weights)
transition_weights = __make_square(transition_weights, nodes, default=0.0)
transition_weights = __ensure_rows_positive(transition_weights)
state = __start_state(nodes)
transition_probabilities = __normalize_rows(transition_weights)
transition_probabilities = __integrate_random_surfer(nodes,
    transition_probabilities, rsp)
"]
		"51_calls" [label="pandas.DataFrame
__extract_nodes
__make_square
__ensure_rows_positive
__start_state
__normalize_rows
__integrate_random_surfer" shape=box]
		51 -> "51_calls" [label=calls style=dashed]
		52 [label="for iteration in range(max_iterations):
"]
		53 [label="old_state = state.copy()
state = state.dot(transition_probabilities)
delta = state - old_state
if __euclidean_norm(delta) < epsilon:
"]
		"53_calls" [label="state.copy
state.dot" shape=box]
		53 -> "53_calls" [label=calls style=dashed]
		54 [label="return state
"]
		53 -> 54 [label="__euclidean_norm(delta) < epsilon"]
		53 -> 52 [label="(__euclidean_norm(delta) >= epsilon)"]
		52 -> 53 [label="range(max_iterations)"]
		52 -> 54 [label=""]
		51 -> 52 [label=""]
	}
}
