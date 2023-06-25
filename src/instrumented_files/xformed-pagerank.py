import sys
import os
import sys
import math
import numpy
import pandas


def __extract_nodes(matrix):
    print(1, 10)
    print(3, 11)
    nodes = set()
    print(4, 12)
    for col_key in matrix:
        print(4, 12)
        nodes.add(col_key)
    print(6, 14)
    for row_key in matrix.T:
        print(6, 14)
        nodes.add(row_key)
    return nodes


def __make_square(matrix, keys, default=0.0):
    print(1, 18)
    print(12, 19)
    matrix = matrix.copy()

    def insert_missing_columns(matrix):
        print(12, 21)
        print(14, 22)
        for key in keys:
            print(14, 22)
            if not key in matrix:
                print(15, 23)
                print(17, 24)
                matrix[key] = pandas.Series(default, index=matrix.index)
            else:
                print(15, 23)
        return matrix
    print(12, 27)
    matrix = insert_missing_columns(matrix)
    print(12, 28)
    matrix = insert_missing_columns(matrix.T).T
    return matrix.fillna(default)


def __ensure_rows_positive(matrix):
    print(1, 32)
    print(24, 33)
    matrix = matrix.T
    print(25, 34)
    for col_key in matrix:
        print(25, 34)
        if matrix[col_key].sum() == 0.0:
            print(26, 35)
            print(28, 36)
            matrix[col_key] = pandas.Series(numpy.ones(len(matrix[col_key])
                ), index=matrix.index)
        else:
            print(26, 35)
    return matrix.T


def __normalize_rows(matrix):
    print(1, 39)
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclidean_norm(series):
    print(1, 42)
    return math.sqrt(series.dot(series))


def __start_state(nodes):
    print(1, 47)
    if len(nodes) == 0:
        print(41, 48)
        raise ValueError('There must be at least one node.')
    else:
        print(41, 48)
    print(43, 49)
    start_prob = 1.0 / float(len(nodes))
    return pandas.Series({node: start_prob for node in nodes})


def __integrate_random_surfer(nodes, transition_probabilities, rsp):
    print(1, 52)
    print(47, 53)
    alpha = 1.0 / float(len(nodes)) * rsp
    return transition_probabilities.copy().multiply(1.0 - rsp) + alpha


def power_iteration(transition_weights, rsp=0.15, epsilon=1e-05,
    max_iterations=1000):
    print(1, 56)
    print(51, 58)
    transition_weights = pandas.DataFrame(transition_weights)
    print(51, 59)
    nodes = __extract_nodes(transition_weights)
    print(51, 60)
    transition_weights = __make_square(transition_weights, nodes, default=0.0)
    print(51, 61)
    transition_weights = __ensure_rows_positive(transition_weights)
    print(51, 64)
    state = __start_state(nodes)
    print(51, 65)
    transition_probabilities = __normalize_rows(transition_weights)
    print(51, 66)
    transition_probabilities = __integrate_random_surfer(nodes,
        transition_probabilities, rsp)
    print(52, 69)
    for iteration in range(max_iterations):
        print(52, 69)
        print(53, 70)
        old_state = state.copy()
        print(53, 71)
        state = state.dot(transition_probabilities)
        print(53, 72)
        delta = state - old_state
        if __euclidean_norm(delta) < epsilon:
            print(53, 73)
            break
        else:
            print(53, 73)
    return state


if __name__ == '__main__':
    print(1, 78)
    power_iteration([[1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5,
        23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 
        7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7,
        8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3,
        5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 
        4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23,
        4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7],
        [1, 3, 5, 5, 23, 4, 7, 8, 9, 2, 4, 6, 7], [1, 3, 5, 5, 23, 4, 7, 8,
        9, 2, 4, 6, 7]])
else:
    print(1, 78)
