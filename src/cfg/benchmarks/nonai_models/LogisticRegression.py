import csv
import numpy as np
import math

demographic = []

def log_like(theta, data):
    result = 0
    print(theta)
    for i in range(len(data)):
        y = data[i][-1]
        y_hat = sigmoid(dot_prod(theta, data[i, :-1]))
        result += y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
    return result

def sigmoid(x):
    return 1.0 / (1 + math.exp(-1 * x))

def make_arr(train, noDem):
    data = []
    i = 0
    for row in train:
        data.append([])
        j = 0
        for col in row:
            if j == 0:
                data[i].append(1)
            data[i].append(int(row[col]))
            j += 1
        i += 1
    col = len(data[0])
    if noDem:
        col -= 1
    npdata = np.zeros((len(data), col))
    for i in range(len(data)):
        a = 0
        for j in range(len(data[0])):
            if noDem and j == len(data[0]) - 2:
                demographic.append(data[i][j])
                a -= 1
            else:
                npdata[i][a] = data[i][j]
            a += 1
    return npdata

def dot_prod(x, y):
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result

def logistic_regression(train, test, step, iters, noDem):
    # move data into array so its not annoying to iterate through
    data = make_arr(train, noDem)
    theta = np.zeros(len(data[0]) - 1)
    for a in range(iters):
        gradient = np.zeros(len(data[0]) - 1)
        for i in range(len(data)):
            for j in range(len(data[0]) - 1):
                gradient[j] += data[i][j] * (data[i][-1] - sigmoid(dot_prod(theta, data[i])))
        for i in range(len(theta)):
            theta[i] += step * gradient[i]
    print(test_data(data, theta, noDem, True))
    #return test_data(test, theta, noDem, False)

def test_data(test, theta, noDem, datatype):
    if datatype == True:
        data = test
    else:
        data = make_arr(test, noDem)
    predict = np.zeros(len(data))
    for i in range(len(predict)):
        if sigmoid(dot_prod(data[i, :-1], theta)) > 0.5:
            predict[i] = 1
    correct = []
    for i in range(len(predict)):
        if predict[i] == data[i][-1]:
            correct.append(1)
        else:
            correct.append(0)
    if noDem == True:
        D0 = 0
        P0 = 0
        D1 = 0
        P1 = 0
        for i in range(len(data)):
            if demographic[i] == 1:
                D1 += 1
                if data[i][-1] == 1:
                    P1 += 1
            else:
                D0 += 1
                if data[i][-1] == 1:
                    P0 += 1
        print(P0/D0, P1/D1)
    print(log_like(theta, data))
    toSort = []
    for i in range(len(theta)):
        toSort.append([theta[i], i])
    toSort.sort()
    #print(toSort[len(theta)-1], toSort[len(theta)-2], toSort[len(theta)-3])
    return sum(correct) / len(predict)

def main():
    atrain = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/ancestry-train.csv'))
    atest = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/ancestry-test.csv'))
    htrain = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/heart-train.csv'))
    htest = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/heart-test.csv'))
    ntrain = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/netflix-train.csv'))
    ntest = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/netflix-test.csv'))
    strain = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/simple-train.csv'))
    stest = csv.DictReader(open('/Users/PatrickMcEwen/109pset6/venv/pset6data/simple-test.csv'))
    #print(logistic_regression(strain, stest, 0.0001, 1000,  False))
    print(logistic_regression(ntrain, ntest, 0.00625, 100, True))
    print(logistic_regression(atrain, atest, 0.0001, 1000, False))
    print(logistic_regression(htrain, htest, 0.00001, 1000, True))
    return 0

if __name__ == '__main__':
    main()