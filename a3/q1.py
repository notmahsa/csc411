#
# CSC411
# Assignment 3
# Mahsa Naserifar
#
#

import numpy as np

def gradient_descent(X, y, w, b, learning_rate, iter, zeta):
    x_trans = X.transpose()
    for i in range(0, iter):
        a = (np.dot(w, X) + b) - y
        if np.abs(a) <= zeta:
            loss = np.sum(a ** 2) / (2 * len(X))
        else:
            loss = zeta * (np.abs(a) - zeta / 2)
        gradient = np.dot(x_trans, loss) / len(X)
        w = w - learning_rate * gradient
    return w

