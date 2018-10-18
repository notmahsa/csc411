# -*- coding: utf-8 -*-
#
# CSC411
# Assignment 3
# Mahsa Naserifar
#
#

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import scipy as sc

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    val = []
    for x_t in x_train:
        val += [- np.power(np.linalg.norm(test_datum - x_t), 2) / (2 * tau * tau)]
    denom = np.exp(sc.misc.logsumexp(np.array(val) - max(val)))
    a = []
    for x_t in x_train:
        a.append(np.exp(- np.power(np.linalg.norm(test_datum - x_t), 2) / (2 * tau * tau) - max(val)) / denom)
    A = np.diag(a)
    lhs = x_train.transpose().dot(A).dot(x_train) + lam * np.identity(len(x_train.transpose()))
    rhs = x_train.transpose().dot(A).dot(y_train)
    w = np.linalg.solve(lhs, rhs)
    y = test_datum.transpose().dot(w)
    return y


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    np.random.shuffle(x_t)
    np.random.shuffle(y_t)
    val_x, train_x = x[:int(len(x) * val_frac)], x[int(len(x) * val_frac):]
    val_y, train_y = y[:int(len(y) * val_frac)], y[int(len(y) * val_frac):]
    val_loss = []
    for tau in taus:
        pred = np.array([LRLS(i, train_x, train_y, tau) for i in val_x])
        loss = np.average(pred - np.array(val_y))
        val_loss.append(loss)
    train_loss = []
    for tau in taus:
        pred = np.array([LRLS(i, train_x, train_y, tau) for i in train_x])
        loss = np.average(pred - np.array(train_y))
        train_loss.append(loss)
    return train_loss, val_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
    plt.show()
