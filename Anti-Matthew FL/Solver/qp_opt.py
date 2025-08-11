import torch
import numpy as np
import cvxopt
from cvxopt import matrix
import os
import random


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = 0.5 * (P + P.T)
    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if "optimal" not in sol["status"]:
        optimal_flag = 0
    return np.array(sol["x"]).reshape((P.shape[1],)), optimal_flag


def setup_qp_and_solve(vec):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = -np.eye(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    cvxopt.solvers.options["show_progress"] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag
