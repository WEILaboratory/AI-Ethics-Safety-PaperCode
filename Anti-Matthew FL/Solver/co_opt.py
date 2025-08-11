import numpy as np
import cvxpy as cp
import cvxopt
import pdb

# from cvxpylayers.torch import CvxpyLayer
import torch


class CO_OPT(object):  # hard-constrained optimization
    def __init__(self, n, eps_delta):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.eps_delta = eps_delta
        # self.objs = objs # the two objs [l, g].
        self.n = n  # the dimension of \theta
        self.deltas = cp.Parameter(2)  # the two deltas of the objectives [l1, l2]
        self.Ca1 = cp.Parameter((2, 1))  # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.Ca2 = cp.Parameter((2, 1))

        self.alpha = cp.Variable((1, 2))  # Variable to optimize
        # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Maximize(self.alpha @ self.Ca1)
        obj_fair = cp.Maximize(self.alpha @ self.Ca2)

        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1]
        constraints_fair = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,
            self.alpha @ self.Ca1 >= 0,
        ]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.disparity = 0  # Stores the latest maximum of selected K disparities

    def get_alpha(
        self, dis_max, d_gradient, deltas, eps, factor_delta, lr_delta, goal_index
    ):
        # # norm
        # for i in range(len(d_gradient)):
        #     gn = np.sqrt(np.mean([gr.pow(2).sum().data.cpu() for gr in d_gradient[i]]))
        #     d_gradient[i] /= gn

        d_ls = torch.cat(d_gradient)
        if dis_max[1] <= eps:  # [l, g] disparities < eps0
            # self.Ca1.value = d_ls @ d_l1
            self.Ca1.value = (d_ls @ d_gradient[0].t()).cpu().numpy()
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
            if self.eps_delta != None:
                for i in range(len(d_gradient)):
                    if (
                        self.eps_delta[goal_index[i]] < deltas[goal_index[i]]
                        and np.linalg.norm(d_gradient[i].cpu()) * factor_delta
                        <= deltas[goal_index[i]]
                    ):
                        deltas[goal_index[i]] = lr_delta * deltas[goal_index[i]]

            return self.alpha.value, deltas

        else:
            self.Ca1.value = (d_ls @ d_gradient[0].t()).cpu().numpy()
            self.Ca2.value = (d_ls @ d_gradient[1].t()).cpu().numpy()
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            if self.eps_delta != None:
                for i in range(len(d_gradient)):
                    if (
                        self.eps_delta[goal_index[i]] < deltas[goal_index[i]]
                        and np.linalg.norm(d_gradient[i].cpu()) * factor_delta
                        <= deltas[goal_index[i]]
                    ):
                        deltas[goal_index[i]] = lr_delta * deltas[goal_index[i]]
            self.last_move = "fair"
            return self.alpha.value, deltas


class EFO_OPT(object):  # hard-constrained optimization
    def __init__(self, n, eps_delta):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.eps_delta = eps_delta
        # self.objs = objs # the two objs [l, g].
        self.n = n  # the dimension of \theta
        self.deltas = cp.Parameter(4)  # the two deltas of the objectives [l1, l2]
        self.Ca1 = cp.Parameter((4, 1))  # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.Ca2 = cp.Parameter((4, 1))
        self.Ca3 = cp.Parameter((4, 1))  # [d_l, d_g] * d_l or [d_l, d_g] * d_g.
        self.Ca4 = cp.Parameter((4, 1))

        self.alpha = cp.Variable((1, 4))  # Variable to optimize
        # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Maximize(self.alpha @ self.Ca3)
        obj_fair = cp.Maximize(self.alpha @ self.Ca4)

        constraints_dom = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,
            self.alpha @ self.Ca1 >= 0,
            self.alpha @ self.Ca2 >= 0,
        ]
        constraints_fair = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,
            self.alpha @ self.Ca1 >= 0,
            self.alpha @ self.Ca2 >= 0,
            self.alpha @ self.Ca3 >= 0,
        ]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.disparity = 0  # Stores the latest maximum of selected K disparities

    def get_alpha(
        self, dis_max, d_l1, d_l2, d_l3, d_l4, deltas, eps, factor_delta, lr_delta
    ):
        d_ls = torch.cat((d_l1, d_l2, d_l3, d_l4))

        self.Ca1.value = (d_ls @ d_l1.t()).cpu().numpy()
        self.Ca2.value = (d_ls @ d_l2.t()).cpu().numpy()
        self.Ca3.value = (d_ls @ d_l3.t()).cpu().numpy()
        self.Ca4.value = (d_ls @ d_l4.t()).cpu().numpy()
        if dis_max[1] <= self.eps[0]:  # [l, g] disparities < eps0
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
            if (
                self.eps[3] < deltas[2]
                and np.linalg.norm(d_l3.cpu()) * factor_delta <= deltas[2]
            ):
                deltas[2] = lr_delta * deltas[2]
            if (
                self.eps[4] < deltas[3]
                and np.linalg.norm(d_l4.cpu()) * factor_delta <= deltas[3]
            ):
                deltas[3] = lr_delta * deltas[3]
            return self.alpha.value, deltas

        else:
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            if (
                self.eps[3] < deltas[2]
                and np.linalg.norm(d_l3.cpu()) * factor_delta <= deltas[2]
            ):
                deltas[2] = lr_delta * deltas[2]
            if (
                self.eps[4] < deltas[3]
                and np.linalg.norm(d_l4.cpu()) * factor_delta <= deltas[3]
            ):
                deltas[3] = lr_delta * deltas[3]
            self.last_move = "fair"
            return self.alpha.value, deltas
