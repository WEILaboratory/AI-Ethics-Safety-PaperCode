from sklearn.metrics import accuracy_score
import numpy as np


def Ua(dataset, M, z_index):
    y_pred = M(dataset[:, :-1])
    accuracy = accuracy_score(dataset[:, -1], y_pred)
    return accuracy


def pz_1(dataset, M, z_index):
    y_pred = M(dataset[:, :-1])
    y_pred_z1 = y_pred[np.where(dataset[:, z_index] > 0)[0]]

    pz_1 = np.count_nonzero(y_pred_z1 == 1) / (
        np.count_nonzero(dataset[:, z_index] > 0) + 0.0001
    )
    return pz_1


def pz_0(dataset, M, z_index):
    y_pred = M(dataset[:, :-1])
    y_pred_z0 = y_pred[np.where(dataset[:, z_index] <= 0)[0]]
    pz_0 = np.count_nonzero(y_pred_z0 == 1) / (
        np.count_nonzero(dataset[:, z_index] <= 0) + 0.0001
    )
    return pz_0


def Uf(dataset, M, z_index):
    Uf = pz_1(dataset, M, z_index) - pz_0(dataset, M, z_index)
    return Uf
