#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Upper and lower MMSE bounds under Kullback-Leiber constraints
on the joint input-output distrbution
"""

import numpy as np

from numpy.linalg import inv
from scipy import optimize
from scipy.special import lambertw
from scipy.linalg import eigvalsh


def phi(t):
    return np.log(1 + t) - t / (1 + t)


def w(t, i):
    if t == 0.0:
        return 1.0
    else:
        return -np.real(lambertw(-np.exp(-(1 + 2 * t)), i))


def lower(Sigma0, K, M, eps):
    assert np.minimum(K, M) > 0
    assert eps >= 0
    assert Sigma0.shape == (K + M, K + M)
    assert np.allclose(Sigma0, Sigma0.T)

    A0 = Sigma0[0:K, 0:K]
    B0 = Sigma0[0:K, K:K + M]
    C0 = Sigma0[K:K + M, K:K + M]

    Xi0 = A0 - B0 @ inv(C0) @ B0.T
    xi0 = eigvalsh(Xi0)
    assert np.all(xi0 >= 0)

    if eps == 0.0:
        mmse = np.sum(xi0)
        return mmse, Sigma0
    elif K == 1:
        w0 = w(eps, 0)
        mmse_lb = w0 * xi0[0]
        Sigma = Sigma0.copy()
        Sigma[0, 0] -= (1 - w0) * xi0
        return mmse_lb, Sigma
    else:
        w0_min, w0_max = w(eps / K, 0), w(eps, 0)
        gamma_pos_max = (1 - w0_max) / (xi0[-1] * w0_max)
        gamma_pos_min = (1 - w0_min) / (xi0[-1] * w0_min)

        def phi_sum(gamma):
            return np.sum(phi(gamma * xi0)) - 2 * eps

        gamma_pos = optimize.brentq(phi_sum, gamma_pos_min, gamma_pos_max)
        mmse_lb = np.sum(xi0 / (1 + gamma_pos * xi0))
        Sigma = Sigma0.copy()
        Sigma[0:K, 0:K] -= gamma_pos * Xi0 * inv(np.eye(K) + gamma_pos * Xi0) * Xi0

        return mmse_lb, Sigma


def upper(Sigma0, K, M, eps):
    assert np.minimum(K, M) > 0
    assert eps >= 0
    assert Sigma0.shape == (K + M, K + M)
    assert np.allclose(Sigma0, Sigma0.T)

    A0 = Sigma0[0:K, 0:K]
    B0 = Sigma0[0:K, K:K + M]
    C0 = Sigma0[K:K + M, K:K + M]

    Xi0 = A0 - B0 @ inv(C0) @ B0.T
    xi0 = eigvalsh(Xi0)
    assert np.all(xi0 >= 0)

    if eps == 0.0:
        mmse = np.sum(xi0)
        return mmse, Sigma0
    elif K == 1:
        w1 = w(eps, -1)
        mmse_ub = w1 * xi0[0]
        Sigma = Sigma0.copy()
        Sigma[0, 0] -= (1 - w1) * xi0
        return mmse_ub, Sigma
    else:
        w1_min, w1_max = w(eps, -1), w(eps / K, -1)
        gamma_neg_min = (1 - w1_min) / (xi0[-1] * w1_min)
        gamma_neg_max = (1 - w1_max) / (xi0[-1] * w1_max)

        def phi_sum(gamma):
            return np.sum(phi(gamma * xi0)) - 2 * eps

        gamma_neg = optimize.brentq(phi_sum, gamma_neg_min, gamma_neg_max)
        mmse_ub = np.sum(xi0 / (1 + gamma_neg * xi0))
        Sigma = Sigma0.copy()
        Sigma[0:K, 0:K] -= gamma_neg * Xi0 * inv(np.eye(K) + gamma_neg * Xi0) * Xi0

        return mmse_ub, Sigma


if __name__ == "__main__":
    # run a simple test
    K, M = 6, 4
    eps = 0.1
    R = np.random.randn(K + M, K + M)
    Sigma0 = R.T @ R
    lb = lower(Sigma0, K, M, eps)[0]
    ub = upper(Sigma0, K, M, eps)[0]
    assert lb < ub

    # check eps = 0 special case
    lb = lower(Sigma0, K, M, 0.0)[0]
    ub = upper(Sigma0, K, M, 0.0)[0]
    assert lb == ub

    # check K=1 special case
    K = 1
    R = np.random.randn(K + M, K + M)
    Sigma0 = R.T @ R
    lb = lower(Sigma0, K, M, eps)[0]
    ub = upper(Sigma0, K, M, eps)[0]
    assert lb < ub
