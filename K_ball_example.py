#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import integrate
from scipy.special import gamma
from scipy.special import gammaln

import mmse_bounds


def H_func(a, b):
    def func(x, a, b):
        return np.log(1 + a * x) * (1 - x ** 2) ** (b - 1)

    if np.isscalar(b) and np.isscalar(a):
        return integrate.quad(lambda x: func(x, a, b), -1, 1)[0]
    else:
        if np.isscalar(b):
            b = np.ones_like(a) * b
        elif np.isscalar(a):
            a = np.ones_like(b) * a
        K = b.size
        return np.array(
            [integrate.quad(lambda x: func(x, a[i], b[i]), -1, 1)[0] for i in range(K)]
        )


def mean_log(c, r, K):
    B = gamma((K + 2) / 2) / (gamma((K + 1) / 2) * np.sqrt(np.pi))
    return np.log(c) + B * H_func(r / c, (K + 1) / 2)


def kl_div_input(K):
    return K / 2 - K / 2 * np.log(1 + K / 2) + gammaln(1 + K / 2)


def kl_div_channel(c, r, K):
    return 0.5 * (np.log(c ** 2 + r ** 2 / (K + 2)) - 2 * mean_log(c, r, K))


def kl_div_channel_apx(c, r, K):
    return 0.5 * (np.log(c ** 2 + r ** 2 / (K + 2)) - np.log(c ** 2 - r ** 2))


def kl_div_joint(c, r, K):
    return kl_div_input(K) + np.sum(kl_div_channel(c, r, K))


def kl_div_joint_apx(c, r, K):
    return kl_div_input(K) + np.sum(kl_div_channel_apx(c, r, K))


def mse_vs_K(c0, r, K_max):
    mmse_lb = np.zeros(K_max)
    mmse_ub = np.zeros(K_max)
    mmse_lb_apx = np.zeros(K_max)

    for K in range(1, K_max + 1):
        c = c0 * np.ones(K)
        s_X = r ** 2 / (K + 2)

        eps = kl_div_joint(c, r, K)
        eps_apx = kl_div_joint_apx(c, r, K)

        mmse_ub[K - 1] = K * s_X
        mmse_lb[K - 1] = K * mmse_bounds.lower(np.diag([s_X, 1]), 1, 1, eps / K)[0]
        mmse_lb_apx[K - 1] = (
            K * mmse_bounds.lower(np.diag([s_X, 1]), 1, 1, eps_apx / K)[0]
        )

    return mmse_ub, mmse_lb, mmse_lb_apx


def mse_vs_c(c1, c2, r):
    s_X = r ** 2 / 4
    C1, C2 = np.meshgrid(c1, c2)
    mmse_lb = np.zeros_like(C1)

    for i in range(C1.size):
        c = np.array([C1.flat[i], C2.flat[i]])
        eps = kl_div_joint(c, r, 4)
        mmse_lb.flat[i] = 4 * mmse_bounds.lower(np.diag([s_X, 1]), 1, 1, eps / 4)[0]

    return mmse_lb
