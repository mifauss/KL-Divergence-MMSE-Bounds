#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import toeplitz

import mmse_bounds


def sigma_lfd_2d(snr_db, eps):
    snr = 10 ** (snr_db / 10)
    xi0 = 1 / (1 + snr)

    gamma = (1 / xi0) * (1 - mmse_bounds.w(eps, -1)) / mmse_bounds.w(eps, -1)

    Sigma_lfd = np.array([[1, 1], [1, 1 + 1 / snr]])
    Sigma_lfd[0, 0] -= gamma * xi0 ** 2 / (1 + gamma * xi0)

    return Sigma_lfd


def mmse_upper_bound(snr_db, K, eps_vec):
    snr = 10 ** (snr_db / 10)
    SigmaX0 = np.exp(-0.9 * toeplitz(np.arange(K), np.arange(K)))
    Sigma0 = np.block([[SigmaX0, SigmaX0], [SigmaX0, SigmaX0 + np.eye(K) / snr]])

    i = 0
    mmse_ub = np.zeros_like(eps_vec)
    for eps in eps_vec:
        mmse_ub[i] = mmse_bounds.upper(Sigma0, K, K, eps)[0]
        i += 1

    return mmse_ub
