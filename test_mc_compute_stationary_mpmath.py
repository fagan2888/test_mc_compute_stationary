"""
Filename: test_mc_compute_stationary_mpmath.py
Author: Daisuke Oyama

Nose test for mc_compute_stationary_mpmath

Input Markov matrices defined by the Kandori-Mailath-Rob model with
- two actions (0 and 1),
- payoffs being characterized by the level of p-dominance of action 1,
- N players, and
- mutation probability epsilon.

References
----------

    https://github.com/oyamad/test_mc_compute_stationary

"""

from __future__ import division

import sys
import random
import argparse
import numpy as np
from scipy.stats import binom
import unittest
import nose
from nose.tools import ok_, eq_
from mc_compute_stationary_mpmath import mc_compute_stationary_mpmath


# Sets of parameter values
pvalues_list = [
                {'N': 27, 'epsilon': 1e-2, 'move': 'sequential'},
                {'N': 3, 'epsilon': 1e-14, 'move': 'sequential'},
                {'N': 5, 'epsilon': 1e-15, 'move': 'simultaneous'}
                ]

# Value of p-dominance of action 1
p = 1/3  # action 1 is risk-dominant

# Tolerance level
TOL = 1e-17


def KMR_Markov_matrix_simultaneous(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *simultaneous* move

    Parameters
    ----------
    N : int
        Number of players

    p : float
        Level of p-dominance of action 1, i.e.,
        the value of p such that action 1 is the BR for (1-q, q) for any q > p,
        where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)

    epsilon : float
        Probability of mutation

    Returns
    -------
    P : numpy.ndarray
        Markov matrix for the KMR model with simultaneous move

    Notes
    -----
    For simplicity, the transition probabilities are computed under the assumption
    that a player is allowed to be matched to play with himself.

    """
    P = np.empty((N+1, N+1), dtype=float)
    for n in range(N+1):
        P[n, :] = \
            (n/N < p) * binom.pmf(range(N+1), N, epsilon/2) + \
            (n/N == p) * binom.pmf(range(N+1), N, 1/2) + \
            (n/N > p) * binom.pmf(range(N+1), N, 1-epsilon/2)
    return P


def KMR_Markov_matrix_sequential(N, p, epsilon):
    """
    Generate the Markov matrix for the KMR model with *sequential* move

    Parameters
    ----------
    N : int
        Number of players

    p : float
        Level of p-dominance of action 1, i.e.,
        the value of p such that action 1 is the BR for (1-q, q) for any q > p,
        where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)

    epsilon : float
        Probability of mutation

    Returns
    -------
    P : numpy.ndarray
        Markov matrix for the KMR model with simultaneous move

    """
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = \
            (n/N) * (epsilon * (1/2) +
                     (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = \
            ((N-n)/N) * (epsilon * (1/2) +
                         (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                         )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


def test_compute_stationary():
    for i, pvalues in enumerate(pvalues_list):
        if pvalues['move'] == 'simultaneous':
            P = KMR_Markov_matrix_simultaneous(pvalues['N'], p, pvalues['epsilon'])
        else:
            P = KMR_Markov_matrix_sequential(pvalues['N'], p, pvalues['epsilon'])
        vecs = mc_compute_stationary_mpmath(P)  # List of eigenvectors
        v = np.array(random.choice(vecs), dtype=float)
        # Pick a random element from vecs and convert it numpy.float

        print '===='
        print 'Testing with prameter values set %d\n' % i
        print 'N =', pvalues['N'], ', epsilon =', pvalues['epsilon'], ', move =', pvalues['move'], '\n'
        if pvalues['N'] <= 5:
            print 'P =\n', P, '\n'
        print 'v (converted to numpy.float) =\n', v, '\n'
        print 'TOL =', TOL, '\n'

        yield MarkovMatrix(), P
        yield SumOne(), v
        yield Nonnegative(), v
        yield LeftEigenVec(), P, v


class MarkovMatrix:
    def __init__(self):
        self.description = 'Elements in each row of P sum to one'
    def __call__(self, P):
        for i in range(len(P)):
            eq_(sum(P[i, :]), 1)

class SumOne:
    def __init__(self):
        self.description = 'Elements of v sum to one'
    def __call__(self, v):
        ok_(np.allclose(sum(v), 1, atol=TOL))

class Nonnegative:
    def __init__(self):
        self.description = 'All the elements of v are nonnegative'
    def __call__(self, v):
        eq_(np.prod(v >= 0-TOL), 1)

class LeftEigenVec:
    def __init__(self):
        self.description = 'v is a left eigen vector'
    def __call__(self, P, v):
        ok_(np.allclose(np.dot(v, P), v, atol=TOL))


if __name__ == '__main__':
    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
