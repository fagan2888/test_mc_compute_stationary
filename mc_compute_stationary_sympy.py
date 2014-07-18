"""
Filename: mc_compute_stationary_sympy.py

Author: Daisuke Oyama

"""
import numpy as np
from sympy.matrices import Matrix


def mc_compute_stationary_sympy(P, tol=1e-10):
    """
    Computes the stationary distribution(s) of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    tol : scalar(float) optional
        Tolerance level
        Find eigenvectors for eigenvalues > 1 - tol

    Returns
    -------
    ndarray_eigenvecs : list of numpy.arrays(float, ndim=1)
        A list of the stationary distribution(s) of P

    """
    P = Matrix(P)  # type(P): sympy.matrices.dense.MutableDenseMatrix
    outputs = P.transpose().eigenvects()  # TODO: Raise exception when empty

    eigenvecs = []

    # output = (eigenvalue, algebraic multiplicity, [eigenvectors])
    for output in outputs:
        if output[0] > 1 - tol:
            eigenvecs.extend(output[2])

    # type(eigenvec): sympy.matrices.dense.MutableDenseMatrix
    ndarray_eigenvecs = \
        [np.array(eigenvec).flatten().astype(float) for eigenvec in eigenvecs]

    return [v/sum(v) for v in ndarray_eigenvecs]
