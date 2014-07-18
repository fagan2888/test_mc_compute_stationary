"""
Filename: mc_compute_stationary_sympy.py

Author: Daisuke Oyama

"""
import numpy as np
from sympy.matrices import Matrix
from sympy.utilities.iterables import flatten


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
    vals : list of sympy.core.numbers.Float
        A list of eigenvalues of P > 1 - tol

    ndarray_vects : list of numpy.arrays of sympy.core.numbers.Float
        A list of the corresponding eigenvectors of P

    """
    P = Matrix(P)  # type(P): sympy.matrices.dense.MutableDenseMatrix
    outputs = P.transpose().eigenvects()  # TODO: Raise exception when empty

    vals = []
    vecs = []

    # output = (eigenvalue, algebraic multiplicity, [eigenvectors])
    for output in outputs:
        if output[0] > 1 - tol:
            vals.append(output[0])
            vecs.extend(output[2])

    # type(vec): sympy.matrices.dense.MutableDenseMatrix
    vecs_flattened = \
        np.array([flatten(vec) for vec in vecs])

    return vals, [vec/sum(vec) for vec in vecs_flattened]
