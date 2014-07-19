"""
Filename: mc_compute_stationary_mpmath.py

Author: Daisuke Oyama

"""
import sys
import numpy as np

try:
    try:
        import mpmath as mp
    except ImportError:
        import sympy.mpmath as mp
except ImportError:
    sys.stderr.write('Failed to import mpmath\n')
    sys.exit(1)


def mc_compute_stationary_mpmath(P, precision=None, tol=1e-17):
    """
    Computes the stationary distributions of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    precision : scalar(int), optional
        Decimal precision in float-point arithemetic with mpmath

    tol: scalar(float), optional
        Tolerance level
        Find eigenvectors for eigenvalues in [1-tol, 1+tol]

    Returns
    -------
    vecs : list of numpy.arrays of sympy.mpmath.ctx_mp_python.mpf
        A list of the eigenvectors of whose eigenvalues > 1 - 1e-(precision)

    References
    ----------

        http://mpmath.org/doc/current/matrices.html#the-eigenvalue-problem
        http://mpmath.org/doc/current/basics.html#setting-the-precision

    """
    if precision:
        tmp = mp.mp.dps  # Store the current decimal precision
        mp.mp.dps = precision  # Set decimal precision to precision

    TOL = str(tol)

    if not isinstance(P, np.ndarray):
        Q = np.empty([len(P), len(P)], dtype='|S{0}'.format(mp.mp.dps))
        for i, row in enumerate(P):
            Q[i] = ['{0:.{1}f}'.format(x, mp.mp.dps) for x in row]
        E, EL = mp.eig(mp.matrix(Q), left=True, right=False)
    else:
        E, EL = mp.eig(mp.matrix(P), left=True, right=False)

    vecs = []

    for i, val in enumerate(E):
        if mp.mpf(1) - mp.mpf(TOL) <= val <= mp.mpf(1) + mp.mpf(TOL):
            vec = np.array(EL[i, :].tolist()[0])
            vecs.append(vec/sum(vec))

    if precision:
        mp.mp.dps = tmp  # Restore the current decimal precision

    return vecs
