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


def mc_compute_stationary_mpmath(P, dps=17):
    """
    Computes the stationary distributions of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    dps : scalar(int), optional
        Decimal precision
        Find eigenvectors for eigenvalues > 1 - 1e-(dps)

    Returns
    -------
    vecs : list of numpy.arrays of sympy.mpmath.ctx_mp_python.mpf
        A list of the eigenvectors of whose eigenvalues > 1 - 1e-(dps)

    References
    ----------

        http://mpmath.org/doc/current/matrices.html#the-eigenvalue-problem
        http://mpmath.org/doc/current/basics.html#setting-the-precision

    """
    tmp = mp.mp.dps  # Store the current decimal precision
    mp.mp.dps = dps  # Set decimal precision to dps

    TOL = '1e-{0}'.format(dps)  # Tolerance

    E, EL = mp.eig(mp.matrix(P), left=True, right=False)

    vecs = []

    for i, val in enumerate(E):
        if val > mp.mpf(1) - mp.mpf(TOL):
            vec = np.array(EL[i, :].tolist()[0])
            vecs.append(vec/sum(vec))

    mp.mp.dps = tmp  # Restore the current decimal precision

    return vecs
