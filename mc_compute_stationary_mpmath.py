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


def mc_compute_stationary_mpmath(P, precision=17, ltol=0, utol=None):
    """
    Computes the stationary distributions of Markov matrix P.

    Parameters
    ----------
    P : array_like(float, ndim=2)
        A discrete Markov transition matrix

    precision : scalar(int), optional(default: 17)
        Decimal precision in float-point arithemetic with mpmath
        mp.mp.dps is set to *precision*

    ltol, utol: scalar(float), optional(default: ltol=0, utol=inf)
        Lower and upper tolerance levels
        Find eigenvectors for eigenvalues in [1-ltol, 1+utol]

    Returns
    -------
    vecs : list of numpy.arrays of sympy.mpmath.ctx_mp_python.mpf
        A list of the eigenvectors of whose eigenvalues in [1-ltol, 1+utol]

    Notes
    -----
    mpmath 0.18 or above is required.

    References
    ----------

        http://mpmath.org/doc/current

    """
    LTOL = ltol  # Lower tolerance level
    if utol is None:  # Upper tolerance level
        UTOL = 'inf'
    else:
        UTOL = utol

    with mp.workdps(precision):  # Temporarily change the working precision
        E, EL = mp.eig(mp.matrix(P), left=True, right=False)
        # E  : a list of length n containing the eigenvalues of A
        # EL : a matrix whose rows contain the left eigenvectors of A
        # See: github.com/fredrik-johansson/mpmath/blob/master/mpmath/matrices/eigen.py
        E, EL = mp.eig_sort(E, EL)  # Sorted in a descending order

        num_eigval_one = sum(
            mp.mpf(1) - mp.mpf(LTOL) <= val <= mp.mpf(1) + mp.mpf(UTOL)
            for val in E
            )

        vecs = [np.array((EL[i, :]/sum(EL[i, :])).tolist()[0])
                for i in range(len(EL)-num_eigval_one, len(EL))]

    return vecs
