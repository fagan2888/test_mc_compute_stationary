test_mc_compute_stationary.py
=============================

Nose test for `mc_compute_stationary`
in [mc_tools.py](https://github.com/jstac/quant-econ/blob/master/quantecon/mc_tools.py)
from [quantecon](https://github.com/jstac/quant-econ)

Check whether all the elements of the computed stationary distribution are nonnegative
for the Markov matrix generated by the KMR model
with a large number of players or a very small probability of mutation.

(Instances of outputs with negative values have initially been found by students from my undergrad seminar,
[Takayasu Ohno](https://github.com/beeleb) and [Atsushi Yamagishi](https://github.com/haru110jp).)

Run

```sh
$ python test_mc_compute_stationary.py
```

or

```sh
$ nosetests -v -s
```


## IPython Notebooks

* [Outcome 1](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/test_mc_compute_stationary_2_7_6.ipynb):
  Python 2.7.6 (installed via Homebrew) on Mac OS X 10.6.8 (Snow Leopard)
* [Outcome 2](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/test_mc_compute_stationary_2_7_6_anaconda.ipynb):
  Python 2.7.6 in Anaconda on Mac OS X 10.9.4 (Marvericks)
* [Outcome 3](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/test_mc_compute_stationary_2_7_8.ipynb):
  Python 2.7.8 (installed via Homebrew) on Mac OS X 10.9.4 (Marvericks)

The outputs in general differ across different environment,
while for some cases, 1 and 3 yield exactly the same outputs.


unittest_mc_compute_stationary.py
=================================

Manual Unittest for `mc_compute_stationary`
in [mc_tools.py](https://github.com/jstac/quant-econ/blob/master/quantecon/mc_tools.py)
from [quantecon](https://github.com/jstac/quant-econ)

Run

```sh
$ python unittest_mc_compute_stationary.py
```

or with parameter values, for example,

```sh
$ python unittest_mc_compute_stationary.py --N=3 --epsilon=1e-14
```

or

```sh
$ python unittest_mc_compute_stationary.py --move='simultaneous' --N=5 --epsilon=1e-15
```


## IPython Notebooks

* [Outcome 1](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/unittest_mc_compute_stationary_2_7_6.ipynb):
  Python 2.7.6 (installed via Homebrew) on Mac OS X 10.6.8 (Snow Leopard)
* [Outcome 2](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/unittest_mc_compute_stationary_2_7_6_anaconda.ipynb):
  Python 2.7.6 in Anaconda on Mac OS X 10.9.4 (Marvericks)
* [Outcome 3](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/unittest_mc_compute_stationary_2_7_8.ipynb):
  Python 2.7.8 (installed via Homebrew) on Mac OS X 10.9.4 (Marvericks)


mc_compute_stationary_sympy.py
==============================

Using `Matrix.eigenvects()` from [SymPy](http://sympy.org),
it returns (computed) eigenvectors whose (computed) eigenvalues are close to 1.

* [Demonstration](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/mc_compute_stationary_sympy_demo.ipynb)

In some cases, this does not work well, returning an empty list...


mc_compute_stationary_mpmath.py
===============================

This uses [mpmath](http://mpmath.org),
which allows arbitrary precision in floating-point arithmetic.

* [Main documentation](http://mpmath.org/doc/current/)
* [Eigenvalue problem](http://mpmath.org/doc/current/matrices.html#the-eigenvalue-problem)
* [Setting the precision](http://mpmath.org/doc/current/basics.html#setting-the-precision)

This seems to work very well:

* [Demonstration](http://nbviewer.ipython.org/github/oyamad/test_mc_compute_stationary/blob/master/mc_compute_stationary_mpmath_demo01.ipynb)
  (with sympy.mpmath)
