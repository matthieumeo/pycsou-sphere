.. image:: https://matthieumeo.github.io/pycsou/html/_images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://github.com/matthieumeo/pycsou-sphere

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


*Pycsou-sphere* is an extension module of the Python 3 package `Pycsou <https://github.com/matthieumeo/pycsou>`_ for solving linear inverse problems on the sphere. The extension offers implementations of spherical zonal *convolution* operators as well as the spherical harmonic and Fourier-Legendre transforms (all compatible with Pycsou's interface for linear operators). It also provides numerical routines for computing the Green kernels of common spherical pseudo-differential operators and generating spherical meshes/point sets. 

This module heavily relies and follows similar conventions as the `healpy <https://healpy.readthedocs.io/en/latest/index.html>`_ package for spherical signal processing with Python. 

Content
=======

The package, named `pycsphere <https://pypi.org/project/pycsphere>`_,  is organised as follows:

1. The subpackage ``pycsphere.linop`` implements the following common spherical linear operators:
  
   * Convolution/pooling operators,
   * Spherical transforms and their inverses,
   * Finite-difference spherical differential operators.

2. The subpackage ``pycsphere.mesh`` provides routines for generating spherical meshes. 
3. The subpackage ``pycsphere.green`` provides numerical routines for computing the Green  kernels of common spherical pseudo-differential operators.

Installation
============

Pycsou-sphere requires Python 3.6 or greater. It is developed and tested on x86_64 systems running MacOS and Linux.


Dependencies
------------

Before installing Pycsou-sphere, make sure that the base package `Pycsou <https://github.com/matthieumeo/pycsou>`_ is correctly installed on your machine.
Installation instructions for Pycsou are available at `that link <https://matthieumeo.github.io/pycsou/html/general/install.html>`_.

The package extra dependencies are listed in the files ``requirements.txt`` and ``requirements-conda.txt``.
It is recommended to install those extra dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
`Anaconda <https://www.anaconda.com/download/#linux>`_. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (a highly optimized BLAS library created by Intel).

.. code-block:: bash

   >> conda install --channel=conda-forge --file=requirements-conda.txt


Quick Install
-------------

Pycsou-sphere is also available on `Pypi <https://pypi.org/project/pycsou-sphere/>`_. You can hence install it very simply via the command:

.. code-block:: bash

   >> pip install pycsou-sphere

If you have previously activated your conda environment ``pip`` will install Pycsou in said environment.
Otherwise it will install it in your ``base`` environment together with the various dependencies obtained from the file ``requirements.txt``.


Developer Install
------------------

It is also possible to install Pycsou-sphere from the source for developers:


.. code-block:: bash

   >> git clone https://github.com/matthieumeo/pycsou-sphere
   >> cd <repository_dir>/
   >> pip install -e .

The package documentation can be generated with:

.. code-block:: bash

   >> conda install sphinx=='2.1.*'            \
                    sphinx_rtd_theme=='0.4.*'
   >> python3 setup.py build_sphinx

You can verify that the installation was successful by running the package doctests:

.. code-block:: bash

   >> python3 test.py


Cite
====

For citing this package, please see: http://doi.org/10.5281/zenodo.4486431



