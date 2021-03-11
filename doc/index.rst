.. image:: /images/pycsou.png
  :width: 50 %
  :align: center
  :target: https://github.com/matthieumeo/pycsou-sphere

.. image:: https://zenodo.org/badge/277582581.svg
   :target: https://zenodo.org/badge/latestdoi/277582581


*Pycsou-sphere* is an extension module of the Python 3 package `Pycsou <https://github.com/matthieumeo/pycsou>`_ for solving linear inverse problems on the sphere. The extension offers implementations of spherical zonal *convolution* operators as well as the spherical harmonic and Fourier-Legendre transforms (all compatible with Pycsou's interface for linear operators). It also provides numerical routines for computing the Green kernels of common spherical pseudo-differential operators and generating spherical meshes/point sets. 

This module heavily relies and follows similar conventions as the `healpy <https://healpy.readthedocs.io/en/latest/index.html>`_ package for spherical signal processing with Python. 

Content
-------

The package is organised as follows:

1. The module ``pycsphere.linop`` implements the following common spherical linear operators:
  
.. autosummary::

  pycsphere.linop.ZonalSphericalConvolution
  pycsphere.linop.SphericalPooling
  pycsphere.linop.DiscreteSphericalLaplacian
  pycsphere.linop.DiscreteSphericalGradient
  pycsphere.linop.SphericalHarmonicTransform
  pycsphere.linop.FourierLegendreTransform

2. The module ``pycsphere.mesh`` provides routines for generating spherical meshes:

.. autosummary::

  pycsphere.mesh.FibonacciPointSet
  pycsphere.mesh.HEALPixPointSet
  pycsphere.mesh.RandomPointSet

3. The module ``pycsphere.green`` provides numerical routines for computing the Green  kernels of common spherical pseudo-differential operators:

.. autosummary::

  pycsphere.green.ZonalGreenSobolev
  pycsphere.green.ZonalGreenFractionalLaplaceBeltrami
  pycsphere.green.ZonalGreenIteratedLaplaceBeltrami
  pycsphere.green.ZonalGreenBeltrami
  pycsphere.green.ZonalGreenIteratedBeltrami
  pycsphere.green.ZonalMatern
  pycsphere.green.ZonalWendland


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   general/install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: More

   notes/index

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
