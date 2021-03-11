# #############################################################################
# linop.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Common spherical linear operators.
"""

from pycsou.core import LinearOperator
import numpy as np
import healpy as hp
from healpy.pixelfunc import ud_grade
from typing import Optional, Callable, Union
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from pycgsp.linop.diff import GraphLaplacian, GraphGradient, GeneralisedGraphLaplacian
from pycgsp.graph import cvxhull_graph, healpix_nngraph
from pycsphere.mesh import SphericalPointSet, HEALPixPointSet


class ZonalSphericalConvolution(LinearOperator):
    r"""
    Zonal spherical convolution.

    Compute the convolution between a bandlimited *zonal kernel* :math:`\psi(\langle\mathbf{r},\mathbf{s}\rangle)` and a bandlimited *spherical map* :math:`f(\mathbf{r})`:

    .. math::

        \left\{\psi\ast f\right\}(\mathbf{r})=\int_{\mathbb{S}^{2}}\psi(\langle\mathbf{r},\mathbf{s}\rangle)f(\mathbf{s})\,d\mathbf{s},\quad \forall \mathbf{r}\in\mathbb{S}^{2}.

    Examples
    --------

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import SHT, FLT, ZonalSphericalConvolution
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        n_max = 30
        nside = SHT.nmax2nside(n_max)
        rng = np.random.default_rng(0)
        map_in = 100 * rng.binomial(n=1, p=0.01, size=int(hp.nside2npix(nside=nside)))
        n=np.arange(2000)
        spectral_window=1/(100+n*(n+1))**2
        flt=FLT(n_max=1999, t=np.linspace(-1,1,2048))
        zonal_filter=flt.adjoint(spectral_window)
        zonal_filter_interp=interp1d(flt.t, zonal_filter, assume_sorted=True)
        convOp = ZonalSphericalConvolution(size=map_in.size, spectral_window=spectral_window)
        map_smoothed = convOp(map_in)
        map_bismoothed = convOp.adjoint(map_smoothed)
        hp.mollview(map=map_in, title='Input Map', cmap='viridis')
        plt.figure()
        theta=np.linspace(-np.pi, np.pi, 1024)
        plt.plot(theta, zonal_filter_interp(np.cos(theta)))
        plt.title('Angular section of Zonal Filter')
        hp.mollview(map=map_smoothed, title='Smoothed Map', cmap='viridis')
        hp.mollview(map=map_bismoothed, title='Backprojected Smoothed Map', cmap='viridis')


    Notes
    -----
    The ``ZonalSphericalConvolution`` operator is *self-adjoint* and can be computed efficiently in the spherical harmonic domain:

    .. math::
        \left\{\psi\ast f\right\}(\mathbf{r})&=\int_{\mathbb{S}^{2}}\psi(\langle\mathbf{r},\mathbf{s}\rangle)f(\mathbf{s})\,d\mathbf{s}\\
        &= \sum_{n=0}^N\hat{\psi}_n\sum_{m=-n}^n \hat{f}_n^m Y_n^m(\mathbf{r}),\quad \forall \mathbf{r}\in\mathbb{S}^{2},

    where :math:`N` is the maximum between the bandwidth of :math:`f` and :math:`\psi`. To perform this computation, we use
    the routine :py:func:`healpy.sphtfunc.smoothing` which assumes a RING-ordered HEALPix discretisation of :math:`f`.

    Warnings
    --------
    * This class is for *real* spherical maps :math:`f` **only**.
    * Using this operator on non-bandlimited spherical maps :math:`f` incurs aliasing.
    * HEALPix maps used as inputs must be **RING ordered**.

    See Also
    --------
    :py:class:`~pycsphere.linop.SphericalHarmonicTransform`, :py:class:`~pycsphere.linop.FourierLegendreTransform`,
    :py:class:`~pycsphere.linop.BiZonalSphericalConvolution`
    """

    def __init__(self, size: int, spectral_window: Optional[np.ndarray] = None,
                 zonal_filter: Optional[np.ndarray] = None,
                 n_filter: Optional[int] = None, sigma: Optional[float] = None, use_weights: bool = False):
        r"""

        Parameters
        ----------
        size: int
            Size of the RING-ordered, HEALPix-discretised sherical map :math:`f`.
        spectral_window: Optional[np.ndarray]
            Fourier-Legendre coefficients :math:`\hat{\psi}_n` of the zonal filter. Overrides ``zonal_filter``, ``n_filter``
            and ``sigma``.
        zonal_filter: Optional[np.ndarray]
            Zonal filter :math:`\psi` discretised on [-1,1]. Overrides ``sigma``.
        n_filter: Optional[int]
            Bandwidth of the zonal filter :math:`\psi`. Only used if ``zonal_filter`` is specified.
        sigma: float
             Standard deviation of a Gaussian filter in radians.
        use_weights: bool
            If ``True``, use the ring weighting quadrature rule when computing the spherical harmonic transform.

        Notes
        -----
        The zonal filter can be specified in three ways:

            1. Via its Fourier-Legendre coefficients (keyword ``spectral_window``).
            2. Via its discretisation on [-1,1] and its bandwidth  (keywords ``zonal_filter`` and ``n_filter``).
            3. As a spherical Gaussian filter with standard deviation ``sigma`` in radians.

        If keywords from multiple scenarios are used, 1. overrides  2. and 3. and 2. overrides 3.
        """
        self.size = size
        self.use_weights = use_weights
        self.sigma = None
        if spectral_window is not None:
            self.window = spectral_window
        elif zonal_filter is not None and n_filter is not None:
            flt = FLT(n_max=n_filter, t=np.linspace(-1, 1, zonal_filter.size))
            self.window = flt(zonal_filter)
        elif sigma is not None:
            self.sigma = sigma
            self.window = None
        else:
            raise ValueError('Invalid filter specification.')
        super(ZonalSphericalConvolution, self).__init__(shape=(self.size, self.size))

    def __call__(self, map_in: np.ndarray, verbose: bool = False) -> np.ndarray:
        return hp.smoothing(map_in=map_in, sigma=self.sigma, beam_window=self.window, use_weights=self.use_weights,
                            verbose=verbose)

    def adjoint(self, map_in: np.ndarray) -> np.ndarray:
        return self.__call__(map_in=map_in)


class SphericalPooling(LinearOperator):
    r"""
    Spherical pooling operator.

    Pool an HEALPix map by summing/averaging children pixels nested in a common superpixel.

    Examples
    --------

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import SphericalPooling

        nside = 16
        rng = np.random.default_rng(0)
        map_in = rng.binomial(n=1, p=0.2, size=hp.nside2npix(nside=nside))
        map_in = hp.smoothing(map_in, sigma=10 * np.pi / 180)
        pool = SphericalPooling(nside_in=nside, nside_out=8, pooling_func='sum')
        pooled_map = pool(map_in)
        backprojected_map = pool.adjoint(pooled_map)
        hp.mollview(map=map_in, title='Input Map', cmap='viridis')
        hp.mollview(map=pooled_map, title='Pooled Map', cmap='viridis')
        hp.mollview(map=backprojected_map, title='Backprojected Map', cmap='viridis')

    Notes
    -----
    Pooling is performed via the function :py:func:`healpy.pixelfunc.ud_grade` from Healpy.
    The adjoint (*unpooling*) is performed by assigning the value of the superpixels through the pooling function (e.g. mean, sum) to each children
    pixels of the superpixels.
    """

    def __init__(self, nside_in: int, nside_out: int, order_in: str = 'RING', order_out: str = 'RING',
                 pooling_func: str = 'mean', dtype: type = np.float64):
        r"""

        Parameters
        ----------
        nside_in: int
            Parameter NSIDE of the input HEALPix map.
        nside_out: int
            Parameter NSIDE of the pooled HEALPix map.
        order_in: str ['RING', 'NESTED']
            Ordering of the input HEALPix map.
        order_out: str ['RING', 'NESTED']
            Ordering of the pooled HEALPix map.
        pooling_func: str ['mean', 'sum']
            Pooling function.
        dtype: type
            Data type of the linear operator.

        Raises
        ------
        ValueError
            If ``nside_out >= nside_in``.
        """
        if nside_out >= nside_in:
            raise ValueError('Parameter nside_out must be smaller than nside_in.')
        self.nside_in = nside_in
        self.nside_out = nside_out
        self.order_in = order_in
        self.order_out = order_out
        self._power = None if pooling_func == 'mean' else -2
        super(SphericalPooling, self).__init__(shape=(nside_out, nside_in), dtype=dtype)

    def __call__(self, map_in: np.ndarray) -> np.ndarray:
        return ud_grade(map_in=map_in, nside_out=self.nside_out, order_in=self.order_in, order_out=self.order_out,
                        dtype=self.dtype, power=self._power)

    def adjoint(self, pooled_map: np.ndarray) -> np.ndarray:
        return ud_grade(map_in=pooled_map, nside_out=self.nside_in, order_in=self.order_out, order_out=self.order_in,
                        dtype=self.dtype)


class DiscreteSphericalLaplacian(GraphLaplacian):
    r"""
    Discrete spherical Laplacian.

    Finite-difference approximation of the continuous spherical Laplacian for a map defined over a
    :py:class:`~pycsphere.mesh.SphericalPointSet`.

    Examples
    --------

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.mesh import HEALPixPointSet
        from pycsphere.linop import DiscreteSphericalLaplacian

        nside = 16
        rng = np.random.default_rng(0)
        map_in = rng.binomial(n=1, p=0.005, size=hp.nside2npix(nside=nside))
        map_in = hp.smoothing(map_in, sigma=10 * np.pi / 180)
        laplacian = DiscreteSphericalLaplacian(point_set=HEALPixPointSet(nside=nside))
        map_d2 = laplacian(map_in)
        hp.mollview(map=map_in, title='Input Map', cmap='viridis')
        hp.mollview(map=np.abs(map_d2), title='Magnitude of Laplacian Map', cmap='viridis')

    Notes
    -----
    The discrete Laplacian is computed as the Laplacian of the spherical point set's graph `Pycsou-gsp tessellation graphs <https://matthieumeo.github.io/pycsou-gsp/html/api/graphs/index.html#module-pycgsp.graph.__init__>`_ using
    :py:class:`pycgsp.linop.diff.GraphLaplacian`.

    """

    def __init__(self, point_set: SphericalPointSet, dtype: type = np.float64):
        r"""

        Parameters
        ----------
        point_set: SphericalPointSet
            Spherical point set on which the signal is defined.
        dtype: type
            Input type.
        """
        if isinstance(point_set, HEALPixPointSet):
            graph, _ = healpix_nngraph(nside=point_set.nside, cheb_normalized=False, compute_differential_operator=False)
        else:
            graph, _ = cvxhull_graph(R=point_set.vec.transpose(), cheb_normalized=False,
                                  compute_differential_operator=False)
        super(DiscreteSphericalLaplacian, self).__init__(Graph=graph, dtype=dtype)


class DiscreteSphericalGradient(GraphGradient):
    r"""
    Discrete spherical gradient.

    Finite-difference approximation of the continuous spherical gradient for a map defined over a
    :py:class:`~pycsphere.mesh.SphericalPointSet`.

    Notes
    -----
    The discrete gradient is computed as the gradient of the spherical point set's graph (see `Pycsou-gsp tessellation graphs <https://matthieumeo.github.io/pycsou-gsp/html/api/graphs/index.html#module-pycgsp.graph.__init__>`_)
    using :py:class:`pycgsp.linop.diff.GraphGradient`.

    """

    def __init__(self, point_set: SphericalPointSet, dtype: type = np.float64):
        r"""

        Parameters
        ----------
        point_set: SphericalPointSet
            Spherical point set on which the signal is defined.
        dtype: type
            Input type.
        """
        if isinstance(point_set, HEALPixPointSet):
            graph, _ = healpix_nngraph(nside=point_set.nside, cheb_normalized=False, compute_differential_operator=True)
        else:
            graph, _ = cvxhull_graph(R=point_set.vec.transpose(), cheb_normalized=False,
                                  compute_differential_operator=True)
        super(DiscreteSphericalGradient, self).__init__(Graph=graph, dtype=dtype)


class SphericalHarmonicTransform(LinearOperator):
    r"""
    Spherical Harmonic Transform (SHT).

    Compute the spherical harmonic transform of a **real** bandlimited spherical function :math:`f:\mathbb{S}^2\to\mathbb{R}`.

    Examples
    --------

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import SHT
        import matplotlib.pyplot as plt

        n_max = 20
        nside = SHT.nmax2nside(n_max)
        rng = np.random.default_rng(0)
        map_in = 100 * rng.binomial(n=1, p=0.01, size=int(hp.nside2npix(nside=nside)))
        map_in = hp.smoothing(map_in, beam_window=np.ones(shape=(3*n_max//4,)))
        sht = SHT(n_max=n_max)
        anm = sht(map_in)
        synth_map = sht.adjoint(anm)
        hp.mollview(map=map_in, title='Input Map', cmap='viridis')
        sht.plot_anm(anm)
        hp.mollview(map=synth_map, title='Synthesised Map', cmap='viridis')

    Notes
    -----
    Every function :math:`f\in\mathcal{L}^2(\mathbb{S}^{2})` admits a *spherical Fourier expansion* given by

    .. math::

        f\stackrel{\mathcal{L}^2}{=}\sum_{n=0}^{+\infty}\sum_{m=-n}^{n} \,\hat{a}_n^m \,Y_n^m,

    where the *spherical harmonic coefficients* :math:`\{\hat{a}_n^m\}\subset\mathbb{C}` of :math:`f` are given by the *Spherical Harmonic Transform*:

    .. math::

        \hat{a}_n^m=\int_{0}^\pi\int_{-\pi}^\pi f(\phi,\theta) \overline{Y_n^m(\phi,\theta)} \,\sin(\theta)d\phi d\theta.

    The functions :math:`Y_n^m:[-\pi,\pi[\times [0,\pi]\to \mathbb{C}` are called the *spherical harmonics* and are given by:

    .. math::

        Y_n^m(\phi,\theta):=\sqrt{\frac{(2n+1)(n-m)!}{4\pi (n+m)!}}P_n^m(\cos(\theta))e^{j m\phi}, \;\forall (\phi,\theta)\in[-\pi,\pi[\times [0,\pi],

    where :math:`P_n^m:[-1,1]\rightarrow \mathbb{R}` denote the *associated Legendre functions* (see Chapter 1 of [Rafaely]_).

    For bandlimited functions of order :math:`N\in\mathbb{N}` (:math:`|\hat{a}_n^m|=0\forall n>N`), the spherical harmonic coefficients
    can be approximated very accurately via the spherical quadrature rule (see `HEALPix help <https://healpix.sourceforge.io/html/fac_anafast.htm>`_):

    .. math::

        \hat{a}_n^m=\frac{4\pi}{N_{pix}}\sum_{p=1}^{N_{pix}} f(\phi_p,\theta_p) \overline{Y_n^m(\phi_p,\theta_p)}

    assuming a HEALPix spherical point set  :math:`\left\{\mathbf{r}_p(\phi_p,\theta_p)\in\mathbb{S}^2, p=1, \ldots, N_{pix}=12N_{side}^2\right\}` with
    :math:`2 N_{side}<N\leq 3 N_{side}-1`. The spherical harmonic transform and its inverse (adjoint) are computed with the routines
    :py:func:`healpy.sphtfunc.map2alm` and :py:func:`healpy.sphtfunc.alm2map` which compute the spherical harmonics efficiently via
    recurrence relations for Legendre polynomials on co-latitudes, and Fast Fourier Transforms on longitudes (see `HEALPix help <https://healpix.sourceforge.io/html/fac_anafast.htm>`_).
    If accuracy is a concern, ring-based quadrature rules can also be used with  the keyword ``use_weights=True``.

    Warnings
    --------
    * This class is for real spherical maps **only**. Complex spherical maps are not supported yet by the routines
      :py:func:`healpy.sphtfunc.map2alm` and :py:func:`healpy.sphtfunc.alm2map` which compute only half of the spherical harmonic coefficients,
      assuming symmetry.
    * Using this operator on non-bandlimited spherical maps incurs aliasing.
    * HEALPix maps used as inputs must be **RING ordered**.

    See Also
    --------
    :py:class:`~pycsphere.linop.SHT`, :py:class:`~pycsphere.linop.FourierLegendreTransform`
    """

    @classmethod
    def nmax2nside(cls, n_max: int) -> int:
        r"""
        Compute the critical HEALPix NSIDE parameter for a given bandwidth ``n_max``.

        Parameters
        ----------
        n_max: int
            Bandwidth of the map.

        Returns
        -------
        int
            The critical HEALPix NSIDE parameter.
        """
        return int(2 ** np.ceil(np.log2((n_max + 1) / 3)))

    def __init__(self, n_max: int, use_weights: bool = False, verbose: bool = False):
        r"""

        Parameters
        ----------
        n_max: int
            Bandwidth of the map.
        use_weights: bool
            If ``True``, use ring-based quadrature weights (more accurate), otherwise use uniform quadrature weights.
            See `HEALPix help <https://healpix.sourceforge.io/html/fac_anafast.htm>`_ for more information.
        verbose: bool
            If ``True`` prints diagnostic information.
        """
        if n_max < 0:
            raise ValueError('Parameter n_max must be a positive integer.')
        self.n_max = n_max
        self.use_weights = use_weights
        self.verbose = verbose
        self.nside = int(2 ** np.ceil(np.log2((self.n_max + 1) / 3)))
        self.n_pix = hp.nside2npix(self.nside)
        self.coeffs_size = hp.Alm.getsize(lmax=n_max)
        super(SphericalHarmonicTransform, self).__init__(shape=(self.coeffs_size, self.n_pix), dtype=np.float64,
                                                         lipschitz_cst=1)

    def __call__(self, map_in: np.ndarray) -> np.ndarray:
        r"""
        Compute the spherical harmonic transform.

        Parameters
        ----------
        map_in: np.ndarray
            Bandlimited spherical map discretised on a critical RING ordered HEALPix mesh.

        Returns
        -------
        np.ndarray
            Spherical harmonic coefficients :math:`\{\hat{a}_n^m\}\subset\mathbb{C}`.
        """
        return hp.map2alm(maps=map_in, lmax=self.n_max, use_weights=self.use_weights, verbose=self.verbose)

    def adjoint(self, anm: np.ndarray, nside: Optional[int] = None) -> np.ndarray:
        r"""
        Compute the inverse spherical harmonic transform.

        Parameters
        ----------
        anm: np.ndarray
            Spherical harmonic coefficients :math:`\{\hat{a}_n^m\}\subset\mathbb{C}`.

        Returns
        -------
        np.ndarray
            Synthesised bandlimited spherical map discretised on a critical RING ordered HEALPix mesh.

        """
        nside = self.nside if nside is None else nside
        return hp.alm2map(alms=anm, nside=nside, lmax=self.n_max, verbose=self.verbose)

    def anm2cn(self, anm: np.ndarray) -> np.ndarray:
        r"""
        Compute the angular power spectrum.

        The *angular power spectrum* is defined as:

        .. math::

            \hat{c}_n:=\frac{1}{2n+1}\sum_{m=-n}^n |\hat{a}_n^m|^2, \quad n\in \mathbb{N}.

        Parameters
        ----------
        anm: np.ndarray
            Spherical harmonic coefficients :math:`\{\hat{a}_n^m\}\subset\mathbb{C}`.
        Returns
        -------
        The *angular power spectrum* coefficients :math:`\hat{c}_n`.
        """
        return hp.alm2cl(anm, lmax=self.n_max)

    def anm_triangle(self, anm: np.ndarray) -> np.ndarray:
        r"""
        Arrange the spherical harmonic coefficients in a lower-triangular matrix where each row represents a level :math:`n`.

        Parameters
        ----------
        anm: np.ndarray
            Spherical harmonic coefficients.

        Returns
        -------
        np.ndarray
            Spherical harmonic coefficients arranged in a lower-triangular matrix.
        """
        n, m = hp.Alm.getlm(lmax=self.n_max, i=np.arange(self.coeffs_size))
        Anm = np.asarray(coo_matrix((anm, (n, m)), shape=(self.n_max + 1, self.n_max + 1)).todense())
        return Anm

    def plot_anm(self, anm: np.ndarray, cmap: str = 'viridis', cast: Callable = np.abs):
        r"""
        Plot the spherical harmonic coefficients.

        Parameters
        ----------
        anm: np.ndarray
            Spherical harmonic coefficients.
        cmap: str
            Colormap.
        cast: Callable
            Function to cast the complex coefficients into real coefficients (e.g. ``np.abs``, ``np.real``, ``np.imag``...)

        """
        Anm = self.anm_triangle(anm)
        triu = np.triu(np.ones(shape=Anm.shape), k=1).astype(bool)
        Anm[triu] = np.NaN
        plt.figure()
        plt.imshow(cast(Anm), cmap=cmap)
        plt.colorbar()
        plt.title('Spherical Harmonic Coefficients')
        plt.xlabel('m')
        plt.ylabel('n')


SHT = SphericalHarmonicTransform


class FourierLegendreTransform(LinearOperator):
    r"""
    Fourier Legendre Transform (FLT).

    Compute the Fourier Legendre Transform of a function :math:`f:[-1,1]\to\mathbb{C}`. This is useful for computing the
    spherical harmonics coefficients of spherical zonal functions of the form :math:`g(\mathbf{r})=f(\langle\mathbf{r}, \mathbf{s}\rangle)`.
    Indeed, for such functions, we have:

    .. math::

        \hat{g}_n^m=\hat{f}_n \sqrt{\frac{2n+1}{4\pi}}\delta_n^0, \quad \forall n,m,

    where :math:`\hat{f}_n` are the Fourier-Legendre coefficients of :math:`f`.  Moreover, from the Fourier-Legendre expansion we have also:

    .. math::

        f(\langle\mathbf{r}, \mathbf{s}\rangle)=\sum_{n=0}^{+\infty} \hat{f}_n\frac{2n+1}{4\pi} P_n(\langle\mathbf{r}, \mathbf{s}\rangle).

    Examples
    --------

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import FLT
        import matplotlib.pyplot as plt

        t = np.linspace(-1, 1, 4096)
        b = (np.arccos(t) <= np.pi / 4)
        flt = FLT(n_max=40, t=t)
        bn = flt(b)
        trunc_fl_series = flt.adjoint(bn)
        plt.figure()
        plt.plot(t, b)
        plt.xlabel('$t$')
        plt.title('Original Signal')
        plt.figure()
        plt.stem(np.arange(flt.n_max + 1), bn)
        plt.xlabel('$n$')
        plt.title('Fourier-Legendre coefficients')
        plt.figure()
        plt.plot(t, trunc_fl_series)
        plt.xlabel('$t$')
        plt.title('Truncated Fourier-Legendre Expansion')

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import FLT
        import matplotlib.pyplot as plt

        t = np.linspace(-1, 1, 4096)
        bn = np.ones(21)
        flt = FLT(n_max=20, t=t)
        b = flt.adjoint(bn)
        plt.figure()
        plt.stem(np.arange(flt.n_max + 1), bn)
        plt.xlabel('$n$')
        plt.title('Fourier-Legendre coefficients')
        plt.figure()
        plt.plot(t, b)
        plt.xlabel('$t$')
        plt.title('Fourier-Legendre Expansion')


    Notes
    -----
    Let :math:`\{P_{n}:[-1,1]\rightarrow\mathbb{C}, \, n\in\mathbb{N}\}` be the *Legendre polynomials*.
    Then, any  function :math:`b\in\mathcal{L}^2([-1, 1], \mathbb{C})` admits a *Fourier-Legendre expansion* given by

    .. math::

        b(t)\stackrel{a.e.}{=}\sum_{n=0}^{+\infty} \hat{b}_n\,\frac{2n+1}{4\pi} P_{n}(t),

    where the *Fourier-Legendre coefficients* are given by the *Fourier-Legendre transform*

    .. math::

        \hat{b}_n:=2\pi \int_{-1}^1 b(t) P_{n}(t) \,dt, \quad n\geq 0.

    This implementation of the Fourier-Legendre transform  leverages a
    recurrence relationship for computing efficiently Legendre polynomials, and a trapezoidal rule for approximating the integral.

    Warnings
    --------
    Using this function with ``n_max`` smaller than the function's bandwidth may result in aliasing/smoothing artefacts.

    See Also
    --------
    :py:class:`~pycsphere.linop.FLT`, :py:class:`~pycsphere.linop.SphericalHarmonicTransform`
    """

    @classmethod
    def nmax2t(cls, n_max: int, oversampling: float = 10.) -> np.ndarray:
        r"""
        Generate suitable samples ``t`` for a given ``n_max``.

        Parameters
        ----------
        n_max: int
            Maximal Fourier-Legendre coefficient index :math:`n`.
        oversampling: float
            Oversampling factor.
        Returns
        -------
        np.ndarray
            Samples ``t``.
        """
        critical_res = 4 * SHT.nmax2nside(n_max) - 1
        return np.linspace(-1, 1, np.ceil(oversampling * critical_res).astype(int))

    def __init__(self, n_max: int, t: np.ndarray, dtype: type = np.float64):
        r"""

        Parameters
        ----------
        n_max: int
            Maximal Fourier-Legendre coefficient index :math:`n`.
        t: np.ndarray
            Grid of :math:`[-1,1]` used to approximate the integral when computing the Fourier-Legendre coefficients.
        dtype: type
            Data type of the operator.
        """
        self.n_max = n_max
        self.t = t
        super(FourierLegendreTransform, self).__init__(shape=(self.n_max, self.t.size), dtype=dtype,
                                                       lipschitz_cst=1)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        r"""
        Compute the Fourier-Legendre coefficients :math:`\{\hat{b}_n, n=0,\ldots, n_{max}\}`.

        Parameters
        ----------
        b: np.ndarray
            Function :math:`b` sampled at the points ``t``.

        Returns
        -------
        np.ndarray
            The Fourier-Legendre coefficients :math:`\{\hat{b}_n, n=0,\ldots, n_{max}\}`.
        """

        bn = np.zeros(self.n_max + 1)

        p0 = 0 * self.t + 1
        p1 = self.t.copy()

        bn[0] = np.trapz(b * p0, self.t, dx=2 / self.t.size)
        bn[1] = np.trapz(b * p1, self.t, dx=2 / self.t.size)

        for n in np.arange(2, self.n_max + 1):
            p2 = (self.t * p1 * (2 * n - 1) - p0 * (n - 1)) / n
            bn[n] = np.trapz(b * p2, self.t, dx=2 / self.t.size)
            p0 = p1
            p1 = p2

        bn *= 2 * np.pi

        return bn

    def adjoint(self, bn: np.ndarray) -> np.ndarray:
        r"""
        Compute the Fourier-Legendre series truncated at ``n_max``.

        Parameters
        ----------
        bn: np.ndarray
            Fourier-Legendre coefficients :math:`\{\hat{b}_n, n=0,\ldots, n_{max}\}`.

        Returns
        -------
        np.ndarray
            The Fourier-Legendre series truncated at ``n_max``.
        """
        p0 = 0 * self.t + 1
        p1 = self.t.copy()

        b = bn[0] * p0 + bn[1] * p1 * 3

        for n in np.arange(2, self.n_max + 1):
            p2 = (self.t * p1 * (2 * n - 1) - p0 * (n - 1)) / n
            p0 = p1
            p1 = p2
            b += bn[n] * p2 * (2 * n + 1)

        b /= 4 * np.pi

        return b


FLT = FourierLegendreTransform

if __name__ == '__main__':
    pass

