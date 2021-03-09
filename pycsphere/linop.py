from pycsou.core import LinearOperator
import numpy as np
import healpy as hp
from healpy.pixelfunc import ud_grade
from typing import Optional, Callable, Union
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


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
        map = rng.binomial(n=1, p=0.2, size=hp.nside2npix(nside=nside))
        map = hp.smoothing(map, sigma=10 * np.pi / 180)
        pool = SphericalPooling(nside_in=nside, nside_out=8, pooling_func='sum')
        pooled_map = pool(map)
        backprojected_map = pool.adjoint(pooled_map)
        hp.mollview(map=map, title='Input Map', cmap='viridis')
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

    def __call__(self, map: np.ndarray) -> np.ndarray:
        return ud_grade(map_in=map, nside_out=self.nside_out, order_in=self.order_in, order_out=self.order_out,
                        dtype=self.dtype, power=self._power)

    def adjoint(self, pooled_map: np.ndarray) -> np.ndarray:
        return ud_grade(map_in=pooled_map, nside_out=self.nside_in, order_in=self.order_out, order_out=self.order_in,
                        dtype=self.dtype)


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
        map = 100 * rng.binomial(n=1, p=0.01, size=int(hp.nside2npix(nside=nside)))
        map = hp.smoothing(map, beam_window=np.ones(shape=(3*n_max//4,)))
        sht = SHT(n_max=n_max)
        anm = sht(map)
        synth_map = sht.adjoint(anm)
        hp.mollview(map=map, title='Input Map', cmap='viridis')
        sht.plot_anm(anm)
        hp.mollview(map=map, title='Synthesised Map', cmap='viridis')

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

    def __call__(self, map: np.ndarray) -> np.ndarray:
        r"""
        Compute the spherical harmonic transform.

        Parameters
        ----------
        map: np.ndarray
            Bandlimited spherical map discretised on a critical RING ordered HEALPix mesh.

        Returns
        -------
        np.ndarray
            Spherical harmonic coefficients :math:`\{\hat{a}_n^m\}\subset\mathbb{C}`.
        """
        return hp.map2alm(maps=map, lmax=self.n_max, use_weights=self.use_weights, verbose=self.verbose)

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

    Compute the Fourier Legendre Transform of a function :math:`f:[0,\pi]\to\mathbb{C}`. This is useful for computing the
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

        theta = np.linspace(0, np.pi, 4096)
        b = (theta <= np.pi / 4)
        flt = FLT(n_max=40, theta=theta)
        bn = flt(b)
        trunc_fl_series = flt.adjoint(bn)
        plt.figure()
        plt.plot(theta, b)
        plt.xlabel('$\\theta$')
        plt.title('Original Signal')
        plt.figure()
        plt.stem(np.arange(flt.n_max + 1), bn)
        plt.xlabel('$n$')
        plt.title('Fourier-Legendre coefficients')
        plt.figure()
        plt.plot(theta, trunc_fl_series)
        plt.xlabel('$\\theta$')
        plt.title('Truncated Fourier-Legendre Expansion')

    .. plot::

        import healpy as hp
        import numpy as np
        from pycsphere.linop import FLT
        import matplotlib.pyplot as plt

        theta = np.linspace(0, np.pi, 4096)
        bn = np.ones(21)
        flt = FLT(n_max=20, theta=theta)
        b = flt.adjoint(bn)
        plt.figure()
        plt.stem(np.arange(flt.n_max + 1), bn)
        plt.xlabel('$n$')
        plt.title('Fourier-Legendre coefficients')
        plt.figure()
        plt.plot(theta, b)
        plt.xlabel('$\\theta$')
        plt.title('Fourier-Legendre Expansion')


    Notes
    -----
    Let :math:`\{P_{n,d}:[-1,1]\rightarrow\mathbb{C}, \, n\in\mathbb{N}\}` be the *Legendre polynomials*.
    Then, any  function :math:`b\in\mathcal{L}^2([0, \pi], \mathbb{C})` admits a *Fourier-Legendre expansion* given by

    .. math::

        b(\theta)\stackrel{a.e.}{=}\sum_{n=0}^{+\infty} \hat{b}_n\,\frac{2n+1}{4\pi} P_{n}(\cos\theta),

    where the *Fourier-Legendre coefficients* are given by the *Fourier-Legendre transform*

    .. math::

        \hat{b}_n:=2\pi \int_{0}^\pi b(\theta) P_{n}(\cos\theta) sin\theta \,d\theta, \quad n\geq 0.

    The Fourier-Legendre transform is computed with the routine
    :py:func:`healpy.sphtfunc.beam2bl` which leverages a
    recurrence relationship for computing efficiently Legendre polynomials, and a trapezoidal rule for approximating the integral.
    The inverse (adjoint) Fourier-Legendre transform could be computed via the function :py:func:`healpy.sphtfunc.bl2beam` but the latter
    has `a bug which discards the last coefficient <https://github.com/healpy/healpy/issues/666>`_.
    We therefore propose a correct implementation here, pending a fix in the healpy library.

    Warnings
    --------
    Using this function with ``n_max`` smaller than the function's bandwidth may result in aliasing/smoothing artefacts.

    See Also
    --------
    :py:class:`~pycsphere.linop.FLT`, :py:class:`~pycsphere.linop.SphericalHarmonicTransform`
    """

    def __init__(self, n_max: int, theta: np.ndarray, dtype: type = np.float64):
        r"""

        Parameters
        ----------
        n_max: int
            Maximal Fourier-Legendre coefficient index :math:`n`.
        theta: np.ndarray
            Grid of :math:`[0,\pi]` used to approximate the integral when computing the Fourier-Legendre coefficients.
        dtype: type
            Data type of the operator.

        Raises
        ------
        Warning
            If the resolution of ``theta`` is too crude for the chosen ``n_max``.
        """
        self.n_max = n_max
        self.theta = theta
        self._nside = SphericalHarmonicTransform.nmax2nside(n_max)
        self.min_resolution_theta = 4 * self._nside - 1
        if self.min_resolution_theta > self.theta.size:
            raise Warning('Resolution of the theta grid is too low. Consider increasing it for higher accuracy.')
        super(FourierLegendreTransform, self).__init__(shape=(self.n_max, self.theta.size), dtype=dtype,
                                                       lipschitz_cst=1)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        r"""
        Compute the Fourier-Legendre coefficients :math:`\{\hat{b}_n, n=0,\ldots, n_{max}\}`.

        Parameters
        ----------
        b: np.ndarray
            Function :math:`b` sampled at the points ``theta``.

        Returns
        -------
        np.ndarray
            The Fourier-Legendre coefficients :math:`\{\hat{b}_n, n=0,\ldots, n_{max}\}`.
        """
        return hp.beam2bl(beam=b, theta=self.theta, lmax=self.n_max)

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
        x = np.cos(self.theta)
        p0 = np.zeros(self.theta.size, dtype=np.dtype) + 1
        p1 = x

        b = bn[0] * p0 + bn[1] * p1 * 3

        for n in np.arange(2, self.n_max + 1):
            p2 = (x * p1 * (2 * n - 1) - p0 * (n - 1)) / n
            p0 = p1
            p1 = p2
            b += bn[n] * p2 * (2 * n + 1)

        b /= 4 * np.pi

        return b


FLT = FourierLegendreTransform

if __name__ == '__main__':
    pass