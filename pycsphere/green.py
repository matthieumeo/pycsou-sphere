# #############################################################################
# green.py
# ========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Common zonal green kernels (see Chapter 4 of [FuncSphere]_ for a survey).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Optional, Union, Any, Callable
from pycsphere.linop import FLT
from pycsou.math.green import Matern, Wendland
from abc import ABC, abstractmethod


class TruncatedFourierLegendreSeries:
    r"""
    Truncated Fourier-Legendre series.

    Given coefficients :math:`\{\hat{f}_n, n=0,\ldots, N\}`, compute the *truncated Fourier-Legendre series*:

    .. math::

        f_N(t):=\sum_{n=0}^N \hat{f}_n \frac{2n+1}{4\pi} P_n(t), \quad \forall t\in [-1,1].

    Examples
    --------

    .. plot::

        import numpy as np
        from pycsphere.green import TruncatedFourierLegendreSeries

        n_max=20
        n=np.arange(0, n_max + 1).astype(np.float)
        fn=0*n; fn[n>0]=(n[n>0]*(n[n>0]+1))**(-1)
        fN=TruncatedFourierLegendreSeries(fn)
        theta=np.linspace(-np.pi, np.pi, 1024)
        plt.plot(theta, fN(np.cos(theta)))
        plt.ylabel('$f_N(\\cos\\theta)$')
        plt.xlabel('$\\theta\\in[-\\pi,\\pi]$')

    Notes
    -----
    The ``TruncatedFourierLegendreSeries`` is computed by interpolating linearly the result of :py:class:`~pycsphere.linop.FourierLegendreTransform`.

    """

    def __init__(self, fn: np.ndarray):
        r"""

        Parameters
        ----------
        fn: np.ndarray
         Fourier-Legendre coefficients.
        """
        self.fn = fn
        self.samples_t = FLT.nmax2t(fn.size - 1)
        flt = FLT(n_max=fn.size - 1, t=self.samples_t)
        self.samples_f = flt.adjoint(fn)
        self._interp_f = interp1d(self.samples_t, self.samples_f, kind='linear', assume_sorted=True)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        r"""

        Parameters
        ----------
        t: np.ndarray
            Query points in [-1,1].
        Returns
        -------
        np.ndarray
            Truncated Fourier-Legendre transform evaluated at the query points.
        """
        return self._interp_f(t)


class ZonalGreenFunction(ABC):
    r"""
    Base class for zonal green functions.

    Any subclass/instance of this class must implement the abstract method ``_compute_fl_coefficients``.

    Notes
    -----
    Consider a spherical pseudpo-differential operator acting on a spherical function :math:`h(\mathbf{r})` as:

    .. math::

        \mathcal{D} h:=\sum_{n=0}^{+\infty}\hat{D}_n \left[\sum_{m=-n}^{n}\hat{h}_n^m Y_n^m\right],

    where :math:`\{\hat{D}_n\}_{n\in\mathbb{N}}\in\mathbb{R}^\mathbb{N}` is a sequence of *real numbers* such that the set

	.. math::

	    \mathcal{K}_\mathcal{D}:=\left\{n\in\mathbb{N}:\; \vert \hat{D}_n\vert=0\right\},

	is *finite* and

	.. math::

	    \vert\hat{D}_n\vert=\Theta\left(n^p\right),

	for some real number :math:`p\geq 0`, called the *spectral growth order* of :math:`\mathcal{D}`.
	Then, the zonal Green kernel of :math:`\mathcal{D}` is given by Proposition 4 of [FuncSphere]_:

    .. math::

         \psi_{\mathcal{D}}(\langle\mathbf{r}, \mathbf{s}\rangle) := \sum_{{n\in\mathbb{N}\backslash\mathcal{K}_\mathcal{D}}}  \frac{2n+1}{4\pi \hat{D}_n}P_{n}(\langle\mathbf{r}, \mathbf{s}\rangle),\quad \mathbf{r}\in\mathbb{S}^{2},

    where :math:`P_{n}` are the Legendre polynomials. The summation above is truncated to a large enough :math:`N` called the effective bandwidth of the zonal Green kernel.
    """

    def __init__(self, order: Optional[float] = None, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        order: Optional[float]
            Spectral growth order of the associated pseudo-differential operator.
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.order = order
        if cutoff is None:
            self.rtol = rtol
            self.cutoff = self.get_cutoff()
        else:
            self.cutoff = cutoff
            self.rtol = self.get_tol()
        self.bandwidth = self.cutoff
        self.fourier_legendre_coefficients = self._compute_fl_coefficients()
        self.interp_green_kernel = self.green_kernel()

    def get_cutoff(self, min_cutoff: int = 16) -> int:
        r"""
        Compute the effective bandwidth of the zonal Green kernel.

        Parameters
        ----------
        min_cutoff: int
            Minimal bandwidth.
        Returns
        -------
        int
            Effective bandwidth of the zonal Green kernel.
        """
        cutoff = np.fmax(np.ceil((1 / self.rtol) ** (1 / self.order)), min_cutoff).astype(int)
        return cutoff

    def get_tol(self) -> float:
        return self.cutoff ** (-self.order)

    @abstractmethod
    def _compute_fl_coefficients(self) -> np.ndarray:
        r"""
        Compute the Fourier-Legendre coefficients :math:`1/\hat{D}_n` of the zonal Green kernel.

        Returns
        -------
        np.ndarray
            Fourier-Legendre coefficients :math:`1/\hat{D}_n` of the zonal Green kernel.
        """
        pass

    def green_kernel(self) -> Callable:
        r"""
        Compute the zonal Green kernel.

        Returns
        -------
        Callable
            The zonal Green kernel as a callable function.
        """
        return TruncatedFourierLegendreSeries(fn=self.fourier_legendre_coefficients)

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Evaluate the zonal Green kernel.

        Parameters
        ----------
        t:  Union[float, np.ndarray]
            Query points in :math:`[-1, 1]`.

        Returns
        -------
        Union[float, np.ndarray]

            Evaluation of the zonal Green kernel at the query points.
        """
        return self.interp_green_kernel(t)

    def plot(self, resolution: int = 1024, color: str = None, linewidth: Union[int, float] = 2, angles: bool = False,
             fraction: float = 1, fhandle: Optional[Any]=None):
        r"""
        Plot the zonal Green kernel.

        Parameters
        ----------
        resolution: int
            Resolution of the uniform grid on which the Green kernel is sampled.
        color: str
            Color of the line.
        linewidth: Union[int, float]
            Line width.
        angles: bool
            If ``True`` plot :math:`\psi(\cos(\theta))`, otherwise plot :math:`\psi(t)`.
        fraction: float
            Plot over ``[-fraction, fraction]`` (if ``angles==False``) or ``[-np.pi*fraction, np.pi*fraction]``
            (if ``angles==True``) with ``0<fraction<=1``.
        fhandle: Optional[Any]
            Figure handle in which to plot.
        """
        if not 0 < fraction <= 1:
            raise ValueError('Parameter fraction must be in (0,1]')
        if angles:
            theta = np.linspace(-np.pi * fraction, np.pi * fraction, resolution)
            x = theta * 180 / np.pi
            y = self.__call__(np.cos(theta))
        else:
            x = np.linspace(-fraction, fraction, resolution)
            y = self.__call__(x)
        y /= np.max(y)
        plt.figure(fhandle)
        plt.plot(x, y, '-', color=color, linewidth=linewidth)
        if angles:
            plt.xlabel('$\\theta$')
            plt.ylabel('$\\psi(\\cos\\theta)$')
        else:
            plt.xlabel('$t$')
            plt.ylabel('$\\psi(t)$')
        plt.title('Zonal Function')

    def spectrum(self, n_max=None, cast: Callable = np.real, fhandle: Optional[Any]=None, color_index: int=0):
        r"""
        Plot the Fourier-Legendre spectrum.
        """
        if n_max is None:
            n_max=self.cutoff+1
        plt.figure(fhandle)
        plt.stem(np.arange(0, n_max), cast(self.fourier_legendre_coefficients[:n_max]), use_line_collection=True,
                 linefmt=f'C{color_index}-', markerfmt=f'C{color_index}o')
        plt.xlabel('$n$')
        plt.ylabel('$\\hat{\\psi}_n$')
        plt.title('Fourier-Legendre Spectrum')


class Radial2Zonal(ZonalGreenFunction):
    r"""
    Transforms a radial Green function into a zonal Green kernel.

    Let :math:`\phi:\mathbb{R}_+\to \mathbb{C}` be a radial function. The zonal Green kernel associated to :math:`\phi` is defined as:

    .. math::

        \psi(t):=\phi(\sqrt{2-2t}), \qquad t\in [-1,1].

    Examples
    --------

    .. plot::

        from pycsphere.green import Radial2Zonal
        from pycsou.math import Matern

        radial_green=Matern(k=0, epsilon=1)
        zonal_green=Radial2Zonal(radial_green=radial_green, order=3 / 2)
        plt.plot(np.linspace(0,10,1024), radial_green(np.linspace(0,10,1024)))
        plt.xlabel('$x$')
        plt.ylabel('$\\phi(x)$')
        plt.title('Radial Function')
        zonal_green.plot()
        zonal_green.spectrum(n_max=20)

    """

    def __init__(self, radial_green: Callable, order: Optional[float] = None, rtol: float = 1e-4,
                 cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        radial_green: Callable
            Radial green function (must have a method ``__call__`` for evaluation).
        order: Optional[float]
            Spectral growth order of the associated pseudo-differential operator.
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.radial_green = radial_green
        self.interp_green_kernel = self.green_kernel()
        super(Radial2Zonal, self).__init__(order=order, rtol=rtol, cutoff=cutoff)

    def green_kernel(self):
        return lambda t: self.radial_green(np.sqrt(2 - 2 * t))

    def _compute_fl_coefficients(self):
        self.samples_t = FLT.nmax2t(self.cutoff)
        flt = FLT(n_max=self.cutoff, t=self.samples_t)
        return flt(self(self.samples_t))


class ZonalGreenExponentiated(ZonalGreenFunction):
    r"""
    Exponentiated zonal Green kernel.

    Consider a zonal Green kernel :math:`\psi:[-1,1]\to \mathbb{C}` with Fourier-Legendre expansion:

    .. math::

        \psi(t)=\sum_{n=0}^{+\infty}\frac{2n+1}{4\pi\hat{D}_n} P_n(t), \quad t\in[-1,1].

    Then, the exponentiated zonal Green kernel :math:`\psi^{(\alpha)}:[-1,1]\to \mathbb{C}, \; \alpha\in\mathbb{R}_+`
    has Fourier-Legendre expansion:

    .. math::

        \psi^{(\alpha)}(t)=\sum_{n=0}^{+\infty}\frac{2n+1}{4\pi\hat{D}^\alpha_n} P_n(t), \quad t\in[-1,1].

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalWendland, ZonalGreenExponentiated

        zonal_green=ZonalWendland(k=0)
        iterated_zonal_green=ZonalGreenExponentiated(zonal_green, exponent=2)
        zonal_green.plot(angles=True, fhandle=1)
        iterated_zonal_green.plot(angles=True, fhandle=1)
        plt.legend(['Original Green kernel', 'Iterated Green kernel'])
        zonal_green.spectrum(n_max=20, fhandle=2, color_index=0)
        iterated_zonal_green.spectrum(n_max=20, fhandle=2, color_index=1)
        plt.legend(['Original Green kernel', 'Iterated Green kernel'])

    """
    def __init__(self, base_green_kernel: ZonalGreenFunction, exponent: int, rtol: float = 1e-4,
                 cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        base_green_kernel: ZonalGreenFunction
            Zonal Green kernel to be exponentiated.
        exponent: int
            Exponent :math:`\alpha>0`.
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.base_green_kernel = base_green_kernel
        self.exponent = exponent
        super(ZonalGreenExponentiated, self).__init__(order=base_green_kernel.order + exponent,
                                                 rtol=rtol, cutoff=cutoff)

    def _compute_fl_coefficients(self):
        return self.base_green_kernel._compute_fl_coefficients() ** self.exponent


class ZonalGreenSobolev(ZonalGreenFunction):
    r"""
    Zonal Green kernel of the Sobolev operator :math:`(\alpha^2\mbox{Id} -\Delta_{\mathbb{S}^2})^{\beta/2}`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalGreenSobolev

        for exp in [2,2.5,3,3.5,4]:
            zonal_green=ZonalGreenSobolev(alpha=1., exponent=exp)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$\\beta={np.round(val, 1)}$' for val in [2,2.5,3,3.5,4]])
        plt.title('$\\alpha=1$')

        for exp in [2,2.5,3,3.5,4]:
            zonal_green=ZonalGreenSobolev(alpha=5., exponent=exp)
            zonal_green.plot(angles=True, fhandle=2)
        plt.legend([f'$\\beta={np.round(val, 1)}$' for val in [2,2.5,3,3.5,4]])
        plt.title('$\\alpha=5$')

    Notes
    -----
    We have in this case :math:`\hat{D}_n=(\alpha^2+n(n+1))^{\beta/2},`  :math:`\hat{D}_n>0`, :math:`|\hat{D}_n|=\Theta(n^{\beta})`,
    and :math:`\mathcal{K}_{\mathcal{D}}=\emptyset`.
    """
    def __init__(self, alpha: Union[float, int], exponent: Union[int, float],
                 rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        alpha: Union[complex, float, int]
            Parameter :math:`\alpha\in\mathbb{R}`. Large values of :math:`\alpha` yield more localised Green kernels.
        exponent: Union[int, float]
            Exponent :math:`\beta\geq 1.`
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.alpha = alpha
        self.exponent = exponent
        super(ZonalGreenSobolev, self).__init__(order=self.exponent, rtol=rtol, cutoff=cutoff)

    def _compute_fl_coefficients(self):
        n = np.arange(0, self.cutoff + 1).astype(np.float)
        op_coeffs = (self.alpha ** 2 + n * (n + 1)) ** (self.exponent / 2)
        fl_coeffs = 0 * op_coeffs
        fl_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fl_coeffs


class ZonalGreenFractionalLaplaceBeltrami(ZonalGreenSobolev):
    r"""
    Zonal Green kernel of the Fractional Laplace-Beltrami operator :math:`(-\Delta_{\mathbb{S}^2})^{\beta/2}`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalGreenFractionalLaplaceBeltrami

        for exp in [2,2.5,3,3.5,4]:
            zonal_green=ZonalGreenFractionalLaplaceBeltrami(exponent=exp)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$\\beta={np.round(val, 1)}$' for val in [2,2.5,3,3.5,4]])

    Notes
    -----
    We have in this case :math:`\hat{D}_n=(n(n+1))^{\beta/2},`  :math:`\hat{D}_n\geq 0`, :math:`|\hat{D}_n|=\Theta(n^{\beta})`,
    and :math:`\mathcal{K}_{\mathcal{D}}=\{0\}`.
    See Example 4.2 of [FuncSphere]_ for a closed-form formula of the zonal Green kernel for :math:`\beta=4`.
    The case :math:`\beta=1` yields the square-root of the Laplace-Beltrami operator, which is intimately linked to the spherical
    *divergence* and *gradient* differential operators.

    See Also
    --------
    :py:class:`~pycsphere.green.ZonalGreenIteratedLaplaceBeltrami`
    """
    def __init__(self, exponent: float, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        exponent: Union[int, float]
            Exponent :math:`\beta\geq 1.`
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        super(ZonalGreenFractionalLaplaceBeltrami, self).__init__(alpha=0, exponent=exponent, rtol=rtol, cutoff=cutoff)


class ZonalGreenIteratedLaplaceBeltrami(ZonalGreenFunction):
    r"""
    Zonal Green kernel of the iterated Laplace-Beltrami operator :math:`\Delta_{\mathbb{S}^2}^{k}`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalGreenIteratedLaplaceBeltrami

        for exp in [1,2,3,4,5]:
            zonal_green=ZonalGreenIteratedLaplaceBeltrami(exponent=exp)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [1,2,3,4,5]])

    Notes
    -----
    We have in this case :math:`\hat{D}_n=(-n(n+1))^{k},`  :math:`\hat{D}_n\in \mathbb{R}`, :math:`|\hat{D}_n|=\Theta(n^{2k})`,
    and :math:`\mathcal{K}_{\mathcal{D}}=\{0\}`.
    See Example 4.2 of [FuncSphere]_ for a closed-form formula of the zonal Green kernel for :math:`k=2`.

    See Also
    --------
    :py:class:`~pycsphere.green.ZonalGreenFractionalLaplaceBeltrami`
    """
    def __init__(self, exponent: int, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        exponent: int
            Exponent :math:`k\geq 1.`
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.exponent = exponent
        super(ZonalGreenIteratedLaplaceBeltrami, self).__init__(order=2 * exponent, rtol=rtol, cutoff=cutoff)

    def _compute_fl_coefficients(self):
        n = np.arange(0, self.cutoff + 1).astype(np.float)
        op_coeffs = (-n * (n + 1)) ** self.exponent
        fl_coeffs = 0 * op_coeffs
        fl_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fl_coeffs


class ZonalGreenBeltrami(ZonalGreenFunction):
    r"""
    Zonal Green kernel of the Beltrami operator :math:`\partial_k=k(k+1)\mbox{Id}+\Delta_{\mathbb{S}^2}`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalGreenBeltrami

        for k in [1,2,3,4,5]:
            zonal_green=ZonalGreenBeltrami(k=k)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [1,2,3,4,5]])

    Notes
    -----
    We have in this case :math:`\hat{D}_n=k(k+1)-n(n+1),`  :math:`\hat{D}_n\in \mathbb{R}`, :math:`|\hat{D}_n|=\Theta(n^{2})`,
    and :math:`\mathcal{K}_{\mathcal{D}}=\{k\}`.
    See Chapter 4 of [FuncSphere]_ for properties of Beltrami operators.

    See Also
    --------
    :py:class:`~pycsphere.green.ZonalGreenIteratedBeltrami`
    """
    def __init__(self, k: int, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        k: int
            Parameter :math:`k\geq 1.`
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.k = k
        super(ZonalGreenBeltrami, self).__init__(order=2, rtol=rtol, cutoff=cutoff)

    def _compute_fl_coefficients(self):
        n = np.arange(0, self.cutoff + 1).astype(np.float)
        op_coeffs = (self.k * (self.k + 1) - n * (n + 1))
        fl_coeffs = 0 * op_coeffs
        fl_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fl_coeffs


class ZonalGreenIteratedBeltrami(ZonalGreenFunction):
    r"""
    Zonal Green kernel of the Beltrami operator :math:`\partial_{0\cdots k}=\partial_0\cdots\partial_k`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalGreenIteratedBeltrami

        for k in [1,2,3,4,5]:
            zonal_green=ZonalGreenIteratedBeltrami(k=k)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [1,2,3,4,5]])

    Notes
    -----
    We have in this case :math:`\hat{D}_n=\Pi_{j=0}^k j(j+1)-n(n+1),`  :math:`\hat{D}_n\in \mathbb{R}`, :math:`|\hat{D}_n|=\Theta(n^{2(k+1)})`,
    and :math:`\mathcal{K}_{\mathcal{D}}=\{0,\ldots,k\}`.
    See Chapter 4 of [FuncSphere]_ for properties of Beltrami operators.

    See Also
    --------
    :py:class:`~pycsphere.green.ZonalGreenBeltrami`
    """
    def __init__(self, k: int, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        k: int
            Parameter :math:`k\geq 1.`
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        """
        self.k = k
        super(ZonalGreenIteratedBeltrami, self).__init__(order=2 * (k + 1), rtol=rtol, cutoff=cutoff)

    def _compute_fl_coefficients(self):
        n = np.arange(0, self.cutoff + 1).astype(np.float)
        k = np.arange(0, self.k + 1).astype(np.float)
        op_coeffs = np.prod((k * (k + 1))[None, :] - (n * (n + 1))[:, None], axis=-1)
        fl_coeffs = 0 * op_coeffs
        fl_coeffs[op_coeffs != 0] = 1 / op_coeffs[op_coeffs != 0]
        return fl_coeffs


class ZonalMatern(Radial2Zonal):
    r"""
    Matern zonal Green kernel.

    The Matern zonal Green kernel is obtained by restricting the radial Matern function :py:class:`pycsou.math.green.Matern`
    to the sphere as described in :py:class:`pycsphere.green.Radial2Zonal`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalMatern

        for k in [0, 1,2,3]:
            zonal_green=ZonalMatern(k=k)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [0, 1,2,3]])

        for k in [0, 1,2,3]:
            zonal_green=ZonalMatern(k=k)
            zonal_green.spectrum(n_max=10, fhandle=2, color_index=k)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [0, 1,2,3]])

    Notes
    -----
    See Chapter 8 of [FuncSphere]_ for definitions, closed-form formulas and properties.

    See Also
    --------
    :py:class:`pycsou.math.green.Matern`, :py:class:`~pycsphere.green.Wendland`
    """
    def __init__(self, k: int, epsilon: float = 1.0, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        k: int, [0, 1 ,2 ,3]
            Order of the Matern function.
        epsilon: float
            Spread of the Matern function.
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.

        Notes
        -----
        See the help of :py:class:`pycsou.math.green.Matern` for more details on the parameters ``k`` and ``epsilon``.
        """
        super(ZonalMatern, self).__init__(radial_green=Matern(k=k, epsilon=epsilon), order=k + 3 / 2, cutoff=cutoff,
                                          rtol=rtol)


class ZonalWendland(Radial2Zonal):
    r"""
    Wendland zonal Green kernel.

    The Wendland zonal Green kernel is obtained by restricting the radial Wendland function :py:class:`pycsou.math.green.Wendland`
    to the sphere as described in :py:class:`pycsphere.green.Radial2Zonal`.

    Examples
    --------

    .. plot::

        from pycsphere.green import ZonalWendland

        for k in [0, 1,2,3]:
            zonal_green=ZonalWendland(k=k)
            zonal_green.plot(angles=True, fhandle=1)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [0, 1,2,3]])

        for k in [0, 1,2,3]:
            zonal_green=ZonalWendland(k=k)
            zonal_green.spectrum(n_max=10, fhandle=2, color_index=k)
        plt.legend([f'$k={np.round(val, 1)}$' for val in [0, 1,2,3]])

    Notes
    -----
    See Chapter 8 of [FuncSphere]_ for definitions, closed-form formulas and properties.

    See Also
    --------
    :py:class:`pycsou.math.green.Wendland`, :py:class:`~pycsphere.green.Matern`
    """
    def __init__(self, k: int, epsilon: float = 1.0, rtol: float = 1e-4, cutoff: Optional[int] = None):
        r"""

        Parameters
        ----------
        k: int, [0, 1 ,2 ,3]
            Order of the Wendland function.
        epsilon: float
            Spread of the Wendland function.
        rtol: float
            Threshold for defining the effective bandwidth of the zonal Green kernel.
        cutoff: Optional[int]
            Effective bandwidth of the zonal Green kernel.
        Notes
        -----
        See the help of :py:class:`pycsou.math.green.Wendland` for more details on the parameters ``k`` and ``epsilon``.
        """
        super(ZonalWendland, self).__init__(radial_green=Wendland(k=k, epsilon=epsilon), order=k + 3 / 2, cutoff=cutoff,
                                            rtol=rtol)

if __name__=='__main__':
    pass