# #############################################################################
# mesh.py
# =======
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Routines for deterministic and random spherical point sets.
"""

import numpy as np
from abc import ABC, abstractmethod
import scipy.spatial as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import healpy as hp
from matplotlib.cm import get_cmap
import matplotlib.colors as mplcols
from typing import Optional, List, Union, Tuple
from astropy.coordinates import Angle
import astropy.units as u
from scipy.special import lambertw


class SphericalPointSet(ABC):
    r"""
    Base class for spherical point sets.

    Any subclass/instance of this class must implement the abstract method ``generate_point_set`` which returns the
    point set cartesian/spherical coordinates.

    A spherical point set is a collection :math:`\Theta_N=\{\mathbf{r}_1, \ldots, \mathbf{r}_N\}\subset \mathbb{S}^2` of
    unit vectors (directions).

    Notes
    -----
    The class ``SphericalPointSet`` possesses a few class/object attributes which allow to describe the properties of the point set.
    The terminology is the same as [PointSets]_:

    * A sequence of point sets :math:`\{\Theta_N\}_{n\in\mathbb{N}}\subset \mathcal{P}(\mathbb{S}^2)` is called ``equidistributed``
      if the sequence of *normalised atomic measures* :math:`\nu_N(A):=|A\cap \Theta_N|/N` converges  w.r.t. the weak*-topology
      towards the Lebesgue measure as :math:`N` grows to infinity.
    * The *separation* of a point set is defined as :math:`\delta(\Theta_N):=\min_{\mathbf{r}\neq \mathbf{s}\in\Theta_N} \|\mathbf{r}- \mathbf{s}\|_2`.
      A set is said ``well_separated`` if :math:`\delta(\Theta_N)\geq c N^{-1/2}` for some ``separation_cst`` constant :math:`c>0` and :math:`N\geq 2`.
    * The ``nodal_width`` of a point set is defined as :math:`\eta(\Theta_N):=\max_{\mathbf{s}\in\mathbb{S}^2}\min_{\mathbf{r}\in\Theta_N} \|\mathbf{r}- \mathbf{s}\|_2`.
      A set is said to have ``good_covering`` if :math:`\eta(\Theta_N)\leq C N^{-1/2}` for some ``nodal_width_cst`` :math:`C>0` and :math:`N\geq 2`.
    * The ``mesh_ratio`` of a point set is the ratio between its nodal width and separation. If the mesh ratio is asymptotically
      bounded, then the point set is said ``quasi_uniform``. A lower bound on the mesh ratio is ``minimal_mesh_ratio = (np.sqrt(5) - 1) / 2``.

    """
    minimal_mesh_ratio = (np.sqrt(5) - 1) / 2
    separation_cst = None
    nodal_width_cst = None
    mesh_ratio = None
    well_separated = None
    good_covering = None
    quasi_uniform = None
    equidistributed = None
    equal_area = None
    isolatitude = None
    hierarchical = None

    @classmethod
    def arc2angle(cls, arc_length: Union[np.ndarray, float], deg: bool = False) -> Union[np.ndarray, float]:
        r"""
        Convert an arc length to an angular section.

        Parameters
        ----------
        arc_length: Union[np.ndarray, float]
            Input arc length.
        deg: bool
            Whether the output angle is in degrees (``True``) or radians (``False``, default).

        Returns
        -------
        Union[np.ndarray, float]
            Angle corresponding to the arc.

        """
        if deg is False:
            return 2 * np.arcsin(arc_length / 2)
        else:
            return 2 * np.arcsin(arc_length / 2) * 180 / np.pi

    @classmethod
    def angle2arc(cls, angle: Union[np.ndarray, float], deg: bool = False) -> Union[np.ndarray, float]:
        r"""
        Compute the arc length of a given angular section.

        Parameters
        ----------
        angle: Union[np.ndarray, float]
            Angle defining the arc.
        deg: bool
            Whether the input angle is in degrees (``True``) or radians (``False``, default).

        Returns
        -------
        Union[np.ndarray, float]
            Arc length of the angular section.

        """
        if deg is True:
            return 2 * np.sin((np.pi / 180) * angle / 2)
        else:
            return 2 * np.sin(angle / 2)

    @classmethod
    def angle2res(cls, angular_resolution: float, deg: bool = True) -> int:
        r"""
        Compute the minimal point set size required to achieve a desired (angular) nodal width.

        Parameters
        ----------
        angular_resolution: float
            Desired angular nodal width.
        deg: bool
            Whether the angular nodal width is in degrees or not.

        Returns
        -------
        int
            Minimal point set resolution.
        """
        nodal_width = cls.angle2arc(angular_resolution, deg=deg)
        return np.ceil(cls.nodal_width_cst / nodal_width) ** 2

    def __init__(self, resolution: int, separation: Optional[float] = None,
                 nodal_width: Optional[float] = None, lonlat: bool = False):
        r"""

        Parameters
        ----------
        resolution: int
            Resolution (size) of the point set.
        separation: Optional[float]
            Separation of the point set (``None`` if unknown).
        nodal_width: Optional[float]
            Nodal width of the point set (``None`` if unknown).
        lonlat: bool
            If ``True``, longitude/latitude spherical coordinates (in degrees) are used. Otherwise, longitude/co-latitude (in radians)
            are used.
        """
        self.resolution = resolution
        self.separation = separation
        self.nodal_width = nodal_width
        self._lonlat = lonlat
        self.vec, self.dir = self.generate_point_set()

    @abstractmethod
    def generate_point_set(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Generate the cartesian/spherical coordinates of the point set.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cartesian/spherical coordinates of the point set. The latter are also stored in the attributes ``self.vec`` and ``self.dir`` respectively.
            The attribute ``self.vec`` has shape ``(3, N)`` and contains the cartesian coordinates of the points in the set.
            The attribute ``self.dir`` has shape ``(2, N)`` and contains the spherical coordinates of the points in the set:

            *  If ``self.latlon==True``, longitudes and latitudes in degrees,
            *  If ``self.latlon==False``, longitudes and co-latitudes in radians.
        """
        pass

    def plot_tessellation(self, facecmap: Optional[str] = 'Blues_r',
                          ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                          markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                          seed: int = 0):
        r"""
        Plots the tessellation defined by the point set.

        Parameters
        ----------
        facecmap: Optional[str]
            Color map to be used for random coloring the tessellation faces.
        ncols: int
            Number of colors in the color map.
        edgecolor: str
            Color of the cell edges.
        cell_centers: bool
            Whether or not the cell centers are displayed.
        markercolor: str
            Color of the cell centers.
        markersize: float
            Size of the cell centers.
        linewidths: float
            Width of the cell edges.
        alpha: float
            Transparency of the cell faces.
        seed: int
            Seed for random coloring reproducibility.

        Warnings
        --------
        This method can be **very slow** for large point sets (>10'000 points)!
        """
        self.plot_voronoi_cells(facecmap=facecmap, ncols=ncols, edgecolor=edgecolor,
                                cell_centers=cell_centers, markercolor=markercolor, markersize=markersize,
                                linewidths=linewidths, alpha=alpha, seed=seed)

    def plot_delaunay_cells(self, facecmap: Optional[str] = 'Blues_r',
                            ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                            markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                            seed: int = 0):
        r"""
       Plots the tessellation defined by the set's Delaunay triangulation.

       Parameters
       ----------
       facecmap: Optional[str]
           Color map to be used for random coloring the tessellation faces.
       ncols: int
           Number of colors in the color map.
       edgecolor: str
           Color of the cell edges.
       cell_centers: bool
           Whether or not the cell centers are displayed.
       markercolor: str
           Color of the cell centers.
       markersize: float
           Size of the cell centers.
       linewidths: float
           Width of the cell edges.
       alpha: float
           Transparency of the cell faces.
       seed: int
           Seed for random coloring reproducibility.

       Warnings
       --------
       This method can be **very slow** for large point sets (>10'000 points)!
       """
        vec = self.vec.transpose()
        delaunay = sp.ConvexHull(vec)
        vertices = [[vec[delaunay.simplices[i]]] for i in range(delaunay.simplices.shape[0])]
        self._plot_spherical_polygons(vertices, facecmap=facecmap, ncols=ncols, edgecolor=edgecolor,
                                      cell_centers=cell_centers, markercolor=markercolor, markersize=markersize,
                                      linewidths=linewidths, alpha=alpha, seed=seed)

    def plot_voronoi_cells(self, facecmap: Optional[str] = 'Blues_r',
                           ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                           markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                           seed: int = 0):
        r"""
       Plots the tessellation defined by the set's Voronoi decomposition.

       Parameters
       ----------
       facecmap: Optional[str]
           Color map to be used for random coloring the tessellation faces.
       ncols: int
           Number of colors in the color map.
       edgecolor: str
           Color of the cell edges.
       cell_centers: bool
           Whether or not the cell centers are displayed.
       markercolor: str
           Color of the cell centers.
       markersize: float
           Size of the cell centers.
       linewidths: float
           Width of the cell edges.
       alpha: float
           Transparency of the cell faces.
       seed: int
           Seed for random coloring reproducibility.

       Warnings
       --------
       This method can be **very slow** for large point sets (>10'000 points)!
       """
        voronoi = sp.SphericalVoronoi(self.vec.transpose(), radius=1)
        voronoi.sort_vertices_of_regions()
        vertices = [[voronoi.vertices[region]] for region in voronoi.regions]
        self._plot_spherical_polygons(vertices, facecmap=facecmap, ncols=ncols, edgecolor=edgecolor,
                                      cell_centers=cell_centers, markercolor=markercolor, markersize=markersize,
                                      linewidths=linewidths, alpha=alpha, seed=seed)

    def _plot_spherical_polygons(self, vertices: List[List[np.ndarray]], facecmap: Optional[str] = 'Blues_r',
                                 ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                                 markercolor: str = 'k', markersize: float = 3, linewidths=1, alpha: float = 0.9,
                                 seed: int = 0):
        rng = np.random.default_rng(seed=seed)
        cmap_obj = get_cmap(facecmap, ncols)
        normalise_data = mplcols.Normalize(vmin=0, vmax=ncols - 1)
        fig = plt.figure()
        ax3 = plt3.Axes3D(fig)
        ax3.scatter3D(1, 1, 1, s=0)
        ax3.scatter3D(-1, -1, -1, s=0)
        ax3.view_init(elev=80, azim=0)
        if cell_centers is True:
            ax3.scatter3D(self.vec[0], self.vec[1], self.vec[2], '.', color=markercolor, s=markersize)
        for verts in vertices:
            color_index = rng.integers(0, ncols - 1, size=1)
            polygon = Poly3DCollection(verts, linewidths=linewidths, alpha=alpha)
            polygon.set_facecolor(cmap_obj(normalise_data(color_index)))
            polygon.set_edgecolor(edgecolor)
            ax3.add_collection3d(polygon)
        plt.axis('off')
        plt.show()

    @property
    def angular_nodal_width(self) -> Optional[float]:
        r"""
        Angular nodal width.
        """
        if self.nodal_width is None:
            return None
        else:
            return self.arc2angle(self.nodal_width, deg=self.lonlat)

    @property
    def lonlat(self) -> bool:
        return self._lonlat

    @lonlat.setter
    def lonlat(self, value):
        self._lonlat = value
        self.dir = hp.vec2dir(self.vec, lonlat=value)

    def compute_empirical_nodal_width(self, mode='mean') -> float:
        r"""
        Compute the set's nodal width.
        """
        cvx_hull = sp.ConvexHull(self.vec.transpose())
        cols = np.roll(cvx_hull.simplices, shift=1, axis=-1).reshape(-1)
        rows = cvx_hull.simplices.reshape(-1)
        # Form sparse coo_matrix from extracted pairs
        affinity_matrix = sparse.coo_matrix((cols * 0 + 1, (rows, cols)), shape=(cvx_hull.points.shape[0],
                                                                                 cvx_hull.points.shape[0]))
        # Symmetrize the matrix to obtain an undirected graph.
        extended_row = np.concatenate([affinity_matrix.row, affinity_matrix.col])
        extended_col = np.concatenate([affinity_matrix.col, affinity_matrix.row])
        affinity_matrix.row = extended_row
        affinity_matrix.col = extended_col
        affinity_matrix.data = np.concatenate([affinity_matrix.data, affinity_matrix.data])
        affinity_matrix = affinity_matrix.tocsr().tocoo()  # Delete potential duplicate pairs
        distance = np.linalg.norm(cvx_hull.points[affinity_matrix.row, :] - cvx_hull.points[affinity_matrix.col, :],
                                  axis=-1)
        if mode is 'mean':
            nodal_distance = np.mean(distance)  # average distance to neighbors
        elif mode is 'max':
            nodal_distance = np.max(distance)
        elif mode is 'median':
            nodal_distance = np.median(distance)
        else:
            raise TypeError("Parameter mode must be one of ['mean', 'max', 'median']")
        return nodal_distance


class FibonacciPointSet(SphericalPointSet):
    r"""
    Fibonacci point set.

    Examples
    --------

    .. plot::

        from pycsphere.mesh import FibonacciPointSet
        N = FibonacciPointSet.angle2N(angular_resolution=10)
        fib = FibonacciPointSet(N, lonlat=True)
        fib.plot_delaunay_cells()
        fib.plot_tessellation()

    Notes
    -----
    Points in the Fibonacci point set are arranged uniformly along a spiral pattern on the sphere linking the two poles (see Section 2 of [PointSets]_ for
    a definition of the point set). The Fibonacci point set is defined for odd resolutions :math:`2N+1` only. It is well-separated, has good covering and is hence quasi-uniform. It is also
    equidistributed. Its Voronoi cells are however irregular and do not have the same shape/area. The points are not arranged on isolatitude
    tracks. Finally it is non hierarchical (Fibonacci point sets at two different resolutions are not contained in one another).

    .. todo::

       Add support for local Fibonacci meshes.
    """
    separation_cst = 3.09206862
    nodal_width_cst = 2.72812463
    mesh_ratio = 0.882298
    well_separated = True
    good_covering = True
    quasi_uniform = True
    equidistributed = True
    equal_area = False
    isolatitude = False
    hierarchical = False

    @classmethod
    def angle2N(cls, angular_resolution: float, deg: bool = True) -> int:
        r"""
        Minimal parameter :math:`N` to achieve a prescribed angular nodal width.

        Parameters
        ----------
        angular_resolution: float
            Desired angular nodal width.
        deg: bool
            Whether or not the nodal width is provided in degrees.

        Returns
        -------
        int
            Minimal value of the parameter :math:`N` defining the point set resolution :math:`2N+  1`.

        """
        resolution = cls.angle2res(angular_resolution=angular_resolution, deg=deg)
        return np.ceil((resolution - 1) / 2)

    def __init__(self, N: int, lonlat: bool = False):
        r"""

        Parameters
        ----------
        N: int
            Defines the point set resolution :math:`2N+1`.
        lonlat: bool
            Convention for the spherical coordinates (see help of :py:func:`~pycsphere.mesh.SphericalPointSet.__init__`).
        """
        separation = self.separation_cst / np.sqrt(2 * N + 1)
        nodal_width = self.nodal_width_cst / np.sqrt(2 * N + 1)
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.golden_angle = 2 * np.pi * (1 - 1 / self.golden_ratio)
        super(FibonacciPointSet, self).__init__(resolution=2 * N + 1, separation=separation,
                                                nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self) -> Tuple[np.ndarray, np.ndarray]:
        step = np.arange(self.resolution)
        phi = step * self.golden_angle
        phi = Angle(phi * u.rad).wrap_at(2 * np.pi * u.rad)
        theta = np.arccos(1 - (2 * step / self.resolution))
        vec = hp.dir2vec(theta=theta, phi=phi, lonlat=False)
        dir = hp.vec2dir(vec, lonlat=self.lonlat)
        return vec, dir


class HEALPixPointSet(SphericalPointSet):
    r"""
    HEALPix point set.

    Examples
    --------

    .. plot::

        from pycsphere.mesh import HEALPixPointSet
        nside = HEALPixPointSet.angle2nside(angular_resolution=10)
        healpix = HEALPixPointSet(nside, lonlat=True)
        healpix.plot_delaunay_cells()
        healpix.plot_voronoi_cells()
        healpix.plot_tessellation()

    Notes
    -----
    Developed by NASA for fast data analysis of the cosmic microwave background (CMB), the *Hierarchical Equal Area iso-Latitude Pixelization (HEALPix)*
    was designed to have three properties essential for computational efficiency in discretising functions on :math:`\mathbb{S}^2`:

    1. The sphere is hierarchically tessellated into curvilinear quadrilaterals.
    2. The pixelisation is an equal area partition of :math:`\mathbb{S}^2`.
    3. The point sets are distributed along iso-latitude lines.

    A formal definition of the point set is provided Section 2 of [PointSets]_. We use here the :py:mod:`healpy` package.
    The HEALPix point set is defined for resolutions :math:`N=12k^2` only (parameter :math:`k` is called ``nside``).
    It is well-separated, has good covering and is hence quasi-uniform. It is also equidistributed. Its tessellation cells are
    moreover regular and have the same shape/area. Finally it is hierarchical (point sets at two different resolutions
    are nested in one another).

    .. todo::

       Add spatial filtering routines for local HEALPix meshes.
    """
    separation_cst = 2.8345
    nodal_width_cst = 2.8345
    mesh_ratio = 1
    well_separated = True
    good_covering = True
    quasi_uniform = True
    equidistributed = True
    equal_area = True
    isolatitude = True
    hierarchical = True

    @classmethod
    def angle2nside(cls, angular_resolution: float, deg: bool = True) -> int:
        r"""
        Minimal parameter :math:`k` (``nside``) to achieve a prescribed angular nodal width.

        Parameters
        ----------
        angular_resolution: float
            Desired angular nodal width.
        deg: bool
            Whether or not the nodal width is provided in degrees.

        Returns
        -------
        int
            Minimal value of the parameter :math:`k` defining the point set resolution :math:`N=12k^2`.

        """
        resolution = cls.angle2res(angular_resolution=angular_resolution, deg=deg)
        return (2 ** np.ceil(np.log2(np.sqrt(resolution / 12)))).astype(int)

    def __init__(self, nside: int, lonlat: bool = False, nest=False):
        r"""

        Parameters
        ----------
        nside: int
            Defines the point set resolution ``12 * nside ** 2``.  ``nside`` must be a power of two.
        lonlat: bool
            Convention for the spherical coordinates (see help of :py:func:`~pycsphere.mesh.SphericalPointSet.__init__`).
        nest: bool
            If ``True``, the HEALPix mesh is stored in NESTED ordering (efficient for pixel querying routines).
            If ``False``, the HEALPix mesh is stored in RING ordering (efficient for spherical harmonics transforms).
        """
        resolution = int(hp.nside2npix(nside))
        separation = self.separation_cst / resolution
        nodal_width = self.nodal_width_cst / resolution
        self.nside = nside
        self.nrings = 4 * self.nside - 1
        self.nest = nest
        super(HEALPixPointSet, self).__init__(resolution=resolution, separation=separation,
                                              nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.stack(hp.pix2vec(self.nside, np.arange(self.resolution)), axis=0)
        dir = hp.vec2dir(vec, lonlat=self.lonlat)
        return vec, dir

    def plot_tessellation(self, facecmap: Optional[str] = 'Blues_r',
                          ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                          markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                          seed: int = 0):
        vertices = [[hp.boundaries(self.nside, i).transpose()] for i in range(self.resolution)]
        self._plot_spherical_polygons(vertices, facecmap=facecmap, ncols=ncols, edgecolor=edgecolor,
                                      cell_centers=cell_centers, markercolor=markercolor, markersize=markersize,
                                      linewidths=linewidths, alpha=alpha, seed=seed)


class RandomPointSet(SphericalPointSet):
    r"""
    Random uniform point set.

    Examples
    --------

    .. plot::

        from pycsphere.mesh import RandomPointSet
        res = RandomPointSet.angle2res(angular_resolution=10)
        rnd = RandomPointSet(res, lonlat=True)
        rnd.plot_delaunay_cells()
        rnd.plot_tessellation()

    Notes
    -----
    Properties of spherical random point sets are discussed in Section 2 of [PointSets]_. They are equidistributed but not
    quasi-uniform, well-separated or with good-coverage. They are also of course not hierarchical and yield highly irregular
    tessellation cells with different shapes and areas.
    """
    separation_cst = np.sqrt(2 * np.pi)
    nodal_width_cst = 2
    mesh_ratio = None
    well_separated = False
    good_covering = False
    quasi_uniform = False
    equidistributed = True
    equal_area = False
    isolatitude = False
    hierarchical = False

    @classmethod
    def angle2res(cls, angular_resolution: float, deg: bool = True) -> int:
        nodal_width = cls.angle2arc(angular_resolution, deg=deg)
        return np.ceil(np.exp(-np.real(lambertw(-1 / (cls.nodal_width_cst / nodal_width) ** 2, k=-1)))).astype(int)

    def __init__(self, N: int, seed: int = 0, lonlat: bool = False):
        r"""

        Parameters
        ----------
        N: int
            Resolution of the random point set.
        seed: int
            Seed for reproducibility of the random point set.
        lonlat: bool
            Convention for the spherical coordinates (see help of :py:func:`~pycsphere.mesh.SphericalPointSet.__init__`).
        """
        resolution = N
        separation = self.separation_cst / resolution
        nodal_width = self.nodal_width_cst / np.sqrt(resolution / np.log(resolution))
        self.seed = seed
        super(RandomPointSet, self).__init__(resolution=resolution, separation=separation,
                                             nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        lon = 360 * rng.random(size=self.resolution) - 180
        uniform = rng.random(size=self.resolution)
        lat = np.arcsin(2 * uniform - 1) * 180 / np.pi
        vec = hp.dir2vec(theta=lon, phi=lat, lonlat=True)
        dir = hp.vec2dir(vec, lonlat=self.lonlat)
        return vec, dir
