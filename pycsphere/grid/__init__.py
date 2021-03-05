# #############################################################################
# __init__.py
# ===========
# Author : Matthieu Simeoni [matthieu.simeoni@gmail.com]
# #############################################################################

r"""
Routines for equidistributed spherical point sets.
"""

import numpy as np
from abc import ABC, abstractmethod
import scipy.spatial as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import healpy as hp
from healpy.rotator import Rotator
from matplotlib.cm import get_cmap
import matplotlib.colors as mplcols
from typing import Optional, List, Sequence
from astropy.coordinates import Angle
import astropy.units as u


class SphericalPointSet(ABC):
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
    def angle2res(cls, angular_resolution: float, deg: bool = True) -> int:
        if deg is True:
            nodal_width = 2 * np.sin((np.pi / 180) * angular_resolution / 2)
        else:
            nodal_width = 2 * np.sin(angular_resolution / 2)
        return np.ceil(cls.nodal_width_cst / nodal_width) ** 2

    def __init__(self, resolution: int, separation: Optional[float] = None,
                 nodal_width: Optional[float] = None, lonlat: bool = False):
        self.resolution = resolution
        self.separation = separation
        self.nodal_width = nodal_width
        self._lonlat = lonlat
        self.vec, self.dir = self.generate_point_set()

    @abstractmethod
    def generate_point_set(self):
        pass

    def plot_tessellation(self, facecmap: Optional[str] = 'Blues_r',
                          ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                          markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                          seed: int = 0):
        self.plot_voronoi_cells(facecmap=facecmap, ncols=ncols, edgecolor=edgecolor,
                                cell_centers=cell_centers, markercolor=markercolor, markersize=markersize,
                                linewidths=linewidths, alpha=alpha, seed=seed)

    def plot_delaunay_cells(self, facecmap: Optional[str] = 'Blues_r',
                            ncols: int = 20, edgecolor: str = '#32519D', cell_centers: bool = True,
                            markercolor: str = 'k', markersize: float = 2, linewidths=1, alpha: float = 0.9,
                            seed: int = 0):
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
    def angular_nodal_width(self):
        if self.nodal_width is None:
            return None
        else:
            if self.lonlat is True:
                return 2 * np.arcsin(self.nodal_width / 2) * 180 / np.pi
            else:
                return 2 * np.arcsin(self.nodal_width / 2)

    @property
    def lonlat(self):
        return self._lonlat

    @lonlat.setter
    def lonlat(self, value):
        self._lonlat = value
        self.dir = hp.vec2dir(self.vec, lonlat=value)

    def compute_empirical_nodal_width(self, mode='mean'):
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
        resolution = cls.angle2res(angular_resolution=angular_resolution, deg=deg)
        return np.floor((resolution - 1) / 2)

    def __init__(self, N: int, lonlat: bool = False):
        separation = self.separation_cst / np.sqrt(2 * N + 1)
        nodal_width = self.nodal_width_cst / np.sqrt(2 * N + 1)
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.golden_angle = 2 * np.pi * (1 - 1 / self.golden_ratio)
        super(FibonacciPointSet, self).__init__(resolution=2 * N + 1, separation=separation,
                                                nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self):
        step = np.arange(self.resolution)
        phi = step * self.golden_angle
        phi = Angle(phi * u.rad).wrap_at(2 * np.pi * u.rad)
        theta = np.arccos(1 - (2 * step / self.resolution))
        vec = hp.dir2vec(theta=theta, phi=phi, lonlat=False)
        dir = hp.vec2dir(vec, lonlat=self.lonlat)
        return vec, dir


class HEALPixPointSet(SphericalPointSet):
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
        resolution = cls.angle2res(angular_resolution=angular_resolution, deg=deg)
        return np.ceil(np.sqrt(resolution / 12)).astype(int)

    def __init__(self, nside: int, lonlat: bool = False, nest=False):
        resolution = int(hp.nside2npix(nside))
        separation = self.separation_cst / resolution
        nodal_width = self.nodal_width_cst / resolution
        self.nside = nside
        self.nrings = 4 * self.nside - 1
        self.nest = nest
        super(HEALPixPointSet, self).__init__(resolution=resolution, separation=separation,
                                              nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self):
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
    def angle2N(cls, angular_resolution: float, deg: bool = True) -> int:
        pass

    def __init__(self, N: int, seed: int = 0, lonlat: bool = False):
        resolution = N
        separation = self.separation_cst / resolution
        nodal_width = self.nodal_width_cst / np.sqrt(resolution / np.log(resolution))
        self.seed = seed
        super(RandomPointSet, self).__init__(resolution=resolution, separation=separation,
                                             nodal_width=nodal_width, lonlat=lonlat)

    def generate_point_set(self):
        rng = np.random.default_rng(self.seed)
        lon = 360 * rng.random(size=self.resolution) - 180
        uniform = rng.random(size=self.resolution)
        lat = np.arcsin(2 * uniform - 1) * 180 / np.pi
        vec = hp.dir2vec(theta=lon, phi=lat, lonlat=True)
        dir = hp.vec2dir(vec, lonlat=self.lonlat)
        return vec, dir


if __name__ == '__main__':
    N = FibonacciPointSet.angle2N(angular_resolution=10)
    fib = FibonacciPointSet(N, lonlat=True)
    fib.plot_delaunay_cells()
    fib.plot_tessellation()

    nside = HEALPixPointSet.angle2nside(angular_resolution=10)
    healpix = HEALPixPointSet(nside, lonlat=True)
    healpix.plot_delaunay_cells()
    healpix.plot_tessellation()

    rnd = RandomPointSet(healpix.resolution, lonlat=True)
    rnd.plot_delaunay_cells()
    rnd.plot_tessellation()
