import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

import numpy as np
from scipy.spatial import Voronoi
from seismostats.analysis.estimate_beta import estimate_b
from scipy.spatial import ConvexHull, distance_matrix
import geopandas as gpd
import shapely
from functions.general_functions import transform_n, a_samples, b_samples


def est_morans_i(values, w):
    """
    Estimate the auto correlation (Moran's I) of the b-values of the voronoi
    cells. Applicable in 2+ dimensions.

    Args:
        values:     values for each voronoi cell
        w:          Weight matrix, indicating which of the values are
                neighbors to each other. It should be a square matrix of
                size len(b_vec) x len(b_vec). It is assumed that the neighbor
                matrix is zero at points where values are nan.

    Returns:
        ac:     Auto correlation of the values
        n_p:    Sum of the weight matrix. In the limit the standard deviation
            of the autocorrelation is ~1/sqrt(n_p).

    """
    ac = 0
    ac_0 = 0
    # estimate mean
    n = len(values[~np.isnan(values)])
    mean_v = np.mean(values[~np.isnan(values)])

    for ii, v1 in enumerate(values):
        if np.isnan(v1):
            w[ii, :] = 0
            continue
        ac_0 += (v1 - mean_v) ** 2
        for jj, v2 in enumerate(values):
            if np.isnan(v2):
                w[ii, jj] = 0
                continue
            if w[ii, jj] == 1:
                ac += (v1 - mean_v) * (v2 - mean_v)
    n_p = np.sum(w)
    ac = (n-1) / n_p * ac/ac_0
    return ac, n, n_p


def mac_d_dimensions(
        coords,
        mags,
        delta_m,
        mc,
        times,
        limits,
        n_points,
        n_realizations,
        eval_coords=None,
        min_num=10,
        b_method="positive",
        transform=True,
        scaling_factor=1,
        include_a=True,
):
    """
    This function estimates the mean autocorrelation for the D-dimensional
    case (tested for 2 and 3 dimensions). Additionally, it provides the mean
    a- and b-values for each grid-point. The partitioning method is based on
    voronoi tesselation (random area).

    Args:
        coords:     Coordinates of the earthquakes. It should have the
                structure [x1, ... , xD], where xi are vectors of the same
                length (number of events)
        mags:       Magnitudes of the earthquakes
        delta_m:    Magnitude bin width
        mc:         Completeness magnitude
        times:      Times of the earthquakes
        limits:     Limits of the area of interest. It should be a list with
                the minimum and maximum values of each variable.
                [[x1min, x1max], ..., [xDmin, xDmax]]
                The limits should be such that all the coordinates are within
                the limits.
        n_points:   Number of voronoi nodes
        n_realizations: Number of realizations of random tesselation for the
                estimation of the mean values
        eval_coords: Coordinates of the grid-points where the mean a- and
                b-values are estimated. It should have the structure
                [x1, ..., xD], where xi are vectors of the same length (number
                od points where the mean a- and b-values are estimated)
        min_num:    Minimum number of events to estimate a- and b-values in
                each tile
        b_method:   Method to estimate b-values. Options are "positive" and
                "tinti"

    """
    # 0. preparation
    dim = len(coords[:, 0])
    if eval_coords is None:
        # in this case, the values are estimated at the earthquake locations
        eval_coords = coords

    # 1. some data checks
    if len(mags) != len(coords[0, :]):
        raise ValueError("The number of magnitudes and coordinates do not "
                         "match")
    if len(limits) != dim:
        raise ValueError("The number of limits and dimensions do not match")
    if len(eval_coords[:, 0]) != dim:
        raise ValueError("The number of evaluation coordinates and dimensions "
                         "do not match")
    if min(mags) < mc:
        raise ValueError("The completeness magnitude is larger than the "
                         "smallest magnitude")
    for ii in range(dim):
        if min(eval_coords[ii, :]) < limits[ii][0] or max(
                eval_coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The evaluation coordinates are outside the limits")
        if min(coords[ii, :]) < limits[ii][0] or max(
                coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The earthquake coordinates are outside the limits")

    # 2. estimate a and b values for n realizations
    b_average = np.zeros(len(eval_coords[0, :]))
    average_cnt_b = np.zeros(len(eval_coords[0, :]))
    ac_2D = np.zeros(n_realizations)
    n_p = np.zeros(n_realizations)
    n = np.zeros(n_realizations)
    if include_a is True:
        a_average = np.zeros(len(eval_coords[0, :]))
        average_cnt_a = np.zeros(len(eval_coords[0, :]))
        ac_2D_a = np.zeros(n_realizations)
        n_p_a = np.zeros(n_realizations)
        n_a = np.zeros(n_realizations)
    for ii in range(n_realizations):
        # 2.1 create voronoi nodes (randomly distributed within the limits)
        coords_vor = np.random.rand(dim, n_points)
        for jj in range(dim):
            coords_vor[jj, :] = limits[jj][0] + (
                limits[jj][1] - limits[jj][0]) * coords_vor[jj, :]
        vor = mirror_voronoi(coords_vor, limits)

        # 2.2 estimate areas
        volume = volumes_vor(vor, n_points)
        volume *= scaling_factor
        for jj in range(dim):
            volume /= (limits[jj][1] - limits[jj][0])

        # 2.3 find maggnitudes and times corresponding to the voronoi nodes
        tile_magnitudes, tile_times = find_points_in_tile(
            coords_vor, coords, mags, times)

        # 2.4 estimate a- and b-values
        b_vec,  n_m = b_samples(
            tile_magnitudes, tile_times, delta_m, mc, b_method=b_method)
        
        b_vec[n_m < min_num] = np.nan

        if include_a is True:
            a_vec = a_samples(tile_magnitudes, tile_times, delta_m,
                              mc, volumes=volume, a_method=b_method)
            a_vec[n_m < min_num] = np.nan

        # 2.5 transform the b-values
        if transform:
            idx = np.argsort(times)
            mags_sorted = mags[idx]
            b_all = estimate_b(mags_sorted, mc=mc,
                               delta_m=delta_m, method=b_method)
            b_vec_t = transform_n(b_vec, b_all, n_m, np.max(n_m))

        # 2.6 find the nearest voronoi node for each grid-point
        nearest = find_nearest_vor_node(coords_vor, eval_coords)

        # 2.7 average the b-values and a-values
        b_loop = b_vec[nearest]
        average_cnt_b += ~np.isnan(b_loop)
        b_loop[np.isnan(b_loop)] = 0
        b_average += b_loop

        if include_a is True:
            a_loop = a_vec[nearest]
            average_cnt_a += ~np.isnan(a_loop)
            a_loop[np.isnan(a_loop)] = 0
            a_average += a_loop

        # 2.8 estimate nearest neighbour matrix,  estimate autocorrelation
        nan_idx = np.argwhere(np.isnan(b_vec))
        w = neighbors_vor(vor, len(coords_vor[0, :]), nan_idx)
        ac_2D[ii], n[ii], n_p[ii] = est_morans_i(b_vec_t, w)

        if include_a is True:
            nan_idx = np.argwhere(np.isnan(a_vec))
            w = neighbors_vor(vor, len(coords_vor[0, :]), nan_idx)
            ac_2D_a[ii], n_a[ii], n_p_a[ii] = est_morans_i(a_vec, w)

    # 3. estimate the averages & estimate expected standard deviation of MAC
    b_average = b_average / average_cnt_b
    mac = np.mean(ac_2D)
    mean_n_p = np.mean(n_p)
    mean_n = np.mean(n)
    mu_mac = -1/mean_n
    std_mac = np.sqrt(1/mean_n_p)
    if include_a is True:
        a_average = a_average / average_cnt_a
        mac_a = np.mean(ac_2D_a)
        mean_n_p_a = np.mean(n_p_a)
        mean_n_a = np.mean(n_a)
        mu_mac_a = -1/mean_n_a
        std_mac_a = np.sqrt(1/mean_n_p_a)

        return (mac, mu_mac, std_mac, b_average,
                mac_a, mu_mac_a, std_mac_a, a_average)

    return mac, mu_mac, std_mac, b_average


def find_nearest_vor_node(coords_vor, coords):
    """
    Find the nearest voronoi node for each coordinate in coords.
    coords_vor and coords should have the same number of dimensions D.

    Args:
        coords_vor: coordinates of the voronoi nodes,  [x1, ...,  xD],
                where xi are vectors of the same length (number of events)
        coords: coordinates of the EQs [x1, ...,  xD], where xi are vectors
                of the same length (number of voronoi cells)

    Returns:
        nearest: indices of the nearest voronoi node for each coordinate in
        coords
    """
    distance_matrix(coords_vor.T, coords.T)
    nearest = np.argmin(distance_matrix(coords.T, coords_vor.T), axis=1)
    return nearest


def find_points_in_tile(coords_vor, coords, magnitudes, times):
    """
    Find the magnitudes and times of the earthquakes that are in each tile.

    Args:
        coords_vor:     coordinates of the voronoi nodes,  [x1, ...,  xD],
                    where xi are vectors of the same length (number of events)
        coords:         coordinates of the EQs [x1, ...,  xD], where xi are
                     vectors of the same length (number of voronoi cells)
        magnitudes:     magnitudes of the earthquakes
        times:          times of the earthquakes

    Returns:
        tile_magnitudes:    list of magnitudes of the earthquakes in each tile.
                        The index corresponds to the index of the voronoi node.
        tile_times:         list of times of the earthquakes in each tile
    """
    nearest = find_nearest_vor_node(coords_vor, coords)
    tile_magnitudes = []
    tile_times = []
    for ii in range(len(coords_vor[0, :])):
        idx = nearest == ii
        tile_magnitudes.append(magnitudes[idx])
        tile_times.append(times[idx])

    return tile_magnitudes, tile_times


def mirror_voronoi(coords_vor, limits):
    """
    Mirror the voronoi points such that the cells are confined within the
    limits.
    """
    n = len(coords_vor[0, :])
    mirror_coords = np.zeros([len(limits), (2*len(limits)+1)*n])
    mirror_coords[:, :n] = coords_vor
    for ii, limit in enumerate(limits):
        # make points such that the voronoi cells are confined within the
        # limits
        mirror_coords[:, (2*ii+1)*n:(ii+1)*2*n] = coords_vor
        mirror_coords[ii, (2*ii+1)*n:(ii+1)*2*n] = (
            2 * limit[1] - coords_vor[ii, :])
        mirror_coords[:, (ii+1)*2*n:(2*ii+3)*n] = coords_vor
        mirror_coords[ii, (ii+1)*2*n:(2*ii+3)*n] = (
            2 * limit[0] - coords_vor[ii, :])
    points = np.array(mirror_coords).T
    vor = Voronoi(points)

    return vor


def volumes_vor(vor,  n):
    """
    Estimate area of given voronoi points. indices larger than n or or
    present in nan_idx are not considered."""
    vol = np.zeros(n)
    for ii, reg_num in enumerate(vor.point_region):
        if ii >= n:  # only consider the original points
            continue
        indices = vor.regions[reg_num]
        if -1 in indices:  # some regoins are open, then the area is infinite
            vol[ii] = np.inf
        else:
            vol[ii] = ConvexHull(vor.vertices[indices]).volume
    return vol


def neighbors_vor(vor, n, nan_idx):
    """
    Find neighbours of given voronoi points. indices larger than n or or
    present in nan_idx are not by convention equal to zero"""

    w = np.zeros([n, n])
    for ridge in vor.ridge_points:
        ii = ridge[0]
        jj = ridge[1]
        if ii == -1 or jj == -1:
            continue
        # only consider the original points
        if ii >= n or jj >= n:
            continue
        # filter out the neighbours of tiles that are not to be considered
        if nan_idx is not None:
            if ii in nan_idx or jj in nan_idx:
                continue
        w[max(ii, jj), min(jj, ii)] = 1
    return w


def plot_averagemap(
        values,
        grid,
        vmin=None,
        vmax=None,
        vcenter=None,
        ax=None,
        cmap=None,
        show_colorbar=True):
    """
    Plot the average values on a 2D grid.

    Args:
        values:     Values to be plotted
        grid:       Grid coordinates. It should have the structure [x1, x2],
                where xi are vectors of the same length (number of grid points)
        vmin:       Minimum value of the colormap
        vmax:       Maximum value of the colormap
        vcenter:    Center value of the colormap
        ax:         Axis to plot the figure
        cmap:       Colormap to be used
        show_colorbar:  If true, colorbar is shown
    """
    idx_x = 0
    idx_z = 0
    for ii in range(1, len(grid[0, :])):
        if grid[0, ii] == grid[0, ii-1]:
            idx_z += 1
        else:
            idx_x += 1
            idx_z = 0
    b_matrix = np.zeros([idx_x+1, idx_z+1])

    idx_x = 0
    idx_z = 0
    for ii in range(0, len(grid[0, :])):
        if ii > 0:
            if grid[0, ii] == grid[0, ii-1]:
                idx_z += 1
            else:
                idx_x += 1
                idx_z = 0
            b_matrix[idx_x, idx_z] = values[ii]
        else:
            b_matrix[idx_x, idx_z] = values[ii]
    b_matrix = b_matrix.T

    # colormap definition
    extent = [min(grid[0, :]), max(grid[0, :]),
              min(grid[1, :]), max(grid[1, :])]
    if cmap is None:
        cmap = cm.viridis
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 1.5))
    if vmin is not None and vmax is not None:
        if vcenter is not None:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        im = plt.imshow(b_matrix, cmap=cmap, extent=extent,
                        aspect='equal', norm=norm)
    else:
        im = plt.imshow(b_matrix, cmap=cmap, extent=extent, aspect='equal')

    ax.set_facecolor('xkcd:dark grey')
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    return ax