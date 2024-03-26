import numpy as np
from scipy.spatial import Voronoi
import progressbar

from functions.transformation_functions import (
    translation,
    transform_and_rotate,
    cut_section,
)
from seismostats.analysis.estimate_beta import estimate_b


import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


# ======= attention: not well tested yet ===============================
# =========and partly not working for general cases!=====================
def b_value_imaging_tiles(
    cut_coords,
    cut_mags,
    length,
    depth,
    x_vor,
    y_vor,
    z_vor,
    tesselation="voronoi",
    min_num=2,
    radius=None,
    b_method="pos",
    delta_m=0.1,
    n_points=10,
):
    if tesselation == "radius":
        # create a grid
        delta = 1
        x_vec = np.arange(0, length + delta, delta)
        z_vec = -np.arange(0, depth + delta, delta)
        x_vor = np.array([])
        z_vor = np.array([])
        for x_loop in x_vec:
            x_vor = np.concatenate((x_vor, np.ones(len(z_vec)) * x_loop))
            z_vor = np.concatenate((z_vor, z_vec))
        y_vor = np.zeros(len(z_vor))

    if tesselation == "voronoi":
        # create a grid
        # delta = 4
        # x_vec = np.arange(0, length + delta, delta)
        # z_vec = -np.arange(0, depth + delta, delta)
        # x_vor = np.array([])
        # z_vor = np.array([])
        # for x_loop in x_vec:
        #    x_vor = np.concatenate((x_vor, np.ones(len(z_vec)) * x_loop))
        #    z_vor = np.concatenate((z_vor, z_vec))

        x_vor, z_vor = np.random.rand(2, n_points)
        x_vor = length * x_vor
        z_vor = -depth * z_vor
        y_vor = np.zeros(len(z_vor))

    if tesselation == "x":
        delta = 2
        x_vor = np.arange(0, length + delta, delta)
        # x_vor = np.random.rand(n_points)
        # x_vor = length * x_vor
        z_vor = np.zeros(len(x_vor))
        y_vor = np.zeros(len(x_vor))

    if tesselation == "z":
        delta = 0.2
        z_vor = np.arange(0, -depth - delta, -delta)
        # z_vor = np.random.rand(n_points)
        # z_vor = -depth * z_vor
        x_vor = np.zeros(len(z_vor))
        y_vor = np.zeros(len(z_vor))

    # b values on plane
    b_vec, ks_vec, sig_b_vec, n_est = bvalues_on_plane3(
        [x_vor, y_vor, z_vor],
        cut_coords,
        cut_mags,
        min_num,
        delta_m=delta_m,
        b_method=b_method,
        tesselation=tesselation,
        radius=radius,
    )

    return b_vec, ks_vec, sig_b_vec, x_vor, z_vor, n_est, cut_mags


def bvalues_on_plane3(
    voronoi_coords,
    cartesian,
    magnitudes,
    min_num,
    delta_m,
    b_method="tinti",
    tesselation="tiles",
    radius=None,
):
    """calculates the b-values along the plane defined by x and z
    input:
    - x, z: vectors from zero to length and depth, respectively
    - radius: radius used for b-value calculation
    - cartesian: coordinates of the EQs [x, y, z] (x,y,z are vectors)
    - magnitudes: magnitudes corresponding to the coordinates
    - mc: cut-off magnitude
    - min_num: minimum number of EQs to start calculating b-values
    output:
    - bb: grid of max likelihood b-values
    - ks: kolmogorov–smirnov significance (p-value of the hypothesis that we
      drew from the exponential with estimated b-value)
    - sig_b: certainty itervals of bb, only dependent on n"""
    x_vor = voronoi_coords[0]
    y_vor = voronoi_coords[1]
    z_vor = voronoi_coords[2]
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]

    bb = np.zeros(len(x_vor))
    ks = np.zeros(len(x_vor))
    sig_b = np.zeros(len(x_vor))
    n_est = np.zeros(len(x_vor))

    # calculate squared distance for each EQ
    if tesselation == "voronoi" or tesselation == "x" or tesselation == "z":
        nearest = find_nearest_vor_node(x_vor, y_vor, z_vor, x, y, z)

    # now go through all voronoi indices
    for ii in range(len(x_vor)):
        if tesselation == "radius":
            # get vector of center of circle
            center = np.array([x[ii], 0, z[ii]])
            # cut out circles with radius
            circle_catalogue, idx = cut_sphere(cartesian, radius, center)
        elif (
            tesselation == "voronoi"
            or tesselation == "x"
            or tesselation == "z"
        ):
            idx = nearest == ii
            # circle_catalogue = [x[idx], y[idx], z[idx]]
            # TODO delete this
        else:
            print("choose tesselation pls")
        circle_magnitudes = magnitudes[idx]

        if len(circle_magnitudes) >= min_num:
            # compute b-value
            ks[ii] = ks_test_lillifors(bb[ii], circle_magnitudes)
            n_est[ii] = len(circle_magnitudes)
            bb[ii], sig_b[ii] = estimate_b(
                circle_magnitudes,
                mc=min(circle_magnitudes),
                delta_m=delta_m,
                method=b_method,
                return_std=True,
            )

            # the variance expressed in relation to the b-value
            sig_b[ii] = sig_b[ii] / bb[ii]
        else:
            n_est[ii] = 0
            bb[ii] = None
            ks[ii] = None
            sig_b[ii] = None
    return bb, ks, sig_b, n_est


def find_nearest_vor_node(x_vor, y_vor, z_vor, x, y, z):
    nearest = []
    # TODO distance matrix scipy.spatial, matrix argmin
    for x_el, y_el, z_el in zip(x, y, z):
        dist = (x_vor - x_el) ** 2 + (y_vor - y_el) ** 2 + (z_vor - z_el) ** 2
        nearest.append(np.argmin(dist))
    nearest = np.array(nearest)
    return nearest


def cut_sphere(cart_coords, radius, center):
    """only returns the elements that are within a radius from the center of a
    sphere

    Args:
        cart_coords: cartesian coordinates [x,y,z] (x,y,z are vectors)
        radius:     radius of the sphere, in km
        center: vector of the origin of the sphere

    Returns:
        circle_coords:  coordinates of the points within the sphere (same
                    format as cartesian)
    """
    # translate origin to center of circle
    trans_coords = translation(cart_coords, center)
    trans_coords = trans_coords.transpose()
    # distances to center
    distances = np.linalg.norm(trans_coords, axis=1)
    # compute distances to origin
    idx = [idx for idx, dist in enumerate(distances) if dist <= radius]
    circle_coords = trans_coords[idx]
    circle_coords = circle_coords.transpose()
    # circle_catalogue = catalogue.transpose()
    return circle_coords, idx


def ks_test_lillifors(b, magnitudes):
    """calculates the kolmogorov smirnoff significance that sample was not
    drawn from the fitted distribution.

    Source: Lilliefors (1969) On the Kolmogorov-Smirnov Test for the
    Exponential Distribution with Mean Unknown, Journal of the American
    Statistical Association

    Args:
        b:      b-value
    Returns:
        ks_rel: relative significance of the hypothesis that the sample was
            not drawn. ks_crit is   the ks length for a significance of 0.1.
            ks_rel < 1 means therefore that the hypothesis is more likely,
            ks_rel > 1 means that the hypothesis is less likely.
    -"""
    x = np.sort(magnitudes)
    x = x[::-1]
    y = np.array(range(1, len(x) + 1))
    obs_dist = y / len(y)
    th_dist = 10 ** (-b * (x - min(x)))
    diff = abs(th_dist - obs_dist)
    ks_dist = max(diff)
    ks_crit = 0.96 / np.sqrt(len(x))
    ks_rel = ks_dist / ks_crit
    return ks_rel


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def image_average_voronoi(
    cut_coords, cut_mags, length, depth, n_points, n, delta=1
):
    """computes the average b-value map for a given section

    Args:
        cut_coords:     coordinates of the EQs [x,y,z]. x,y,z are vectors, and
                    y is 0 everywhere
        length:         length of the section
        depth:          depth of the section
        width:          width of the section

    """
    # create a grid
    x_vec = np.arange(0, length + delta, delta)
    z_vec = -np.arange(0, depth + delta, delta)
    # TODO itertools product
    x = np.array([])
    z = np.array([])
    for x_loop in x_vec:
        x = np.concatenate((x, np.ones(len(z_vec)) * x_loop))
        z = np.concatenate((z, z_vec))
    y = np.zeros(len(x))

    b_average = np.zeros(len(x))
    average_cnt = np.zeros(len(x))
    acf_2D = []

    bar = progressbar.ProgressBar(
        maxval=n,
        widgets=[
            progressbar.Bar("=", "[", "]"),
            " ",
            progressbar.Percentage(),
        ],
    )
    bar.start()
    for ii in range(n):
        (
            b_vec,
            ks_vec,
            sig_b_vec,
            x_vor,
            z_vor,
            n_est,
            cut_mags,
        ) = b_value_imaging_tiles(
            cut_coords,
            cut_mags,
            length,
            depth,
            [],
            [],
            [],
            tesselation="voronoi",
            min_num=30,
            radius=None,
            b_method="positive",
            n_points=n_points,
            delta_m=0.01,
        )
        y_vor = np.zeros(len(x_vor))
        nearest = find_nearest_vor_node(x_vor, y_vor, z_vor, x, y, z)
        b_loop = b_vec[nearest]
        b_loop[np.isnan(b_loop)] = 0
        b_average += b_loop
        average_cnt += b_loop > 0

        # estimate correlation
        acf_2D.append(est_acf_2d(b_vec, x_vor, z_vor, length, depth))

        bar.update(ii + 1)
    bar.finish()
    average_cnt[average_cnt == 0] = np.nan
    b_average = b_average / average_cnt

    return b_average, x, z, np.array(acf_2D)


def image_average_voronoi_old(cat_df, p1, p2, width, depth, n_points, n):
    """computes the average b-value map for a given section

    Args:
        cut_coords: coordinates of the EQs [x,y,z] (x,y,z are vectors)
        length:     length of the section
        depth:      depth of the section
        width:      width of the section

    """
    # transform data (only in order to get the length)
    mags = np.array(cat_df["magnitude"])
    lats = np.array(cat_df["latitude"])
    lons = np.array(cat_df["longitude"])
    depths = np.array(cat_df["depth"])
    # transform to cartesian coordinates, to that p1 is on origin, p2 is on y
    # axis and the fault plane is the x-z plane
    cart_coords, cart_p2 = transform_and_rotate(p1, p2, lats, lons, depths)
    # cut data to only the one in the section
    length = np.linalg.norm(cart_p2)
    cut_coords, idx1 = cut_section(cart_coords, length, width, depth)
    cut_mags = mags[idx1]

    # create a grid
    delta = 1
    x_vec = np.arange(0, length + delta, delta)
    z_vec = -np.arange(0, depth + delta, delta)
    # TODO itertools product
    x = np.array([])
    z = np.array([])
    for x_loop in x_vec:
        x = np.concatenate((x, np.ones(len(z_vec)) * x_loop))
        z = np.concatenate((z, z_vec))
    y = np.zeros(len(x))

    b_average = np.zeros(len(x))
    average_cnt = np.zeros(len(x))
    acf_2D = []

    bar = progressbar.ProgressBar(
        maxval=n,
        widgets=[
            progressbar.Bar("=", "[", "]"),
            " ",
            progressbar.Percentage(),
        ],
    )
    bar.start()
    for ii in range(n):
        (
            b_vec,
            ks_vec,
            sig_b_vec,
            x_vor,
            z_vor,
            n_est,
            cut_mags,
        ) = b_value_imaging_tiles(
            cut_coords,
            cut_mags,
            length,
            depth,
            [],
            [],
            [],
            tesselation="voronoi",
            min_num=30,
            radius=None,
            b_method="positive",
            mc_method="constant",
            mc=1.1,
            n_points=n_points,
            delta_m=0.01,
        )
        y_vor = np.zeros(len(x_vor))
        nearest = find_nearest_vor_node(x_vor, y_vor, z_vor, x, y, z)
        b_loop = b_vec[nearest]
        b_loop[np.isnan(b_loop)] = 0
        b_average += b_loop
        average_cnt += b_loop > 0

        # estimate correlation
        acf_2D.append(est_acf_2d(b_vec, x_vor, z_vor, length, depth))

        bar.update(ii + 1)
    bar.finish()
    average_cnt[average_cnt == 0] = np.nan
    b_average = b_average / average_cnt

    return b_average, x, z, np.array(acf_2D)


def est_acf_2d(b_vec, x_vor, z_vor, length, depth):
    # 1. get the vor points, and mirror them in such a way that the cells
    # are confined within the fault plane
    # print(len(b_vec), "number of b-value estimates")
    points = []
    for x_el, z_el in zip(x_vor, z_vor):
        points.append([x_el, z_el])
    points = np.array(points)

    x_max = length
    x_min = 0
    y_max = 0
    y_min = -depth

    mirrorx1 = points * 1
    mirrorx1[:, 0] = x_min - abs(x_min - mirrorx1[:, 0])
    mirrorx2 = points * 1
    mirrorx2[:, 0] = x_max + abs(x_max - mirrorx2[:, 0])
    mirrory1 = points * 1
    mirrory1[:, 1] = y_min - abs(y_min - mirrory1[:, 1])
    mirrory2 = points * 1
    mirrory2[:, 1] = y_max + abs(y_max - mirrory2[:, 1])
    mirror_points = (
        points.tolist()
        + mirrorx1.tolist()
        + mirrorx2.tolist()
        + mirrory1.tolist()
        + mirrory2.tolist()
    )

    vor = Voronoi(mirror_points)
    # 2. compute list of neighbouring pairs, but only where there is a bvalue
    idx_nonan = np.argwhere(~np.isnan(b_vec)).flatten()

    point_idx = []
    check = 0
    for ii in idx_nonan:  # only goes through points that are not mirrored
        idx = vor.point_region[ii]
        region_a = np.array(vor.regions[idx])
        check += 1
        for jj in idx_nonan[check:]:
            idx = vor.point_region[jj]
            region_b = np.array(vor.regions[idx])
            same_points = count_equal_els(
                region_a[region_a > -1], region_b[region_b > -1]
            )
            if same_points >= 2:
                point_idx.append((max(ii, jj), min(jj, ii)))
    point_idx = list(set(point_idx))
    # 3. estimate the auto cross correlation
    acf_pairs = 0
    acf_0 = 0

    mean_b = np.mean(b_vec[~np.isnan(b_vec)])
    # print(len(point_idx), "number of pairs")
    for ii, pair in enumerate(point_idx):
        b_1 = b_vec[pair[0]]
        b_2 = b_vec[pair[1]]
        if not np.isnan(b_1) and not np.isnan(b_2):
            acf_pairs += (b_1 - mean_b) * (b_2 - mean_b)
            acf_0 += ((b_1 - mean_b) ** 2 + (b_2 - mean_b) ** 2) / 2

    acf_pairs /= acf_0
    return acf_pairs


def count_equal_els(v, w):
    """counts the unique number of elements that v and w share"""
    set_v = set(v)
    set_w = set(w)
    return len(set_v.intersection(set_w))


def plot_averagemap(b_average, x, z, squares):
    # construct voronoi nods
    points = []
    for point in zip(x, z):
        points.append(point)

    # compute Voronoi tesselation
    vor = Voronoi(points)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(
        vor, radius=(vor.max_bound[0] - vor.min_bound[0]) * 10000
    )

    # colormap definition
    minima = min(b_average[~np.isnan(b_average)])
    maxima = max(b_average[~np.isnan(b_average)])
    minima = 0.3
    maxima = 1.9
    cmap = cm.viridis
    # cmap = cmr.toxic
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(6, 1.5))

    # colorize
    for ii, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), color=mapper.to_rgba(b_average[ii]))

    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])
    ax.set_aspect("equal")

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    return ax


def plot_voronoi_map(b_vec, x, z):
    # construct voronoi nods
    points = []
    for point in zip(x, z):
        points.append(point)

    # compute Voronoi tesselation
    vor = Voronoi(points)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(
        vor, radius=(vor.max_bound[0] - vor.min_bound[0]) * 10000
    )

    # colormap definition
    minima = min(b_vec[~np.isnan(b_vec)])
    maxima = max(b_vec[~np.isnan(b_vec)])
    cmap = cm.viridis
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    fig, ax = plt.subplots(figsize=(20, 10))

    # colorize
    for ii, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), color=mapper.to_rgba(b_vec[ii]))

    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])
    ax.set_aspect("equal")

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    return ax


# ======= attention: not well tested yet ===============================
# =========and partly not working for general cases!=====================
