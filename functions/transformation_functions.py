import numpy as np


# ======= attention: not well tested yet ===============================
# =========and partly not working for general cases!=====================
def spherical_to_cart(lats, lons, depths):
    """transform spherical coordinates in cartesian ones

    Args:
        lats: vectors of latitude
        lons: vectors of longitude
        depths: vectors of depth

    Returns:
        cart_coords: cartesian coordinates [x,y,z] (x,y,z are vectors) of the
                input data
    """
    r = 6371 - depths  # radius of earth in km
    x = r * np.cos(lats / 180 * np.pi) * np.cos(lons / 180 * np.pi)
    y = r * np.cos(lats / 180 * np.pi) * np.sin(lons / 180 * np.pi)
    z = r * np.sin(lats / 180 * np.pi)
    cart_coords = np.array([x, y, z])
    return cart_coords


def translation(cart_coords, trans_vector):
    """translate data to that trans_vector is the origin

    Args:
        cart_coords:    cartesian coordinates [x,y,z] (x,y,z are vectors)
        trans_vector:   vector by which the data should be translated. after
                    translation, this vector will be in the origin
    Returns:
        trans_coords: translated coordinates of the same format as input
    """
    cart_coords = cart_coords.transpose()
    trans_coords = cart_coords - trans_vector
    trans_coords = trans_coords.transpose()
    return trans_coords


def rotation(cart_coords, rot_vector, k):
    """rotate data so that rot_vector is on k-axis

    Args:
        cart_coords:    cartesian coordinates [x,y,z] (x,y,z are vectors)
        rot_vector:     vector that the coordinate system will be aligned with
        k:              axis along which rot_vector is aligned

    Returns:
        rot_coords:     rotated coordinates, same format as input
    """
    # normalize vector
    n = rot_vector / np.linalg.norm(rot_vector)
    # compute angle of rotation
    theta = np.arccos(np.inner(n, k))
    # compute axis of rotation
    axis_rot = np.cross(k, n) / np.sin(theta)
    # compute rotation matrix
    ct = np.cos(theta)
    st = np.sin(theta)
    # cross product matrix
    cross_p_mat = np.cross(axis_rot, np.identity(axis_rot.shape[0]) * -1)
    outer_mat = np.outer(axis_rot, axis_rot)  # outer product
    r_mat = ct * np.identity(3) + st * cross_p_mat + (1 - ct) * outer_mat
    # rotate vectors
    rot_coords = np.matmul(r_mat.transpose(), cart_coords)
    return rot_coords


def transform_and_rotate(p1, p2, latitudes, longitudes, depths):
    """transforms data to cartesian coordinates and rotates and translates it
    so the line segment is on the x-axis

    Args:
        p1, p2:     endpoints of line segment in spherical coordinates
                [lat, lon, depth = 0] -> should be on the surface
        latidues:   vectors of the latitudes
        longitudes: vectors of the longitudes
        depths:     vectors of the depths

    Returns:
        cart_coords: transformed coordinates EQs [x, y, z] (x, y, z are
                vectors)
        p2: second endpoint of line segment (first endpoint is at origin)
    """
    # ATTENTION: p3 should not be parallel to the segment! in the future this
    # possibility should be included
    p3 = p1 + [0.1, 0.1, 0]
    # transform catalogue data to cartesian coordinates
    cart_coords = spherical_to_cart(latitudes, longitudes, depths)
    cart_p1 = spherical_to_cart(p1[0], p1[1], p1[2])
    cart_p2 = spherical_to_cart(p2[0], p2[1], p2[2])
    cart_p3 = spherical_to_cart(p3[0], p3[1], p3[2])
    # translate catalogue to p1 is the origin
    cart_coords = translation(cart_coords, cart_p1)
    cart_p3 = translation(cart_p3, cart_p1)
    cart_p2 = translation(cart_p2, cart_p1)
    # cart_p1 = translation(cart_p1, cart_p1)
    # make sure that cart_p3 is orthogonal to the line-segment and in the
    # depth=0 plane
    cart_p3 = np.cross(cart_p3, cart_p2)
    cart_p3 = np.cross(cart_p3, cart_p2)
    cart_p3 = cart_p3 / np.linalg.norm(cart_p3) * np.linalg.norm(cart_p2)
    # rotate catalogue so p1-p2 are on the x-axis
    rot_vector = cart_p2
    k = np.array([1, 0, 0])
    cart_coords = rotation(cart_coords, rot_vector, k)
    cart_p2 = rotation(cart_p2, rot_vector, k)  # parallel to x-vector
    cart_p3 = rotation(cart_p3, rot_vector, k)
    # rotate so that secondary vector is on y-axis
    rot_vector = cart_p3
    k = np.array([0, -1, 0])
    cart_coords = rotation(cart_coords, rot_vector, k)
    # cart_p2 = rotation(cart_p2, rot_vector, k)
    # cart_p3 = rotation(cart_p3, rot_vector, k) # parallel to y-vector
    return cart_coords, cart_p2


def cut_section(cart_coords, length, width, depth):
    """cuts out data outside the section. it is assumed that the data are
    rotated such that one endpoint is on the origin, the other on the positive
    y-axis and the segment is aligned with the x-z plane.
    ATTENTION: does not work for obligue sections

    Args:
        cart_coords:    cartesian coordinates [x,y,z] (x,y,z are vectors)
        length, width, depth:   length, width and depth of the section in km

    Returns:
        cut_coords: coordinates of only the points that are in the section
        idx:        indices of the points that are in the section, with
                respect to the original cooordniates -> e.g. you can obtain
                the corresponding magnitudes
    """
    x = cart_coords[0]
    y = cart_coords[1]
    z = cart_coords[2]
    # retrieve indices where the points are inside the section
    idx_x1 = x <= length
    idx_x2 = x >= 0
    idxy = abs(y) <= width / 2
    idx_z = z <= depth
    idx = idx_x1 & idx_x2 & idxy & idx_z
    # cut out data
    x = x[idx]
    y = y[idx]
    z = z[idx]
    cut_coords = np.array([x, y, z])
    return cut_coords, idx


# ======= attention: not well tested yet ===============================
# =========and partly not working for general cases!=====================
