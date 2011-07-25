import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


def find_closest_vertices(surface_coords, point_coords):
    """Return the vertices on a surface mesh closest to some given coordinates.

    The distance metric used is Euclidian distance.

    Parameters
    ----------
    surface_coords : numpy array
        Array of coordinates on a surface mesh
    point_coords : numpy array
        Array of coordinates to map to vertices

    Returns
    -------
    closest_vertices : numpy array
        Array of mesh vertex ids

    """
    point_coords = np.atleast_2d(point_coords)
    return np.argmin(cdist(surface_coords, point_coords), axis=0)


def tal_to_mni(coords):
    """Convert Talairach coords to MNI using the Lancaster transform.

    Parameters
    ----------
    coords : n x 3 numpy array
        Array of Talairach coordinates

    Returns
    -------
    mni_coords : n x 3 numpy array
        Array of coordinates converted to MNI space

    """
    coords = np.atleast_2d(coords)
    xfm = np.array([[  1.06860, -0.00396, 0.00826,  1.07816],
                    [  0.00640,  1.05741, 0.08566,  1.16824],
                    [ -0.01281, -0.08863, 1.10792, -4.17805],
                    [  0.00000,  0.00000, 0.00000,  1.00000]])
    mni_coords = np.dot(np.c_[coords, np.ones(coords.shape[0])], xfm.T)[:, :3]
    return mni_coords


def mesh_edges(faces):
    """Returns sparse matrix with edges as an adjacency matrix

    Parameters
    ----------
    faces : array of shape [n_triangles x 3]
        The mesh faces

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                                            shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                            shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                            shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges
