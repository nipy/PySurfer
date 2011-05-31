import numpy as np
from scipy import sparse


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
