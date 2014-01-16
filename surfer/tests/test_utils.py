from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from surfer import utils

subj_dir = utils._get_subjects_dir()
subject_id = 'fsaverage'
data_path = pjoin(subj_dir, subject_id)


def _slow_compute_normals(rr, tris):
    """Efficiently compute vertex normals for triangulated surface"""
    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = np.cross((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    zidx = np.where(size == 0)[0]
    size[zidx] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    # accumulate the normals
    nn = np.zeros((len(rr), 3))
    for p, verts in enumerate(tris):
        nn[verts] += tri_nn[p, :]
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]
    return nn


@utils.requires_fsaverage
def test_surface():
    """Test IO for Surface class"""
    for subjects_dir in [None, subj_dir]:
        surface = utils.Surface('fsaverage', 'lh', 'inflated',
                                subjects_dir=subjects_dir)
        surface.load_geometry()
        surface.load_label('BA1')
        surface.load_curvature()
        xfm = np.eye(4)
        xfm[:3, -1] += 2  # translation
        x = surface.x
        surface.apply_xfm(xfm)
        x_ = surface.x
        assert_array_almost_equal(x + 2, x_)

        # normals
        nn = _slow_compute_normals(surface.coords, surface.faces[:10000])
        nn_fast = utils._compute_normals(surface.coords, surface.faces[:10000])
        assert_array_almost_equal(nn, nn_fast)


def test_huge_cross():
    """Test cross product with lots of elements
    """
    x = np.random.rand(100000, 3)
    y = np.random.rand(1, 3)
    z = np.cross(x, y)
    zz = utils._fast_cross_3d(x, y)
    assert_array_equal(z, zz)
