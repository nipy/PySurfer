from distutils.version import LooseVersion
import numpy as np
import scipy
from scipy import sparse
import pytest
import matplotlib as mpl
from numpy.testing import assert_allclose, assert_array_equal

from surfer import utils


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


@utils.requires_fsaverage()
def test_surface():
    """Test IO for Surface class"""
    extra, subj_dir = utils._get_extra()
    for subjects_dir in [None, subj_dir]:
        surface = utils.Surface('fsaverage', 'lh', 'inflated',
                                subjects_dir=subjects_dir)
        surface.load_geometry()
        surface.load_label('BA1' + extra)
        surface.load_curvature()
        xfm = np.eye(4)
        xfm[:3, -1] += 2  # translation
        x = surface.x
        surface.apply_xfm(xfm)
        x_ = surface.x
        assert_allclose(x + 2, x_)

        # normals
        nn = _slow_compute_normals(surface.coords, surface.faces[:10000])
        nn_fast = utils._compute_normals(surface.coords, surface.faces[:10000])
        assert_allclose(nn, nn_fast)
        assert 50 < np.linalg.norm(surface.coords, axis=-1).mean() < 100  # mm
    surface = utils.Surface('fsaverage', 'lh', 'inflated',
                            subjects_dir=subj_dir, units='m')
    surface.load_geometry()
    assert 0.05 < np.linalg.norm(surface.coords, axis=-1).mean() < 0.1  # m


def test_huge_cross():
    """Test cross product with lots of elements."""
    x = np.random.rand(100000, 3)
    y = np.random.rand(1, 3)
    z = np.cross(x, y)
    zz = utils._fast_cross_3d(x, y)
    assert_array_equal(z, zz)


def test_create_color_lut():
    """Test various ways of making a colormap."""
    # Test valid lut
    cmap_in = (np.random.rand(256, 4) * 255).astype(int)
    cmap_out = utils.create_color_lut(cmap_in)
    assert_array_equal(cmap_in, cmap_out)

    # Test mostly valid lut
    cmap_in = cmap_in[:, :3]
    cmap_out = utils.create_color_lut(cmap_in)
    assert_array_equal(cmap_in, cmap_out[:, :3])
    assert_array_equal(cmap_out[:, 3], np.ones(256, int) * 255)

    # Test named matplotlib lut
    cmap_out = utils.create_color_lut("BuGn_r")
    assert cmap_out.shape == (256, 4)

    # Test named pysurfer lut
    cmap_out = utils.create_color_lut("icefire_r")
    assert cmap_out.shape == (256, 4)

    # Test matplotlib object lut
    cmap_in = mpl.colors.ListedColormap(["blue", "white", "red"])
    cmap_out = utils.create_color_lut(cmap_in)
    assert_array_equal(cmap_out, (cmap_in(np.linspace(0, 1, 256)) * 255))

    # Test list of colors lut
    cmap_out = utils.create_color_lut(["purple", "pink", "white"])
    assert cmap_out.shape == (256, 4)

    # Test that we can ask for a specific number of colors
    cmap_out = utils.create_color_lut("Reds", 12)
    assert cmap_out.shape == (12, 4)


def test_smooth():
    """Test smoothing support."""
    adj_mat = sparse.csc_matrix(np.repeat(np.repeat(np.eye(2), 2, 0), 2, 1))
    vertices = np.array([0, 2])
    want = np.repeat(np.eye(2), 2, axis=0)
    smooth = utils.smoothing_matrix(vertices, adj_mat).toarray()
    assert_allclose(smooth, want)
    if LooseVersion(scipy.__version__) < LooseVersion('1.3'):
        with pytest.raises(RuntimeError, match='nearest.*requires'):
            utils.smoothing_matrix(vertices, adj_mat, 'nearest')
    else:
        smooth = utils.smoothing_matrix(vertices, adj_mat, 'nearest').toarray()
        assert_allclose(smooth, want)
