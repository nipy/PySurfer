import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from .. import io

if 'SUBJECTS_DIR' not in os.environ:
    raise ValueError('Test suite relies on the definition of SUBJECTS_DIR')

subj_dir = os.environ["SUBJECTS_DIR"]
subject_id = 'fsaverage'
# subject_id = 'sample'
data_path = pjoin(subj_dir, subject_id)


def test_geometry():
    """Test IO of .surf"""
    surf_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "inflated"))
    coords, faces = io.read_geometry(surf_path)
    assert_equal(0, faces.min())
    assert_equal(coords.shape[0], faces.max() + 1)

    # Test quad with sphere
    surf_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "sphere.reg"))
    coords, faces = io.read_geometry(surf_path)
    assert_equal(0, faces.min())
    assert_equal(coords.shape[0], faces.max() + 1)


def test_morph_data():
    """Test IO of morphometry data file (eg. curvature)."""
    curv_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "curv"))
    curv = io.read_morph_data(curv_path)
    assert -1.0 < curv.min() < 0
    assert 0 < curv.max() < 1.0


def test_annot():
    """Test IO of .annot"""
    annots = ['aparc', 'aparc.a2005s']
    for a in annots:
        annot_path = pjoin(data_path, "label", "%s.%s.annot" % ("lh", a))
        labels, ctab, names = io.read_annot(annot_path)
        assert labels.shape == (163842, )
        assert ctab.shape == (len(names), 5)


def test_label():
    """Test IO of .annot"""
    label_path = pjoin(data_path, "label", "lh.BA1.label")
    label = io.read_label(label_path)
    # XXX : test more
    assert np.all(label > 0)


def test_surface():
    """Test IO of .annot"""
    surface = io.Surface('fsaverage', 'lh', 'inflated')
    surface.load_geometry()
    surface.load_label('BA1')
    surface.load_curvature()
    xfm = np.eye(4)
    xfm[:3, -1] += 2  # translation
    x = surface.x
    surface.apply_xfm(xfm)
    x_ = surface.x
    assert_array_almost_equal(x + 2, x_)
