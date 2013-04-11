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


def test_surface():
    """Test of Surface class"""
    for subjects_dir in [None, subj_dir]:
        surface = io.Surface('fsaverage', 'lh', 'inflated',
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
