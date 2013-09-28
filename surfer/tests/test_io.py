from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_array_almost_equal

from surfer import io
from surfer.utils import requires_fsaverage

subj_dir = io._get_subjects_dir()
subject_id = 'fsaverage'
data_path = pjoin(subj_dir, subject_id)


@requires_fsaverage
def test_surface():
    """Test IO for Surface class"""
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
