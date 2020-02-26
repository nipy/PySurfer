import gc
import os
import os.path as op
from os.path import join as pjoin
import sys

import pytest
from mayavi import mlab
import nibabel as nib
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less

from unittest import SkipTest

from surfer import Brain, io, utils
from surfer.utils import (requires_fsaverage, requires_imageio, requires_fs,
                          _get_extra)

subject_id = 'fsaverage'
std_args = [subject_id, 'lh', 'inflated']
data_dir = pjoin(op.dirname(__file__), '..', '..', 'examples', 'example_data')

overlay_fname = pjoin(data_dir, 'lh.sig.nii.gz')


def _set_backend(backend=None):
    """Use testing backend for Windows."""
    only_test = (sys.platform == 'win32' or
                 (os.getenv('TRAVIS', 'false') == 'true' and
                  sys.platform == 'linux') and sys.version[0] == '3')
    if backend is None:
        backend = 'test' if only_test else 'auto'
    if only_test and backend != 'test':
        raise SkipTest('non-testing backend crashes on Windows and '
                       'Travis Py3k')
    mlab.options.backend = backend


def get_view(brain):
    """Setup for view persistence test"""
    fig = brain._figures[0][0]
    if mlab.options.backend == 'test':
        return
    fig.scene.camera.parallel_scale = 50
    assert fig.scene.camera.parallel_scale == 50
    view, roll = brain.show_view()
    return fig.scene.camera.parallel_scale, view, roll


def check_view(brain, view):
    """Test view persistence"""
    fig = brain._figures[0][0]
    if mlab.options.backend == 'test':
        return
    parallel_scale, view, roll = view
    assert fig.scene.camera.parallel_scale == parallel_scale
    view_now, roll_now = brain.show_view()
    assert view_now[:3] == view[:3]
    assert_array_equal(view_now[3], view[3])
    assert roll_now == roll


@requires_fsaverage()
def test_offscreen():
    """Test offscreen rendering."""
    _set_backend()
    brain = Brain(*std_args, offscreen=True)
    shot = brain.screenshot()
    assert_array_less((400, 400, 2), shot.shape)
    assert_array_less(shot.shape, (801, 801, 4))
    brain.close()


@requires_fsaverage()
def test_image(tmpdir):
    """Test image saving."""
    tmp_name = tmpdir.join('temp.png')
    tmp_name = str(tmp_name)  # coerce to str to avoid PIL error

    _set_backend()
    subject_id, _, surf = std_args
    brain = Brain(subject_id, 'both', surf=surf, size=100)
    brain.add_overlay(overlay_fname, hemi='lh', min=5, max=20, sign="pos")
    brain.save_imageset(tmp_name, ['med', 'lat'], 'jpg')
    brain.save_image(tmp_name)
    brain.save_image(tmp_name, 'rgba', True)
    brain.screenshot()
    brain.save_montage(tmp_name, ['l', 'v', 'm'], orientation='v')
    brain.save_montage(tmp_name, ['l', 'v', 'm'], orientation='h')
    brain.save_montage(tmp_name, [['l', 'v'], ['m', 'f']])
    brain.close()


@requires_fsaverage()
def test_brain_separate():
    """Test that Brain does not reuse existing figures by default."""
    _set_backend('auto')
    brain = Brain(*std_args)
    assert brain.brain_matrix.size == 1
    brain_2 = Brain(*std_args)
    assert brain_2.brain_matrix.size == 1
    assert brain._figures[0][0] is not brain_2._figures[0][0]
    brain_3 = Brain(*std_args, figure=brain._figures[0][0])
    assert brain._figures[0][0] is brain_3._figures[0][0]


@requires_fsaverage()
def test_brains():
    """Test plotting of Brain with different arguments."""
    # testing backend breaks when passing in a figure, so we use 'auto' here
    # (shouldn't affect usability, but it makes testing more annoying)
    _set_backend('auto')
    mlab.figure(101)
    surfs = ['inflated', 'white', 'white', 'white', 'white', 'white', 'white']
    hemis = ['lh', 'rh', 'both', 'both', 'rh', 'both', 'both']
    titles = [None, 'Hello', 'Good bye!', 'lut test',
              'dict test', 'None test', 'RGB test']
    cortices = ["low_contrast", ("Reds", 0, 1, False), 'hotpink',
                ['yellow', 'blue'], dict(colormap='Greys'),
                None, (0.5, 0.5, 0.5)]
    sizes = [500, (400, 300), (300, 300), (300, 400), 500, 400, 300]
    backgrounds = ["white", "blue", "black", "0.75",
                   (0.2, 0.2, 0.2), "black", "0.75"]
    foregrounds = ["black", "white", "0.75", "red",
                   (0.2, 0.2, 0.2), "blue", "black"]
    figs = [101, mlab.figure(), None, None, mlab.figure(), None, None]
    subj_dir = utils._get_subjects_dir()
    subj_dirs = [None, subj_dir, subj_dir, subj_dir,
                 subj_dir, subj_dir, subj_dir]
    alphas = [1.0, 0.5, 0.25, 0.7, 0.5, 0.25, 0.7]
    for surf, hemi, title, cort, s, bg, fg, fig, sd, alpha \
            in zip(surfs, hemis, titles, cortices, sizes,
                   backgrounds, foregrounds, figs, subj_dirs, alphas):
        brain = Brain(subject_id, hemi, surf, title=title, cortex=cort,
                      alpha=alpha, size=s, background=bg, foreground=fg,
                      figure=fig, subjects_dir=sd)
        with np.errstate(invalid='ignore'):  # encountered in double_scalars
            brain.set_distance()
        brain.close()
    brain = Brain(subject_id, hemi, surf, subjects_dir=sd,
                  interaction='terrain')
    brain.close()
    pytest.raises(ValueError, Brain, subject_id, 'lh', 'inflated',
                  subjects_dir='')
    pytest.raises(ValueError, Brain, subject_id, 'lh', 'inflated',
                  interaction='foo', subjects_dir=sd)


@requires_fsaverage()
def test_annot():
    """Test plotting of annot."""
    _set_backend()
    annots = ['aparc', 'aparc.a2005s']
    borders = [True, False, 2]
    alphas = [1, 0.5]
    brain = Brain(*std_args)
    view = get_view(brain)

    for a, b, p in zip(annots, borders, alphas):
        brain.add_annotation(a, b, p, opacity=0.8)
    check_view(brain, view)

    brain.set_surf('white')
    with pytest.raises(ValueError):
        brain.add_annotation('aparc', borders=-1)

    subj_dir = utils._get_subjects_dir()
    annot_path = pjoin(subj_dir, subject_id, 'label', 'lh.aparc.a2009s.annot')
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    brain.add_annotation((labels, ctab))

    brain.add_annotation('aparc', color="red", remove_existing=True)
    surf = brain.annot["surface"]
    ctab = surf.module_manager.scalar_lut_manager.lut.table
    for color in ctab:
        assert color[:3] == (255, 0, 0)

    brain.close()


@requires_fsaverage()
def test_contour():
    """Test plotting of contour overlay."""
    _set_backend()
    brain = Brain(*std_args)
    view = get_view(brain)

    overlay_file = pjoin(data_dir, "lh.sig.nii.gz")
    brain.add_contour_overlay(overlay_file)
    brain.add_contour_overlay(overlay_file, max=20, n_contours=9,
                              line_width=2)
    brain.contour['surface'].actor.property.line_width = 1
    brain.contour['surface'].contour.number_of_contours = 10

    check_view(brain, view)
    brain.close()


@requires_fsaverage()
@requires_fs()
def test_data():
    """Test plotting of data."""
    _set_backend()
    brain = Brain(*std_args)
    mri_file = pjoin(data_dir, 'resting_corr.nii.gz')
    reg_file = pjoin(data_dir, 'register.dat')
    surf_data = io.project_volume_data(mri_file, "lh", reg_file)
    brain.add_data(surf_data, -.7, .7, colormap="jet", alpha=.7)
    brain.set_surf('white')
    brain.add_data([], vertices=np.array([], int))
    brain.close()


@requires_fsaverage()
def test_close():
    """Test that close and del actually work."""
    _set_backend()
    brain = Brain('fsaverage', 'both', 'inflated')
    brain.close()
    brain.__del__()
    del brain
    gc.collect()


@requires_fsaverage()
def test_data_limits():
    """Test handling of data limits."""
    _set_backend()
    brain = Brain('fsaverage', 'both', 'inflated')
    surf_data = np.linspace(0, 1, 163842)
    pytest.raises(ValueError, brain.add_data, surf_data, 0, 0)
    brain.add_data(surf_data, 0, 1, hemi='lh')
    assert brain.data_dict['lh']['fmax'] == 1.
    brain.add_data(surf_data, 0, 0.5, hemi='rh')
    assert brain.data_dict['lh']['fmax'] == 1.  # unmodified
    assert brain.data_dict['rh']['fmax'] == 0.5
    brain.close()


@requires_fsaverage()
def test_foci():
    """Test plotting of foci."""
    _set_backend('test')
    brain = Brain(*std_args)
    coords = [[-36, 18, -3],
              [-43, 25, 24],
              [-48, 26, -2]]
    brain.add_foci(coords,
                   map_surface="white",
                   color="gold",
                   name='test1',
                   resolution=25)

    subj_dir = utils._get_subjects_dir()
    annot_path = pjoin(subj_dir, subject_id, 'label', 'lh.aparc.a2009s.annot')
    ids, ctab, names = nib.freesurfer.read_annot(annot_path)
    verts = np.arange(0, len(ids))
    coords = np.random.permutation(verts[ids == 74])[:10]
    scale_factor = 0.7
    brain.add_foci(coords, coords_as_verts=True, scale_factor=scale_factor,
                   color="#A52A2A", name='test2')
    with pytest.raises(ValueError):
        brain.remove_foci(['test4'])
    brain.remove_foci('test1')
    brain.remove_foci()
    assert len(brain.foci_dict) == 0
    brain.close()


@requires_fsaverage()
def test_label():
    """Test plotting of label."""
    _set_backend()
    subject_id = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    brain = Brain(subject_id, hemi, surf)
    view = get_view(brain)

    extra, subj_dir = _get_extra()
    brain.add_label("BA1" + extra)
    check_view(brain, view)
    brain.add_label("BA1" + extra, color="blue", scalar_thresh=.5)
    label_file = pjoin(subj_dir, subject_id,
                       "label", "%s.MT%s.label" % (hemi, extra))
    brain.add_label(label_file)
    brain.add_label("BA44" + extra, borders=True)
    brain.add_label("BA6" + extra, alpha=.7)
    brain.show_view("medial")
    brain.add_label("V1" + extra, color="steelblue", alpha=.6)
    brain.add_label("V2" + extra, color="#FF6347", alpha=.6)
    brain.add_label("entorhinal" + extra, color=(.2, 1, .5), alpha=.6)
    brain.set_surf('white')
    brain.show_view(dict(elevation=40, distance=430), distance=430)
    with pytest.raises(ValueError, match='!='):
        brain.show_view(dict(elevation=40, distance=430), distance=431)

    # remove labels
    brain.remove_labels('V1' + extra)
    assert 'V2' + extra in brain.labels_dict
    assert 'V1' + extra not in brain.labels_dict
    brain.remove_labels()
    assert 'V2' + extra not in brain.labels_dict

    brain.close()


@requires_fsaverage()
def test_meg_inverse():
    """Test plotting of MEG inverse solution."""
    _set_backend()
    brain = Brain(*std_args)
    stc_fname = os.path.join(data_dir, 'meg_source_estimate-lh.stc')
    stc = io.read_stc(stc_fname)
    vertices = stc['vertices']
    colormap = 'hot'
    data = stc['data']
    data_full = (brain.geo['lh'].nn[vertices][..., np.newaxis] *
                 data[:, np.newaxis])
    time = np.linspace(stc['tmin'], stc['tmin'] + data.shape[1] * stc['tstep'],
                       data.shape[1], endpoint=False)

    def time_label(t):
        return 'time=%0.2f ms' % (1e3 * t)

    for use_data in (data, data_full):
        brain.add_data(use_data, colormap=colormap, vertices=vertices,
                       smoothing_steps=1, time=time, time_label=time_label)

    brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)
    assert brain.data_dict['lh']['time_idx'] == 0

    brain.set_time(.1)
    assert brain.data_dict['lh']['time_idx'] == 2
    # viewer = TimeViewer(brain)

    # multiple data layers
    pytest.raises(ValueError, brain.add_data, data, vertices=vertices,
                  time=time[:-1])
    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=1, time=time, time_label=time_label,
                   initial_time=.09)
    assert brain.data_dict['lh']['time_idx'] == 1
    data_dicts = brain._data_dicts['lh']
    assert len(data_dicts) == 3
    assert data_dicts[0]['time_idx'] == 1
    assert data_dicts[1]['time_idx'] == 1

    # shift time in both layers
    brain.set_data_time_index(0)
    assert data_dicts[0]['time_idx'] == 0
    assert data_dicts[1]['time_idx'] == 0
    brain.set_data_smoothing_steps(2)

    # add second data-layer without time axis
    brain.add_data(data[:, 1], colormap=colormap, vertices=vertices,
                   smoothing_steps=2)
    brain.set_data_time_index(2)
    assert len(data_dicts) == 4

    # change surface
    brain.set_surf('white')

    # remove all layers
    brain.remove_data()
    assert brain._data_dicts['lh'] == []

    brain.close()


@requires_fsaverage()
def test_morphometry():
    """Test plotting of morphometry."""
    _set_backend()
    brain = Brain(*std_args)
    brain.add_morphometry("curv")
    brain.add_morphometry("sulc", grayscale=True)
    brain.add_morphometry("thickness")
    brain.close()


@requires_imageio()
@requires_fsaverage()
def test_movie(tmpdir):
    """Test saving a movie of an MEG inverse solution."""
    import imageio
    if sys.version_info < (3,):
        raise SkipTest('imageio ffmpeg requires Python 3')
    # create and setup the Brain instance
    _set_backend()
    brain = Brain(*std_args)
    stc_fname = os.path.join(data_dir, 'meg_source_estimate-lh.stc')
    stc = io.read_stc(stc_fname)
    data = stc['data']
    time = np.arange(data.shape[1]) * stc['tstep'] + stc['tmin']
    brain.add_data(data, colormap='hot', vertices=stc['vertices'],
                   smoothing_steps=10, time=time, time_label='time=%0.2f ms')
    brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)

    # save movies with different options
    dst = str(tmpdir.join('test.mov'))
    # test the number of frames in the movie
    brain.save_movie(dst)
    frames = imageio.mimread(dst)
    assert len(frames) == 2
    brain.save_movie(dst, time_dilation=10)
    frames = imageio.mimread(dst)
    assert len(frames) == 7
    brain.save_movie(dst, tmin=0.081, tmax=0.102)
    frames = imageio.mimread(dst)
    assert len(frames) == 2
    brain.close()


@requires_fsaverage()
def test_overlay():
    """Test plotting of overlay."""
    _set_backend()
    # basic overlay support
    overlay_file = pjoin(data_dir, "lh.sig.nii.gz")
    brain = Brain(*std_args)
    brain.add_overlay(overlay_file)
    brain.overlays["sig"].remove()
    brain.add_overlay(overlay_file, min=5, max=20, sign="pos", opacity=0.7)
    sig1 = io.read_scalar_data(pjoin(data_dir, "lh.sig.nii.gz"))
    sig2 = io.read_scalar_data(pjoin(data_dir, "lh.alt_sig.nii.gz"))

    # two-sided overlay
    brain.add_overlay(sig1, 4, 30, name="two-sided")
    overlay = brain.overlays_dict.pop('two-sided')[0]
    assert_array_equal(overlay.pos_bar.data_range, [4, 30])
    assert_array_equal(overlay.neg_bar.data_range, [-30, -4])
    assert overlay.pos_bar.reverse_lut
    assert not overlay.neg_bar.reverse_lut
    overlay.remove()

    thresh = 4
    sig1[sig1 < thresh] = 0
    sig2[sig2 < thresh] = 0

    conjunct = np.min(np.vstack((sig1, sig2)), axis=0)
    brain.add_overlay(sig1, 4, 30, name="sig1")
    brain.overlays["sig1"].pos_bar.lut_mode = "Reds"
    brain.overlays["sig1"].pos_bar.visible = False

    brain.add_overlay(sig2, 4, 30, name="sig2")
    brain.overlays["sig2"].pos_bar.lut_mode = "Blues"
    brain.overlays["sig2"].pos_bar.visible = False

    brain.add_overlay(conjunct, 4, 30, name="conjunct")
    brain.overlays["conjunct"].pos_bar.lut_mode = "Purples"
    brain.overlays["conjunct"].pos_bar.visible = False

    brain.set_surf('white')

    brain.close()


@requires_fsaverage()
def test_probabilistic_labels():
    """Test plotting of probabilistic labels."""
    _set_backend()
    brain = Brain("fsaverage", "lh", "inflated",
                  cortex="low_contrast")

    extra, subj_dir = _get_extra()
    brain.add_label("BA1" + extra, color="darkblue")
    brain.add_label("BA1" + extra, color="dodgerblue", scalar_thresh=.5)
    brain.add_label("BA45" + extra, color="firebrick", borders=True)
    brain.add_label("BA45" + extra, color="salmon", borders=True,
                    scalar_thresh=.5)

    label_file = pjoin(subj_dir, "fsaverage", "label",
                       "lh.BA6%s.label" % (extra,))
    prob_field = np.zeros_like(brain.geo['lh'].x)
    ids, probs = nib.freesurfer.read_label(label_file, read_scalars=True)
    prob_field[ids] = probs
    brain.add_data(prob_field, thresh=1e-5)

    brain.data["colorbar"].number_of_colors = 10
    brain.data["colorbar"].number_of_labels = 11
    brain.close()


@requires_fsaverage()
def test_text():
    """Test plotting of text."""
    _set_backend('test')
    brain = Brain(*std_args)
    brain.add_text(0.1, 0.1, 'Hello', 'blah')
    brain.close()


@requires_fsaverage()
def test_animate(tmpdir):
    """Test animation."""
    _set_backend('auto')
    brain = Brain(*std_args, size=100)
    brain.add_morphometry('curv')
    tmp_name = str(tmpdir.join('test.avi'))
    brain.animate(["m"] * 3, n_steps=2)
    brain.animate(['l', 'l'], n_steps=2, fname=tmp_name)
    # can't rotate in axial plane
    pytest.raises(ValueError, brain.animate, ['l', 'd'])
    brain.close()


@requires_fsaverage()
def test_views():
    """Test showing different views."""
    _set_backend('test')
    brain = Brain(*std_args)
    brain.show_view('lateral')
    brain.show_view('m')
    brain.show_view('rostral')
    brain.show_view('caudal')
    brain.show_view('ve')
    brain.show_view('frontal')
    brain.show_view('par')
    brain.show_view('dor')
    brain.show_view({'distance': 432})
    brain.show_view({'azimuth': 135, 'elevation': 79}, roll=107)
    brain.close()
