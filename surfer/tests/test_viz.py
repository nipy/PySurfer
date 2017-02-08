import os
import os.path as op
from os.path import join as pjoin
import shutil
from tempfile import mkdtemp, mktemp

from nose.tools import assert_equal
from mayavi import mlab
import nibabel as nib
import numpy as np
from numpy.testing import assert_raises, assert_array_equal

from surfer import Brain, io, utils
from surfer.utils import requires_fsaverage, requires_imageio

subj_dir = utils._get_subjects_dir()
subject_id = 'fsaverage'
std_args = [subject_id, 'lh', 'inflated']
data_dir = pjoin(op.dirname(__file__), '..', '..', 'examples', 'example_data')

overlay_fname = pjoin(data_dir, 'lh.sig.nii.gz')


def has_freesurfer():
    if 'FREESURFER_HOME' not in os.environ:
        return False
    else:
        return True


requires_fs = np.testing.dec.skipif(not has_freesurfer(),
                                    'Requires FreeSurfer command line tools')


@requires_fsaverage
def test_offscreen():
    """Test offscreen rendering."""
    mlab.options.backend = 'auto'
    brain = Brain(*std_args, offscreen=True)
    # Sometimes the first screenshot is rendered with a different
    # resolution on OS X
    brain.screenshot()
    shot = brain.screenshot()
    assert_array_equal(shot.shape, (800, 800, 3))
    brain.close()


@requires_fsaverage
def test_image():
    """Test image saving."""
    tmp_name = mktemp() + '.png'

    mlab.options.backend = 'auto'
    subject_id, _, surf = std_args
    brain = Brain(subject_id, 'both', surf=surf, size=100)
    brain.add_overlay(overlay_fname, hemi='lh', min=5, max=20, sign="pos")
    brain.save_imageset(tmp_name, ['med', 'lat'], 'jpg')

    brain = Brain(*std_args, size=100)
    brain.save_image(tmp_name)
    brain.save_image(tmp_name, 'rgba', True)
    brain.save_montage(tmp_name, ['l', 'v', 'm'], orientation='v')
    brain.save_montage(tmp_name, ['l', 'v', 'm'], orientation='h')
    brain.save_montage(tmp_name, [['l', 'v'], ['m', 'f']])
    brain.screenshot()
    brain.close()


@requires_fsaverage
def test_brains():
    """Test plotting of Brain with different arguments."""
    # testing backend breaks when passing in a figure, so we use 'auto' here
    # (shouldn't affect usability, but it makes testing more annoying)
    mlab.options.backend = 'auto'
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
    figs = [None, mlab.figure(), None, None, mlab.figure(), None, None]
    subj_dirs = [None, subj_dir, subj_dir, subj_dir,
                 subj_dir, subj_dir, subj_dir]
    alphas = [1.0, 0.5, 0.25, 0.7, 0.5, 0.25, 0.7]
    for surf, hemi, title, cort, s, bg, fg, fig, sd, alpha \
            in zip(surfs, hemis, titles, cortices, sizes,
                   backgrounds, foregrounds, figs, subj_dirs, alphas):
        brain = Brain(subject_id, hemi, surf, title=title, cortex=cort,
                      alpha=alpha, size=s, background=bg, foreground=fg,
                      figure=fig, subjects_dir=sd)
        brain.close()
    assert_raises(ValueError, Brain, subject_id, 'lh', 'inflated',
                  subjects_dir='')


@requires_fsaverage
def test_annot():
    """Test plotting of annot."""
    mlab.options.backend = 'test'
    annots = ['aparc', 'aparc.a2005s']
    borders = [True, False, 2]
    alphas = [1, 0.5]
    brain = Brain(*std_args)
    for a, b, p in zip(annots, borders, alphas):
        brain.add_annotation(a, b, p)
    assert_raises(ValueError, brain.add_annotation, 'aparc', borders=-1)
    brain.close()


@requires_fsaverage
def test_contour():
    """Test plotting of contour overlay."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    overlay_file = pjoin(data_dir, "lh.sig.nii.gz")
    brain.add_contour_overlay(overlay_file)
    brain.add_contour_overlay(overlay_file, max=20, n_contours=9,
                              line_width=2)
    brain.contour['surface'].actor.property.line_width = 1
    brain.contour['surface'].contour.number_of_contours = 10
    brain.close()


@requires_fsaverage
@requires_fs
def test_data():
    """Test plotting of data."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    mri_file = pjoin(data_dir, 'resting_corr.nii.gz')
    reg_file = pjoin(data_dir, 'register.dat')
    surf_data = io.project_volume_data(mri_file, "lh", reg_file)
    brain.add_data(surf_data, -.7, .7, colormap="jet", alpha=.7)
    brain.add_data([], vertices=np.array([], int))
    brain.close()


@requires_fsaverage
def test_foci():
    """Test plotting of foci."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    coords = [[-36, 18, -3],
              [-43, 25, 24],
              [-48, 26, -2]]
    brain.add_foci(coords, map_surface="white", color="gold")

    annot_path = pjoin(subj_dir, subject_id, 'label', 'lh.aparc.a2009s.annot')
    ids, ctab, names = nib.freesurfer.read_annot(annot_path)
    verts = np.arange(0, len(ids))
    coords = np.random.permutation(verts[ids == 74])[:10]
    scale_factor = 0.7
    brain.add_foci(coords, coords_as_verts=True,
                   scale_factor=scale_factor, color="#A52A2A")
    brain.close()


@requires_fsaverage
def test_label():
    """Test plotting of label."""
    mlab.options.backend = 'test'
    subject_id = "fsaverage"
    hemi = "lh"
    surf = "inflated"
    brain = Brain(subject_id, hemi, surf)
    brain.add_label("BA1")
    brain.add_label("BA1", color="blue", scalar_thresh=.5)
    label_file = pjoin(subj_dir, subject_id,
                       "label", "%s.MT.label" % hemi)
    brain.add_label(label_file)
    brain.add_label("BA44", borders=True)
    brain.add_label("BA6", alpha=.7)
    brain.show_view("medial")
    brain.add_label("V1", color="steelblue", alpha=.6)
    brain.add_label("V2", color="#FF6347", alpha=.6)
    brain.add_label("entorhinal", color=(.2, 1, .5), alpha=.6)
    brain.close()


@requires_fsaverage
def test_meg_inverse():
    """Test plotting of MEG inverse solution."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    stc_fname = os.path.join(data_dir, 'meg_source_estimate-lh.stc')
    stc = io.read_stc(stc_fname)
    data = stc['data']
    vertices = stc['vertices']
    time = np.linspace(stc['tmin'], stc['tmin'] + data.shape[1] * stc['tstep'],
                       data.shape[1], endpoint=False)
    colormap = 'hot'

    def time_label(t):
        return 'time=%0.2f ms' % (1e3 * t)

    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=10, time=time, time_label=time_label)
    brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)
    assert_equal(brain.data_dict['lh']['time_idx'], 0)

    brain.set_time(.1)
    assert_equal(brain.data_dict['lh']['time_idx'], 2)
    # viewer = TimeViewer(brain)

    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=10, time=time, time_label=time_label,
                   initial_time=.09, remove_existing=True)
    assert_equal(brain.data_dict['lh']['time_idx'], 1)
    brain.close()


@requires_fsaverage
def test_morphometry():
    """Test plotting of morphometry."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    brain.add_morphometry("curv")
    brain.add_morphometry("sulc", grayscale=True)
    brain.add_morphometry("thickness")
    brain.close()


@requires_imageio
@requires_fsaverage
def test_movie():
    """Test saving a movie of an MEG inverse solution."""
    import imageio

    # create and setup the Brain instance
    mlab.options.backend = 'auto'
    brain = Brain(*std_args)
    stc_fname = os.path.join(data_dir, 'meg_source_estimate-lh.stc')
    stc = io.read_stc(stc_fname)
    data = stc['data']
    time = np.arange(data.shape[1]) * stc['tstep'] + stc['tmin']
    brain.add_data(data, colormap='hot', vertices=stc['vertices'],
                   smoothing_steps=10, time=time, time_label='time=%0.2f ms')
    brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)

    # save movies with different options
    tempdir = mkdtemp()
    try:
        dst = os.path.join(tempdir, 'test.mov')
        # test the number of frames in the movie
        brain.save_movie(dst)
        frames = imageio.mimread(dst)
        assert_equal(len(frames), 2)
        brain.save_movie(dst, time_dilation=10)
        frames = imageio.mimread(dst)
        assert_equal(len(frames), 7)
        brain.save_movie(dst, tmin=0.081, tmax=0.102)
        frames = imageio.mimread(dst)
        assert_equal(len(frames), 2)
    finally:
        # clean up
        shutil.rmtree(tempdir)
    brain.close()


@requires_fsaverage
def test_overlay():
    """Test plotting of overlay."""
    mlab.options.backend = 'test'
    # basic overlay support
    overlay_file = pjoin(data_dir, "lh.sig.nii.gz")
    brain = Brain(*std_args)
    brain.add_overlay(overlay_file)
    brain.overlays["sig"].remove()
    brain.add_overlay(overlay_file, min=5, max=20, sign="pos")
    sig1 = io.read_scalar_data(pjoin(data_dir, "lh.sig.nii.gz"))
    sig2 = io.read_scalar_data(pjoin(data_dir, "lh.alt_sig.nii.gz"))

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
    brain.close()


@requires_fsaverage
def test_probabilistic_labels():
    """Test plotting of probabilistic labels."""
    mlab.options.backend = 'test'
    brain = Brain("fsaverage", "lh", "inflated",
                  cortex="low_contrast")

    brain.add_label("BA1", color="darkblue")

    brain.add_label("BA1", color="dodgerblue", scalar_thresh=.5)

    brain.add_label("BA45", color="firebrick", borders=True)
    brain.add_label("BA45", color="salmon", borders=True, scalar_thresh=.5)

    label_file = pjoin(subj_dir, "fsaverage", "label", "lh.BA6.label")
    prob_field = np.zeros_like(brain._geo.x)
    ids, probs = nib.freesurfer.read_label(label_file, read_scalars=True)
    prob_field[ids] = probs
    brain.add_data(prob_field, thresh=1e-5)

    brain.data["colorbar"].number_of_colors = 10
    brain.data["colorbar"].number_of_labels = 11
    brain.close()


@requires_fsaverage
def test_text():
    """Test plotting of text."""
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    brain.add_text(0.1, 0.1, 'Hello', 'blah')
    brain.close()


@requires_fsaverage
def test_animate():
    """Test animation."""
    mlab.options.backend = 'auto'
    brain = Brain(*std_args, size=100)
    brain.add_morphometry('curv')
    tmp_name = mktemp() + '.avi'
    brain.animate(["m"] * 3, n_steps=2)
    brain.animate(['l', 'l'], n_steps=2, fname=tmp_name)
    # can't rotate in axial plane
    assert_raises(ValueError, brain.animate, ['l', 'd'])
    brain.close()


@requires_fsaverage
def test_views():
    """Test showing different views."""
    mlab.options.backend = 'test'
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
