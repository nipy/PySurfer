import numpy as np
import os
from os.path import join as pjoin
from numpy.testing import assert_raises
from tempfile import mktemp
import nibabel as nib

from surfer import Brain
from surfer import io
try:
    from mayavi import mlab
    assert mlab
except ImportError:
    from enthought.mayavi import mlab

if 'SUBJECTS_DIR' not in os.environ:
    raise ValueError('Test suite relies on the definition of SUBJECTS_DIR')

subj_dir = os.environ["SUBJECTS_DIR"]
subject_id = 'fsaverage'
std_args = [subject_id, 'lh', 'inflated']
data_dir = pjoin(os.path.split(__file__)[0], '..', '..',
                 'examples', 'example_data')
small_brain = dict(size=100)


def has_freesurfer():
    if 'FREESURFER_HOME' not in os.environ:
        return False
    else:
        return True

requires_fs = np.testing.dec.skipif(not has_freesurfer(),
                                    'Requires FreeSurfer command line tools')


def test_image():
    """Test image saving
    """
    mlab.options.backend = 'auto'
    brain = Brain(*std_args, config_opts=small_brain)
    tmp_name = mktemp() + '.png'
    brain.save_image(tmp_name)
    brain.save_imageset(tmp_name, ['med', 'lat'], 'jpg')
    brain.save_montage(tmp_name, ['l', 'v', 'm'], orientation='v')
    brain.screenshot()
    brain.close()


def test_brains():
    """Test plotting of Brain with different arguments
    """
    # testing backend breaks when passing in a figure, so we use 'auto' here
    # (shouldn't affect usability, but it makes testing more annoying)
    mlab.options.backend = 'auto'
    surfs = ['inflated', 'sphere']
    hemis = ['lh', 'rh']
    curvs = [True, False]
    titles = [None, 'Hello']
    config_opts = [{}, dict(size=(800, 800))]
    figs = [None, mlab.figure()]
    subj_dirs = [None, subj_dir]
    for surf, hemi, curv, title, co, fig, sd \
            in zip(surfs, hemis, curvs, titles, config_opts, figs, subj_dirs):
        print 'hello'
        brain = Brain(subject_id, hemi, surf, curv, title, co, fig, sd)
        brain.close()
    assert_raises(ValueError, Brain, subject_id, 'lh', 'inflated',
                  subjects_dir='')


def test_annot():
    """Test plotting of annot
    """
    mlab.options.backend = 'test'
    annots = ['aparc', 'aparc.a2005s']
    borders = [True, False]
    alphas = [1, 0.5]
    brain = Brain(*std_args)
    for a, b, p in zip(annots, borders, alphas):
        brain.add_annotation(a, b, p)
    brain.close()


def test_contour():
    """Test plotting of contour overlay
    """
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    overlay_file = pjoin(data_dir, "lh.sig.nii.gz")
    brain.add_contour_overlay(overlay_file)
    brain.add_contour_overlay(overlay_file, max=20, n_contours=9,
                              line_width=2)
    brain.contour['surface'].actor.property.line_width = 1
    brain.contour['surface'].contour.number_of_contours = 10
    brain.close()


@requires_fs
def test_data():
    """Test plotting of data
    """
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    mri_file = pjoin(data_dir, 'resting_corr.nii.gz')
    reg_file = pjoin(data_dir, 'register.dat')
    surf_data = io.project_volume_data(mri_file, "lh", reg_file)
    brain.add_data(surf_data, -.7, .7, colormap="jet", alpha=.7)
    brain.close()


def test_foci():
    """Test plotting of foci
    """
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


def test_label():
    """Test plotting of label
    """
    mlab.options.backend = 'test'
    subject_id = "fsaverage"
    hemi = "lh"
    surf = "smoothwm"
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


def test_meg_inverse():
    """Test plotting of MEG inverse solution
    """
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    stc_fname = os.path.join(data_dir, 'meg_source_estimate-lh.stc')
    stc = io.read_stc(stc_fname)
    data = stc['data']
    vertices = stc['vertices']
    time = 1e3 * np.linspace(stc['tmin'],
                             stc['tmin'] + data.shape[1] * stc['tstep'],
                             data.shape[1])
    colormap = 'hot'
    time_label = 'time=%0.2f ms'
    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=10, time=time, time_label=time_label)
    brain.set_data_time_index(2)
    brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)
    # viewer = TimeViewer(brain)
    brain.close()


def test_morphometry():
    """Test plotting of morphometry
    """
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    brain.add_morphometry("curv")
    brain.add_morphometry("sulc", grayscale=True)
    brain.add_morphometry("thickness")
    brain.close()


def test_overlay():
    """Test plotting of overlay
    """
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


def test_probabilistic_labels():
    """Test plotting of probabilistic labels
    """
    mlab.options.backend = 'test'
    brain = Brain("fsaverage", "lh", "inflated",
                  config_opts=dict(cortex="low_contrast"))

    brain.add_label("BA1", color="darkblue")

    brain.add_label("BA1", color="dodgerblue", scalar_thresh=.5)

    brain.add_label("BA45", color="firebrick", borders=True)
    brain.add_label("BA45", color="salmon", borders=True, scalar_thresh=.5)

    label_file = pjoin(subj_dir, "fsaverage", "label", "lh.BA6.label")
    prob_field = np.zeros_like(brain._geo.x)
    ids, probs = io.read_label(label_file, read_scalars=True)
    prob_field[ids] = probs
    brain.add_data(prob_field, thresh=1e-5)

    brain.data["colorbar"].number_of_colors = 10
    brain.data["colorbar"].number_of_labels = 11
    brain.close()


def test_text():
    """Test plotting of text
    """
    mlab.options.backend = 'test'
    brain = Brain(*std_args)
    brain.add_text(0.1, 0.1, 'Hello', 'blah')
    brain.close()


def test_animate():
    """Test animation
    """
    mlab.options.backend = 'auto'
    brain = Brain(*std_args, config_opts=small_brain)
    brain.add_morphometry('curv')
    tmp_name = mktemp() + '.avi'
    brain.animate(["m"] * 3, n_steps=2)
    brain.animate(['l', 'l'], n_steps=2, fname=tmp_name)
    # can't rotate in axial plane
    assert_raises(ValueError, brain.animate, ['l', 'd'])
    brain.close()


def test_views():
    """Test showing different views
    """
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
