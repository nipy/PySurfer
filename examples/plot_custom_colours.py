"""
=================================
Plot RGBA values on brain surface
=================================

"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from copy import deepcopy
from tvtk.api import tvtk
from tvtk.common import configure_input_data
from surfer.utils import smoothing_matrix, mesh_edges
from surfer import Brain

print(__doc__)

###############################################################################
# define functions

def norm(x, lims=[1, 99]):
    x = np.asarray(x, dtype=float)
    m, M = np.percentile(x, lims)
    x -= m
    x /= M-m
    x[x < 0] = 0
    x[x > 1] = 1
    return x


def plot_rgba_meshbrain(rgba_vals, verts, hemi='lh', surf='pial',
                        **kwargs):

    # where to load up surfaces from
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'

    # make colours copy
    colors = deepcopy(rgba_vals)
    n_verts = colors.shape[0]

    # load surface
    rr, tris = mne.read_surface('%s/fsaverage/surf/%s.%s' % (subjects_dir,
                                                             hemi, surf))
    tris = tris.astype(np.uint32)
    x, y, z = rr.T

    # interpolate values from ico4 -> max
    adj_mat = mesh_edges(tris)
    smooth_mat = smoothing_matrix(verts[0], adj_mat,
                                  20, verbose=False)

    colors = smooth_mat.dot(colors[:n_verts])

    # init figure
    fig = mlab.figure()
    b = Brain('fsaverage', hemi, surf, subjects_dir=subjects_dir,
              background='white',
              figure=fig, **kwargs)

    # plot points in x,y,z
    mesh = mlab.pipeline.triangular_mesh_source(
        x, y, z, tris, figure=fig, **kwargs)
    mesh.data.point_data.scalars.number_of_components = 4  # r, g, b, a
    mesh.data.point_data.scalars = (colors * 255).astype('ubyte')

    # tvtk for vis
    mapper = tvtk.PolyDataMapper()
    configure_input_data(mapper, mesh.data)
    actor = tvtk.Actor()
    actor.mapper = mapper
    fig.scene.add_actor(actor)
    return b

###############################################################################
# generate an rgba matrix, of shape n_vertices x 4

# define color map
cmap = plt.cm.rainbow

# load data from mne sample dataset
stc_dir = (mne.datasets.sample.data_path() +
           '/MEG/sample/fsaverage_audvis-meg-lh.stc')
stc = mne.read_source_estimate(stc_dir)
data = stc.lh_data

# extract latency idx of the peak in each source for *hue*
hue = map(np.argmax, data)

# extract actual value in the peak of each source for *alpha*
alpha = norm([data[ii, w] for ii, w in enumerate(hue)])

# normalise hue and alpha to be between 0-1
alpha = norm(alpha, [1, 80])
hue = norm(hue, [1, 80])

# map the colormap to the vertex latency values
colors = cmap(hue)[:, :3]

# combine hue and alpha into a Nx4 matrix
rgba_vals = np.concatenate((colors, alpha[:, None]), axis=1)

###############################################################################
# plot the rgba values on a PySurfer brain

fig = plot_rgba_meshbrain(rgba_vals, stc.vertices)

