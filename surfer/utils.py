try:
    from collections.abc import Sequence
except ImportError:  # Py 2.7
    from collections import Sequence
from distutils.version import LooseVersion
import logging
import warnings
import sys
import os
from os import path as op
import inspect
from functools import wraps

import mayavi
from mayavi import mlab
from mayavi.filters.api import Threshold
import numpy as np
import nibabel as nib
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib as mpl
from matplotlib import cm as mpl_cm
from . import cm as surfer_cm

logger = logging.getLogger('surfer')


# Py3k compat
if sys.version[0] == '2':
    string_types = basestring  # noqa, analysis:ignore
else:
    string_types = str


if LooseVersion(mayavi.__version__) == LooseVersion('4.5.0'):
    # Monkey-patch Mayavi 4.5:
    # In Mayavi 4.5, filters seem to be missing a .point_data attribute that
    # Threshold accesses on initialization.
    _orig_meth = Threshold._get_data_range

    def _patch_func():
        return []

    def _patch_meth(self):
        return []

    class _MayaviThresholdPatch(object):

        def __enter__(self):
            Threshold._get_data_range = _patch_meth

        def __exit__(self, exc_type, exc_val, exc_tb):
            Threshold._get_data_range = _orig_meth

    _mayavi_threshold_patch = _MayaviThresholdPatch()

    def threshold_filter(*args, **kwargs):
        with _mayavi_threshold_patch:
            thresh = mlab.pipeline.threshold(*args, **kwargs)
        thresh._get_data_range = _patch_func
        return thresh
else:
    threshold_filter = mlab.pipeline.threshold


class Surface(object):
    """Container for surface object

    Attributes
    ----------
    subject_id : string
        Name of subject
    hemi : {'lh', 'rh'}
        Which hemisphere to load
    surf : string
        Name of the surface to load (eg. inflated, orig ...)
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    offset : float | None
        If float, align inside edge of each hemisphere to center + offset.
        If None, do not change coordinates (default).
    units : str
        Can be 'm' or 'mm' (default).
    """

    def __init__(self, subject_id, hemi, surf, subjects_dir=None,
                 offset=None, units='mm'):
        """Surface

        Parameters
        ----------
        subject_id : string
            Name of subject
        hemi : {'lh', 'rh'}
            Which hemisphere to load
        surf : string
            Name of the surface to load (eg. inflated, orig ...)
        offset : float | None
            If 0.0, the surface will be offset such that the medial
            wall is aligned with the origin. If None, no offset will
            be applied. If != 0.0, an additional offset will be used.
        """
        if hemi not in ['lh', 'rh']:
            raise ValueError('hemi must be "lh" or "rh')
        self.subject_id = subject_id
        self.hemi = hemi
        self.surf = surf
        self.offset = offset
        self.coords = None
        self.faces = None
        self.nn = None
        self.units = _check_units(units)

        subjects_dir = _get_subjects_dir(subjects_dir)
        self.data_path = op.join(subjects_dir, subject_id)

    def load_geometry(self):
        surf_path = op.join(self.data_path, "surf",
                            "%s.%s" % (self.hemi, self.surf))
        coords, faces = nib.freesurfer.read_geometry(surf_path)
        if self.units == 'm':
            coords /= 1000.
        if self.offset is not None:
            if self.hemi == 'lh':
                coords[:, 0] -= (np.max(coords[:, 0]) + self.offset)
            else:
                coords[:, 0] -= (np.min(coords[:, 0]) + self.offset)
        nn = _compute_normals(coords, faces)

        if self.coords is None:
            self.coords = coords
            self.faces = faces
            self.nn = nn
        else:
            self.coords[:] = coords
            self.faces[:] = faces
            self.nn[:] = nn

    @property
    def x(self):
        return self.coords[:, 0]

    @property
    def y(self):
        return self.coords[:, 1]

    @property
    def z(self):
        return self.coords[:, 2]

    def load_curvature(self):
        """Load in curvature values from the ?h.curv file."""
        curv_path = op.join(self.data_path, "surf", "%s.curv" % self.hemi)
        self.curv = nib.freesurfer.read_morph_data(curv_path)
        self.bin_curv = np.array(self.curv > 0, int)

    def load_label(self, name):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.

        """
        label = nib.freesurfer.read_label(op.join(self.data_path, 'label',
                                          '%s.%s.label' % (self.hemi, name)))
        label_array = np.zeros(len(self.x), int)
        label_array[label] = 1
        try:
            self.labels[name] = label_array
        except AttributeError:
            self.labels = {name: label_array}

    def apply_xfm(self, mtx):
        """Apply an affine transformation matrix to the x,y,z vectors."""
        self.coords = np.dot(np.c_[self.coords, np.ones(len(self.coords))],
                             mtx.T)[:, :3]


def _fast_cross_3d(x, y):
    """Compute cross product between list of 3D vectors

    Much faster than np.cross() when the number of cross products
    becomes large (>500). This is because np.cross() methods become
    less memory efficient at this stage.

    Parameters
    ----------
    x : array
        Input array 1.
    y : array
        Input array 2.

    Returns
    -------
    z : array
        Cross product of x and y.

    Notes
    -----
    x and y must both be 2D row vectors. One must have length 1, or both
    lengths must match.
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == 3
    assert y.shape[1] == 3
    assert (x.shape[0] == 1 or y.shape[0] == 1) or x.shape[0] == y.shape[0]
    if max([x.shape[0], y.shape[0]]) >= 500:
        return np.c_[x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1],
                     x[:, 2] * y[:, 0] - x[:, 0] * y[:, 2],
                     x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]]
    else:
        return np.cross(x, y)


def _compute_normals(rr, tris):
    """Efficiently compute vertex normals for triangulated surface"""
    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = _fast_cross_3d((r2 - r1), (r3 - r1))

    #   Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    zidx = np.where(size == 0)[0]
    size[zidx] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    npts = len(rr)

    # the following code replaces this, but is faster (vectorized):
    #
    # for p, verts in enumerate(tris):
    #     nn[verts, :] += tri_nn[p, :]
    #
    nn = np.zeros((npts, 3))
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(verts, tri_nn[:, idx], minlength=npts)
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]
    return nn


###############################################################################
# LOGGING (courtesy of mne-python)

def set_log_level(verbose=None, return_old_level=False):
    """Convenience function for setting the logging level

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable MNE_LOG_LEVEL is read, and if
        it doesn't exist, defaults to INFO.
    return_old_level : bool
        If True, return the old verbosity level.
    """
    if verbose is None:
        verbose = "INFO"
    elif isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, string_types):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)


class WrapStdOut(object):
    """Ridiculous class to work around how doctest captures stdout"""
    def __getattr__(self, name):
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        return getattr(sys.stdout, name)


def set_log_file(fname=None, output_format='%(message)s', overwrite=None):
    """Convenience function for setting the log to print to a file

    Parameters
    ----------
    fname : str, or None
        Filename of the log to print to. If None, stdout is used.
        To suppress log outputs, use set_log_level('WARN').
    output_format : str
        Format of the output messages. See the following for examples:
            http://docs.python.org/dev/howto/logging.html
        e.g., "%(asctime)s - %(levelname)s - %(message)s".
    overwrite : bool, or None
        Overwrite the log file (if it exists). Otherwise, statements
        will be appended to the log (default). None is the same as False,
        but additionally raises a warning to notify the user that log
        entries will be appended.
    """
    handlers = logger.handlers
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
        logger.removeHandler(h)
    if fname is not None:
        if op.isfile(fname) and overwrite is None:
            warnings.warn('Log entries will be appended to the file. Use '
                          'overwrite=False to avoid this message in the '
                          'future.')
        mode = 'w' if overwrite is True else 'a'
        lh = logging.FileHandler(fname, mode=mode)
    else:
        """ we should just be able to do:
                lh = logging.StreamHandler(sys.stdout)
            but because doctests uses some magic on stdout, we have to do this:
        """
        lh = logging.StreamHandler(WrapStdOut())

    lh.setFormatter(logging.Formatter(output_format))
    # actually add the stream handler
    logger.addHandler(lh)


if hasattr(inspect, 'signature'):  # py35
    def _get_args(function, varargs=False):
        params = inspect.signature(function).parameters
        args = [key for key, param in params.items()
                if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
        if varargs:
            varargs = [param.name for param in params.values()
                       if param.kind == param.VAR_POSITIONAL]
            if len(varargs) == 0:
                varargs = None
            return args, varargs
        else:
            return args
else:
    def _get_args(function, varargs=False):
        out = inspect.getargspec(function)  # args, varargs, keywords, defaults
        if varargs:
            return out[:2]
        else:
            return out[0]


def verbose(function):
    """Decorator to allow functions to override default log level

    Do not call this function directly to set the global verbosity level,
    instead use set_log_level().

    Parameters (to decorated function)
    ----------------------------------
    verbose : bool, str, int, or None
        The level of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        None defaults to using the current log level [e.g., set using
        mne.set_log_level()].
    """
    arg_names = _get_args(function)
    # this wrap allows decorated functions to be pickled (e.g., for parallel)

    @wraps(function)
    def dec(*args, **kwargs):
        # Check if the first arg is "self", if it has verbose, make it default
        if len(arg_names) > 0 and arg_names[0] == 'self':
            default_level = getattr(args[0], 'verbose', None)
        else:
            default_level = None
        verbose_level = kwargs.get('verbose', default_level)
        if verbose_level is not None:
            old_level = set_log_level(verbose_level, True)
            # set it back if we get an exception
            try:
                ret = function(*args, **kwargs)
            except Exception:
                set_log_level(old_level)
                raise
            set_log_level(old_level)
            return ret
        else:
            return function(*args, **kwargs)

    # set __wrapped__ attribute so ?? in IPython gets the right source
    dec.__wrapped__ = function

    return dec


###############################################################################
# USEFUL FUNCTIONS

def _check_units(units):
    if units not in ('m', 'mm'):
        raise ValueError('Units must be "m" or "mm", got %r' % (units,))
    return units


def find_closest_vertices(surface_coords, point_coords):
    """Return the vertices on a surface mesh closest to some given coordinates.

    The distance metric used is Euclidian distance.

    Parameters
    ----------
    surface_coords : numpy array
        Array of coordinates on a surface mesh
    point_coords : numpy array
        Array of coordinates to map to vertices

    Returns
    -------
    closest_vertices : numpy array
        Array of mesh vertex ids

    """
    point_coords = np.atleast_2d(point_coords)
    return np.argmin(cdist(surface_coords, point_coords), axis=0)


def tal_to_mni(coords, units='mm'):
    """Convert Talairach coords to MNI using the Lancaster transform.

    Parameters
    ----------
    coords : n x 3 numpy array
        Array of Talairach coordinates
    units : str
        Can be 'm' or 'mm' (default).

    Returns
    -------
    mni_coords : n x 3 numpy array
        Array of coordinates converted to MNI space.
    """
    coords = np.atleast_2d(coords)
    xfm = np.array([[1.06860, -0.00396, 0.00826,  1.07816],
                    [0.00640,  1.05741, 0.08566,  1.16824],
                    [-0.01281, -0.08863, 1.10792, -4.17805],
                    [0.00000,  0.00000, 0.00000,  1.00000]])
    units = _check_units(units)
    if units == 'm':
        xfm[:3, 3] /= 1000.
    mni_coords = np.dot(np.c_[coords, np.ones(coords.shape[0])], xfm.T)[:, :3]
    return mni_coords


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


def create_color_lut(cmap, n_colors=256, center=None):
    """Return a colormap suitable for setting as a Mayavi LUT.

    Parameters
    ----------
    cmap : string, list of colors, n x 3 or n x 4 array
        Input colormap definition. This can be the name of a matplotlib
        colormap, a list of valid matplotlib colors, or a suitable
        mayavi LUT (possibly missing the alpha channel).

        if value is "auto", a default sequential or divergent colormap is
        returned
    n_colors : int, optional
        Number of colors in the resulting LUT. This is ignored if cmap
        is a 2d array.
    center : double, optional
        indicates whether desired colormap should be for divergent values,
        currently only used to select default colormap for cmap='auto'

    Returns
    -------
    lut : n_colors x 4 integer array
        Color LUT suitable for passing to mayavi
    """
    if isinstance(cmap, np.ndarray):
        if np.ndim(cmap) == 2:
            if cmap.shape[1] == 4:
                # This looks likes a LUT that's ready to go
                lut = cmap.astype(int)
            elif cmap.shape[1] == 3:
                # This looks like a LUT, but it's missing the alpha channel
                alpha = np.ones(len(cmap), int) * 255
                lut = np.c_[cmap, alpha]

            return lut

    # choose default colormaps (REMEMBER to change doc, e.g., in
    # Brain.add_data, when changing these defaults)
    if isinstance(cmap, string_types) and cmap == "auto":
        if center is None:
            cmap = "rocket"
        else:
            cmap = "icefire"

    surfer_cmaps = ["rocket", "mako", "icefire", "vlag"]
    surfer_cmaps += [name + "_r" for name in surfer_cmaps]

    if not isinstance(cmap, string_types) and isinstance(cmap, Sequence):
        colors = list(map(mpl.colors.colorConverter.to_rgba, cmap))
        cmap = mpl.colors.ListedColormap(colors)
    elif cmap in surfer_cmaps:
        cmap = getattr(surfer_cm, cmap)
    else:
        try:
            # Try to get a named matplotlib colormap
            # This will also pass Colormap object back out
            cmap = mpl_cm.get_cmap(cmap)
        except (TypeError, ValueError):
            # If we get here, it's a bad input
            # but don't raise the matplotlib error as it is less accurate
            raise ValueError("Input %r was not valid for making a lut" % cmap)

    # Convert from a matplotlib colormap to a lut array
    lut = (cmap(np.linspace(0, 1, n_colors)) * 255).astype(int)

    return lut


@verbose
def smoothing_matrix(vertices, adj_mat, smoothing_steps=20, verbose=None):
    """Create a smoothing matrix which can be used to interpolate data defined
       for a subset of vertices onto mesh with an adjancency matrix given by
       adj_mat.

       If smoothing_steps is None, as many smoothing steps are applied until
       the whole mesh is filled with with non-zeros. Only use this option if
       the vertices correspond to a subsampled version of the mesh.

    Parameters
    ----------
    vertices : 1d array
        vertex indices
    adj_mat : sparse matrix
        N x N adjacency matrix of the full mesh
    smoothing_steps : int or None
        number of smoothing steps (Default: 20)
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).

    Returns
    -------
    smooth_mat : sparse matrix
        smoothing matrix with size N x len(vertices)
    """
    if smoothing_steps == 'nearest':
        mat = _nearest(vertices, adj_mat)
    else:
        mat = _smooth(vertices, adj_mat, smoothing_steps)
    return mat


def _nearest(vertices, adj_mat):
    import scipy
    from scipy.sparse.csgraph import dijkstra
    if LooseVersion(scipy.__version__) < LooseVersion('1.3'):
        raise RuntimeError('smoothing_steps="nearest" requires SciPy >= 1.3')
    # Vertices can be out of order, so sort them to start ...
    order = np.argsort(vertices)
    vertices = vertices[order]
    _, _, sources = dijkstra(adj_mat, False, indices=vertices, min_only=True,
                             return_predecessors=True)
    col = np.searchsorted(vertices, sources)
    # ... then get things back to the correct configuration.
    col = order[col]
    row = np.arange(len(col))
    data = np.ones(len(col))
    mat = sparse.coo_matrix((data, (row, col)))
    assert mat.shape == (adj_mat.shape[0], len(vertices)), mat.shape
    return mat


def _smooth(vertices, adj_mat, smoothing_steps):
    from scipy import sparse
    logger.debug("Updating smoothing matrix, be patient..")
    e = adj_mat.copy()
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    e = e + sparse.eye(n_vertices, n_vertices)
    idx_use = vertices
    smooth_mat = 1.0
    n_iter = smoothing_steps if smoothing_steps is not None else 1000
    for k in range(n_iter):
        e_use = e[:, idx_use]

        data1 = e_use * np.ones(len(idx_use))
        idx_use = np.where(data1)[0]
        scale_mat = sparse.dia_matrix((1 / data1[idx_use], 0),
                                      shape=(len(idx_use), len(idx_use)))

        smooth_mat = scale_mat * e_use[idx_use, :] * smooth_mat

        logger.debug("Smoothing matrix creation, step %d" % (k + 1))
        if smoothing_steps is None and len(idx_use) >= n_vertices:
            break

    # Make sure the smoothing matrix has the right number of rows
    # and is in COO format
    smooth_mat = smooth_mat.tocoo()
    smooth_mat = sparse.coo_matrix((smooth_mat.data,
                                    (idx_use[smooth_mat.row],
                                     smooth_mat.col)),
                                   shape=(n_vertices,
                                          len(vertices)))

    return smooth_mat


@verbose
def coord_to_label(subject_id, coord, label, hemi='lh', n_steps=30,
                   map_surface='white', coord_as_vert=False, units='mm',
                   verbose=None):
    """Create label from MNI coordinate

    Parameters
    ----------
    subject_id : string
        Use if file is in register with subject's orig.mgz
    coord : numpy array of size 3 | int
        One coordinate in MNI space or the vertex index.
    label : str
        Label name
    hemi : [lh, rh]
        Hemisphere target
    n_steps : int
        Number of dilation iterations
    map_surface : str
        The surface name used to find the closest point
    coord_as_vert : bool
        whether the coords parameter should be interpreted as vertex ids
    units : str
        Can be 'm' or 'mm' (default).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).
    """
    geo = Surface(subject_id, hemi, map_surface, units=units)
    geo.load_geometry()

    coords = geo.coords
    # work in mm from here on
    if geo.units == 'm':
        coords = coords * 1000
    if coord_as_vert:
        coord = coords[coord]

    n_vertices = len(coords)
    adj_mat = mesh_edges(geo.faces)
    foci_vtxs = find_closest_vertices(coords, [coord])
    data = np.zeros(n_vertices)
    data[foci_vtxs] = 1.
    smooth_mat = smoothing_matrix(np.arange(n_vertices), adj_mat, 1)
    for _ in range(n_steps):
        data = smooth_mat * data
    idx = np.where(data.ravel() > 0)[0]
    # Write label
    label_fname = label + '-' + hemi + '.label'
    logger.debug("Saving label : %s" % label_fname)
    f = open(label_fname, 'w')
    f.write('#label at %s from subject %s\n' % (coord, subject_id))
    f.write('%d\n' % len(idx))
    for i in idx:
        x, y, z = coords[i]
        f.write('%d  %f  %f  %f 0.000000\n' % (i, x, y, z))


def _get_subjects_dir(subjects_dir=None, raise_error=True):
    """Get the subjects directory from parameter or environment variable

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory.
    raise_error : bool
        If True, raise a ValueError if no value for SUBJECTS_DIR can be found
        or the corresponding directory does not exist.

    Returns
    -------
    subjects_dir : str
        The subjects directory. If the subjects_dir input parameter is not
        None, its value will be returned, otherwise it will be obtained from
        the SUBJECTS_DIR environment variable.
    """
    if subjects_dir is None:
        subjects_dir = os.environ.get("SUBJECTS_DIR", "")
        if not subjects_dir and raise_error:
            raise ValueError('The subjects directory has to be specified '
                             'using the subjects_dir parameter or the '
                             'SUBJECTS_DIR environment variable.')

    if raise_error and not os.path.exists(subjects_dir):
        raise ValueError('The subjects directory %s does not exist.'
                         % subjects_dir)

    return subjects_dir


def has_fsaverage(subjects_dir=None, raise_error=True, return_why=False):
    """Determine whether the user has a usable fsaverage"""
    subjects_dir = _get_subjects_dir(subjects_dir, raise_error=raise_error)
    out = ''
    if not op.isdir(subjects_dir):
        out = 'SUBJECTS_DIR not found: %s' % (subjects_dir,)
    else:
        fs_dir = op.join(_get_subjects_dir(subjects_dir, False), 'fsaverage')
        surf_dir = op.join(fs_dir, 'surf')
        if not op.isdir(fs_dir):
            out = 'fsaverage not found in SUBJECTS_DIR: %s' % (fs_dir,)
        elif not op.isdir(surf_dir):
            out = 'fsaverage has no "surf" directory: %s' % (surf_dir,)
    out = (out == '', out) if return_why else (out == '')
    return out


def requires_fsaverage():
    import pytest
    has, why = has_fsaverage(raise_error=False, return_why=True)
    return pytest.mark.skipif(
        not has, reason='Requires fsaverage subject data (%s)' % why)


def requires_imageio():
    import pytest
    try:
        from imageio.plugins.ffmpeg import get_exe  # noqa, analysis:ignore
    except ImportError:
        has = False
    else:
        has = True
    return pytest.mark.skipif(not has, reason="Requires imageio with ffmpeg")


def requires_fs():
    import pytest
    has = ('FREESURFER_HOME' in os.environ)
    return pytest.mark.skipif(
        not has, reason='Requires FreeSurfer command line tools')


def _get_extra():
    # Get extra label for newer freesurfer
    subj_dir = _get_subjects_dir()
    fname = op.join(subj_dir, 'fsaverage', 'label', 'lh.BA1.label')
    return '_exvivo' if not op.isfile(fname) else '', subj_dir
