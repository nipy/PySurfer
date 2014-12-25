import logging
import warnings
import sys
import os
from os import path as op
import inspect
from functools import wraps
import subprocess

import numpy as np
import nibabel as nib
from scipy import sparse
from scipy.spatial.distance import cdist
import matplotlib as mpl
from matplotlib import cm

from .config import config

logger = logging.getLogger('surfer')


# Py3k compat
if sys.version[0] == '2':
    string_types = basestring  # noqa
else:
    string_types = str


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
    data_path : string
        Path where to look for data
    x: 1d array
        x coordinates of vertices
    y: 1d array
        y coordinates of vertices
    z: 1d array
        z coordinates of vertices
    coords : 2d array of shape [n_vertices, 3]
        The vertices coordinates
    faces : 2d array
        The faces ie. the triangles
    nn : 2d array
        Normalized surface normals for vertices.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment variable.
    """

    def __init__(self, subject_id, hemi, surf, subjects_dir=None,
                 offset=None):
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

        subjects_dir = _get_subjects_dir(subjects_dir)
        self.data_path = op.join(subjects_dir, subject_id)

    def load_geometry(self):
        surf_path = op.join(self.data_path, "surf",
                            "%s.%s" % (self.hemi, self.surf))
        self.coords, self.faces = nib.freesurfer.read_geometry(surf_path)
        if self.offset is not None:
            if self.hemi == 'lh':
                self.coords[:, 0] -= (np.max(self.coords[:, 0]) + self.offset)
            else:
                self.coords[:, 0] -= (np.min(self.coords[:, 0]) + self.offset)
        self.nn = _compute_normals(self.coords, self.faces)

    def save_geometry(self):
        surf_path = op.join(self.data_path, "surf",
                            "%s.%s" % (self.hemi, self.surf))
        nib.freesurfer.write_geometry(surf_path, self.coords, self.faces)

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
        self.bin_curv = np.array(self.curv > 0, np.int)

    def load_label(self, name):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.

        """
        label = nib.freesurfer.read_label(op.join(self.data_path, 'label',
                                          '%s.%s.label' % (self.hemi, name)))
        label_array = np.zeros(len(self.x), np.int)
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
        verbose = config.get('options', 'logging_level')
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
    arg_names = inspect.getargspec(function).args
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
            except:
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


def tal_to_mni(coords):
    """Convert Talairach coords to MNI using the Lancaster transform.

    Parameters
    ----------
    coords : n x 3 numpy array
        Array of Talairach coordinates

    Returns
    -------
    mni_coords : n x 3 numpy array
        Array of coordinates converted to MNI space

    """
    coords = np.atleast_2d(coords)
    xfm = np.array([[1.06860, -0.00396, 0.00826,  1.07816],
                    [0.00640,  1.05741, 0.08566,  1.16824],
                    [-0.01281, -0.08863, 1.10792, -4.17805],
                    [0.00000,  0.00000, 0.00000,  1.00000]])
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


def create_color_lut(cmap, n_colors=256):
    """Return a colormap suitable for setting as a Mayavi LUT.

    Parameters
    ----------
    cmap : string, list of colors, n x 3 or n x 4 array
        Input colormap definition. This can be the name of a matplotlib
        colormap, a list of valid matplotlib colors, or a suitable
        mayavi LUT (possibly missing the alpha channel).
    n_colors : int, optional
        Number of colors in the resulting LUT. This is ignored if cmap
        is a 2d array.
    Returns
    -------
    lut : n_colors x 4 integer array
        Color LUT suitable for passing to mayavi
    """
    if isinstance(cmap, np.ndarray):
        if np.ndim(cmap) == 2:
            if cmap.shape[1] == 4:
                # This looks likes a LUT that's ready to go
                lut = cmap.astype(np.int)
            elif cmap.shape[1] == 3:
                # This looks like a LUT, but it's missing the alpha channel
                alpha = np.ones(len(cmap), np.int) * 255
                lut = np.c_[cmap, alpha]

            return lut

    # Otherwise, we're going to try and use matplotlib to create it

    if cmap in dir(cm):
        # This is probably a matplotlib colormap, so build from that
        # The matplotlib colormaps are a superset of the mayavi colormaps
        # except for one or two cases (i.e. blue-red, which is a crappy
        # rainbow colormap and shouldn't be used for anything, although in
        # its defense it's better than "Jet")
        cmap = getattr(cm, cmap)

    elif np.iterable(cmap):
        # This looks like a list of colors? Let's try that.
        colors = list(map(mpl.colors.colorConverter.to_rgb, cmap))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("_", colors)

    else:
        # If we get here, it's a bad input
        raise ValueError("Input %s was not valid for making a lut" % cmap)

    # Convert from a matplotlib colormap to a lut array
    lut = (cmap(np.linspace(0, 1, n_colors)) * 255).astype(np.int)

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
    from scipy import sparse

    logger.info("Updating smoothing matrix, be patient..")

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

        logger.info("Smoothing matrix creation, step %d" % (k + 1))
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
                   map_surface='white', coord_as_vert=False, verbose=None):
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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).
    """
    geo = Surface(subject_id, hemi, map_surface)
    geo.load_geometry()

    if coord_as_vert:
        coord = geo.coords[coord]

    n_vertices = len(geo.coords)
    adj_mat = mesh_edges(geo.faces)
    foci_vtxs = find_closest_vertices(geo.coords, [coord])
    data = np.zeros(n_vertices)
    data[foci_vtxs] = 1.
    smooth_mat = smoothing_matrix(np.arange(n_vertices), adj_mat, 1)
    for _ in range(n_steps):
        data = smooth_mat * data
    idx = np.where(data.ravel() > 0)[0]
    # Write label
    label_fname = label + '-' + hemi + '.label'
    logger.info("Saving label : %s" % label_fname)
    f = open(label_fname, 'w')
    f.write('#label at %s from subject %s\n' % (coord, subject_id))
    f.write('%d\n' % len(idx))
    for i in idx:
        x, y, z = geo.coords[i]
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
        the SUBJECTS_DIR environment variable. If that does not exist,
        it will be obtained from the configuration file.
    """
    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            subjects_dir = config.get('options', 'subjects_dir')
            if raise_error and subjects_dir == '':
                raise ValueError('The subjects directory has to be specified '
                                 'using the subjects_dir parameter, the '
                                 'SUBJECTS_DIR environment variable, or the '
                                 '"subjects_dir" entry in the config file')

    if raise_error and not os.path.exists(subjects_dir):
        raise ValueError('The subjects directory %s does not exist.'
                         % subjects_dir)

    return subjects_dir


def has_fsaverage(subjects_dir=None):
    """Determine whether the user has a usable fsaverage"""
    fs_dir = op.join(_get_subjects_dir(subjects_dir, False), 'fsaverage')
    if not op.isdir(fs_dir):
        return False
    if not op.isdir(op.join(fs_dir, 'surf')):
        return False
    return True

requires_fsaverage = np.testing.dec.skipif(not has_fsaverage(),
                                           'Requires fsaverage subject data')


def has_ffmpeg():
    """Test whether the FFmpeg is available in a subprocess

    Returns
    -------
    ffmpeg_exists : bool
        True if FFmpeg can be successfully called, False otherwise.
    """
    try:
        subprocess.call(["ffmpeg"], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        return True
    except OSError:
        return False


def assert_ffmpeg_is_available():
    "Raise a RuntimeError if FFmpeg is not in the PATH"
    if not has_ffmpeg():
        err = ("FFmpeg is not in the path and is needed for saving "
               "movies. Install FFmpeg and try again. It can be "
               "downlaoded from http://ffmpeg.org/download.html.")
        raise RuntimeError(err)

requires_ffmpeg = np.testing.dec.skipif(not has_ffmpeg(), 'Requires FFmpeg')


def ffmpeg(dst, frame_path, framerate=24, codec='mpeg4'):
    """Run FFmpeg in a subprocess to convert an image sequence into a movie

    Parameters
    ----------
    dst : str
        Destination path. If the extension is not ".mov" or ".avi", ".mov" is
        added. If the file already exists it is overwritten.
    frame_path : str
        Path to the source frames (with a frame number field like '%04d').
    framerate : float
        Framerate of the movie (frames per second, default 24).
    codec : str
        Codec to use (default 'mpeg4').

    Notes
    -----
    Requires FFmpeg to be in the path. FFmpeg can be downlaoded from `here
    <http://ffmpeg.org/download.html>`_. Stdout and stderr are written to the
    logger. If the movie file is not created, a RuntimeError is raised.
    """
    assert_ffmpeg_is_available()

    # find target path
    dst = os.path.expanduser(dst)
    dst = os.path.abspath(dst)
    root, ext = os.path.splitext(dst)
    dirname = os.path.dirname(dst)
    if ext not in ['.mov', '.avi']:
        dst += '.mov'

    if os.path.exists(dst):
        os.remove(dst)
    elif not os.path.exists(dirname):
        os.mkdir(dirname)

    frame_dir, frame_fmt = os.path.split(frame_path)

    # make the movie
    cmd = ['ffmpeg', '-i', frame_fmt, '-r', str(framerate), '-c', codec, dst]
    logger.info("Running FFmpeg with command: %s", ' '.join(cmd))
    sp = subprocess.Popen(cmd, cwd=frame_dir, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    # log stdout and stderr
    stdout, stderr = sp.communicate()
    std_info = os.linesep.join(("FFmpeg stdout", '=' * 25, stdout))
    logger.info(std_info)
    if stderr.strip():
        err_info = os.linesep.join(("FFmpeg stderr", '=' * 27, stderr))
        logger.error(err_info)

    # check that movie file is created
    if not os.path.exists(dst):
        err = ("FFmpeg failed, no file created; see log for more more "
               "information.")
        raise RuntimeError(err)
