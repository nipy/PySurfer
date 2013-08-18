import os
from os.path import join as pjoin
from tempfile import mktemp

from subprocess import Popen, PIPE
import gzip
import numpy as np
import nibabel as nib
from nibabel.spatialimages import ImageFileError

import logging
logger = logging.getLogger('surfer')


def _get_subjects_dir(subjects_dir=None):
    """Get the subjects directory from parameter or environment variable

    Parameters
    ----------
    subjects_dir : str | None
        The subjects directory.

    Returns
    -------
    subjects_dir : str
        The subjects directory. If the subjects_dir input parameter is not
        None, its value will be returned, otherwise it will be obtained from
        the SUBJECTS_DIR environment variable.
    """

    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('The subjects directory has to be specified '
                             'using either the subjects_dir parameter or the '
                             'SUBJECTS_DIR environment variable.')

    if not os.path.exists(subjects_dir):
        raise ValueError('The subjects directory %s does not exist.'
                         % subjects_dir)

    return subjects_dir


def read_scalar_data(filepath):
    """Load in scalar data from an image.

    Parameters
    ----------
    filepath : str
        path to scalar data file

    Returns
    -------
    scalar_data : numpy array
        flat numpy array of scalar data
    """
    try:
        scalar_data = nib.load(filepath).get_data()
        scalar_data = np.ravel(scalar_data, order="F")
        return scalar_data

    except ImageFileError:
        ext = os.path.splitext(filepath)[1]
        if ext == ".mgz":
            openfile = gzip.open
        elif ext == ".mgh":
            openfile = open
        else:
            raise ValueError("Scalar file format must be readable "
                             "by Nibabel or .mg{hz} format")

    fobj = openfile(filepath, "rb")
    # We have to use np.fromstring here as gzip fileobjects don't work
    # with np.fromfile; same goes for try/finally instead of with statement
    try:
        v = np.fromstring(fobj.read(4), ">i4")[0]
        if v != 1:
            # I don't actually know what versions this code will read, so to be
            # on the safe side, let's only let version 1 in for now.
            # Scalar data might also be in curv format (e.g. lh.thickness)
            # in which case the first item in the file is a magic number.
            raise NotImplementedError("Scalar data file version not supported")
        ndim1 = np.fromstring(fobj.read(4), ">i4")[0]
        ndim2 = np.fromstring(fobj.read(4), ">i4")[0]
        ndim3 = np.fromstring(fobj.read(4), ">i4")[0]
        nframes = np.fromstring(fobj.read(4), ">i4")[0]
        datatype = np.fromstring(fobj.read(4), ">i4")[0]
        # Set the number of bytes per voxel and numpy data type according to
        # FS codes
        databytes, typecode = {0: (1, ">i1"), 1: (4, ">i4"), 3: (4, ">f4"),
                               4: (2, ">h")}[datatype]
        # Ignore the rest of the header here, just seek to the data
        fobj.seek(284)
        nbytes = ndim1 * ndim2 * ndim3 * nframes * databytes
        # Read in all the data, keep it in flat representation
        # (is this ever a problem?)
        scalar_data = np.fromstring(fobj.read(nbytes), typecode)
    finally:
        fobj.close()

    return scalar_data


def read_label(filepath, read_scalars=False):
    """Load in a Freesurfer .label file.

    Parameters
    ----------
    filepath : str
        Path to label file
    read_scalars : bool
        If true, read and return scalars associated with each vertex

    Returns
    -------
    label_array : numpy array (ints)
        Array with indices of vertices included in label
    scalar_array : numpy array (floats)
        If read_scalars is True, array of scalar data for each vertex

    """
    label_array = np.loadtxt(filepath, dtype=np.int, skiprows=2, usecols=[0])
    if read_scalars:
        scalar_array = np.loadtxt(filepath, skiprows=2, usecols=[-1])
        return label_array, scalar_array
    return label_array


def read_stc(filepath):
    """Read an STC file from the MNE package

    STC files contain activations or source reconstructions
    obtained from EEG and MEG data.

    Parameters
    ----------
    filepath: string
        Path to STC file

    Returns
    -------
    data: dict
        The STC structure. It has the following keys:
           tmin           The first time point of the data in seconds
           tstep          Time between frames in seconds
           vertices       vertex indices (0 based)
           data           The data matrix (nvert * ntime)
    """
    fid = open(filepath, 'rb')

    stc = dict()

    fid.seek(0, 2)  # go to end of file
    file_length = fid.tell()
    fid.seek(0, 0)  # go to beginning of file

    # read tmin in ms
    stc['tmin'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tmin'] /= 1000.0

    # read sampling rate in ms
    stc['tstep'] = float(np.fromfile(fid, dtype=">f4", count=1))
    stc['tstep'] /= 1000.0

    # read number of vertices/sources
    vertices_n = int(np.fromfile(fid, dtype=">u4", count=1))

    # read the source vector
    stc['vertices'] = np.fromfile(fid, dtype=">u4", count=vertices_n)

    # read the number of timepts
    data_n = int(np.fromfile(fid, dtype=">u4", count=1))

    if ((file_length / 4 - 4 - vertices_n) % (data_n * vertices_n)) != 0:
        raise ValueError('incorrect stc file size')

    # read the data matrix
    stc['data'] = np.fromfile(fid, dtype=">f4", count=vertices_n * data_n)
    stc['data'] = stc['data'].reshape([data_n, vertices_n]).T

    # close the file
    fid.close()
    return stc


def project_volume_data(filepath, hemi, reg_file=None, subject_id=None,
                        projmeth="frac", projsum="avg", projarg=[0, 1, .1],
                        surf="white", smooth_fwhm=3, mask_label=None,
                        target_subject=None, verbose=False):
    """Sample MRI volume onto cortical manifold.

    Note: this requires Freesurfer to be installed with correct
    SUBJECTS_DIR definition (it uses mri_vol2surf internally).

    Parameters
    ----------
    filepath : string
        Volume file to resample (equivalent to --mov)
    hemi : [lh, rh]
        Hemisphere target
    reg_file : string
        Path to TKreg style affine matrix file
    subject_id : string
        Use if file is in register with subject's orig.mgz
    projmeth : [frac, dist]
        Projection arg should be understood as fraction of cortical
        thickness or as an absolute distance (in mm)
    projsum : [avg, max, point]
        Average over projection samples, take max, or take point sample
    projarg : single float or sequence of three floats
        Single float for point sample, sequence for avg/max specifying
        start, stop, and stop
    surf : string
        Target surface
    smooth_fwhm : float
        FWHM of surface-based smoothing to apply; 0 skips smoothing
    mask_label : string
        Path to label file to constrain projection; otherwise uses cortex
    target_subject : string
        Subject to warp data to in surface space after projection
    verbose : bool
        If True, print the command used
    """
    # Set the basic commands
    cmd_list = ["mri_vol2surf",
                "--mov", filepath,
                "--hemi", hemi,
                "--surf", surf]

    # Specify the affine registration
    if reg_file is not None:
        cmd_list.extend(["--reg", reg_file])
    elif subject_id is not None:
        cmd_list.extend(["--regheader", subject_id])
    else:
        raise ValueError("Must specify reg_file or subject_id")

    # Specify the projection
    proj_flag = "--proj" + projmeth
    if projsum != "point":
        proj_flag += "-"
        proj_flag += projsum
    if hasattr(projarg, "__iter__"):
        proj_arg = map(str, projarg)
    else:
        proj_arg = [str(projarg)]
    cmd_list.extend([proj_flag] + proj_arg)

    # Set misc args
    if smooth_fwhm:
        cmd_list.extend(["--surf-fwhm", str(smooth_fwhm)])
    if mask_label is not None:
        cmd_list.extend(["--mask", mask_label])
    if target_subject is not None:
        cmd_list.extend(["--trgsubject", target_subject])

    # Execute the command
    out_file = mktemp(prefix="pysurfer-v2s", suffix='.mgz')
    cmd_list.extend(["--o", out_file])
    if verbose:
        print " ".join(cmd_list)
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    out = p.returncode
    if out:
        raise RuntimeError(("mri_vol2surf command failed "
                            "with command-line: ") + " ".join(cmd_list))

    # Read in the data
    surf_data = read_scalar_data(out_file)
    os.remove(out_file)
    return surf_data


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
        self.data_path = pjoin(subjects_dir, subject_id)

    def load_geometry(self):
        surf_path = pjoin(self.data_path, "surf",
                          "%s.%s" % (self.hemi, self.surf))
        self.coords, self.faces = nib.freesurfer.read_geometry(surf_path)
        if self.offset is not None:
            if self.hemi == 'lh':
                self.coords[:, 0] -= (np.max(self.coords[:, 0]) + self.offset)
            else:
                self.coords[:, 0] -= (np.min(self.coords[:, 0]) + self.offset)

    def save_geometry(self):
        surf_path = pjoin(self.data_path, "surf",
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
        curv_path = pjoin(self.data_path, "surf", "%s.curv" % self.hemi)
        self.curv = nib.freesurfer.read_morph_data(curv_path)
        self.bin_curv = np.array(self.curv > 0, np.int)

    def load_label(self, name):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.

        """
        label = nib.freesurfer.read_label(pjoin(self.data_path, 'label',
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
