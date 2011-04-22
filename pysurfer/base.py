import os
import numpy as np
import gzip
from os.path import join as pjoin


def read_geometry(filepath):
    """Load in a Freesurfer surface mesh in triangular format."""
    with open(filepath, "rb") as fobj:
        magic = fread3(fobj)
        if magic == 16777215:
            raise NotImplementedError("Quadrangle surface format reading "
                                      "not implemented")
        elif magic != 16777214:
            raise ValueError("File does not appear to be a Freesurfer surface")
        create_stamp = fobj.readline()
        _ = fobj.readline()
        vnum = np.fromfile(fobj, ">i4", 1)[0]
        fnum = np.fromfile(fobj, ">i4", 1)[0]
        coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
        faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

    coords = coords.astype(np.float)  # XXX : due to mayavi bug on mac 32bits
    return coords, faces


def read_curvature(filepath):
    """Load in curavature values from the ?h.curv file."""

    with open(filepath, "rb") as fobj:
        magic = fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            _ = fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
        return curv


class Surface(object):
    """Basic object to read and view a freesurfer surface
    """

    def __init__(self, subject, hemi='lh', surface='inflated', curv="curv"):
        """Initialize a Surface

        Parameters
        ----------
        subject : string
            subject name as found in SUBJECTS_DIR
        hemi : 'lh' or 'rh'
            Left or right hemisphere
        surface : 'inflated' | 'white' | 'pial' | 'orig' | 'sphere'
            Name of the surface to load
        curv : 'curv' | 'curv.pial'
            Type of curvature
        """
        if subject and hemi and surface and curv:
            # SUBJECTS_DIR from Freesurfer is (most likely) set and accurate
            subj_dir = os.environ["SUBJECTS_DIR"]

            # Geometry
            surface_path = pjoin(subj_dir, subject, "surf",
                                 ".".join([hemi, surface]))
            if not os.path.exists(surface_path):
                raise IOError("%s doesn't exist" % surface_path)
            self.coords, self.faces = read_geometry(surface_path)

            # Curvature
            curv_path = pjoin(subj_dir, subject, "surf",
                                                ".".join([hemi, curv]))
            if not os.path.exists(curv_path):
                raise IOError("%s doesn't exist" % curv_path)
            self.curv = read_curvature(curv_path)
            self.bin_curv = np.array(self.curv > 0, np.int)

            # Init label
            self.labels = dict()

    def load_annot(self, filepath):
        """Load in a Freesurfer annotation from a .annot file."""
        with open(filepath, "rb") as fobj:
            dt = ">i4"
            vnum = np.fromfile(fobj, dt, 1)[0]
            data = np.fromfile(fobj, dt, vnum * 2).reshape(vnum, 2)
            self.annot = data[:, 1]
            ctab_exists = np.fromfile(fobj, dt, 1)[0]
            if not ctab_exists:
                return
            ctab_version = np.fromfile(fobj, dt, 1)[0]
            if ctab_version != -2:
                return
            n_entries = np.fromfile(fobj, dt, 1)[0]
            self.ctab = np.zeros((n_entries, 5), np.int)
            length = np.fromfile(fobj, dt, 1)[0]
            orig_tab = np.fromfile(fobj, "|S%d" % length, 1)[0]
            entries_to_read = np.fromfile(fobj, dt, 1)[0]
            for i in xrange(entries_to_read):
                structure = np.fromfile(fobj, dt, 1)[0]
                name_length = np.fromfile(fobj, dt, 1)[0]
                struct_name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
                self.ctab[i, :4] = np.fromfile(fobj, dt, 4)
                self.ctab[i, 4] = (self.ctab[i, 0] + self.ctab[i, 1] * 2 ** 8 +
                                   self.ctab[i, 2] * 2 ** 16 +
                                   self.ctab[i, 3] * 2 ** 24)

    def load_scalar_data(self, filepath):
        """Load in scalar data from an mg(h,z) image."""
        ext = os.path.splitext(filepath)[1]
        if ext == ".mgz":
            openfile = gzip.open
        elif ext == ".mgh":
            openfile = open
        else:
            raise ValueError("Scalar file format must be .mgh or .mgz")
        fobj = openfile(filepath, "rb")
        # We have to use np.fromstring here as gzip fileobjects don't work
        # with np.fromfile; same goes for try/finally instead of with statement
        try:
            v = np.fromstring(fobj.read(4), ">i4")[0]
            if v != 1:
                # I don't actually know what versions this code will read, so
                # to be on the safe side, let's only let version 1 in for now.
                # Scalar data might also be in curv format (e.g. lh.thickness)
                # in which case the first item in the file is a magic number.
                # We'll have to think about how to deal with that, although
                # currently trying to load one of those files will just error
                # out when openfile doesn't get defined.
                raise NotImplementedError("Scalar data file version not "
                                          "supported")
            ndim1 = np.fromstring(fobj.read(4), ">i4")[0]
            ndim2 = np.fromstring(fobj.read(4), ">i4")[0]
            ndim3 = np.fromstring(fobj.read(4), ">i4")[0]
            nframes = np.fromstring(fobj.read(4), ">i4")[0]
            datatype = np.fromstring(fobj.read(4), ">i4")[0]
            # Set the number of bytes per voxel and numpy data type according
            # to FS codes
            databytes, typecode = {0: (1, ">i1"), 1: (4, ">i4"), 3: (4, ">f4"),
                                   4: (2, ">h")}[datatype]
            # Ignore the rest of the header here, just seek to the data
            fobj.seek(284)
            nbytes = ndim1 * ndim2 * ndim3 * nframes * databytes
            # Read in all the data, keep it in flat representation
            # (is this ever a problem?)
            self.scalar_data = np.fromstring(fobj.read(nbytes), typecode)
        finally:
            fobj.close()

    def load_label(self, filepath, name=None):
        """Load in a Freesurfer .label file.

        Label files are just text files indicating the vertices included
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an
        argument.

        """
        label_array = np.loadtxt(filepath, dtype=np.int, skiprows=2,
                                usecols=[0])
        label = np.zeros(len(self.coords), np.int)
        label[label_array] = 1
        if name is None:
            name = os.path.basename(filepath)
            if name.endswith("label"):
                name = os.path.split(name)[0]
        self.labels[name] = label

    def apply_xfm(self, mtx):
        """Apply an affine transformation matrix to the x,y,z vectors."""
        self.coords = np.dot(np.c_[self.coords, np.ones(len(self.coords))],
                             mtx.T)[:, :3]

    def get_mesh(self, curv=True):
        """Return an mlab pipeline mesh object"""
        try:  # lazy import for speed when no plotting
            import enthought.mayavi.mlab as mlab
            use_mlab = True
        except ImportError:
            use_mlab = False
            print "Not using EPD environment; display won't work"

        if hasattr(self, "bin_curv") and curv:
            curv_scalars = self.bin_curv
        else:
            curv_scalars = None
        x, y, z = self.coords.T
        return mlab.pipeline.triangular_mesh_source(x, y, z, self.faces,
                                                    scalars=curv_scalars)


def fread3(fobj):
    """Docstring"""
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3
