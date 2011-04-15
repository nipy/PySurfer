import os
import numpy as np
import gzip
from os.path import join as pjoin
try:
    import enthought.mayavi.mlab as mlab
    use_mlab = True
except ImportError:
    use_mlab = False
    print "Not using EPD environment; display won't work"

class Surface(object):

    def __init__(self, subject=None, hemi=None, surface=None,curv="curv"):
        """Initialize a Surface with (optionally) subject/hemi/surface/curvature
        subject (string): subject name as found in SUBJECTS_DIR
        hemi (string):lh or rh
        surface (string): inflated,white,pial,orig,sphere
        curv (string):curv,curv.pial"""
        if subject and hemi and surface and curv:
            #If using FreeSurfer, SUBJECTS_DIR is (most likely) set and accurate
            subj_dir = os.environ["SUBJECTS_DIR"]
            surface_path = pjoin(subj_dir,subject,"surf",".".join([hemi,surface]))
            if not os.path.exists(surface_path):
                raise IOError("%s doesn't exist"%surface_path)
            self.load_geometry(surface_path)
            curv_path = pjoin(subj_dir,subject,"surf",".".join([hemi,curv]))
            if not os.path.exists(curv_path):
                raise IOError("%s doesn't exist"%curv_path)
            self.load_curvature(curv_path)
            

    def load_geometry(self, filepath):
        """Load in a Freesurfer surface mesh in triangular format."""
        with open(filepath, "rb") as fobj:
            magic = fread3(fobj)
            if magic == 16777215:
                raise NotImplementedError("Quadrangle surface format reading not implemented")
            elif magic != 16777214:
                raise ValueError("File does not appear to be a Freesurfer surface")
            self.create_stamp = fobj.readline()
            blankline = fobj.readline()
            del blankline
            self.vnum = np.fromfile(fobj, ">i4", 1)[0]
            self.fnum = np.fromfile(fobj, ">i4", 1)[0]
            self.vertex_coords = np.fromfile(fobj, ">f4", self.vnum*3).reshape(self.vnum, 3)
            self.faces = np.fromfile(fobj, ">i4", self.fnum*3).reshape(self.fnum, 3)
        self.x = self.vertex_coords[:,0]
        self.y = self.vertex_coords[:,1]
        self.z = self.vertex_coords[:,2]

    def load_curvature(self, filepath):
        """Load in curavature values from the ?h.curv file."""

        with open(filepath, "rb") as fobj:
            magic = fread3(fobj)
            if magic == 16777215:
                vnum = np.fromfile(fobj, ">i4", 3)[0]
                self.curv = np.fromfile(fobj, ">f4", vnum)
            else:
                vnum = magic
                fnum = fread3(fobj)
                self.curv = np.fromfile(fobj, ">i2", vnum)/100
            self.bin_curv = np.array(self.curv > 0, np.int)
            
    def load_annot(self, filepath):
        """Load in a Freesurfer annotation from a .annot file."""
        with open(filepath, "rb") as fobj:
            dt = ">i4"
            vnum = np.fromfile(fobj, dt, 1)[0]
            data = np.fromfile(fobj, dt, vnum*2).reshape(vnum, 2)
            self.annot = data[:,1]
            ctab_exists = np.fromfile(fobj, dt, 1)[0]
            if not ctab_exists:
                return
            ctab_version = np.fromfile(fobj, dt, 1)[0]
            if ctab_version != -2:
                return
            n_entries = np.fromfile(fobj, dt, 1)[0]
            self.ctab = np.zeros((n_entries, 5),np.int)
            length = np.fromfile(fobj, dt, 1)[0]
            orig_tab = np.fromfile(fobj, "|S%d"%length, 1)[0]
            entries_to_read = np.fromfile(fobj, dt, 1)[0]
            for i in xrange(entries_to_read):
                structure = np.fromfile(fobj, dt, 1)[0]
                name_length = np.fromfile(fobj, dt, 1)[0]
                struct_name = np.fromfile(fobj, "|S%d"%name_length, 1)[0]
                self.ctab[i,:4] = np.fromfile(fobj, dt, 4)
                self.ctab[i,4] = (self.ctab[i,0] + self.ctab[i,1]*2**8 +
                                  self.ctab[i,2]*2**16 + self.ctab[i,3]*2**24)

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
                # I don't actually know what versions this code will read, so to be
                # on the safe side, let's only let version 1 in for now.
                # Scalar data might also be in curv format (e.g. lh.thickness)
                # in which case the first item in the file is a magic number.
                # We'll have to think about how to deal with that, although 
                # currently trying to load one of those files will just error
                # out when openfile doesn't get defined.
                raise NotImplementedError("Scalar data file version not supported")
            ndim1 = np.fromstring(fobj.read(4), ">i4")[0]
            ndim2 = np.fromstring(fobj.read(4), ">i4")[0]
            ndim3 = np.fromstring(fobj.read(4), ">i4")[0]
            nframes = np.fromstring(fobj.read(4), ">i4")[0]
            datatype = np.fromstring(fobj.read(4), ">i4")[0]
            # Set the number of bytes per voxel and numpy data type according to FS codes
            databytes, typecode = {0:(1,">i1"), 1:(4,">i4"), 3:(4,">f4"), 4:(2,">h")}[datatype]
            # Ignore the rest of the header here, just seek to the data
            fobj.seek(284)
            nbytes = ndim1*ndim2*ndim3*nframes*databytes
            # Read in all the data, keep it in flat representation (is this ever a problem?)
            self.scalar_data = np.fromstring(fobj.read(nbytes), typecode)
        finally:
            fobj.close()

    def load_label(self, filepath, name=None):
        """Load in a Freesurfer .label file.
        
        Label files are just text files indicating the vertices included 
        in the label. Each Surface instance has a dictionary of labels, keyed
        by the name (which is taken from the file name if not given as an argument.
        
        """
        labelarray = np.loadtxt(filepath, dtype=np.int, skiprows=2, usecols=[0])
        label = np.zeros(self.vnum, np.int)
        label[labelarray] = 1
        if name is None:
            name = os.path.basename(filepath)
            if name.endswith("label"):
                name = os.path.split(name)[0]
        try:
            self.labels[name] = label
        except AttributeError:
            self.labels = dict(name=label)


    def apply_xfm(self, mtx):
        """Apply an affine transformation matrix to the x,y,z vectors."""
        mtx = np.asmatrix(mtx)
        for v in range(self.vnum):
            coords = np.matrix([self.x[v], self.y[v], self.z[v], 1]).T
            coords = mtx * coords
            self.x[v] = coords[0]
            self.y[v] = coords[1]
            self.z[v] = coords[2]

    def get_mesh(self, curv=True):
        """Return an mlab pipeline mesh object""" 
        if hasattr(self, "bin_curv") and curv:
            curv_scalars = self.bin_curv
        else:
            curv_scalars=None
        return mlab.pipeline.triangular_mesh_source(self.x, 
                                                    self.y,
                                                    self.z, 
                                                    self.faces,
                                                    scalars=curv_scalars)


def fread3(fobj):
    """Docstring"""
    b1 = np.fromfile(fobj, ">u1", 1)[0]
    b2 = np.fromfile(fobj, ">u1", 1)[0]
    b3 = np.fromfile(fobj, ">u1", 1)[0]
    return (b1 << 16) + (b2 << 8) + b3
