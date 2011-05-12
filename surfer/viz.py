import os
from os.path import join as pjoin

import numpy as np

from . import io
from .io import Surface

lh_viewdict = {'lateral': (0, -90),
                'medial': (0, 90),
                'anterior': (90, 90),
                'posterior': (90, -90),
                'dorsal': (90, 0),
                'ventral': (-90, 180)}
rh_viewdict = {'lateral': (0, 90),
                'medial': (0, -90),
                'anterior': (90, 90),
                'posterior': (-90, -90),
                'dorsal': (-90, 0),
                'ventral': (90, 0)}


class Brain(object):
    """Brain object for visualizing with mlab."""

    def __init__(self, subject_id, hemi, surf, curv=True):
        """Initialize a Brain object with Freesurfer-specific data.

        Parameters
        ----------
        subject_id : str
            subject name in Freesurfer subjects dir
        hemi : str
            hemisphere id (ie 'lh' or 'rh')
        surf :  geometry name
            freesurfer surface mesh name (ie 'white', 'inflated', etc.)
        curv : boolean
            if true, loads curv file and displays binary curvature
            (default: True)
        overlay : filepath
            path to overlay file
        """
        from enthought.mayavi import mlab

        # Set the identifying info
        self.subject_id = subject_id
        self.hemi = hemi
        if self.hemi == 'lh':
            self.viewdict = lh_viewdict
        else:
            self.viewdict = rh_viewdict
        self.surf = surf

        # Initialize an mlab figure
        self._f = mlab.figure(np.random.randint(1, 1000),
                              bgcolor=(12. / 256, 0. / 256, 25. / 256),
                              size=(800, 800))
        mlab.clf()
        self._f.scene.disable_render = True

        # Initialize a Surface object as the geometry
        self._geo = Surface(subject_id, hemi, surf)

        # Load in the geometry and (maybe) curvature
        self._geo.load_geometry()
        if curv:
            self._geo.load_curvature()
            curv_data = self._geo.bin_curv
        else:
            curv_data = None

        # mlab pipeline mesh for geomtery
        self._geo_mesh = mlab.pipeline.triangular_mesh_source(
                                        self._geo.x, self._geo.y, self._geo.z,
                                        self._geo.faces, scalars=curv_data)

        # mlab surface for the geometry
        colormap, vmin, vmax, reverse = self.__get_geo_colors()
        self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                    colormap=colormap, vmin=vmin, vmax=vmax)
        if reverse:
            curv_bar = mlab.scalarbar(self._geo_surf)
            curv_bar.reverse_lut = True
            curv_bar.visible = False

        # Initialize the overlay dictionary
        self.overlays = dict()

        # Turn disable render off so that it displays
        self._f.scene.disable_render = False

        #make lateral view
        self.show_view("lat")

    def show_view(self, view):
        """Orient camera to display view

        Parameters
        ----------
        view : {'lateral' | 'medial' | 'anterior' |
                'posterior' | 'superior' | 'inferior'}
              desired viewing angle (can be leading substring of above list)
        """
        from enthought.mayavi import mlab

        if not view in self.viewdict:
            good_view = [k for k in self.viewdict.keys()
                        if view == k[:len(view)]]
            if len(good_view) != 1:
                raise ValueError("Available views are %s " %
                                " ".join(self.viewdict.keys()))
            view = good_view[0]
        mlab.view(*self.viewdict[view])

    def add_overlay(self, filepath, range, sign="abs",
                    name=None, visible=True):
        """Add an overlay to the overlay dict.

        Parameters
        ----------
        filepath : str
            path to the overlay file (must be readable by Nibabel, or .mgh
        range : (min, max)
            threshold and saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary
        visible : boolean
            whether the overlay should be visible upon load

        """
        if name is None:
            basename = os.path.basename(filepath)
            if basename.endswith(".gz"):
                basename = basename[:-3]
            name = os.path.splitext(basename)[0]

        if name in self.overlays:
            raise NameError("Overlay with name %s already exists. "
                            "Please provide a name for this overlay" % name)

        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")

        self._f.scene.disable_render = False
        self.overlays[name] = Overlay(self._geo, filepath, range, sign)

    def __get_geo_colors(self):
        """Return an mlab colormap name, vmin, and vmax for binary curvature.

        At the moment just return a default.  Get from the config eventually

        Returns
        -------
        colormap : string
            mlab colormap name
        vmin : float
            curv colormap minimum
        vmax : float
            curv colormap maximum
        reverse : boolean
            boolean indicating whether the colormap should be reversed

        """
        return "gray", -1., 2., True

    def save_image(self, fname):
        """Save current view to disk

        Only mayavi image types are supported:
        (png jpg bmp tiff ps eps pdf rib  oogl iv  vrml obj

        Parameters
        ----------
        filename: string
            path to new image file

        """
        from enthought.mayavi import mlab
        ftype = fname[fname.rfind('.') + 1:]
        good_ftypes = ['png', 'jpg', 'bmp', 'tiff', 'ps',
                        'eps', 'pdf', 'rib', 'oogl', 'iv', 'vrml', 'obj']
        if not ftype in good_ftypes:
            raise ValueError("Supported image types are %s"
                                % " ".join(good_ftypes))
        mlab.savefig(fname)

    def save_imageset(self, prefix, views, filetype='png'):
        """Convience wrapper for save_image

        Files created are prefix+'_$view'+filetype

        Parameters
        ----------
        prefix: string
            filename prefix for image to be created
        views: list
            desired views for images
        filetype: string
            image type

        Returns
        -------
        images_written: list
            all filenames written
        """
        if isinstance(views, basestring):
            raise ValueError("Views must be a non-string sequence"
                             "Use show_view & save_image for a single view")
        images_written = []
        for view in views:
            try:
                fname = "%s_%s.%s" % (prefix, view, filetype)
                images_written.append(fname)
                self.show_view(view)
                try:
                    self.save_image(fname)
                except ValueError:
                    print("Bad image type")
            except ValueError:
                print("Skipping %s: not in view dict" % view)
        return images_written

    def save_montage(self, filename, order=['lat', 'ven', 'med'], shape='h'):
        """Create a montage from a given order of images

        Parameters
        ----------
        filename: string
            path to final image
        order: list
            order of views to build montage
        shape: {'h' | 'v'}

        """
        import Image
        fnames = self.save_imageset("tmp", order)
        images = map(Image.open, fnames)
        if shape == 'h':
            w = sum(i.size[0] for i in images)
            h = max(i.size[1] for i in images)
        else:
            h = sum(i.size[1] for i in images)
            w = max(i.size[0] for i in images)
        new = Image.new("RGBA", (w, h))
        x = 0
        for i in images:
            if shape == 'h':
                pos = (x, 0)
                x += i.size[0]
            else:
                pos = (0, x)
                x += i.size[1]
            new.paste(i, pos)
        try:
            new.save(filename)
        except Exception:
            print("Error saving %s" % filename)


class Overlay(object):

    def __init__(self, geo, filepath, range, sign):
        """
        """
        from enthought.mayavi import mlab

        scalar_data = io.read_scalar_data(filepath)
        if scalar_data.dtype.byteorder == '>':
            scalar_data.byteswap(True)  # byte swap inplace
        if sign in ["abs", "pos"]:
            pos_mesh = mlab.pipeline.triangular_mesh_source(geo.x, geo.y,
                                                        geo.z, geo.faces,
                                                        scalars=scalar_data)
            pos_thresh = mlab.pipeline.threshold(pos_mesh, low=range[0])
            pos_surf = mlab.pipeline.surface(pos_thresh, colormap="YlOrRd",
                                             vmin=range[0], vmax=range[1])
            pos_bar = mlab.scalarbar(pos_surf)
            pos_bar.reverse_lut = True
            pos_bar.visible = False

            self.pos = pos_surf

        if sign in ["abs", "neg"]:
            neg_mesh = mlab.pipeline.triangular_mesh_source(geo.x, geo.y,
                                                        geo.z, geo.faces,
                                                        scalars=scalar_data)
            neg_thresh = mlab.pipeline.threshold(neg_mesh, up=-range[0])
            neg_surf = mlab.pipeline.surface(neg_thresh, colormap="Blues",
                                             vmin=-range[1], vmax=-range[0])
            neg_bar = mlab.scalarbar(neg_surf)
            neg_bar.visible = False

            self.neg = neg_surf
