import os
from os.path import join as pjoin
from warnings import warn

import numpy as np
from scipy import stats

from . import io
from .io import Surface
from .config import config

lh_viewdict = {'lateral': (180, 90),
                'medial': (0, 90),
                'anterior': (90, 90),
                'posterior': (-90, 90),
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

    def __init__(self, subject_id, hemi, surf, curv=True, title=None):
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
        """
        from enthought.mayavi import mlab

        # Set the identifying info
        self.subject_id = subject_id
        self.hemi = hemi
        if self.hemi == 'lh':
            self.viewdict = lh_viewdict
        elif self.hemi == 'rh':
            self.viewdict = rh_viewdict
        self.surf = surf

        # Initialize an mlab figure
        bg_color_code, size = self.__get_scene_properties()
        if title is None:
            title = subject_id
        self._f = mlab.figure(title,
                              bgcolor=bg_color_code,
                              size=size)
        mlab.clf()
        self._f.scene.disable_render = True

        # Initialize a Surface object as the geometry
        self._geo = Surface(subject_id, hemi, surf)

        # Load in the geometry and (maybe) curvature
        self._geo.load_geometry()
        if curv:
            self._geo.load_curvature()
            curv_data = self._geo.bin_curv
            meshargs = dict(scalars=curv_data)
        else:
            curv_data = None
            meshargs = dict()

        # mlab pipeline mesh for geomtery
        self._geo_mesh = mlab.pipeline.triangular_mesh_source(
                                        self._geo.x, self._geo.y, self._geo.z,
                                        self._geo.faces, **meshargs)

        # mlab surface for the geometry
        if curv:
            colormap, vmin, vmax, reverse = self.__get_geo_colors()
            self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                colormap=colormap, vmin=vmin, vmax=vmax)
            if reverse:
                curv_bar = mlab.scalarbar(self._geo_surf)
                curv_bar.reverse_lut = True
                curv_bar.visible = False
        else:
            self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                                   color=(.5, .5, .5))

        # Initialize the overlay dictionaries
        self.overlays = dict()

        # Bring up the lateral view
        self.show_view(config.get("visual", "default_view"))
        #self.show_view("lat")

        # Turn disable render off so that it displays
        self._f.scene.disable_render = False

    def show_view(self, view):
        """Orient camera to display view

        Parameters
        ----------
        view : {'lateral' | 'medial' | 'anterior' |
                'posterior' | 'dorsal' | 'ventral' | tuple}
            brain surface to view, or tuple to pass to mlab.view()
        """
        from enthought.mayavi import mlab

        if isinstance(view, str):
            if view in self.viewdict:
                mlab.view(*self.viewdict[view])
            else:
                try:
                    view = self.__xfm_view(view)
                except ValueError:
                    print("Cannot display %s view. Must be preset view "
                          "name or leading substring" % view)
        elif isinstance(view, tuple):
            mlab.view(*view)
        else:
            raise ValueError("View must be one of the preset view names "
                             "or a tuple to be passed to mlab.view()")

    def add_overlay(self, filepath, min=None, max=None, sign="abs",
                    name=None, visible=True):
        """Add an overlay to the overlay dict.

        Parameters
        ----------
        filepath : str
            path to the overlay file (must be readable by Nibabel, or .mgh
        min : float
            threshold for overlay display
        max : float
            saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary
        visible : boolean
            whether the overlay should be visible upon load

        """
        from enthought.mayavi import mlab
        if name is None:
            basename = os.path.basename(filepath)
            if basename.endswith(".gz"):
                basename = basename[:-3]
            if basename.startswith("%s." % self.hemi):
                basename = basename[3:]
            name = os.path.splitext(basename)[0]

        if name in self.overlays:
            raise NameError("Overlay with name '%s' already exists. "
                            "Please provide a name for this overlay" % name)

        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")

        self._f.scene.disable_render = True
        scalar_data = io.read_scalar_data(filepath)
        view = mlab.view()
        self.overlays[name] = Overlay(scalar_data, self._geo, min, max, sign)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_morphometry(self, measure, visible=True):
        """Add a morphometry overlay to the image.

        Parameters
        ----------
        measure : {'area' | 'curv' | 'jacobian_white' | 'sulc' | 'thickness'}
            which measure to load

        """
        from enthought.mayavi import mlab
        surf_dir = pjoin(os.environ['SUBJECTS_DIR'], self.subject_id, 'surf')
        morph_file = pjoin(surf_dir, '.'.join([self.hemi, measure]))
        if not os.path.exists(morph_file):
            raise ValueError(
                'Could not find %s in subject directory' % morph_file)

        cmap_dict = dict(area="pink",
                         curv="RdBu",
                         jacobian_white="pink",
                         sulc="RdBu",
                         thickness="pink")

        self._f.scene.disable_render = True
        view = mlab.view()
        morph_data = io.read_morph_data(morph_file)
        cortex = self._geo.load_label("cortex")
        ctx_idx = np.where(cortex == 1)
        min = stats.scoreatpercentile(morph_data[ctx_idx], 2)
        max = stats.scoreatpercentile(morph_data[ctx_idx], 98)
        if morph_data.dtype.byteorder == '>':
            morph_data.byteswap(True)  # byte swap inplace; due to mayavi bug
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=morph_data)
        surf = mlab.pipeline.surface(mesh, colormap=cmap_dict[measure],
                                     vmin=min, vmax=max,
                                     name=measure)
        bar = mlab.scalarbar(surf)
        bar.scalar_bar_representation.position2 = .8, 0.09
        self.morphometry = dict(surface=surf,
                                colorbar=bar,
                                measure=measure)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def __get_scene_properties(self):
        """Get the background color and size from the config parser."""
        bg_colors = dict(black=[0, 0, 0],
                         white=[256, 256, 256],
                         midnight=[12, 7, 32],
                         slate=[112, 128, 144],
                         sand=[245, 222, 179])

        bg_color_name = config.get("visual", "background")
        bg_color_code = bg_colors[bg_color_name]
        bg_color_code = tuple(map(lambda x: float(x) / 256, bg_color_code))

        size = config.getfloat("visual", "size")
        size = (size, size)

        return bg_color_code, size

    def __get_geo_colors(self):
        """Return an mlab colormap name, vmin, and vmax for binary curvature.

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
        colormap_map = dict(classic=("Greys", -1, 2, False),
                            high_contrast=("Greys", -.1, 1.3, False),
                            low_contrast=("Greys", -5, 5, False),
                            bone=("bone", -.2, 2, True))

        cortex_color = config.get("visual", "cortex")
        try:
            color_data = colormap_map[cortex_color]
        except KeyError:
            warn(""
                 "The 'cortex' setting in your config file must be one of "
                 "'classic', 'high_contrast', 'low_contrast', or 'bone', "
                 "but your value is '%s'. I'm setting the cortex colormap "
                 "to the 'classic' setting." % cortex_color)
            color_data = colormap_map['classic']

        return color_data

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
            montage image shape

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
        for f in fnames:
            os.remove(f)

    def __t2v(self, t):
        """Transform tuple to view

        Parameters
        ----------
        t: tuple
            camera angle

        Returns
        -------
        k: string
            corresponding view

        """
        for k in self.viewdict:
            if self.viewdict[k] == t:
                return k
        raise ValueError("Tuple not found in viewdict")

    def __min_diff(self, beg, end):
        """Determine minimum "camera distance" between two views

        Parameters
        ----------
        beg: string
            beginning anatomical view
        end: string
            ending anatomical view

        Returns
        -------
        new_diff: np.array
            shortest camera "path" between two views

        """
        if beg == end:
            v = self.__xfm_view(beg)
            if v in ['lateral', 'medial', 'anterior', 'posterior']:
                new_diff = [360, 0]
            else:
                new_diff[1] = [0, 360]
        else:
            gv = map(self.__xfm_view, [beg, end])
            d = np.array(self.viewdict[gv[1]]) - np.array(self.viewdict[gv[0]])
            new_diff = []
            for x in d:
                if x > 180:
                    new_diff.append(x - 360)
                elif x < -180:
                    new_diff.append(x + 360)
                else:
                    new_diff.append(x)
        return np.array(new_diff)

    def animate(self, views, n=180):
        """Animate a rotation

        Parameters
        ----------
        views: sequence
            views to animate through
        n: int
            number of steps to take in between
        save_gif: bool
            save the animation
        fname: string
            file to save gif image

        """
        import numpy as np
        from enthought.mayavi import mlab
        for i, v in enumerate(views):
            try:
                if isinstance(v, str):
                    b = self.__xfm_view(v)
                end = views[i + 1]
                if isinstance(end, str):
                    e = self.__xfm_view(end)
                d = self.__min_diff(b, e)
                dx = d / np.array((float(n)))
                ov = np.array(mlab.view(*self.viewdict[b])[:2])
                for i in range(n):
                    nv = ov + i * dx
                    mlab.view(*nv)
            except IndexError:
                pass

    def __xfm_view(self, view, out='s'):
        """Normalize a given string to available view

        Parameters
        ----------
        view: string
            view which may match leading substring of available views

        Returns
        -------
        good: string
            matching view string
        out: {'s' | 't'}
            's' to return string, 't' to return tuple

        """
        if not view in self.viewdict:
            good_view = [k for k in self.viewdict if view == k[:len(view)]]
            if len(good_view) != 1:
                raise ValueError("bad view")
            view = good_view[0]
        if out == 't':
            return self.viewdict[view]
        else:
            return view


class Overlay(object):

    def __init__(self, scalar_data, geo, min, max, sign):
        """
        """
        from enthought.mayavi import mlab

        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"
        self.sign = sign

        # Get data with a range that will make sense for automatic thresholding
        if sign == "neg":
            range_data = np.abs(scalar_data[np.where(scalar_data < 0)])
        elif sign == "pos":
            range_data = scalar_data[np.where(scalar_data > 0)]
        else:
            range_data = np.abs(scalar_data)

        if min is None:
            try:
                min = config.getfloat("overlay", "min_thresh")
            except ValueError:
                min_str = config.get("overlay", "min_thresh")
                if min_str == "robust_min":
                    min = stats.scoreatpercentile(range_data, 2)
                elif min_str == "actual_min":
                    min = range_data.min()
                else:
                    min = 2.0
                    warn("The 'min_thresh' value in your config value must be "
                "a float, 'robust_min', or 'actual_min', but it is %s. "
                "I'm setting the overlay min to the config default of 2" % min)

        if max is None:
            try:
                max = config.getfloat("overlay", "max_thresh")
            except ValueError:
                max_str = config.get("overlay", "max_thresh")
                if max_str == "robust_max":
                    max = stats.scoreatpercentile(range_data, 98)
                elif max_str == "actual_max":
                    max = range_data.max()
                else:
                    max = stats.scoreatpercentile(range_data, 98)
                    warn("The 'max_thresh' value in your config value must be "
                "a float, 'robust_min', or 'actual_min', but it is %s. "
                "I'm setting the overlay min to the config default "
                "of robust_max" % max)

        # Clean up range_data since we don't need it and it might be big
        del range_data

        # Byte swap inplace; due to mayavi bug
        mlab_data = scalar_data.copy()
        if scalar_data.dtype.byteorder == '>':
            mlab_data.byteswap(True)

        if sign in ["abs", "pos"]:
            pos_mesh = mlab.pipeline.triangular_mesh_source(geo.x,
                                                           geo.y,
                                                           geo.z,
                                                           geo.faces,
                                                           scalars=mlab_data)

            # Figure out the correct threshold to avoid TraitErrors
            # This seems like not the cleanest way to do this
            pos_data = scalar_data[np.where(scalar_data > 0)]
            try:
                pos_max = pos_data.max()
            except ValueError:
                pos_max = 0
            if pos_max < min:
                thresh_low = pos_max
            else:
                thresh_low = min
            pos_thresh = mlab.pipeline.threshold(pos_mesh, low=thresh_low)
            pos_surf = mlab.pipeline.surface(pos_thresh, colormap="YlOrRd",
                                             vmin=min, vmax=max)
            pos_bar = mlab.scalarbar(pos_surf)
            pos_bar.reverse_lut = True

            self.pos = pos_surf
            self.pos_bar = pos_bar

        if sign in ["abs", "neg"]:
            neg_mesh = mlab.pipeline.triangular_mesh_source(geo.x,
                                                           geo.y,
                                                           geo.z,
                                                           geo.faces,
                                                           scalars=mlab_data)

            # Figure out the correct threshold to avoid TraitErrors
            # This seems even less clean due to negative convolutedness
            neg_data = scalar_data[np.where(scalar_data < 0)]
            try:
                neg_min = neg_data.min()
            except ValueError:
                neg_min = 0
            if neg_min > -min:
                thresh_up = neg_min
            else:
                thresh_up = -min
            neg_thresh = mlab.pipeline.threshold(neg_mesh, up=thresh_up)
            neg_surf = mlab.pipeline.surface(neg_thresh, colormap="PuBu",
                                             vmin=-max, vmax=-min)
            neg_bar = mlab.scalarbar(neg_surf)

            self.neg = neg_surf
            self.neg_bar = neg_bar

        self.__format_colorbar()

    def toggle_visibility(self):

        if self.sign in ["pos", "abs"]:
            self.pos.visible = not self.pos.visible
            self.pos_bar.visible = False
        if self.sign in ["neg", "abs"]:
            self.neg.visible = not self.neg.visible
            self.neg_bar.visible = False

    def __format_colorbar(self):

        if self.sign in ["abs", "neg"]:
            self.neg_bar.scalar_bar_representation.position = (0.05, 0.01)
            self.neg_bar.scalar_bar_representation.position2 = (0.42, 0.09)
        if self.sign in ["abs", "pos"]:
            self.pos_bar.scalar_bar_representation.position = (0.53, 0.01)
            self.pos_bar.scalar_bar_representation.position2 = (0.42, 0.09)
