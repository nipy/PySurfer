import os
from os.path import join as pjoin
from warnings import warn

import numpy as np
from scipy import stats
from scipy import ndimage

from . import io
from . import utils
from .io import Surface
from .config import config

lh_viewdict = {'lateral': {'v': (180., 90.), 'r': 90.},
                'medial': {'v': (0., 90.), 'r': -90.},
                'rostral': {'v': (90., 90.), 'r': -180.},
                'caudal': {'v': (270., 90.), 'r': 0.},
                'dorsal': {'v': (180., 0.), 'r': 90.},
                'ventral': {'v': (180., 180.), 'r': 90.},
                'frontal': {'v': (120., 80.), 'r': 106.739},
                'parietal': {'v': (-120., 60.), 'r': 49.106}}
rh_viewdict = {'lateral': {'v': (180., -90.), 'r': -90.},
                'medial': {'v': (0., -90.), 'r': 90.},
                'rostral': {'v': (-90., -90.), 'r': 180.},
                'caudal': {'v': (90., -90.), 'r': 0.},
                'dorsal': {'v': (180., 0.), 'r': 90.},
                'ventral': {'v': (180., 180.), 'r': 90.},
                'frontal': {'v': (60., 80.), 'r': -106.739},
                'parietal': {'v': (-60., 60.), 'r': -49.106}}


class Brain(object):
    """Brain object for visualizing with mlab."""

    def __init__(self, subject_id, hemi, surf,
                 curv=True, title=None, config_opts={}):
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
        title : str
            title for the mayavi figure
        config_opts : dict
            options to override visual options in config file
        """
        try:
            from mayavi import mlab
        except ImportError:
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
        bg_color_code, size = self.__get_scene_properties(config_opts)
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
            colormap, vmin, vmax, reverse = self.__get_geo_colors(config_opts)
            self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                colormap=colormap, vmin=vmin, vmax=vmax)
            if reverse:
                curv_bar = mlab.scalarbar(self._geo_surf)
                curv_bar.reverse_lut = True
                curv_bar.visible = False
        else:
            self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                                   color=(.5, .5, .5))

        # Initialize the overlay and label dictionaries
        self.overlays = dict()
        self.labels = dict()
        self.foci = dict()

        # Bring up the lateral view
        self.show_view(config.get("visual", "default_view"))

        # Turn disable render off so that it displays
        self._f.scene.disable_render = False

    def show_view(self, view=None, roll=None):
        """Orient camera to display view

        Parameters
        ----------
        view : {'lateral' | 'medial' | 'rostral' | 'caudal' |
                'dorsal' | 'ventral' | 'frontal' | 'parietal' |
                dict}
            brain surface to view or kwargs to pass to mlab.view()

        Returns
        -------
        cv: tuple
            tuple returned from mlab.view
        cr: float
            current camera roll

        """
        if isinstance(view, basestring):
            try:
                vd = self.xfm_view(view, 'd')
                view = dict(azimuth=vd['v'][0], elevation=vd['v'][1])
                roll = vd['r']
            except ValueError as v:
                print(v)
                raise
        cv, cr = self.__view(view, roll)
        return (cv, cr)

    def __view(self, viewargs=None, roll=None):
        """Wrapper for mlab.view()

        Parameters
        ----------
        viewargs: dict
            mapping with keys corresponding to mlab.view args
        roll: num
            int or float to set camera roll

        Returns
        -------
        camera settings: tuple
            view settings, roll setting

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        if viewargs:
            viewargs['reset_roll'] = True
            mlab.view(**viewargs)
        if not roll is None:
            mlab.roll(roll)
        return mlab.view(), mlab.roll()

    def add_overlay(self, source, min=None, max=None, sign="abs",
                    name=None, visible=True):
        """Add an overlay to the overlay dict from a file or array.

        Parameters
        ----------
        src : str or numpy array
            path to the overlay file or numpy array with data
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
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        # If source is a string, try to load a file
        if isinstance(source, basestring):
            if name is None:
                basename = os.path.basename(source)
                if basename.endswith(".gz"):
                    basename = basename[:-3]
                if basename.startswith("%s." % self.hemi):
                    basename = basename[3:]
                name = os.path.splitext(basename)[0]
            scalar_data = io.read_scalar_data(source)
        else:
            # Can't think of a good way to check that this will work nicely
            scalar_data = source

        if name in self.overlays:
            "%s%d" % (name, len(self.overlays) + 1)

        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")

        self._f.scene.disable_render = True
        view = mlab.view()
        self.overlays[name] = Overlay(scalar_data, self._geo, min, max, sign)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_data(self, array, min=None, max=None, colormap="blue-red"):
        """Display data from a numpy array on the surface.

        Parameters
        ----------
        array : numpy array
            data array (nvtx vector)
        min : float
            min value in colormap (uses real min if None)
        max : float
            max value in colormap (uses real max if None)
        colormap : str
            name of Mayavi colormap to use

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        self._f.scene.disable_render = True
        view = mlab.view()

        # Possibly remove old data
        if hasattr(self, "data"):
            self.data["surface"].remove()
            self.data["colorbar"].remove()

        if min is None:
            min = array.min()
        if max is None:
            max = array.max()

        # Set up the visualization pipeline
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=array)
        surf = mlab.pipeline.surface(mesh, colormap=colormap,
                                     vmin=min, vmax=max)

        # Get the colorbar
        bar = mlab.scalarbar(surf)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Fil in the data dict
        self.data = dict(surface=surf, colorbar=bar)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_annotation(self, annot, borders=True):
        """Add an annotation file.

        Parameters
        ----------
        annot : str
            Either path to annotation file or annotation name
        borders : bool
            Show only borders of regions

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        self._f.scene.disable_render = True
        view = mlab.view()

        # Figure out where the data is coming from
        if os.path.isfile(annot):
            filepath = annot
            annot = os.path.basename(filepath).split('.')[1]
        else:
            filepath = pjoin(os.environ['SUBJECTS_DIR'],
                             self.subject_id,
                             'label',
                             ".".join([self.hemi, annot, 'annot']))
            if not os.path.exists(filepath):
                raise ValueError('Annotation file %s does not exist'
                                 % filepath)

        # Read in the data
        labels, cmap, _ = io.read_annot(filepath, orig_ids=True)

        # Maybe zero-out the non-border vertices
        if borders:
            n_vertices = labels.size
            edges = utils.mesh_edges(self._geo.faces)
            border_edges = labels[edges.row] != labels[edges.col]
            show = np.zeros(n_vertices, dtype=np.int)
            show[np.unique(edges.row[border_edges])] = 1
            labels *= show

        # Handle null labels properly
        # (tksurfer doesn't use the alpha channel, so sometimes this
        # is set weirdly. For our purposes, it should always be 0.
        # Unless this sometimes causes problems?
        cmap[np.where(cmap[:, 4] == 0), 3] = 0
        if np.any(labels == 0) and not np.any(cmap[:, -1] == 0):
            cmap = np.vstack((cmap, np.zeros(5, int)))

        # Set label ids sensibly
        ord = np.argsort(cmap[:, -1])
        ids = ord[np.searchsorted(cmap[ord, -1], labels)]
        cmap = cmap[:, :4]

        # Maybe get rid of old annot
        if hasattr(self, "annot"):
            self.annot['surface'].remove()

        # Create an mlab surface to visualize the annot
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                   self._geo.y,
                                                   self._geo.z,
                                                   self._geo.faces,
                                                   scalars=ids)
        surf = mlab.pipeline.surface(mesh, name=annot)

        # Set the color table
        surf.module_manager.scalar_lut_manager.lut.table = cmap

        # Set the brain attributes
        self.annot = dict(surface=surf, name=annot, colormap=cmap)

        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_label(self, label, borders=True, color=(76, 169, 117, 255)):
        """Add an ROI label to the image.

        Parameters
        ----------
        label : str
            label filepath or name
        borders : bool
            show only label borders
        color : (float, float, float, float)
            RGBA color tuple

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        self._f.scene.disable_render = True
        view = mlab.view()

        # Figure out where the data is coming from
        if os.path.isfile(label):
            filepath = label
            label_name = os.path.basename(filepath).split('.')[1]
        else:
            label_name = label
            filepath = pjoin(os.environ['SUBJECTS_DIR'],
                             self.subject_id,
                             'label',
                             ".".join([self.hemi, label_name, 'label']))
            if not os.path.exists(filepath):
                raise ValueError('Label file %s does not exist'
                                 % filepath)

        ids = (io.read_label(filepath),)
        label = np.zeros(self._geo.coords.shape[0])
        label[ids] = 1

        if borders:
            n_vertices = label.size
            edges = utils.mesh_edges(self._geo.faces)
            border_edges = label[edges.row] != label[edges.col]
            show = np.zeros(n_vertices, dtype=np.int)
            show[np.unique(edges.row[border_edges])] = 1
            label *= show

        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                   self._geo.y,
                                                   self._geo.z,
                                                   self._geo.faces,
                                                   scalars=label)
        surf = mlab.pipeline.surface(mesh, name=label_name)

        if not isinstance(color, tuple) or len(color) != 4:
            raise TypeError("'color' parameter must be a 4-tuple")
        cmap = np.array([(0, 0, 0, 0,), color])
        surf.module_manager.scalar_lut_manager.lut.table = cmap

        self.labels[label_name] = surf

        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_morphometry(self, measure, grayscale=False):
        """Add a morphometry overlay to the image.

        Parameters
        ----------
        measure : {'area' | 'curv' | 'jacobian_white' | 'sulc' | 'thickness'}
            which measure to load
        grayscale : bool
            whether to load the overlay with a grayscale colormap

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Find the source data
        surf_dir = pjoin(os.environ['SUBJECTS_DIR'], self.subject_id, 'surf')
        morph_file = pjoin(surf_dir, '.'.join([self.hemi, measure]))
        if not os.path.exists(morph_file):
            raise ValueError(
                'Could not find %s in subject directory' % morph_file)

        # Preset colormaps
        cmap_dict = dict(area="pink",
                         curv="RdBu",
                         jacobian_white="pink",
                         sulc="RdBu",
                         thickness="pink")

        self._f.scene.disable_render = True

        # Maybe get rid of an old overlay
        if hasattr(self, "morphometry"):
            self.morphometry['surface'].remove()
            self.morphometry['colorbar'].visible = False

        # Save the inital view
        view = mlab.view()

        # Read in the morphometric data
        morph_data = io.read_morph_data(morph_file)

        # Get a cortex mask for robust range
        self._geo.load_label("cortex")
        ctx_idx = self._geo.labels["cortex"]

        # Get the display range
        if measure == "thickness":
            min, max = 1, 4
        else:
            min, max = stats.describe(morph_data[ctx_idx])[1]

        # Set up the Mayavi pipeline
        if morph_data.dtype.byteorder == '>':
            morph_data.byteswap(True)  # byte swap inplace; due to mayavi bug
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=morph_data)
        if grayscale:
            colormap = "gray"
        else:
            colormap = cmap_dict[measure]
        surf = mlab.pipeline.surface(mesh, colormap=colormap,
                                     vmin=min, vmax=max,
                                     name=measure)

        # Get the colorbar
        bar = mlab.scalarbar(surf)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Fil in the morphometry dict
        self.morphometry = dict(surface=surf,
                                colorbar=bar,
                                measure=measure)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_foci(self, coords, coords_as_verts=False, map_surface=None,
                 scale_factor=1, color=(1, 1, 1), name=None):
        """Add spherical foci, possibly mapping to displayed surf.

        The foci spheres can be displayed at the coordinates given, or
        mapped through a surface geometry. In other words, coordinates
        from a volume-based analysis in MNI space can be displayed on an
        inflated average surface by finding the closest vertex on the
        white surface and mapping to that vertex on the inflated mesh.

        Parameters
        ----------
        coords : numpy array
            x, y, z coordinates in stereotaxic space or array of vertex ids
        coords_as_verts : bool
            whether the coords parameter should be interpreted as vertex ids
        map_surface : Freesurfer surf or None
            surface to map coordinates through, or None to use raw coords
        scale_factor : int
            controls the size of the foci spheres
        color : 3-tuple
            RGB color code for foci spheres
        name : str
            internal name to use

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self._geo.coords[coords]
            map_surface = None

        # Possibly map the foci coords through a surface
        if map_surface is None:
            foci_coords = np.atleast_2d(coords)
        else:
            foci_surf = io.Surface(self.subject_id, self.hemi, map_surface)
            foci_surf.load_geometry()
            foci_vtxs = utils.find_closest_vertices(foci_surf.coords, coords)
            foci_coords = self._geo.coords[foci_vtxs]

        # Get a unique name (maybe should take this approach elsewhere)
        if name is None:
            name = "foci_%d" % (len(self.foci) + 1)

        # Create the visualization
        self._f.scene.disable_render = True
        view = mlab.view()
        points = mlab.points3d(foci_coords[:, 0],
                               foci_coords[:, 1],
                               foci_coords[:, 2],
                               np.ones(foci_coords.shape[0]),
                               scale_factor=(5. * scale_factor),
                               color=color, name=name)
        self.foci[name] = points
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_contour_overlay(self, filepath, min=None, max=None,
                            n_contours=7, line_width=1.5):
        """Add a topographic contour overlay.

        Note: This visualization will look best when using the "low_contrast"
        cortical curvature colorscheme.

        Parameters
        ----------
        filepath : str
            path to the overlay file (must be readable by Nibabel, or .mgh)
        min : float
            threshold for overlay display
        max : float
            saturation point for overlay display
        n_contours : int
            number of contours to use in the display
        line_width : float
            width of contour lines

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Read the scalar data
        scalar_data = io.read_scalar_data(filepath)

        #TODO find a better place for this; duplicates code in Overlay object
        if min is None:
            try:
                min = config.getfloat("overlay", "min_thresh")
            except ValueError:
                min_str = config.get("overlay", "min_thresh")
                if min_str == "robust_min":
                    min = stats.scoreatpercentile(scalar_data, 2)
                elif min_str == "actual_min":
                    min = scalar_data.min()
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
                    max = stats.scoreatpercentile(scalar_data, 98)
                elif max_str == "actual_max":
                    max = scalar_data.max()
                else:
                    max = stats.scoreatpercentile(scalar_data, 98)
                    warn("The 'max_thresh' value in your config value must be "
                "a float, 'robust_min', or 'actual_min', but it is %s. "
                "I'm setting the overlay min to the config default "
                "of robust_max" % max)

        # Prep the viz
        self._f.scene.disable_render = True
        view = mlab.view()

        # Maybe get rid of an old overlay
        if hasattr(self, "contour"):
            self.contour['surface'].remove()
            self.contour['colorbar'].visible = False

        # Deal with Mayavi bug
        if scalar_data.dtype.byteorder == '>':
            scalar_data.byteswap(True)

        # Set up the pipeline
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x, self._geo.y,
                                                  self._geo.z, self._geo.faces,
                                                  scalars=scalar_data)
        thresh = mlab.pipeline.threshold(mesh, low=min)
        surf = mlab.pipeline.contour_surface(thresh, contours=n_contours,
                                             line_width=line_width)

        # Set the colorbar and range correctly
        bar = mlab.scalarbar(surf)
        bar.data_range = min, max
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Set up a dict attribute with pointers at important things
        self.contour = dict(surface=surf, colorbar=bar)

        # Show the new overlay
        mlab.view(*view)
        self._f.scene.disable_render = False

    def __get_scene_properties(self, config_opts):
        """Get the background color and size from the config parser.

        Parameters
        ----------
        config_opts : dict
            dictionary of config file "visual" options

        Returns
        -------
        bg_color_code : (float, float, float)
            RBG code for background color
        size : (float, float)
            viewer window size

        """
        bg_colors = dict(black=[0, 0, 0],
                         white=[256, 256, 256],
                         midnight=[12, 7, 32],
                         slate=[112, 128, 144],
                         charcoal=[59, 69, 79],
                         sand=[245, 222, 179])

        try:
            bg_color_name = config_opts['background']
        except KeyError:
            bg_color_name = config.get("visual", "background")
        bg_color_code = bg_colors[bg_color_name]
        bg_color_code = tuple(map(lambda x: float(x) / 256, bg_color_code))

        try:
            size = config_opts['size']
        except KeyError:
            size = config.getfloat("visual", "size")
        size = (size, size)

        return bg_color_code, size

    def __get_geo_colors(self, config_opts):
        """Return an mlab colormap name, vmin, and vmax for binary curvature.

        Parameters
        ----------
        config_opts : dict
            dictionary of config file "visual" options

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

        try:
            cortex_color = config_opts['cortex']
        except KeyError:
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
        try:
            from mayavi import mlab
        except ImportError:
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

    def save_montage(self, filename, order=['lat', 'ven', 'med'],
                     orientation='h', border_size=15):
        """Create a montage from a given order of images

        Parameters
        ----------
        filename: string
            path to final image
        order: list
            order of views to build montage
        orientation: {'h' | 'v'}
            montage image orientation (horizontal of vertical alignment)
        border_size: int
            Size of image border (more or less space between images)
        """
        assert orientation in ['h', 'v']
        import Image
        fnames = self.save_imageset("tmp", order)
        images = map(Image.open, fnames)
        # get bounding box for cropping
        boxes = []
        for im in images:
            labels, n_labels = ndimage.label(np.array(im)[:, :, 0])
            s = ndimage.find_objects(labels, n_labels)[0]  # slice roi
            # box = (left, top, width, height)
            boxes.append([s[1].start - border_size, s[0].start - border_size,
                          s[1].stop + border_size, s[0].stop + border_size])
        if orientation == 'v':
            min_left = min(box[0] for box in boxes)
            max_width = max(box[2] for box in boxes)
            for box in boxes:
                box[0] = min_left
                box[2] = max_width
        else:
            min_top = min(box[1] for box in boxes)
            max_height = max(box[3] for box in boxes)
            for box in boxes:
                box[1] = min_top
                box[3] = max_height
        # crop images
        cropped_images = []
        for im, box in zip(images, boxes):
            cropped_images.append(im.crop(box))
        images = cropped_images
        # Get full image size
        if orientation == 'h':
            w = sum(i.size[0] for i in images)
            h = max(i.size[1] for i in images)
        else:
            h = sum(i.size[1] for i in images)
            w = max(i.size[0] for i in images)
        new = Image.new("RGBA", (w, h))
        x = 0
        for i in images:
            if orientation == 'h':
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

    def min_diff(self, beg, end):
        """Determine minimum "camera distance" between two views.

        Parameters
        ----------
        beg: string
            origin anatomical view
        end: string
            destination anatomical view

        Returns
        -------
        diffs: tuple
            (min view "distance", min roll "distance")

        """
        beg = self.xfm_view(beg)
        end = self.xfm_view(end)
        if beg == end:
            dv = [360., 0.]
            dr = 0
        else:
            end_d = self.xfm_view(end, 'd')
            beg_d = self.xfm_view(beg, 'd')
            dv = []
            for b, e in zip(beg_d['v'], end_d['v']):
                diff = e - b
                # to minimize the rotation we need -180 <= diff <= 180
                if diff > 180:
                    dv.append(diff - 360)
                elif diff < -180:
                    dv.append(diff + 360)
                else:
                    dv.append(diff)
            dr = np.array(end_d['r']) - np.array(beg_d['r'])
        return (np.array(dv), dr)

    def animate(self, views, n_steps=180., fname=None, use_cache=False):
        """Animate a rotation.

        Currently only rotations through the axial plane are allowed.

        Parameters
        ----------
        views: sequence
            views to animate through
        n_steps: float
            number of steps to take in between
        fname: string
            If not None, it saves the animation as a movie.
            fname should end in '.avi' as only the AVI format is supported
        use_cache: bool
            Use previously generated images in ./.tmp/
        """
        gviews = map(self.xfm_view, views)
        allowed = ('lateral', 'caudal', 'medial', 'rostral')
        if not len([v for v in gviews if v in allowed]) == len(gviews):
            raise ValueError('Animate through %s views.' % ' '.join(allowed))
        if fname is not None:
            if not fname.endswith('.avi'):
                raise ValueError('Can only output to AVI currently.')
            tmp_dir = './.tmp'
            tmp_fname = pjoin(tmp_dir, '%05d.png')
            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)
        for i, beg in enumerate(gviews):
            try:
                end = gviews[i + 1]
                dv, dr = self.min_diff(beg, end)
                dv /= np.array((n_steps))
                dr /= np.array((n_steps))
                self.show_view(beg)
                for i in range(int(n_steps)):
                    self._f.scene.camera.orthogonalize_view_up()
                    self._f.scene.camera.azimuth(dv[0])
                    self._f.scene.camera.elevation(dv[1])
                    self._f.scene.renderer.reset_camera_clipping_range()
                    self._f.scene.render()
                    if fname is not None:
                        if not (os.path.isfile(tmp_fname % i) and use_cache):
                            self.save_image(tmp_fname % i)
            except IndexError:
                pass
        if fname is not None:
            fps = 10
            # we'll probably want some config options here
            enc_cmd = " ".join(["mencoder",
                                "-ovc lavc",
                                "-mf fps=%d" % fps,
                                "mf://%s" % tmp_fname,
                                "-of avi",
                                "-lavcopts vcodec=mjpeg",
                                "-ofps %d" % fps,
                                "-noskip",
                                "-o %s" % fname])
            ret = os.system(enc_cmd)
            if ret:
                print("\n\nError occured when exporting movie\n\n")

    def xfm_view(self, view, out='s'):
        """Normalize a given string to available view

        Parameters
        ----------
        view: string
            view which may match leading substring of available views

        Returns
        -------
        good: string
            matching view string
        out: {'s' | 'd'}
            's' to return string, 'd' to return dict

        """
        if not view in self.viewdict:
            good_view = [k for k in self.viewdict if view == k[:len(view)]]
            if len(good_view) == 0:
                raise ValueError('No views exist with this substring')
            if len(good_view) > 1:
                raise ValueError("Multiple views exist with this substring."
                                 "Try a longer substring")
            view = good_view[0]
        if out == 'd':
            return self.viewdict[view]
        else:
            return view

    def close(self):
        """Close the figure and cleanup data structure."""
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        mlab.close(self._f)
        #should we tear down other variables?


class Overlay(object):

    def __init__(self, scalar_data, geo, min, max, sign):
        """
        """
        try:
            from mayavi import mlab
        except ImportError:
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

    def remove(self):

        if self.sign in ["pos", "abs"]:
            self.pos.remove()
            self.pos_bar.visible = False
        if self.sign in ["neg", "abs"]:
            self.neg.remove()
            self.neg_bar.visible = False

    def __format_colorbar(self):

        if self.sign in ["abs", "neg"]:
            self.neg_bar.scalar_bar_representation.position = (0.05, 0.01)
            self.neg_bar.scalar_bar_representation.position2 = (0.42, 0.09)
        if self.sign in ["abs", "pos"]:
            self.pos_bar.scalar_bar_representation.position = (0.53, 0.01)
            self.pos_bar.scalar_bar_representation.position2 = (0.42, 0.09)
