import os
from os.path import join as pjoin
from warnings import warn

import numpy as np
from scipy import stats
from scipy import ndimage
from matplotlib.colors import colorConverter

from . import io
from . import utils
from .io import Surface
from .config import config

try:
    from traits.api import (HasTraits, Range, Int, Float,
                            Bool, Enum, on_trait_change)
except ImportError:
    from enthought.traits.api import (HasTraits, Range, Int, Float, \
                                      Bool, Enum, on_trait_change)

try:
    from traits.ui.api import View, Item, VSplit, HSplit, Group
except ImportError:
    from enthought.traits.ui.api import View, Item, VSplit, HSplit, Group

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

        # Initialize the mlab figure
        if title is None:
            title = subject_id
        self._set_scene_properties(config_opts)
        self._f = mlab.figure(title,
                              **self.scene_properties)
        mlab.clf()
        self._f.scene.disable_render = True

        # Set the lights so they are oriented by hemisphere
        self._orient_lights()

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
            colormap, vmin, vmax, reverse = self._get_geo_colors(config_opts)
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
        self.texts = dict()

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

    def _read_scalar_data(self, source, name=None, cast=True):
        """Load in scalar data from an image stored in a file or an array

        Parameters
        ----------
        source : str or numpy array
            path to scalar data file or a numpy array
        name : str or None, optional
            name for the overlay in the internal dictionary
        cast : bool, optional
            either to cast float data into 64bit datatype as a
            workaround. cast=True can fix a rendering problem with
            certain versions of Mayavi

        Returns
        -------
        scalar_data : numpy array
            flat numpy array of scalar data
        name : str
            if no name was provided, deduces the name if filename was given
            as a source
        """

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

        if cast:
            if (scalar_data.dtype.char == 'f' and
                scalar_data.dtype.itemsize < 8):
                scalar_data = scalar_data.astype(np.float)

        return scalar_data, name

    def add_overlay(self, source, min=None, max=None, sign="abs", name=None):
        """Add an overlay to the overlay dict from a file or array.

        Parameters
        ----------
        source : str or numpy array
            path to the overlay file or numpy array with data
        min : float
            threshold for overlay display
        max : float
            saturation point for overlay display
        sign : {'abs' | 'pos' | 'neg'}
            whether positive, negative, or both values should be displayed
        name : str
            name for the overlay in the internal dictionary

        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        scalar_data, name = self._read_scalar_data(source, name)

        min, max = self._get_display_range(scalar_data, min, max, sign)

        if name in self.overlays:
            "%s%d" % (name, len(self.overlays) + 1)

        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")

        self._f.scene.disable_render = True
        view = mlab.view()
        self.overlays[name] = Overlay(scalar_data, self._geo, min, max, sign)
        for bar in ["pos_bar", "neg_bar"]:
            try:
                self._format_cbar_text(getattr(self.overlays[name], bar))
            except AttributeError:
                pass

        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_data(self, array, min=None, max=None, thresh=None,
                 colormap="blue-red", alpha=1,
                 vertices=None, smoothing_steps=20, time=None,
                 time_label="time index=%d"):
        """Display data from a numpy array on the surface.

        This provides a similar interface to add_overlay, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four dimensional data
        (i.e. a timecourse).

        Note that min sets the low end of the colormap, and is separate
        from thresh (this is a different convention from add_overlay)

        Parameters
        ----------
        array : numpy array
            data array (nvtx vector)
        min : float
            min value in colormap (uses real min if None)
        max : float
            max value in colormap (uses real max if None)
        thresh : None or float
            if not None, values below thresh will not be visible
        colormap : str
            name of Mayavi colormap to use
        alpha : float in [0, 1]
            alpha level to control opacity
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int
            number of smoothing steps (smooting is used if len(data) < nvtx)
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D)
        time_label : str
            format of the time label
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

        # Create smoothing matrix if necessary
        if len(array) < self._geo.x.shape[0]:
            if vertices == None:
                raise ValueError("len(data) < nvtx: need vertices")
            adj_mat = utils.mesh_edges(self._geo.faces)
            smooth_mat = utils.smoothing_matrix(vertices, adj_mat,
                                                smoothing_steps)
        else:
            smooth_mat = None

        # Calculate initial data to plot
        if array.ndim == 1:
            array_plot = array
        elif array.ndim == 2:
            array_plot = array[:, 0]
        else:
            raise ValueError("data has to be 1D or 2D")

        if smooth_mat != None:
            array_plot = smooth_mat * array_plot

        # Copy and byteswap to deal with Mayavi bug
        if array_plot.dtype.byteorder == '>':
            mlab_plot = array_plot.copy()
            mlab_plot.byteswap(True)
        else:
            mlab_plot = array_plot

        # Set up the visualization pipeline
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=mlab_plot)
        if thresh is not None:
            if array_plot.min() >= thresh:
                warn("Data min is greater than threshold.")
            else:
                mesh = mlab.pipeline.threshold(mesh, low=thresh)
        surf = mlab.pipeline.surface(mesh, colormap=colormap,
                                     vmin=min, vmax=max,
                                     opacity=float(alpha))

        # Get the colorbar
        bar = mlab.scalarbar(surf)
        self._format_cbar_text(bar)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Get the original colormap table
        orig_ctable = \
            surf.module_manager.scalar_lut_manager.lut.table.to_array().copy()

        # Fill in the data dict
        self.data = dict(surface=surf, colorbar=bar, orig_ctable=orig_ctable,
                         array=array, smoothing_steps=smoothing_steps,
                         fmin=min, fmid=(min + max) / 2, fmax=max,
                         transparent=False, time=0, time_idx=0)
        if vertices != None:
            self.data["vertices"] = vertices
            self.data["smooth_mat"] = smooth_mat

        mlab.view(*view)

        # Create time array and add label if 2D
        if array.ndim == 2:
            if time == None:
                time = np.arange(array.shape[1])
            self.data["time_label"] = time_label
            self.data["time"] = time
            self.data["time_idx"] = 0
            self.add_text(0.05, 0.1, time_label % time[0], name="time_label")

        self._f.scene.disable_render = False

    def add_annotation(self, annot, borders=True, alpha=1):
        """Add an annotation file.

        Parameters
        ----------
        annot : str
            Either path to annotation file or annotation name
        borders : bool
            Show only borders of regions
        alpha : float in [0, 1]
            Alpha level to control opacity

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

        #  Set the alpha level
        alpha_vec = cmap[:, 3]
        alpha_vec[alpha_vec > 0] = alpha * 255

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

    def add_label(self, label, color="crimson", alpha=1,
                  scalar_thresh=None, borders=False):
        """Add an ROI label to the image.

        Parameters
        ----------
        label : str
            label filepath or name
        color : matplotlib-style color
            anything matplotlib accepts: string, RGB, hex, etc.
        alpha : float in [0, 1]
            alpha level to control opacity
        scalar_thresh : None or number
            threshold the label ids using this value in the label
            file's scalar field (i.e. label only vertices with
            scalar >= thresh)
        borders : bool
            show only label borders

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

        # Load the label data and create binary overlay
        if scalar_thresh is None:
            ids = io.read_label(filepath)
        else:
            ids, scalars = io.read_label(filepath, read_scalars=True)
            ids = ids[scalars >= scalar_thresh]
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

        color = colorConverter.to_rgba(color, alpha)
        cmap = np.array([(0, 0, 0, 0,), color]) * 255
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
        self._format_cbar_text(bar)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Fil in the morphometry dict
        self.morphometry = dict(surface=surf,
                                colorbar=bar,
                                measure=measure)
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_foci(self, coords, coords_as_verts=False, map_surface=None,
                 scale_factor=1, color="white", alpha=1, name=None):
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
        color : matplotlib color code
            HTML name, RBG tuple, or hex code
        alpha : float in [0, 1]
            opacity of focus gylphs
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

        # Convert the color code
        color = colorConverter.to_rgb(color)

        # Create the visualization
        self._f.scene.disable_render = True
        view = mlab.view()
        points = mlab.points3d(foci_coords[:, 0],
                               foci_coords[:, 1],
                               foci_coords[:, 2],
                               np.ones(foci_coords.shape[0]),
                               scale_factor=(10. * scale_factor),
                               color=color, opacity=alpha, name=name)
        self.foci[name] = points
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_contour_overlay(self, source, min=None, max=None,
                            n_contours=7, line_width=1.5):
        """Add a topographic contour overlay of the positive data.

        Note: This visualization will look best when using the "low_contrast"
        cortical curvature colorscheme.

        Parameters
        ----------
        source : str or array
            path to the overlay file or numpy array
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
        scalar_data, _ = self._read_scalar_data(source)

        min, max = self._get_display_range(scalar_data, min, max, "pos")

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
        bar = mlab.scalarbar(surf,
                             nb_colors=n_contours,
                             nb_labels=n_contours + 1)
        bar.data_range = min, max
        self._format_cbar_text(bar)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Set up a dict attribute with pointers at important things
        self.contour = dict(surface=surf, colorbar=bar)

        # Show the new overlay
        mlab.view(*view)
        self._f.scene.disable_render = False

    def add_text(self, x, y, text, name, color=(1, 1, 1), opacity=1.0):
        """ Add a text to the visualization

        Parameters
        ----------
        x : Float
            x coordinate
        y : Float
            y coordinate
        text : str
            Text to add
        name : str
            Name of the text (text label can be updated using update_text())
        color : Tuple
            Color of the text. Default: (1, 1, 1)
        opacity : Float
            Opacity of the text. Default: 1.0
        """
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        text = mlab.text(x, y, text, figure=None, name=name,
                         color=color, opacity=opacity)

        self.texts[name] = text

    def _set_scene_properties(self, config_opts):
        """Set the scene_prop dict from the config parser.

        Parameters
        ----------
        config_opts : dict
            dictionary of config file "visual" options

        """
        colors = dict(black=[0, 0, 0],
                      white=[256, 256, 256],
                      midnight=[12, 7, 32],
                      slate=[112, 128, 144],
                      charcoal=[59, 69, 79],
                      sand=[245, 222, 179])

        try:
            bg_color_name = config_opts['background']
        except KeyError:
            bg_color_name = config.get("visual", "background")
        bg_color_code = colorConverter.to_rgb(bg_color_name)

        try:
            fg_color_name = config_opts['foreground']
        except KeyError:
            fg_color_name = config.get("visual", "foreground")
        fg_color_code = colorConverter.to_rgb(fg_color_name)

        try:
            size = config_opts['size']
        except KeyError:
            size = config.getfloat("visual", "size")
        size = (size, size)

        self.scene_properties = dict(fgcolor=fg_color_code,
                                     bgcolor=bg_color_code,
                                     size=size)

    def _orient_lights(self):
        """Set lights to come from same direction relative to brain."""
        if self.hemi == "rh":
            for light in self._f.scene.light_manager.lights:
                light.azimuth *= -1

    def _get_geo_colors(self, config_opts):
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

    def get_data_properties(self):
        """ Get properties of the data shown

        Returns
        -------
        props : dict
            Dictionary with data properties

            props["fmin"] : minimum colormap
            props["fmid"] : midpoint colormap
            props["fmax"] : maximum colormap
            props["transparent"] : lower part of colormap transparent?
            props["time"] : time points
            props["time_idx"] : current time index
            props["smoothing_steps"] : number of smoothing steps
        """
        props = dict()
        try:
            props["fmin"] = self.data["fmin"]
            props["fmid"] = self.data["fmid"]
            props["fmax"] = self.data["fmax"]
            props["transparent"] = self.data["transparent"]
            props["time"] = self.data["time"]
            props["time_idx"] = self.data["time_idx"]
            props["smoothing_steps"] = self.data["smoothing_steps"]
        except KeyError:
            # The user has not added any data
            props["fmin"] = 0
            props["fmid"] = 0
            props["fmax"] = 0
            props["transparent"] = 0
            props["time"] = 0
            props["time_idx"] = 0
            props["smoothing_steps"] = 0

        return props

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

    def save_image_sequence(self, time_idx, fname_pattern, use_abs_idx=True):
        """Save a temporal image sequence

        The files saved are named "fname_pattern % (pos)" where "pos" is a
        relative or absolute index (controlled by "use_abs_idx")

        Parameters
        ----------
        time_idx : array-like
            time indices to save
        fname_pattern : str
            filename pattern, e.g. 'movie-frame_%0.4d.png'
        use_abs_idx : boolean
            if True the indices given by "time_idx" are used in the filename
            if False the index in the filename starts at zero and is
            incremented by one for each image (Default: True)

        Returns
        -------
        images_written: list
            all filenames written
        """

        current_time_idx = self.data["time_idx"]

        images_written = list()
        rel_pos = 0
        for idx in time_idx:
            self.set_data_time_index(idx)
            fname = fname_pattern % (idx if use_abs_idx else rel_pos)
            self.save_image(fname)
            images_written.append(fname)
            rel_pos += 1

        # Restore original time index
        self.set_data_time_index(current_time_idx)

        return images_written

    def scale_data_colormap(self, fmin, fmid, fmax, transparent):
        """Scale the data colormap.

        Parameters
        ----------
        fmin : float
            minimum value of colormap
        fmid : float
            value corresponding to color midpoint
        fmax : float
            maximum value for colormap
        transparent : boolean
            if True: use a linear transparency between fmin and fmid
        """

        if not (fmin < fmid) and (fmid < fmax):
            raise ValueError("Invalid colormap, we need fmin<fmid<fmax")

        # Cast inputs to float to prevent integer division
        fmin = float(fmin)
        fmid = float(fmid)
        fmax = float(fmax)

        print "colormap: fmin=%0.2e fmid=%0.2e fmax=%0.2e transparent=%d" \
              % (fmin, fmid, fmax, transparent)

        # Get the original colormap
        table = self.data["orig_ctable"].copy()

        # Add transparency if needed
        if  transparent:
            n_colors = table.shape[0]
            n_colors2 = int(n_colors / 2)
            table[:n_colors2, -1] = np.linspace(0, 255, n_colors2)
            table[n_colors2:, -1] = 255 * np.ones(n_colors - n_colors2)

        # Scale the colormap
        table_new = table.copy()
        n_colors = table.shape[0]
        n_colors2 = int(n_colors / 2)

        # Index of fmid in new colorbar
        fmid_idx = np.round(n_colors * ((fmid - fmin) / (fmax - fmin))) - 1

        # Go through channels
        for i in range(4):
            part1 = np.interp(np.linspace(0, n_colors2 - 1, fmid_idx + 1),
                              np.arange(n_colors),
                              table[:, i])
            table_new[:fmid_idx + 1, i] = part1
            part2 = np.interp(np.linspace(n_colors2, n_colors - 1,
                                          n_colors - fmid_idx - 1),
                              np.arange(n_colors),
                              table[:, i])
            table_new[fmid_idx + 1:, i] = part2

        # Get the new colormap
        cmap = self.data["surface"].module_manager.scalar_lut_manager
        cmap.lut.table = table_new
        cmap.data_range = np.array([fmin, fmax])

        # Update the data properties
        self.data["fmin"] = fmin
        self.data["fmid"] = fmid
        self.data["fmax"] = fmax
        self.data["transparent"] = transparent

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

    def set_data_time_index(self, time_idx):
        """ Set the data time index to show

        Parameters
        ----------
        time_idx : int
            time index
        """
        if time_idx < 0 or time_idx >= self.data["array"].shape[1]:
            raise ValueError("time index out of range")

        plot_data = self.data["array"][:, time_idx]

        if "smooth_mat" in self.data:
            plot_data = self.data["smooth_mat"] * plot_data
        self.data["surface"].mlab_source.scalars = plot_data
        self.data["time_idx"] = time_idx

        # Update time label
        self.update_text(self.data["time_label"] % self.data["time"][time_idx],
                         "time_label")

    def set_data_smoothing_steps(self, smoothing_steps):
        """ Set the number of smoothing steps

        Parameters
        ----------
        smoothing_steps : int
            Number of smoothing steps
        """

        adj_mat = utils.mesh_edges(self._geo.faces)
        smooth_mat = utils.smoothing_matrix(self.data["vertices"], adj_mat,
                                            smoothing_steps)

        self.data["smooth_mat"] = smooth_mat

        # Redraw
        if self.data["array"].ndim == 1:
            plot_data = self.data["array"]
        else:
            plot_data = self.data["array"][:, self.data["time_idx"]]

        plot_data = self.data["smooth_mat"] * plot_data

        self.data["surface"].mlab_source.scalars = plot_data

        # Update data properties
        self.data["smoothing_steps"] = smoothing_steps

    def update_text(self, text, name):
        """ Update text label

        Parameters
        ----------
        text : str
            New text for label
        name : str
            Name of text label
        """
        self.texts[name].text = text

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

    def _get_display_range(self, scalar_data, min, max, sign):

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

        # Get the min and max from among various places
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
                    max = stats.scoreatpercentile(scalar_data, 98)
                elif max_str == "actual_max":
                    max = range_data.max()
                else:
                    max = stats.scoreatpercentile(range_data, 98)
                    warn("The 'max_thresh' value in your config value must be "
                "a float, 'robust_min', or 'actual_min', but it is %s. "
                "I'm setting the overlay min to the config default "
                "of robust_max" % max)

        return min, max

    def _format_cbar_text(self, cbar):

        bg_color = self.scene_properties["bgcolor"]
        text_color = (1., 1., 1.) if sum(bg_color) < 2 else (0., 0., 0.)
        cbar.label_text_property.color = text_color


class Overlay(object):
    """Encapsulation of statistical neuroimaging overlay visualization."""

    def __init__(self, scalar_data, geo, min, max, sign):
        try:
            from mayavi import mlab
        except ImportError:
            from enthought.mayavi import mlab

        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"
        self.sign = sign

        # Byte swap copy; due to mayavi bug
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
            pos_bar = mlab.scalarbar(pos_surf, nb_labels=5)
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
            neg_bar = mlab.scalarbar(neg_surf, nb_labels=5)

            self.neg = neg_surf
            self.neg_bar = neg_bar

        self._format_colorbar()

    def remove(self):

        if self.sign in ["pos", "abs"]:
            self.pos.remove()
            self.pos_bar.visible = False
        if self.sign in ["neg", "abs"]:
            self.neg.remove()
            self.neg_bar.visible = False

    def _format_colorbar(self):

        if self.sign in ["abs", "neg"]:
            self.neg_bar.scalar_bar_representation.position = (0.05, 0.01)
            self.neg_bar.scalar_bar_representation.position2 = (0.42, 0.09)
        if self.sign in ["abs", "pos"]:
            self.pos_bar.scalar_bar_representation.position = (0.53, 0.01)
            self.pos_bar.scalar_bar_representation.position2 = (0.42, 0.09)


class TimeViewer(HasTraits):
    """ TimeViewer object providing a GUI for visualizing time series, such
        as M/EEG inverse solutions, on Brain object(s)
    """
    min_time = Int(0)
    max_time = Int(1E9)
    current_time = Range(low="min_time", high="max_time", value=0)
    # colormap: only update when user presses Enter
    fmax = Float(enter_set=True, auto_set=False)
    fmid = Float(enter_set=True, auto_set=False)
    fmin = Float(enter_set=True, auto_set=False)
    transparent = Bool(True)
    smoothing_steps = Int(20, enter_set=True, auto_set=False)
    orientation = Enum("lateral", "medial", "rostral", "caudal",
                       "dorsal", "ventral", "frontal", "parietal")

    # GUI layout
    view = View(VSplit(Item(name="current_time"),
                       Group(HSplit(Item(name="fmin"),
                                    Item(name="fmid"),
                                    Item(name="fmax"),
                                    Item(name="transparent"),
                                   ),
                             label="Color scale",
                             show_border=True
                            ),
                        Item(name="smoothing_steps"),
                        Item(name="orientation")
                      )
                )

    def __init__(self, brain):
        """Initialize TimeViewer

        Parameters
        ----------
        brain : Brain
            brain to control
        """
        super(TimeViewer, self).__init__()

        self.brain = brain

        # Initialize GUI with values from brain
        props = brain.get_data_properties()

        self._disable_updates = True
        self.max_time = len(props["time"]) - 1
        self.current_time = props["time_idx"]
        self.fmin = props["fmin"]
        self.fmid = props["fmid"]
        self.fmax = props["fmax"]
        self.transparent = props["transparent"]
        self.smoothing_steps = props["smoothing_steps"]
        self._disable_updates = False

        # Show GUI
        self.configure_traits()

    @on_trait_change("smoothing_steps")
    def set_smoothing_steps(self):
        """ Change number of smooting steps
        """
        if self._disable_updates:
            return

        self.brain.set_data_smoothing_steps(self.smoothing_steps)

    @on_trait_change("orientation")
    def set_orientation(self):
        """ Set the orientation
        """
        if self._disable_updates:
            return

        self.brain.show_view(view=self.orientation)

    @on_trait_change("current_time")
    def set_time_point(self):
        """ Set the time point shown
        """
        if self._disable_updates:
            return

        self.brain.set_data_time_index(self.current_time)

    @on_trait_change("fmin, fmid, fmax, transparent")
    def scale_colormap(self):
        """ Scale the colormap
        """
        if self._disable_updates:
            return

        self.brain.scale_data_colormap(self.fmin, self.fmid, self.fmax,
                                       self.transparent)
