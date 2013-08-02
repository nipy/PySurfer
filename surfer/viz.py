import os
from os.path import join as pjoin
from warnings import warn

import numpy as np
from scipy import stats, ndimage, misc
from matplotlib.colors import colorConverter

import nibabel as nib

from . import io
from . import utils
from .io import Surface, _get_subjects_dir
from .config import config
from .utils import verbose

import logging
logger = logging.getLogger('surfer')

try:
    from traits.api import (HasTraits, Range, Int, Float,
                            Bool, Enum, on_trait_change, Instance)
except ImportError:
    from enthought.traits.api import (HasTraits, Range, Int, Float,
                                      Bool, Enum, on_trait_change, Instance)

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
viewdicts = dict(lh=lh_viewdict, rh=rh_viewdict)


def make_montage(filename, fnames, orientation='h', colorbar=None,
                 border_size=15):
    """Save montage of current figure

    Parameters
    ----------
    filename : str
        The name of the file, e.g, 'montage.png'. If None, the image
        will not be saved.
    fnames : list of str | list of array
        The images to make the montage of. Can be a list of filenames
        or a list of image data arrays.
    orientation : 'h' | 'v'
        The orientation of the montage: horizontal or vertical
    colorbar : None | list of int
        If None remove colorbars, else keep the ones whose index
        is present.
    border_size : int
        The size of the border to keep.

    Returns
    -------
    out : array
        The montage image data array.
    """
    import Image
    # This line is only necessary to overcome a PIL bug, see:
    #     http://stackoverflow.com/questions/10854903/what-is-causing-
    #          dimension-dependent-attributeerror-in-pil-fromarray-function
    fnames = [f if isinstance(f, basestring) else f.copy() for f in fnames]
    if isinstance(fnames[0], basestring):
        images = map(Image.open, fnames)
    else:
        images = map(Image.fromarray, fnames)
    # get bounding box for cropping
    boxes = []
    for ix, im in enumerate(images):
        # sum the RGB dimension so we do not miss G or B-only pieces
        gray = np.sum(np.array(im), axis=-1)
        gray[gray == gray[0, 0]] = 0  # hack for find_objects that wants 0
        labels, n_labels = ndimage.label(gray.astype(np.float))
        slices = ndimage.find_objects(labels, n_labels)  # slice roi
        if colorbar is not None and ix in colorbar:
            # we need all pieces so let's compose them into single min/max
            slices_a = np.array([[[xy.start, xy.stop] for xy in s]
                                 for s in slices])
            # TODO: ideally gaps could be deduced and cut out with
            #       consideration of border_size
            # so we need mins on 0th and maxs on 1th of 1-nd dimension
            mins = np.min(slices_a[:, :, 0], axis=0)
            maxs = np.max(slices_a[:, :, 1], axis=0)
            s = (slice(mins[0], maxs[0]), slice(mins[1], maxs[1]))
        else:
            # we need just the first piece
            s = slices[0]
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
    if filename is not None:
        try:
            new.save(filename)
        except Exception:
            print("Error saving %s" % filename)
    return np.array(new)


def _prepare_data(data):
    """Ensure data is float64 and has proper endianness.

    Note: this is largely aimed at working around a Mayavi bug.

    """
    data = data.copy()
    data = data.astype(np.float64)
    if data.dtype.byteorder == '>':
        data.byteswap(True)
    return data


def _force_render(figures, backend):
    """Ensure plots are updated before properties are used"""
    try:
        from mayavi import mlab
        assert mlab
    except:
        from enthought.mayavi import mlab
    if not isinstance(figures, list):
        figures = [[figures]]
    for ff in figures:
        for f in ff:
            f.render()
            mlab.draw(figure=f)
    if backend == 'TraitsUI':
        from pyface.api import GUI
        _gui = GUI()
        orig_val = _gui.busy
        _gui.set_busy(busy=True)
        _gui.process_events()
        _gui.set_busy(busy=orig_val)
        _gui.process_events()


def _make_viewer(figure, n_row, n_col, title, scene_size, offscreen):
    """Triage viewer creation

    If n_row == n_col == 1, then we can use a Mayavi figure, which
    generally guarantees that things will be drawn before control
    is returned to the command line. With the multi-view, TraitsUI
    unfortunately has no such support, so we only use it if needed.
    """
    try:
        from mayavi import mlab
        assert mlab
    except:
        from enthought.mayavi import mlab
    if figure is None:
        # spawn scenes
        h, w = scene_size
        if offscreen is True:
            orig_val = mlab.options.offscreen
            mlab.options.offscreen = True
            figures = [[mlab.figure(size=(h / n_row, w / n_col))
                        for _ in xrange(n_col)] for __ in xrange(n_row)]
            mlab.options.offscreen = orig_val
            _v = None
        else:
            # Triage: don't make TraitsUI if we don't have to
            if n_row == 1 and n_col == 1:
                figure = mlab.figure(title, size=(w, h))
                mlab.clf(figure)
                figures = [[figure]]
                _v = None
            else:
                window = _MlabGenerator(n_row, n_col, w, h, title)
                figures, _v = window._get_figs_view()
    else:
        if not isinstance(figure, (list, tuple)):
            figure = [figure]
        if not len(figure) == n_row * n_col:
            raise ValueError('For the requested view, figure must be a '
                             'list or tuple with exactly %i elements, '
                             'not %i' % (n_row * n_col, len(figure)))
        _v = None
        figures = [figure[slice(ri * n_col, (ri + 1) * n_col)]
                   for ri in range(n_row)]
    return figures, _v


class _MlabGenerator(HasTraits):
    """TraitsUI mlab figure generator"""
    try:
        from traitsui.api import View
    except ImportError:
        try:
            from traits.ui.api import View
        except ImportError:
            from enthought.traits.ui.api import View

    view = Instance(View)

    def __init__(self, n_row, n_col, width, height, title, **traits):
        HasTraits.__init__(self, **traits)
        try:
            from mayavi.tools.mlab_scene_model import MlabSceneModel
            assert MlabSceneModel
        except:
            from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel

        self.mlab_names = []
        self.n_row = n_row
        self.n_col = n_col
        self.width = width
        self.height = height
        for fi in xrange(n_row * n_col):
            name = 'mlab_view%03g' % fi
            self.mlab_names.append(name)
            self.add_trait(name, Instance(MlabSceneModel, ()))
        self.view = self._get_gen_view()
        self._v = self.edit_traits(view=self.view)
        self._v.title = title

    def _get_figs_view(self):
        figures = []
        ind = 0
        for ri in range(self.n_row):
            rfigs = []
            for ci in range(self.n_col):
                x = getattr(self, self.mlab_names[ind])
                rfigs.append(x.mayavi_scene)
                ind += 1
            figures.append(rfigs)
        return figures, self._v

    def _get_gen_view(self):
        try:
            from mayavi.core.ui.api import SceneEditor
            from mayavi.core.ui.mayavi_scene import MayaviScene
            assert SceneEditor
            assert MayaviScene
        except:
            from enthought.mayavi.core.ui.api import SceneEditor
            from enthought.mayavi.core.ui.mayavi_scene import MayaviScene
        try:
            from traitsui.api import (View, Item, VGroup, HGroup)
        except ImportError:
            try:
                from traits.ui.api import (View, Item, VGroup, HGroup)
            except ImportError:
                from enthought.traits.ui.api import (View, Item,
                                                     VGroup, HGroup)
        ind = 0
        va = []
        for ri in xrange(self.n_row):
            ha = []
            for ci in xrange(self.n_col):
                ha += [Item(name=self.mlab_names[ind], style='custom',
                            resizable=True, show_label=False,
                            editor=SceneEditor(scene_class=MayaviScene))]
                ind += 1
            va += [HGroup(*ha)]
        view = View(VGroup(*va), resizable=True,
                    height=self.height, width=self.width)
        return view


class Brain(object):
    """Class for visualizing a brain using multiple views in mlab

    Parameters
    ----------
    subject_id : str
        subject name in Freesurfer subjects dir
    hemi : str
        hemisphere id (ie 'lh', 'rh', 'both', or 'split'). In the case
        of 'both', both hemispheres are shown in the same window.
        In the case of 'split' hemispheres are displayed side-by-side
        in different viewing panes.
    surf :  geometry name
        freesurfer surface mesh name (ie 'white', 'inflated', etc.)
    curv : boolean
        if true, loads curv file and displays binary curvature
        (default: True)
    title : str
        title for the window
    config_opts : dict
        options to override visual options in config file
    figure : list of instances of mayavi.core.scene.Scene | None
        If None, a new window will be created with the appropriate
        views.
    subjects_dir : str | None
        If not None, this directory will be used as the subjects directory
        instead of the value set using the SUBJECTS_DIR environment
        variable.
    views : list | str
        views to use
    show_toolbar : bool
        If True, toolbars will be shown for each view.
    offscreen : bool
        If True, rendering will be done offscreen (not shown). Useful
        mostly for generating images or screenshots, but can be buggy.
        Use at your own risk.

    Attributes
    ----------
    brains : list
        List of the underlying brain instances.
    """
    def __init__(self, subject_id, hemi, surf, curv=True, title=None,
                 config_opts={}, figure=None, subjects_dir=None,
                 views=['lat'], show_toolbar=False, offscreen=False):
        col_dict = dict(lh=1, rh=1, both=1, split=2)
        n_col = col_dict[hemi]
        if not hemi in col_dict.keys():
            raise ValueError('hemi must be one of [%s], not %s'
                             % (', '.join(col_dict.keys()), hemi))
        # Get the subjects directory from parameter or env. var
        subjects_dir = _get_subjects_dir(subjects_dir=subjects_dir)

        self._hemi = hemi
        if title is None:
            title = subject_id
        self.subject_id = subject_id

        if not isinstance(views, list):
            views = [views]
        n_row = len(views)

        # load geometry for one or both hemispheres as necessary
        offset = None if hemi != 'both' else 0.0
        self.geo = dict()
        if hemi in ['split', 'both']:
            geo_hemis = ['lh', 'rh']
        elif hemi == 'lh':
            geo_hemis = ['lh']
        elif hemi == 'rh':
            geo_hemis = ['rh']
        else:
            raise ValueError('bad hemi value')
        for h in geo_hemis:
            # Initialize a Surface object as the geometry
            geo = Surface(subject_id, h, surf, subjects_dir, offset)
            # Load in the geometry and (maybe) curvature
            geo.load_geometry()
            if curv:
                geo.load_curvature()
            self.geo[h] = geo

        # deal with making figures
        self._set_window_properties(config_opts)
        figures, _v = _make_viewer(figure, n_row, n_col, title,
                                   self._scene_size, offscreen)
        self._figures = figures
        self._v = _v
        self._window_backend = 'Mayavi' if self._v is None else 'TraitsUI'
        for ff in self._figures:
            for f in ff:
                if f.scene is not None:
                    f.scene.background = self._bg_color
                    f.scene.foreground = self._fg_color
        # force rendering so scene.lights exists
        _force_render(self._figures, self._window_backend)
        self.toggle_toolbars(show_toolbar)
        _force_render(self._figures, self._window_backend)
        self._toggle_render(False)

        # fill figures with brains
        kwargs = dict(surf=surf, curv=curv, title=None,
                      config_opts=config_opts, subjects_dir=subjects_dir,
                      bg_color=self._bg_color, offset=offset)
        brains = []
        brain_matrix = []
        for ri, view in enumerate(views):
            brain_row = []
            for hi, h in enumerate(['lh', 'rh']):
                if not (hemi in ['lh', 'rh'] and h != hemi):
                    ci = hi if hemi == 'split' else 0
                    kwargs['hemi'] = h
                    kwargs['geo'] = self.geo[h]
                    kwargs['figure'] = figures[ri][ci]
                    kwargs['backend'] = self._window_backend
                    brain = _Hemisphere(subject_id, **kwargs)
                    brain.show_view(view)
                    brains += [dict(row=ri, col=ci, brain=brain, hemi=h)]
                    brain_row += [brain]
            brain_matrix += [brain_row]
        self._toggle_render(True)
        self._original_views = views
        self._brain_list = brains
        for brain in self._brain_list:
            brain['brain']._orient_lights()
        self.brains = [b['brain'] for b in brains]
        self.brain_matrix = np.array(brain_matrix)
        self.subjects_dir = subjects_dir
        # Initialize the overlay and label dictionaries
        self.foci_dict = dict()
        self.labels_dict = dict()
        self.overlays_dict = dict()
        self.contour_list = []
        self.morphometry_list = []
        self.annot_list = []
        self.data_dict = dict(lh=None, rh=None)
        # note that texts gets treated differently
        self.texts_dict = dict()
        self.n_times = None

    ###########################################################################
    # HELPERS
    def _toggle_render(self, state, views=None):
        """Turn rendering on (True) or off (False)"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        figs = []
        [figs.extend(f) for f in self._figures]
        if views is None:
            views = [None] * len(figs)
        for vi, (_f, view) in enumerate(zip(figs, views)):
            if state is False and view is None:
                views[vi] = mlab.view(figure=_f)

            # Testing backend doesn't have this option
            if mlab.options.backend != 'test':
                _f.scene.disable_render = not state

            if state is True and view is not None:
                mlab.draw(figure=_f)
                mlab.view(*view, figure=_f)
        # let's do the ugly force draw
        if state is True:
            _force_render(self._figures, self._window_backend)
        return views

    def _set_window_properties(self, config_opts):
        """Set window properties using config_opts"""
        # old option "size" sets both width and height
        if "size" in config_opts:
            width = config_opts["size"]
            height = width

        if "width" in config_opts:
            width = config_opts["width"]
        else:
            width = config.getfloat("visual", "width")
        if 'height' in config_opts:
            height = config_opts['height']
        else:
            height = config.getfloat("visual", "height")
        self._scene_size = (height, width)

        try:
            bg_color_name = config_opts['background']
        except KeyError:
            bg_color_name = config.get("visual", "background")
        if bg_color_name is not None:
            bg_color_code = colorConverter.to_rgb(bg_color_name)
        else:
            bg_color_code = None
        self._bg_color = bg_color_code

        try:
            fg_color_name = config_opts['foreground']
        except KeyError:
            fg_color_name = config.get("visual", "foreground")
        fg_color_code = colorConverter.to_rgb(fg_color_name)
        self._fg_color = fg_color_code

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
        keys = ['fmin', 'fmid', 'fmax', 'transparent', 'time', 'time_idx',
                'smoothing_steps']
        try:
            if self.data_dict['lh'] is not None:
                hemi = 'lh'
            else:
                hemi = 'rh'
            for key in keys:
                props[key] = self.data_dict[hemi][key]
        except KeyError:
            # The user has not added any data
            for key in keys:
                props[key] = 0
        return props

    def toggle_toolbars(self, show=None):
        """Toggle toolbar display

        Parameters
        ----------
        show : bool | None
            If None, the state is toggled. If True, the toolbar will
            be shown, if False, hidden.
        """
        # don't do anything if testing is on
        if self._figures[0][0].scene is not None:
            # this may not work if QT is not the backend (?), or in testing
            if hasattr(self._figures[0][0].scene, 'scene_editor'):
                # Within TraitsUI
                bars = [f.scene.scene_editor._tool_bar
                        for ff in self._figures for f in ff]
            else:
                # Mayavi figure
                bars = [f.scene._tool_bar for ff in self._figures for f in ff]

            if show is None:
                if hasattr(bars[0], 'isVisible'):
                    # QT4
                    show = not bars[0].isVisible()
                else:
                    # WX
                    show = not bars[0].Shown()
            for bar in bars:
                if hasattr(bar, 'setVisible'):
                    bar.setVisible(show)
                else:
                    bar.Show(show)

    def _get_one_brain(self, d, name):
        """Helper for various properties"""
        if len(self.brains) > 1:
            raise ValueError('Cannot access brain.%s when more than '
                             'one view is plotted. Use brain.brain_matrix '
                             'or brain.brains.' % name)
        if isinstance(d, dict):
            out = dict()
            for key, value in d.iteritems():
                out[key] = value[0]
        else:
            out = d[0]
        return out

    @property
    def overlays(self):
        """Wrap to overlays"""
        return self._get_one_brain(self.overlays_dict, 'overlays')

    @property
    def foci(self):
        """Wrap to foci"""
        return self._get_one_brain(self.foci_dict, 'foci')

    @property
    def labels(self):
        """Wrap to labels"""
        return self._get_one_brain(self.labels_dict, 'labels')

    @property
    def contour(self):
        """Wrap to contour"""
        return self._get_one_brain(self.contour_list, 'contour')

    @property
    def annot(self):
        """Wrap to annot"""
        return self._get_one_brain(self.annot_list, 'contour')

    @property
    def texts(self):
        """Wrap to texts"""
        self._get_one_brain([[]], 'texts')
        out = dict()
        for key, val in self.texts_dict.iteritems():
            out[key] = val['text']
        return out

    @property
    def _geo(self):
        """Wrap to _geo"""
        self._get_one_brain([[]], '_geo')
        if self.geo['lh'] is not None:
            return self.geo['lh']
        else:
            return self.geo['rh']

    @property
    def data(self):
        """Wrap to data"""
        self._get_one_brain([[]], 'data')
        if self.data_dict['lh'] is not None:
            data = self.data_dict['lh'].copy()
        else:
            data = self.data_dict['rh'].copy()
        if 'colorbars' in data:
            data['colorbar'] = data['colorbars'][0]
        return data

    def _check_hemi(self, hemi):
        """Check for safe single-hemi input, returns str"""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                raise ValueError('hemi must not be None when both '
                                 'hemispheres are displayed')
            else:
                hemi = self._hemi
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        return hemi

    def _check_hemis(self, hemi):
        """Check for safe dual or single-hemi input, returns list"""
        if hemi is None:
            if self._hemi not in ['lh', 'rh']:
                hemi = ['lh', 'rh']
            else:
                hemi = [self._hemi]
        elif hemi not in ['lh', 'rh']:
            extra = ' or None' if self._hemi in ['lh', 'rh'] else ''
            raise ValueError('hemi must be either "lh" or "rh"' + extra)
        else:
            hemi = [hemi]
        return hemi

    def _read_scalar_data(self, source, hemi, name=None, cast=True):
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
                if basename.startswith("%s." % hemi):
                    basename = basename[3:]
                name = os.path.splitext(basename)[0]
            scalar_data = io.read_scalar_data(source)
        else:
            # Can't think of a good way to check that this will work nicely
            scalar_data = source

        if cast:
            if (scalar_data.dtype.char == 'f'
                    and scalar_data.dtype.itemsize < 8):
                scalar_data = scalar_data.astype(np.float)

        return scalar_data, name

    def _get_display_range(self, scalar_data, min, max, sign):
        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"

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
                         "a float, 'robust_min', or 'actual_min', but it is "
                         "%s. I'm setting the overlay min to the config "
                         "default of 2" % min)

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
                         "a float, 'robust_min', or 'actual_min', but it is "
                         "%s. I'm setting the overlay min to the config "
                         "default of robust_max" % max)

        return min, max

    ###########################################################################
    # ADDING DATA PLOTS
    def add_overlay(self, source, min=None, max=None, sign="abs", name=None,
                    hemi=None):
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
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)
        # load data here
        scalar_data, name = self._read_scalar_data(source, hemi, name=name)
        min, max = self._get_display_range(scalar_data, min, max, sign)
        if not sign in ["abs", "pos", "neg"]:
            raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
        old = OverlayData(scalar_data, self.geo[hemi], min, max, sign)
        ol = []
        views = self._toggle_render(False)
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                ol.append(brain['brain'].add_overlay(old))
        if name in self.overlays_dict:
            name = "%s%d" % (name, len(self.overlays_dict) + 1)
        self.overlays_dict[name] = ol
        self._toggle_render(True, views)

    def add_data(self, array, min=None, max=None, thresh=None,
                 colormap="blue-red", alpha=1,
                 vertices=None, smoothing_steps=20, time=None,
                 time_label="time index=%d", colorbar=True,
                 hemi=None):
        """Display data from a numpy array on the surface.

        This provides a similar interface to add_overlay, but it displays
        it with a single colormap. It offers more flexibility over the
        colormap, and provides a way to display four dimensional data
        (i.e. a timecourse).

        Note that min sets the low end of the colormap, and is separate
        from thresh (this is a different convention from add_overlay)

        Note: If the data is defined for a subset of vertices (specified
        by the "vertices" parameter), a smoothing method is used to interpolate
        the data onto the high resolution surface. If the data is defined for
        subsampled version of the surface, smoothing_steps can be set to None,
        in which case only as many smoothing steps are applied until the whole
        surface is filled with non-zeros.

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
        colormap : str | array [256x4]
            name of Mayavi colormap to use, or a custom look up table (a 256x4
            array, with the columns representing RGBA (red, green, blue, alpha)
            coded with integers going from 0 to 255).
        alpha : float in [0, 1]
            alpha level to control opacity
        vertices : numpy array
            vertices for which the data is defined (needed if len(data) < nvtx)
        smoothing_steps : int or None
            number of smoothing steps (smooting is used if len(data) < nvtx)
            Default : 20
        time : numpy array
            time points in the data array (if data is 2D)
        time_label : str | None
            format of the time label (or None for no label)
        colorbar : bool
            whether to add a colorbar to the figure
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)

        if min is None:
            min = array.min()
        if max is None:
            max = array.max()

        # Create smoothing matrix if necessary
        if len(array) < self.geo[hemi].x.shape[0]:
            if vertices is None:
                raise ValueError("len(data) < nvtx: need vertices")
            adj_mat = utils.mesh_edges(self.geo[hemi].faces)
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

        if smooth_mat is not None:
            array_plot = smooth_mat * array_plot

        # Copy and byteswap to deal with Mayavi bug
        mlab_plot = _prepare_data(array_plot)

        # process colormap argument
        if isinstance(colormap, basestring):
            lut = None
        else:
            lut = np.asarray(colormap)
            if lut.shape != (256, 4):
                err = ("colormap argument must be mayavi colormap (string) or"
                       " look up table (array of shape (256, 4))")
                raise ValueError(err)
            colormap = "blue-red"

        data = dict(array=array, smoothing_steps=smoothing_steps,
                    fmin=min, fmid=(min + max) / 2, fmax=max,
                    transparent=False, time=0, time_idx=0,
                    vertices=vertices, smooth_mat=smooth_mat)

        # Create time array and add label if 2D
        if array.ndim == 2:
            if time is None:
                time = np.arange(array.shape[1])
            self._times = time
            self.n_times = array.shape[1]
            if not self.n_times == len(time):
                raise ValueError('time is not the same length as '
                                 'array.shape[1]')
            data["time_label"] = time_label
            data["time"] = time
            data["time_idx"] = 0
            y_txt = 0.05 + 0.05 * bool(colorbar)
        else:
            self._times = None
            self.n_times = None

        surfs = []
        bars = []
        views = self._toggle_render(False)
        for bi, brain in enumerate(self._brain_list):
            if brain['hemi'] == hemi:
                out = brain['brain'].add_data(array, mlab_plot, vertices,
                                              smooth_mat, min, max, thresh,
                                              lut, colormap, alpha, time,
                                              time_label, colorbar)
                s, ct, bar = out
                surfs.append(s)
                bars.append(bar)
                row, col = np.unravel_index(bi, self.brain_matrix.shape)
                if array.ndim == 2 and time_label is not None:
                    self.add_text(0.05, y_txt, time_label % time[0],
                                  name="time_label", row=row, col=col)
        self._toggle_render(True, views)
        data['surfaces'] = surfs
        data['colorbars'] = bars
        data['orig_ctable'] = ct
        self.data_dict[hemi] = data

    def add_annotation(self, annot, borders=True, alpha=1, hemi=None,
                       remove_existing=True):
        """Add an annotation file.

        Parameters
        ----------
        annot : str
            Either path to annotation file or annotation name
        borders : bool
            Show only borders of regions
        alpha : float in [0, 1]
            Alpha level to control opacity
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, data must exist
            for both hemispheres.
        remove_existing : bool
            If True (default), remove old annotations.
        """
        hemis = self._check_hemis(hemi)

        # Figure out where the data is coming from
        if os.path.isfile(annot):
            filepath = annot
            path = os.path.split(filepath)[0]
            file_hemi, annot = os.path.basename(filepath).split('.')[:2]
            if len(hemis) > 1:
                if annot[:2] == 'lh.':
                    filepaths = [filepath, pjoin(path, 'rh' + annot[2:])]
                elif annot[:2] == 'rh.':
                    filepaths = [pjoin(path, 'lh' + annot[2:], filepath)]
                else:
                    raise RuntimeError('To add both hemispheres '
                                       'simultaneously, filename must '
                                       'begin with "lh." or "rh."')
            else:
                filepaths = [filepath]
        else:
            filepaths = []
            for hemi in hemis:
                filepath = pjoin(self.subjects_dir,
                                 self.subject_id,
                                 'label',
                                 ".".join([hemi, annot, 'annot']))
                if not os.path.exists(filepath):
                    raise ValueError('Annotation file %s does not exist'
                                     % filepath)
                filepaths += [filepath]

        views = self._toggle_render(False)
        if remove_existing is True:
            # Get rid of any old annots
            for a in self.annot_list:
                a['surface'].remove()
            self.annot_list = []

        al = self.annot_list
        for hemi, filepath in zip(hemis, filepaths):
            # Read in the data
            labels, cmap, _ = nib.freesurfer.read_annot(filepath,
                                                        orig_ids=True)

            # Maybe zero-out the non-border vertices
            if borders:
                n_vertices = labels.size
                edges = utils.mesh_edges(self.geo[hemi].faces)
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

            for brain in self._brain_list:
                if brain['hemi'] == hemi:
                    al.append(brain['brain'].add_annotation(annot, ids, cmap))
        self.annot_list = al
        self._toggle_render(True, views)

    def add_label(self, label, color="crimson", alpha=1,
                  scalar_thresh=None, borders=False, hemi=None):
        """Add an ROI label to the image.

        Parameters
        ----------
        label : str | instance of Label
            label filepath or name. Can also be an instance of
            an object with attributes "hemi", "vertices", "name",
            and (if scalar_thresh is not None) "values".
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
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.

        Notes
        -----
        To remove previously added labels, run Brain.remove_labels().
        """
        hemi = self._check_hemi(hemi)
        if isinstance(label, basestring):
            if os.path.isfile(label):
                filepath = label
                label_name = os.path.basename(filepath).split('.')[1]
            else:
                label_name = label
                filepath = pjoin(self.subjects_dir,
                                 self.subject_id,
                                 'label',
                                 ".".join([hemi, label_name, 'label']))
                if not os.path.exists(filepath):
                    raise ValueError('Label file %s does not exist'
                                     % filepath)
            # Load the label data and create binary overlay
            if scalar_thresh is None:
                ids = io.read_label(filepath)
            else:
                ids, scalars = io.read_label(filepath, read_scalars=True)
                ids = ids[scalars >= scalar_thresh]
        else:
            # try to extract parameters from label instance
            try:
                lhemi = label.hemi
                ids = label.vertices
                if label.name is None:
                    label_name = 'unnamed'
                else:
                    label_name = str(label.name)
                if scalar_thresh is not None:
                    scalars = label.values
            except Exception:
                raise ValueError('Label was not a filename (str), and could '
                                 'not be understood as a class. The class '
                                 'must have attributes "hemi", "vertices", '
                                 '"name", and (if scalar_thresh is not None)'
                                 '"values"')
            if not lhemi == hemi:
                raise ValueError('label hemisphere (%s) and brain hemisphere '
                                 '(%s) must match' % (lhemi, hemi))
            if scalar_thresh is not None:
                ids = ids[scalars >= scalar_thresh]

        label = np.zeros(self.geo[hemi].coords.shape[0])
        label[ids] = 1

        # make sure we have a unique name
        if label_name in self.labels_dict:
            i = 2
            name = label_name + '_%i'
            while name % i in self.labels_dict:
                i += 1
            label_name = name % i

        if borders:
            n_vertices = label.size
            edges = utils.mesh_edges(self.geo[hemi].faces)
            border_edges = label[edges.row] != label[edges.col]
            show = np.zeros(n_vertices, dtype=np.int)
            show[np.unique(edges.row[border_edges])] = 1
            label *= show

        # make a list of all the plotted labels
        ll = []
        views = self._toggle_render(False)
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                ll.append(brain['brain'].add_label(label, label_name,
                          color, alpha))
        self.labels_dict[label_name] = ll
        self._toggle_render(True, views)

    def remove_labels(self, labels=None, hemi=None):
        """Remove one or more previously added labels from the image.

        Parameters
        ----------
        labels : None | str | list of str
            Labels to remove. Can be a string naming a single label, or None to
            remove all labels. Possible names can be found in the Brain.labels
            attribute.
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)
        if labels is None:
            labels = self.labels_dict.keys()
        elif isinstance(labels, str):
            labels = [labels]

        for key in labels:
            label = self.labels_dict.pop(key)
            for ll in label:
                ll.remove()

    def add_morphometry(self, measure, grayscale=False, hemi=None,
                        remove_existing=True):
        """Add a morphometry overlay to the image.

        Parameters
        ----------
        measure : {'area' | 'curv' | 'jacobian_white' | 'sulc' | 'thickness'}
            which measure to load
        grayscale : bool
            whether to load the overlay with a grayscale colormap
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, data must exist
            for both hemispheres.
        remove_existing : bool
            If True (default), remove old annotations.
        """
        hemis = self._check_hemis(hemi)
        morph_files = []
        for hemi in hemis:
            # Find the source data
            surf_dir = pjoin(self.subjects_dir, self.subject_id, 'surf')
            morph_file = pjoin(surf_dir, '.'.join([hemi, measure]))
            if not os.path.exists(morph_file):
                raise ValueError(
                    'Could not find %s in subject directory' % morph_file)
            morph_files += [morph_file]

        views = self._toggle_render(False)
        if remove_existing is True:
            # Get rid of any old overlays
            for m in self.morphometry_list:
                m['surface'].remove()
                m['colorbar'].visible = False
            self.morphometry_list = []
        ml = self.morphometry_list
        for hemi, morph_file in zip(hemis, morph_files):
            # Preset colormaps
            cmap_dict = dict(area="pink",
                             curv="RdBu",
                             jacobian_white="pink",
                             sulc="RdBu",
                             thickness="pink")

            # Read in the morphometric data
            morph_data = nib.freesurfer.read_morph_data(morph_file)

            # Get a cortex mask for robust range
            self.geo[hemi].load_label("cortex")
            ctx_idx = self.geo[hemi].labels["cortex"]

            # Get the display range
            if measure == "thickness":
                min, max = 1, 4
            else:
                min, max = stats.describe(morph_data[ctx_idx])[1]

            # Set up the Mayavi pipeline
            morph_data = _prepare_data(morph_data)

            if grayscale:
                colormap = "gray"
            else:
                colormap = cmap_dict[measure]

            for brain in self._brain_list:
                if brain['hemi'] == hemi:
                    ml.append(brain['brain'].add_morphometry(morph_data,
                                                             colormap, measure,
                                                             min, max))
        self.morphometry_list = ml
        self._toggle_render(True, views)

    def add_foci(self, coords, coords_as_verts=False, map_surface=None,
                 scale_factor=1, color="white", alpha=1, name=None,
                 hemi=None):
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
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)

        # Figure out how to interpret the first parameter
        if coords_as_verts:
            coords = self.geo[hemi].coords[coords]
            map_surface = None

        # Possibly map the foci coords through a surface
        if map_surface is None:
            foci_coords = np.atleast_2d(coords)
        else:
            foci_surf = io.Surface(self.subject_id, hemi, map_surface,
                                   subjects_dir=self.subjects_dir)
            foci_surf.load_geometry()
            foci_vtxs = utils.find_closest_vertices(foci_surf.coords, coords)
            foci_coords = self.geo[hemi].coords[foci_vtxs]

        # Get a unique name (maybe should take this approach elsewhere)
        if name is None:
            name = "foci_%d" % (len(self.foci_dict) + 1)

        # Convert the color code
        if not isinstance(color, tuple):
            color = colorConverter.to_rgb(color)

        views = self._toggle_render(False)
        fl = []
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                fl.append(brain['brain'].add_foci(foci_coords, scale_factor,
                                                  color, alpha, name))
        self.foci_dict[name] = fl
        self._toggle_render(True, views)

    def add_contour_overlay(self, source, min=None, max=None,
                            n_contours=7, line_width=1.5, hemi=None):
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
        hemi : str | None
            If None, it is assumed to belong to the hemipshere being
            shown. If two hemispheres are being shown, an error will
            be thrown.
        """
        hemi = self._check_hemi(hemi)

        # Read the scalar data
        scalar_data, _ = self._read_scalar_data(source, hemi)
        min, max = self._get_display_range(scalar_data, min, max, "pos")

        # Deal with Mayavi bug
        scalar_data = _prepare_data(scalar_data)

        # Maybe get rid of an old overlay
        if hasattr(self, "contour"):
            for c in self.contour_list:
                c['surface'].remove()
                c['colorbar'].visible = False

        views = self._toggle_render(False)
        cl = []
        for brain in self._brain_list:
            if brain['hemi'] == hemi:
                cl.append(brain['brain'].add_contour_overlay(scalar_data,
                                                             min, max,
                                                             n_contours,
                                                             line_width))
        self.contour_list = cl
        self._toggle_render(True, views)

    def add_text(self, x, y, text, name, color=None, opacity=1.0,
                 row=-1, col=-1):
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
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        if name in self.texts_dict:
            self.texts_dict[name]['text'].remove()
        text = self.brain_matrix[row, col].add_text(x, y, text,
                                                    name, color, opacity)
        self.texts_dict[name] = dict(row=row, col=col, text=text)

    def update_text(self, text, name, row=-1, col=-1):
        """Update text label

        Parameters
        ----------
        text : str
            New text for label
        name : str
            Name of text label
        """
        if name not in self.texts_dict:
            raise KeyError('text name "%s" unknown' % name)
        self.texts_dict[name]['text'].text = text

    ###########################################################################
    # DATA SCALING / DISPLAY
    def reset_view(self):
        """Orient camera to display original view
        """
        for view, brain in zip(self._original_views, self._brain_list):
            brain['brain'].show_view(view)

    def show_view(self, view=None, roll=None, distance=None, row=-1, col=-1):
        """Orient camera to display view

        Parameters
        ----------
        view : {'lateral' | 'medial' | 'rostral' | 'caudal' |
                'dorsal' | 'ventral' | 'frontal' | 'parietal' |
                dict}
            brain surface to view or kwargs to pass to mlab.view()

        Returns
        -------
        view : tuple
            tuple returned from mlab.view
        roll : float
            camera roll
        distance : float | 'auto' | None
            distance from the origin
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        return self.brain_matrix[row][col].show_view(view, roll, distance)

    def set_distance(self, distance=None):
        """Set view distances for all brain plots to the same value

        Parameters
        ----------
        distance : float | None
            Distance to use. If None, brains are set to the farthest
            "best fit" distance across all current views; note that
            the underlying "best fit" function can be buggy.

        Returns
        -------
        distance : float
            The distance used.
        """
        try:
            from mayavi import mlab
            assert mlab
        except:
            from enthought.mayavi import mlab
        if distance is None:
            distance = []
            for ff in self._figures:
                for f in ff:
                    mlab.view(figure=f, distance='auto')
                    v = mlab.view(figure=f)
                    # This should only happen for the test backend
                    if v is None:
                        v = [0, 0, 100]
                    distance += [v[2]]
            distance = max(distance)

        for ff in self._figures:
            for f in ff:
                mlab.view(distance=distance, figure=f)
        return distance

    @verbose
    def scale_data_colormap(self, fmin, fmid, fmax, transparent, verbose=None):
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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).
        """
        if not (fmin < fmid) and (fmid < fmax):
            raise ValueError("Invalid colormap, we need fmin<fmid<fmax")

        # Cast inputs to float to prevent integer division
        fmin = float(fmin)
        fmid = float(fmid)
        fmax = float(fmax)

        logger.info("colormap: fmin=%0.2e fmid=%0.2e fmax=%0.2e "
                    "transparent=%d" % (fmin, fmid, fmax, transparent))

        # Get the original colormap
        for h in ['lh', 'rh']:
            data = self.data_dict[h]
            if data is not None:
                table = data["orig_ctable"].copy()

        # Add transparency if needed
        if transparent:
            n_colors = table.shape[0]
            n_colors2 = int(n_colors / 2)
            table[:n_colors2, -1] = np.linspace(0, 255, n_colors2)
            table[n_colors2:, -1] = 255 * np.ones(n_colors - n_colors2)

        # Scale the colormap
        table_new = table.copy()
        n_colors = table.shape[0]
        n_colors2 = int(n_colors / 2)

        # Index of fmid in new colorbar
        fmid_idx = int(np.round(n_colors * ((fmid - fmin)
                                            / (fmax - fmin))) - 1)

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

        views = self._toggle_render(False)
        # Use the new colormap
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                for surf in data['surfaces']:
                    cmap = surf.module_manager.scalar_lut_manager
                    cmap.lut.table = table_new
                    cmap.data_range = np.array([fmin, fmax])

                # Update the data properties
                data["fmin"], data['fmid'], data['fmax'] = fmin, fmid, fmax
                data["transparent"] = transparent
        self._toggle_render(True, views)

    def set_data_time_index(self, time_idx):
        """Set the data time index to show

        Parameters
        ----------
        time_idx : int
            time index
        """
        if self.n_times is None:
            raise RuntimeError('cannot set time index with no time data')
        if time_idx < 0 or time_idx >= self.n_times:
            raise ValueError("time index out of range")

        views = self._toggle_render(False)
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                plot_data = data["array"][:, time_idx]
                if data["smooth_mat"] is not None:
                    plot_data = data["smooth_mat"] * plot_data
                for surf in data["surfaces"]:
                    surf.mlab_source.scalars = plot_data
                data["time_idx"] = time_idx

                # Update time label
                if data["time_label"]:
                    time = data["time"][time_idx]
                    self.update_text(data["time_label"] % time, "time_label")
        self._toggle_render(True, views)

    @verbose
    def set_data_smoothing_steps(self, smoothing_steps, verbose=None):
        """Set the number of smoothing steps

        Parameters
        ----------
        smoothing_steps : int
            Number of smoothing steps
        verbose : bool, str, int, or None
            If not None, override default verbose level (see surfer.verbose).
        """
        views = self._toggle_render(False)
        for hemi in ['lh', 'rh']:
            data = self.data_dict[hemi]
            if data is not None:
                adj_mat = utils.mesh_edges(self.geo[hemi].faces)
                smooth_mat = utils.smoothing_matrix(data["vertices"],
                                                    adj_mat, smoothing_steps)
                data["smooth_mat"] = smooth_mat

                # Redraw
                if data["array"].ndim == 1:
                    plot_data = data["array"]
                else:
                    plot_data = data["array"][:, data["time_idx"]]

                plot_data = data["smooth_mat"] * plot_data
                for surf in data["surfaces"]:
                    surf.mlab_source.scalars = plot_data

                # Update data properties
                data["smoothing_steps"] = smoothing_steps
        self._toggle_render(True, views)

    def set_time(self, time):
        """Set the data time index to the time point closest to time

        Parameters
        ----------
        time : scalar
            Time.
        """
        if self.n_times is None:
            raise RuntimeError("Brain has no time axis")
        times = self._times

        # Check that time is in range
        tmin = np.min(times)
        tmax = np.max(times)
        max_diff = (tmax - tmin) / (len(times) - 1) / 2
        if time < tmin - max_diff or time > tmax + max_diff:
            err = ("time = %s lies outside of the time axis "
                   "[%s, %s]" % (time, tmin, tmax))
            raise ValueError(err)

        idx = np.argmin(np.abs(times - time))
        self.set_data_time_index(idx)

    def _get_colorbars(self, row, col):
        shape = self.brain_matrix.shape
        row = row % shape[0]
        col = col % shape[1]
        ind = np.ravel_multi_index((row, col), self.brain_matrix.shape)
        colorbars = []
        h = self._brain_list[ind]['hemi']
        if self.data_dict[h] is not None and 'colorbars' in self.data_dict[h]:
            colorbars.append(self.data_dict[h]['colorbars'][row])
        if len(self.morphometry_list) > 0:
            colorbars.append(self.morphometry_list[ind]['colorbar'])
        if len(self.contour_list) > 0:
            colorbars.append(self.contour_list[ind]['colorbar'])
        if len(self.overlays_dict) > 0:
            for name, obj in self.overlays_dict.items():
                for bar in ["pos_bar", "neg_bar"]:
                    try:
                        colorbars.append(getattr(obj[ind], bar))
                    except AttributeError:
                        pass
        return colorbars

    def _colorbar_visibility(self, visible, row, col):
        for cb in self._get_colorbars(row, col):
            cb.visible = visible

    def show_colorbar(self, row=-1, col=-1):
        """Show colorbar(s) for given plot

        Parameters
        ----------
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        self._colorbar_visibility(True, row, col)

    def hide_colorbar(self, row=-1, col=-1):
        """Hide colorbar(s) for given plot

        Parameters
        ----------
        row : int
            Row index of which brain to use
        col : int
            Column index of which brain to use
        """
        self._colorbar_visibility(False, row, col)

    def close(self):
        """Close all figures and cleanup data structure."""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        for ri, ff in enumerate(self._figures):
            for ci, f in enumerate(ff):
                if f is not None:
                    mlab.close(f)
                    self._figures[ri][ci] = None

        #should we tear down other variables?
        if self._v is not None:
            self._v.dispose()
            self._v = None

    def __del__(self):
        if hasattr(self, '_v') and self._v is not None:
            self._v.dispose()
            self._v = None


    ###########################################################################
    # SAVING OUTPUT
    def save_single_image(self, filename, row=-1, col=-1):
        """Save view from one panel to disk

        Only mayavi image types are supported:
        (png jpg bmp tiff ps eps pdf rib  oogl iv  vrml obj

        Parameters
        ----------
        filename: string
            path to new image file
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

        Due to limitations in TraitsUI, if multiple views or hemi='split'
        is used, there is no guarantee painting of the windows will
        complete before control is returned to the command line. Thus
        we strongly recommend using only one figure window (which uses
        a Mayavi figure to plot instead of TraitsUI) if you intend to
        script plotting commands.
        """
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        brain = self.brain_matrix[row, col]
        ftype = filename[filename.rfind('.') + 1:]
        good_ftypes = ['png', 'jpg', 'bmp', 'tiff', 'ps',
                       'eps', 'pdf', 'rib', 'oogl', 'iv', 'vrml', 'obj']
        if not ftype in good_ftypes:
            raise ValueError("Supported image types are %s"
                             % " ".join(good_ftypes))
        mlab.draw(brain._f)
        mlab.savefig(filename, figure=brain._f)

    def save_image(self, filename):
        """Save view from all panels to disk

        Only mayavi image types are supported:
        (png jpg bmp tiff ps eps pdf rib  oogl iv  vrml obj

        Parameters
        ----------
        filename: string
            path to new image file

        Due to limitations in TraitsUI, if multiple views or hemi='split'
        is used, there is no guarantee painting of the windows will
        complete before control is returned to the command line. Thus
        we strongly recommend using only one figure window (which uses
        a Mayavi figure to plot instead of TraitsUI) if you intend to
        script plotting commands.
        """
        misc.imsave(filename, self.screenshot())

    def screenshot(self, mode='rgb', antialiased=False):
        """Generate a screenshot of current view

        Wraps to mlab.screenshot for ease of use.

        Parameters
        ----------
        mode: string
            Either 'rgb' or 'rgba' for values to return
        antialiased: bool
            Antialias the image (see mlab.screenshot() for details)
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

        Returns
        -------
        screenshot: array
            Image pixel values

        Notes
        -----
        Due to limitations in TraitsUI, if multiple views or hemi='split'
        is used, there is no guarantee painting of the windows will
        complete before control is returned to the command line. Thus
        we strongly recommend using only one figure window (which uses
        a Mayavi figure to plot instead of TraitsUI) if you intend to
        script plotting commands.
        """
        row = []
        for ri in range(self.brain_matrix.shape[0]):
            col = []
            n_col = 2 if self._hemi == 'split' else 1
            for ci in range(n_col):
                col += [self.screenshot_single(mode, antialiased,
                                               ri, ci)]
            row += [np.concatenate(col, axis=1)]
        data = np.concatenate(row, axis=0)
        return data

    def screenshot_single(self, mode='rgb', antialiased=False, row=-1, col=-1):
        """Generate a screenshot of current view from a single panel

        Wraps to mlab.screenshot for ease of use.

        Parameters
        ----------
        mode: string
            Either 'rgb' or 'rgba' for values to return
        antialiased: bool
            Antialias the image (see mlab.screenshot() for details)
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

        Returns
        -------
        screenshot: array
            Image pixel values

        Notes
        -----
        Due to limitations in TraitsUI, if multiple views or hemi='split'
        is used, there is no guarantee painting of the windows will
        complete before control is returned to the command line. Thus
        we strongly recommend using only one figure window (which uses
        a Mayavi figure to plot instead of TraitsUI) if you intend to
        script plotting commands.
        """
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab
        brain = self.brain_matrix[row, col]
        return mlab.screenshot(brain._f, mode, antialiased)

    def save_imageset(self, prefix, views,  filetype='png', colorbar='auto',
                      row=-1, col=-1):
        """Convenience wrapper for save_image

        Files created are prefix+'_$view'+filetype

        Parameters
        ----------
        prefix: string | None
            filename prefix for image to be created. If None, a list of
            arrays representing images is returned (not saved to disk).
        views: list
            desired views for images
        filetype: string
            image type
        colorbar: None | 'auto' | [int], optional
            if None no colorbar is visible. If 'auto' is given the colorbar
            is only shown in the middle view. Otherwise on the listed
            views when a list of int is passed.
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

        Returns
        -------
        images_written: list
            all filenames written
        """
        if isinstance(views, basestring):
            raise ValueError("Views must be a non-string sequence"
                             "Use show_view & save_image for a single view")
        if colorbar == 'auto':
            colorbar = [len(views) // 2]
        images_written = []
        for iview, view in enumerate(views):
            try:
                if colorbar is not None and iview in colorbar:
                    self.show_colorbar(row, col)
                else:
                    self.hide_colorbar(row, col)
                self.show_view(view, row=row, col=col)
                if prefix is not None:
                    fname = "%s_%s.%s" % (prefix, view, filetype)
                    images_written.append(fname)
                    self.save_single_image(fname, row, col)
                else:
                    images_written.append(self.screenshot_single(row=row,
                                                                 col=col))
            except ValueError:
                print("Skipping %s: not in view dict" % view)
        return images_written

    def save_image_sequence(self, time_idx, fname_pattern, use_abs_idx=True,
                            row=-1, col=-1):
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
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

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
            self.save_single_image(fname, row, col)
            images_written.append(fname)
            rel_pos += 1

        # Restore original time index
        self.set_data_time_index(current_time_idx)

        return images_written

    def save_montage(self, filename, order=['lat', 'ven', 'med'],
                     orientation='h', border_size=15, colorbar='auto',
                     row=-1, col=-1):
        """Create a montage from a given order of images

        Parameters
        ----------
        filename: string | None
            path to final image. If None, the image will not be saved.
        order: list
            order of views to build montage
        orientation: {'h' | 'v'}
            montage image orientation (horizontal of vertical alignment)
        border_size: int
            Size of image border (more or less space between images)
        colorbar: None | 'auto' | [int], optional
            if None no colorbar is visible. If 'auto' is given the colorbar
            is only shown in the middle view. Otherwise on the listed
            views when a list of int is passed.
        row : int
            row index of the brain to use
        col : int
            column index of the brain to use

        Returns
        -------
        out : array
            The montage image, useable with matplotlib.imshow().
        """
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        assert orientation in ['h', 'v']
        if colorbar == 'auto':
            colorbar = [len(order) // 2]
        brain = self.brain_matrix[row, col]

        # store current view + colorbar visibility
        current_view = mlab.view(figure=brain._f)
        colorbars = self._get_colorbars(row, col)
        colorbars_visibility = dict()
        for cb in colorbars:
            colorbars_visibility[cb] = cb.visible

        images = self.save_imageset(None, order, colorbar=colorbar,
                                    row=row, col=col)
        out = make_montage(filename, images, orientation, colorbar,
                           border_size)

        # get back original view and colorbars
        mlab.view(*current_view, figure=brain._f)
        for cb in colorbars:
            cb.visible = colorbars_visibility[cb]
        return out

    def animate(self, views, n_steps=180., fname=None, use_cache=False,
                row=-1, col=-1):
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
        row : int
            Row index of the brain to use
        col : int
            Column index of the brain to use
        """
        brain = self.brain_matrix[row, col]
        gviews = map(brain._xfm_view, views)
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
                dv, dr = brain._min_diff(beg, end)
                dv /= np.array((n_steps))
                dr /= np.array((n_steps))
                brain.show_view(beg)
                for i in range(int(n_steps)):
                    brain._f.scene.camera.orthogonalize_view_up()
                    brain._f.scene.camera.azimuth(dv[0])
                    brain._f.scene.camera.elevation(dv[1])
                    brain._f.scene.renderer.reset_camera_clipping_range()
                    _force_render([[brain._f]], self._window_backend)
                    if fname is not None:
                        if not (os.path.isfile(tmp_fname % i) and use_cache):
                            self.save_single_image(tmp_fname % i, row, col)
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


class _Hemisphere(object):
    """Object for visualizing one hemisphere with mlab"""
    def __init__(self, subject_id, hemi, surf, figure, geo, curv, title,
                 config_opts, subjects_dir, bg_color, offset, backend):
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab
        if not hemi in ['lh', 'rh']:
            raise ValueError('hemi must be either "lh" or "rh"')
        # Set the identifying info
        self.subject_id = subject_id
        self.hemi = hemi
        self.subjects_dir = subjects_dir
        self.viewdict = viewdicts[hemi]
        self.surf = surf
        self._f = figure
        self._bg_color = bg_color
        self._backend = backend

        # mlab pipeline mesh and surface for geomtery
        self._geo = geo
        if curv:
            curv_data = self._geo.bin_curv
            meshargs = dict(scalars=curv_data)
            colormap, vmin, vmax, reverse = self._get_geo_colors(config_opts)
            kwargs = dict(colormap=colormap, vmin=vmin, vmax=vmax)
        else:
            curv_data = None
            meshargs = dict()
            kwargs = dict(color=(.5, .5, .5))
        meshargs['figure'] = self._f
        x, y, z, f = self._geo.x, self._geo.y, self._geo.z, self._geo.faces
        self._geo_mesh = mlab.pipeline.triangular_mesh_source(x, y, z, f,
                                                              **meshargs)
        self._geo_surf = mlab.pipeline.surface(self._geo_mesh,
                                               figure=self._f, reset_zoom=True,
                                               **kwargs)
        if curv and reverse:
            curv_bar = mlab.scalarbar(self._geo_surf)
            curv_bar.reverse_lut = True
            curv_bar.visible = False

    def show_view(self, view=None, roll=None, distance=None):
        """Orient camera to display view"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        if isinstance(view, basestring):
            try:
                vd = self._xfm_view(view, 'd')
                view = dict(azimuth=vd['v'][0], elevation=vd['v'][1])
                roll = vd['r']
            except ValueError as v:
                print(v)
                raise

        _force_render(self._f, self._backend)
        if view is not None:
            view['reset_roll'] = True
            view['figure'] = self._f
            view['distance'] = distance
            # DO NOT set focal point, can screw up non-centered brains
            #view['focalpoint'] = (0.0, 0.0, 0.0)
            mlab.view(**view)
        if roll is not None:
            mlab.roll(roll=roll, figure=self._f)
        _force_render(self._f, self._backend)

        view = mlab.view(figure=self._f)
        roll = mlab.roll(figure=self._f)

        return view, roll

    def _xfm_view(self, view, out='s'):
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

    def _min_diff(self, beg, end):
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
        beg = self._xfm_view(beg)
        end = self._xfm_view(end)
        if beg == end:
            dv = [360., 0.]
            dr = 0
        else:
            end_d = self._xfm_view(end, 'd')
            beg_d = self._xfm_view(beg, 'd')
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

    def add_overlay(self, old):
        """Add an overlay to the overlay dict from a file or array"""
        surf = OverlayDisplay(old, figure=self._f)
        for bar in ["pos_bar", "neg_bar"]:
            try:
                self._format_cbar_text(getattr(surf, bar))
            except AttributeError:
                pass
        return surf

    @verbose
    def add_data(self, array, mlab_plot, vertices, smooth_mat, min, max,
                 thresh, lut, colormap, alpha, time, time_label, colorbar):
        """Add data to the brain"""
        # Calculate initial data to plot
        if array.ndim == 1:
            array_plot = array
        elif array.ndim == 2:
            array_plot = array[:, 0]
        else:
            raise ValueError("data has to be 1D or 2D")

        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab
        # Set up the visualization pipeline
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=mlab_plot,
                                                    figure=self._f)
        if thresh is not None:
            if array_plot.min() >= thresh:
                warn("Data min is greater than threshold.")
            else:
                mesh = mlab.pipeline.threshold(mesh, low=thresh)

        surf = mlab.pipeline.surface(mesh, colormap=colormap,
                                     vmin=min, vmax=max,
                                     opacity=float(alpha), figure=self._f)

        # apply look up table if given
        if lut is not None:
            surf.module_manager.scalar_lut_manager.lut.table = lut

        # Get the original colormap table
        orig_ctable = \
            surf.module_manager.scalar_lut_manager.lut.table.to_array().copy()

        # Get the colorbar
        if colorbar:
            bar = mlab.scalarbar(surf)
            self._format_cbar_text(bar)
            bar.scalar_bar_representation.position2 = .8, 0.09
        else:
            bar = None

        return surf, orig_ctable, bar

    def add_annotation(self, annot, ids, cmap):
        """Add an annotation file"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Create an mlab surface to visualize the annot
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=ids,
                                                    figure=self._f)
        surf = mlab.pipeline.surface(mesh, name=annot, figure=self._f)

        # Set the color table
        surf.module_manager.scalar_lut_manager.lut.table = cmap

        # Set the brain attributes
        annot = dict(surface=surf, name=annot, colormap=cmap)
        return annot

    def add_label(self, label, label_name, color, alpha):
        """Add an ROI label to the image"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=label,
                                                    figure=self._f)
        surf = mlab.pipeline.surface(mesh, name=label_name, figure=self._f)
        color = colorConverter.to_rgba(color, alpha)
        cmap = np.array([(0, 0, 0, 0,), color]) * 255
        surf.module_manager.scalar_lut_manager.lut.table = cmap
        return surf

    def add_morphometry(self, morph_data, colormap, measure, min, max):
        """Add a morphometry overlay to the image"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x,
                                                    self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=morph_data,
                                                    figure=self._f)

        surf = mlab.pipeline.surface(mesh, colormap=colormap,
                                     vmin=min, vmax=max,
                                     name=measure, figure=self._f)

        # Get the colorbar
        bar = mlab.scalarbar(surf)
        self._format_cbar_text(bar)
        bar.scalar_bar_representation.position2 = .8, 0.09

        # Fil in the morphometry dict
        return dict(surface=surf, colorbar=bar, measure=measure)

    def add_foci(self, foci_coords, scale_factor, color, alpha, name):
        """Add spherical foci, possibly mapping to displayed surf"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Create the visualization
        points = mlab.points3d(foci_coords[:, 0],
                               foci_coords[:, 1],
                               foci_coords[:, 2],
                               np.ones(foci_coords.shape[0]),
                               scale_factor=(10. * scale_factor),
                               color=color, opacity=alpha, name=name,
                               figure=self._f)
        return points

    def add_contour_overlay(self, scalar_data, min=None, max=None,
                            n_contours=7, line_width=1.5):
        """Add a topographic contour overlay of the positive data"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        # Set up the pipeline
        mesh = mlab.pipeline.triangular_mesh_source(self._geo.x, self._geo.y,
                                                    self._geo.z,
                                                    self._geo.faces,
                                                    scalars=scalar_data,
                                                    figure=self._f)
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
        return dict(surface=surf, colorbar=bar)

    def add_text(self, x, y, text, name, color=None, opacity=1.0):
        """ Add a text to the visualization"""
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab

        return mlab.text(x, y, text, name=name, color=color,
                         opacity=opacity, figure=self._f)

    def _orient_lights(self):
        """Set lights to come from same direction relative to brain."""
        if self.hemi == "rh":
            if self._f.scene is not None and \
                    self._f.scene.light_manager is not None:
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

    def _format_cbar_text(self, cbar):
        bg_color = self._bg_color
        if bg_color is None or sum(bg_color) < 2:
            text_color = (1., 1., 1.)
        else:
            text_color = (0., 0., 0.)
        cbar.label_text_property.color = text_color


class OverlayData(object):
    """Encapsulation of statistical neuroimaging overlay viz data"""

    def __init__(self, scalar_data, geo, min, max, sign):
        if scalar_data.min() >= 0:
            sign = "pos"
        elif scalar_data.max() <= 0:
            sign = "neg"
        self.geo = geo

        if sign in ["abs", "pos"]:
            # Figure out the correct threshold to avoid TraitErrors
            # This seems like not the cleanest way to do this
            pos_max = np.max((0.0, np.max(scalar_data)))
            if pos_max < min:
                thresh_low = pos_max
            else:
                thresh_low = min
            self.pos_lims = [thresh_low, min, max]
        else:
            self.pos_lims = None

        if sign in ["abs", "neg"]:
            # Figure out the correct threshold to avoid TraitErrors
            # This seems even less clean due to negative convolutedness
            neg_min = np.min((0.0, np.min(scalar_data)))
            if neg_min > -min:
                thresh_up = neg_min
            else:
                thresh_up = -min
            self.neg_lims = [thresh_up, -max, -min]
        else:
            self.neg_lims = None
        # Byte swap copy; due to mayavi bug
        self.mlab_data = _prepare_data(scalar_data)


class OverlayDisplay():
    """Encapsulation of overlay viz plotting"""

    def __init__(self, ol, figure):
        try:
            from mayavi import mlab
            assert mlab
        except ImportError:
            from enthought.mayavi import mlab
        args = [ol.geo.x, ol.geo.y, ol.geo.z, ol.geo.faces]
        kwargs = dict(scalars=ol.mlab_data, figure=figure)
        if ol.pos_lims is not None:
            pos_mesh = mlab.pipeline.triangular_mesh_source(*args, **kwargs)
            pos_thresh = mlab.pipeline.threshold(pos_mesh, low=ol.pos_lims[0])
            self.pos = mlab.pipeline.surface(pos_thresh, colormap="YlOrRd",
                                             vmin=ol.pos_lims[1],
                                             vmax=ol.pos_lims[2],
                                             figure=figure)
            self.pos_bar = mlab.scalarbar(self.pos, nb_labels=5)
            self.pos_bar.reverse_lut = True
        else:
            self.pos = None

        if ol.neg_lims is not None:
            neg_mesh = mlab.pipeline.triangular_mesh_source(*args, **kwargs)
            neg_thresh = mlab.pipeline.threshold(neg_mesh,
                                                 up=ol.neg_lims[0])
            self.neg = mlab.pipeline.surface(neg_thresh, colormap="PuBu",
                                             vmin=ol.neg_lims[1],
                                             vmax=ol.neg_lims[2],
                                             figure=figure)
            self.neg_bar = mlab.scalarbar(self.neg, nb_labels=5)
        else:
            self.neg = None
        self._format_colorbar()

    def remove(self):
        if self.pos is not None:
            self.pos.remove()
            self.pos_bar.visible = False
        if self.neg is not None:
            self.neg.remove()
            self.neg_bar.visible = False

    def _format_colorbar(self):
        if self.pos is not None:
            self.pos_bar.scalar_bar_representation.position = (0.53, 0.01)
            self.pos_bar.scalar_bar_representation.position2 = (0.42, 0.09)
        if self.neg is not None:
            self.neg_bar.scalar_bar_representation.position = (0.05, 0.01)
            self.neg_bar.scalar_bar_representation.position2 = (0.42, 0.09)


class TimeViewer(HasTraits):
    """TimeViewer object providing a GUI for visualizing time series

    Useful for visualizing M/EEG inverse solutions on Brain object(s).

    Parameters
    ----------
    brain : Brain (or list of Brain)
        brain(s) to control
    """
     # Nested import of traisui for setup.py without X server
    try:
        from traitsui.api import (View, Item, VSplit, HSplit, Group)
    except ImportError:
        try:
            from traits.ui.api import (View, Item, VSplit, HSplit, Group)
        except ImportError:
            from enthought.traits.ui.api import (View, Item, VSplit,
                                                 HSplit, Group)
    min_time = Int(0)
    max_time = Int(1E9)
    current_time = Range(low="min_time", high="max_time", value=0)
    # colormap: only update when user presses Enter
    fmax = Float(enter_set=True, auto_set=False)
    fmid = Float(enter_set=True, auto_set=False)
    fmin = Float(enter_set=True, auto_set=False)
    transparent = Bool(True)
    smoothing_steps = Int(20, enter_set=True, auto_set=False,
                          desc="number of smoothing steps. Use -1 for"
                               "automatic number of steps")
    orientation = Enum("lateral", "medial", "rostral", "caudal",
                       "dorsal", "ventral", "frontal", "parietal")

    # GUI layout
    view = View(VSplit(Item(name="current_time"),
                       Group(HSplit(Item(name="fmin"),
                                    Item(name="fmid"),
                                    Item(name="fmax"),
                                    Item(name="transparent")
                                    ),
                             label="Color scale",
                             show_border=True),
                       Item(name="smoothing_steps"),
                       Item(name="orientation")
                       )
                )

    def __init__(self, brain):
        super(TimeViewer, self).__init__()

        if isinstance(brain, (list, tuple)):
            self.brains = brain
        else:
            self.brains = [brain]

        # Initialize GUI with values from first brain
        props = self.brains[0].get_data_properties()

        self._disable_updates = True
        self.max_time = len(props["time"]) - 1
        self.current_time = props["time_idx"]
        self.fmin = props["fmin"]
        self.fmid = props["fmid"]
        self.fmax = props["fmax"]
        self.transparent = props["transparent"]
        if props["smoothing_steps"] is None:
            self.smoothing_steps = -1
        else:
            self.smoothing_steps = props["smoothing_steps"]
        self._disable_updates = False

        # Make sure all brains have the same time points
        for brain in self.brains[1:]:
            this_props = brain.get_data_properties()
            if not np.all(props["time"] == this_props["time"]):
                raise ValueError("all brains must have the same time"
                                 "points")

        # Show GUI
        self.configure_traits()

    @on_trait_change("smoothing_steps")
    def set_smoothing_steps(self):
        """ Change number of smooting steps
        """
        if self._disable_updates:
            return

        smoothing_steps = self.smoothing_steps
        if smoothing_steps < 0:
            smoothing_steps = None

        for brain in self.brains:
            brain.set_data_smoothing_steps(self.smoothing_steps)

    @on_trait_change("orientation")
    def set_orientation(self):
        """ Set the orientation
        """
        if self._disable_updates:
            return

        for brain in self.brains:
            brain.show_view(view=self.orientation)

    @on_trait_change("current_time")
    def set_time_point(self):
        """ Set the time point shown
        """
        if self._disable_updates:
            return

        for brain in self.brains:
            brain.set_data_time_index(self.current_time)

    @on_trait_change("fmin, fmid, fmax, transparent")
    def scale_colormap(self):
        """ Scale the colormap
        """
        if self._disable_updates:
            return

        for brain in self.brains:
            brain.scale_data_colormap(self.fmin, self.fmid, self.fmax,
                                      self.transparent)
