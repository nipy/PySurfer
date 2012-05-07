try:
    from traits.api import (HasTraits, Range, Int, Float,
                            Bool, Enum, on_trait_change)
except ImportError:
    from enthought.traits.api import (HasTraits, Range, Int, Float, \
                                      Bool, Enum, on_trait_change)

try:
    from traitsui.api import View, Item, VSplit, HSplit, Group
except ImportError:
    try:
        from traits.ui.api import View, Item, VSplit, HSplit, Group
    except ImportError:
        from enthought.traits.ui.api import View, Item, VSplit, HSplit, Group


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
