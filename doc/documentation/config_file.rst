.. _config_file:

The Config File
===============

There are several aspects of how PySurfer looks and behaves that you may
wish to customize but do not want to have to alter every time you
instantiate a Brain object or use the command-line interface. To
facilitate this, PySurfer allows you to set certain options with a
standard Python config file. 

When a new Brain object is created (either in a script or via the
command-line), it can read configuration options from one of two places:
first, in the local folder in a file called ``surfer.cfg``, and next in
your home directory, from a file called ``.surfer.cfg``. The file is
divided into the following sections:

Visual
------
*background*
    What color the display background should be. (Possible values:
    any HTML color name or 256 color hex code.

*cortex*
    What colormap should be used for the binarized curvature on the
    cortex. (Possible values: any cortical curvature color scheme name.)

*size*
    How large the sides of the display window should be (measured in
    pixels.) The window is always square, so just give one value, and 
    it will be used for the height and width. (Possible values: any
    positive number.)

*default_view*
    Which view should be shown at the beginning of a visualization
    session. (Possible values: ``lateral``, ``medial``, ``rostral``,
    ``caudal``, ``dorsal``, ``ventral``, ``frontal``, ``parietal``.)


Overlay
-------
*min_thresh*
    What the default minimum threshold should be if not provided when
    calling the :meth:`Brain.add_overlay` method. (Possible values: any
    float, ``robust_min``, ``actual_min``.)

*max_thresh*
    What the default saturation point of the colormap should be if not
    provided when calling the :meth:`Brain.add_overlay` method.
    (Possible values: any float, ``robust_max``, ``actual_max``.)

