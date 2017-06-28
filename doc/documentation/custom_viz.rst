.. _custom_viz:

.. currentmodule:: surfer

Customizing the Visualization
=============================

One advantage to PySurfer over Tksurfer is that you are not
limited to a single look for the visualization. Of course, being
built on Mayavi, PySurfer is in theory completely customizable.
However, we also offer a few preset options so that you do not
have to delve into the underlying engine to get a different look.

Changing the display background
-------------------------------

The display background can take any valid matplotlib color (i.e.,
it can be a tuple of rgb values, an rgb hex string, or a named HTML
color).

Changing the display size
-------------------------

The default display window is 800px by 800px, but this can be configured
using the ``size`` keyword argument in the Brain constructor. ``size``
should either be a single number to make a square window, or a pair of
values, ``(width, height)``, to make a rectangular window.

Changing the curvature color scheme
-----------------------------------

By default, a new :class:`Brain` instance displays the binarized
cortical curvature values, so you can see which patches of cortex
are gyri and which are sulci (pass ``curv=False`` to the
:class:`Brain` constructor, or use the ``-no-curv`` switch in the
command-line interface to turn this off). There are four preset
themes for the curvature color scheme, which you can pass to the
``cortex`` parameter in the :class:`Brain` constructor: ``classic``,
``bone``, ``high_contrast``, and ``low_contrast``:

.. image:: ../_static/cortex_options.png

Note that, in each theme, the darker color signifies sulci.

It's also possible to customize this further by passing the name of
a mayavi colormap or a colormap name along with the endpoints of the
colormap and whether it should be reversed.

Additionally, you can load a continuous curvature map with the
:meth:`Brain.add_morphometry` method.

How to use these themes
-----------------------

These options can be selected either as keyword arguments to the
:class:`Brain` constructor,

.. code-block:: python

    >>> from surfer import Brain
    >>> b = Brain('fsaverage', 'lh', 'inflated', cortex='bone')

or as options in the command-line interface::

.. code-block:: bash

    $ pysurfer fsaverage lh inflated -background slategray -size 400

