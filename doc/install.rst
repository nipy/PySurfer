:orphan:

.. _install:

Installing and Getting Started
==============================

PySurfer can be installed with pip_. Note that the package name on PyPi is different from the library name that you import::

    pip install pysurfer

If you already have PySurfer installed, you can also use pip to update it::

    pip install -U --no-deps pysurfer

If you would like to save movies of time course data, it is necessary to include the optional dependency ``imageio`` with::

    pip install pysurfer[save_movie]

If you'd like to install the development version, you have two options. You can
install straight from github::

    pip install https://api.github.com/repos/nipy/PySurfer/zipball/master

Or you can clone the `git repository <https://github.com/nipy/PySurfer>`_ and
install from your local source directory::

    pip install .

Dependencies
~~~~~~~~~~~~

PySurfer works on Python 2.7 and 3.6+.
(Older Python 3 versions will probably work, but are not tested.)

To use PySurfer, you will need to have the following Python packages:

* numpy_
* scipy_
* nibabel_
* mayavi_
* matplotlib_

Some input/output functions also make use of the Python Imaging Library (PIL_)
and ``imageio``, although they are not mandatory.

Getting started
~~~~~~~~~~~~~~~

Because PySurfer relies on some complicated dependencies (Mayavi, VTK and a GUI
library), it can be more difficult to get started with than is the case with
other Python libraries. Consider using the Anaconda_ distribution
or Enthough Canopy_ environment. The difficulty on these
platforms is generally getting Mayavi and VTK installed; see their
installation instructions for information.

PySurfer generally works out of the box on Linux systems. Getting started on
OSX may be trickier. We have had success using the Anaconda distribution with
the additional step of setting the environment variables ``QT_API`` and ``ETS_TOOLKIT``, e.g.::

    export QT_API=pyqt
    export ETS_TOOLKIT=qt4

The values you set should match the GUI library you are using.

You may wish to consult the `Mayavi installation docs
<http://docs.enthought.com/mayavi/mayavi/installation.html>`_ if you are having
trouble getting things working.

If you are using PySurfer interactively in IPython_/Jupyter, you should
activate one of the GUI event loops so that the Mayavi window runs in a
separate process. After starting IPython (either in the terminal, qtconsole, or
notebook), you have to activate the correct GUI backend, which is probably qt::

    %gui qt

This will allow you to have an open PySurfer window while still being able to
execute code in the console/notebook.

If you are having trouble getting started using PySurfer, please describe the problem on the `nipy mailing list`_.

.. include:: links_names.txt

