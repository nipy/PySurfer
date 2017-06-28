:orphan:

.. _install:

Installing and Getting Started
==============================

Installing PySurfer is quite simple with pip_::

    pip install pysurfer

If you already have PySurfer installed, you can also use pip to update it::

    pip install -U pysurfer

If you would like to save movies of time course data, also include the
optional dependency `imageio` with::

    pip install -U pysurfer[save_movie]

If you'd like to install the development version, you have two options. You can
use pip::

    pip install git+git://github.com/nipy/pysurfer.git#egg=pysurfer

Or you can clone the `git repository <https://github.com/nipy/PySurfer>`_ and
install from the source directory::

    python setup.py install

If you don't have pip, you can also install PySurfer with easy_install_, or use
easy_install to get pip.

Dependencies
~~~~~~~~~~~~

PySurfer requires Python 2.7, and it does not work on Python 3.

To use PySurfer, you will need to have the following Python packages:

* numpy_
* scipy_
* ipython_
* nibabel_
* mayavi_
* matplotlib_

Some functions also make use of the Python Imaging Library (PIL_), although
it's not mandatory.

An easy option to set up this environment is the Anaconda_ distribution, which
is free and ships with many of the required packages. If you use Anaconda,
you'll need to install Mayavi separately. This can be done using the ``conda``
command::

    conda install mayavi

You'll also need to install nibabel, which can be done using ``pip`` as above.

Another option for getting set up is the Enthough Canopy_ environment, which is
similar to Anaconda and free for academic use.

Getting started
~~~~~~~~~~~~~~~

PySurfer generally works out of the box on Linux systems. Getting started on
OSX can be a bit more difficult. We have had success using the Anaconda
distribution with the additional step of setting the environment variable
``QT_API`` to ``pyqt``::

    export QT_API=pyqt

If you're using Canopy, you need to do something similar with the
``ETS_TOOLKIT`` variable::

    export ETS_TOOLKIT=qt4

If you want to use PySurfer interactively, you should do so in ipython_. After
starting ipython (either in the terminal, qtconsole, or notebook), you have to
activate the correct GUI backend, which is probably qt::

    %gui qt

This will allow you to have an open PySurfer window while still being able to
execute code in the console/notebook.

If you are having trouble getting started using PySurfer, please describe the problem on the `nipy mailing list`_.

.. include:: links_names.txt

