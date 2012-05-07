from io import Surface
from viz import Brain
try:
    from gui import TimeViewer
except SystemError as e:
    import warnings
    warnings.warn("Could not import surfer.gui module: %s" % e)

__version__ = "0.4.dev"
