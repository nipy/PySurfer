from io import Surface
from viz import Brain
try:
    from gui import TimeViewer
except SystemError:
    import warnings
    warnings.warn("Could not import surfer/gui module")

__version__ = "0.4.dev"
