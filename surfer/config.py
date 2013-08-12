import ConfigParser
import os
from StringIO import StringIO

homedir = os.environ['HOME']
default_cfg = StringIO("""
[visual]
background = black
foreground = white
height = 800
width = 800
cortex = classic
default_view = lateral

[overlay]
min_thresh = 2.0
max_thresh = robust_max

[options]
logging_level = INFO
""")

config = ConfigParser.ConfigParser()
config.readfp(default_cfg)
config.read([os.path.expanduser('~/.surfer.cfg'), 'surfer.cfg'])
