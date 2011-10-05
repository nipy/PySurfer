"""
Plot MEG inverse solution
=========================

Data were computed using mne-python (http://martinos.org/mne)

"""
print __doc__

import os
import numpy as np

from surfer import Brain, TimeViewer
from surfer.io import read_stc

"""
define subject, surface and hemisphere
"""
subject_id, surface, hemi = 'fsaverage', 'inflated', 'lh'

"""
create Brain object for visualization
"""
brain = Brain(subject_id, hemi, surface)

"""
read MNE dSPM inverse solution
"""
stc_fname = os.path.join('auto_examples', 'data',
                         'meg_source_estimate-' + hemi + '.stc')
stc = read_stc(stc_fname)

"""
data and vertices for which the data is defined
"""
data = stc['data']
vertices = stc['vertices']

"""
time points in milliseconds
"""
time = 1000 * np.linspace(stc['tmin'],
                          stc['tmin'] + data.shape[1] * stc['tstep'],
                          data.shape[1])
"""
colormap to use
"""
colormap = 'hot'

"""
label for time annotation
"""
time_label = 'time=%0.2f ms'

"""
create viewer
"""
viewer = TimeViewer(brain, data, vertices, time, colormap, time_label)

"""
set minimum of colormap and time (index) to display
"""
viewer.current_time = 8
viewer.fthresh = 4

"""
uncomment this line to enable interactive configuration using GUI
"""
#viewer.configure_traits()
