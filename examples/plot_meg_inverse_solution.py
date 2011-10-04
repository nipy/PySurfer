"""
Plot MEG inverse soltuion computed using mne-python (http://martinos.org/mne)
===========================

Note: In order for this example to work you need mne-python installed
"""

print __doc__

import numpy as np

from surfer import Brain, TimeViewer
from mne.source_estimate import read_stc

"""
define subject, surface and hemisphere
"""
subject_id = 'sample'
surface    = 'inflated'
hemi       = 'lh'

"""
create Brain object for visualization
"""
brain = Brain(subject_id, hemi, surface)

"""
read MNE dSPM inverse solution
"""
stc_fname = './mne_dSPM_inverse-' + hemi + '.stc'
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
                          stc['tmin'] + data.shape[1]*stc['tstep'],
                          data.shape[1])
"""
colormap to use
"""
colormap='hot'

"""
label for time annotation
"""
time_label='time=%0.2f ms'

"""
create viewer
"""
viewer = TimeViewer(brain, data, vertices, time, colormap, time_label)

"""
enable interactive configuration
"""
viewer.configure_traits()