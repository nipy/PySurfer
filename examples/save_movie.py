"""
Create movie from  MEG inverse solution
=======================================

Data were computed using mne-python (http://martinos.org/mne)

"""
print __doc__

import os
import numpy as np

from surfer import Brain
from surfer.io import read_stc

"""
create Brain object for visualization
"""
brain = Brain('fsaverage', 'split', 'inflated',
              config_opts=dict(width=800, height=400))

"""
read MNE dSPM inverse solution
"""
for hemi in ['lh', 'rh']:
    stc_fname = os.path.join('example_data',
                             'meg_source_estimate-' + hemi + '.stc')
    stc = read_stc(stc_fname)
    data = stc['data']

    """
    time points in milliseconds
    """
    time = 1e3 * np.linspace(stc['tmin'],
                             stc['tmin'] + data.shape[1] * stc['tstep'],
                             data.shape[1])

    brain.add_data(data, colormap='hot', vertices=stc['vertices'],
                   smoothing_steps=10, time=time, time_label='time=%0.2f ms',
                   hemi=hemi)

"""
scale colormap
"""
brain.scale_data_colormap(fmin=13, fmid=18, fmax=22, transparent=True)

"""
Save movies with different combinations of views
"""
brain.save_movie('example_current.mov')
brain.save_movie('example_single.mov', montage='single')
brain.save_movie('example_h.mov', montage=['lat', 'med'], orientation='h')
brain.save_movie('example_v.mov', montage=[['lat'], ['med']])

brain.close()
