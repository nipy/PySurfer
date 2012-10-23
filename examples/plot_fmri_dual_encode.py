"""
Display Dual-Encoded Data
=========================

Plot data such that both effect size and reliability are independently
represented, with effect size encoded with color and reliability encoded in
data opacity.

"""
print __doc__

from surfer import Brain, io

import numpy as np
from scipy.stats import scoreatpercentile

# Read in the effect size and variance data
eff = io.read_scalar_data("example_data/rh.gamma.mgh")
var = io.read_scalar_data("example_data/rh.gammavar.mgh")
t = np.abs(eff / var)
t[var == 0] = 0

# Set the min and max values to be equal
eff_range = eff.min(), eff.max()
range_end = .75 * max(map(abs, eff_range))
off_range = eff.min() - 1

# Open up the visualization
b = Brain("fsaverage", "rh", "inflated",
          config_opts=dict(cortex="low_contrast"))

# Set up the percentile bins
n_bins = 25
bins = np.linspace(0, 100, n_bins)

# Iterate through the bins and add data with progressively richer color
for percentile in bins[1:-1]:
    sub_thresh_mask = t < scoreatpercentile(t, percentile)
    eff_thresh = eff.copy()
    eff_thresh[sub_thresh_mask] = off_range - 1
    b.add_data(eff_thresh,
               min=-range_end, max=range_end,
               thresh=off_range,
               alpha=1. / (n_bins - 2),
               keep_existing=True)

for data_dict in b.data[:-1]:
    data_dict["colorbar"].visible = False
