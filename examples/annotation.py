import os
from os.path import join as pjoin
from surfer import io
from surfer import viz

subj_dir = os.environ["SUBJECTS_DIR"]
subject_id = 'fsaverage'

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

data_path = pjoin(subj_dir, subject_id)
annot_path = pjoin(data_path, "label", "%s.aparc.annot" % "lh")

brain = viz.Brain(sub, hemi, surf)
brain.add_annotation(annot_path, borders=True)
# show all views
brain.show_view('lateral')

