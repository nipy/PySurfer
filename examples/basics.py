import surfer as sf


sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

b = sf.Brain(sub, hemi, surf)

# show all views
b.show_view('lat')
b.show_view('med')
b.show_view('ant')
b.show_view('pos')
b.show_view('sup')
b.show_view('inf')

b.show_view('bar')

#save some images
b.show_view('lat')
b.save_image("%s_lat.png" % sub)

b.save_imageset(sub,['med','lat','ant','pos'],'jpg')

b.save_imageset(sub,['foo'])