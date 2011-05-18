from surfer import Brain

sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

brain = Brain(sub, hemi, surf)

# show all views
brain.show_view('lateral')
brain.show_view('m')
brain.show_view('anter')
brain.show_view('post')
brain.show_view('dor')
brain.show_view('ve')


try:
    brain.show_view('bar')
except ValueError as ve:
    print(ve)

#save some images
brain.show_view('lat')
brain.save_image("%s_lat.png" % sub)

brain.save_imageset(sub, ['med', 'lat', 'ant', 'pos'], 'jpg')

# mlab.view() is exposed
brain.show_view({'distance': 375})
# with great power comes great responsibility
brain.show_view({'azimuth':20, 'elevation': 30}, roll=20)

#even rocky had a montage
brain.save_montage('fsaverage_h_montage.png', ['l', 'v', 'm'])
brain.save_montage('fsaverage_v_montage.png', ['l', 'v', 'm'], 'v')
