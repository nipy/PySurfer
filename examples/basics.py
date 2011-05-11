import surfer as sf


sub = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

b = sf.Brain(sub, hemi, surf)

# show all views
b.show_view('lateral')
b.show_view('m')
b.show_view('anter')
b.show_view('post')
b.show_view('dor')
b.show_view('ve')


try:
    b.show_view('bar')
except ValueError as ve:
    print(ve)

#save some images
b.show_view('lat')
b.save_image("%s_lat.png" % sub)

b.save_imageset(sub, ['med', 'lat', 'ant', 'pos'], 'jpg')

b.save_imageset(sub, ['foo'])
