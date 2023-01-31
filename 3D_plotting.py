import json
import time
import numpy as np
import constants as const
from mayavi import mlab

def plot_3D(scalars):
    x, y, z = np.mgrid[-5:5:51j, -5:5:51j, -5:5:51j]

    print(x.shape, y.shape, z.shape, scalars.shape)

    #obj = mlab.volume_slice(scalars, plane_orientation='x_axes')
    #obj = mlab.volume_slice(x, y, z, scalars, plane_orientation='x_axes')
    obj = mlab.points3d(x, y, z, scalars, mode='cube')
    return obj

def slice_3D(scalars):
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(scalars),
                                     plane_orientation='x_axes',
                                     slice_index=10,
                                     )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(scalars),
                                     plane_orientation='y_axes',
                                     slice_index=10,
                                     )
    mlab.outline()
    mlab.show()


with open('test_gran.json') as json_file:
    data_vis = json.load(json_file)

'''#sample = plot_3D(np.array(data_vis['Temperature']))
sample_and_surface = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
surface = np.array(data_vis['Surface'])
for i in range(0, const.n_z):
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            if np.sum(surface[i][j][k]) > 0:
                sample_and_surface[i][j][k] = 50

for each in data_vis['RSurface']:
    sample_and_surface[each[2]][each[1]][each[0]] = 70
    
sample = plot_3D(sample_and_surface)
'''

sample = plot_3D(np.array(data_vis['Temperature'][len(data_vis['Temperature'])-1]))

@mlab.animate(delay=10)
def animate():
    for i in range(len(data_vis['Temperature'])):
        time.sleep(2)
        sample.mlab_source.scalars = np.array(data_vis['Temperature'][i])
        yield

#animate()
mlab.show()

#slice_3D(data_vis['Temperature'][0])