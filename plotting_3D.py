import json
import time
import numpy as np
import constants as const
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from data_input import getPath
from boundary_conditions import twoD_gaussian

def plot_3D(scalars):
    nx, ny, nz = const.n_x * 1j, const.n_y * 1j, const.n_z * 1j
    x, y, z = np.mgrid[-5:5:nz, -5:5:ny, -5:5:nx]
    #x, y, z = np.mgrid[-5:5:53j, -5:5:1j, -5:5:1j]

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


def bar_chart_2D(dx, dy, scalars):
    x_positions = np.zeros((const.n_y, const.n_x), dtype=np.float64)
    y_positions = np.zeros((const.n_y, const.n_x), dtype=np.float64)
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            x_positions[j][k] = np.sum([dx[1][j][val] for val in range(0, k)])/2
            y_positions[j][k] = np.sum([dx[1][val][k] for val in range(0, j)])/2
    obj = mlab.barchart(x_positions, y_positions, scalars[1])
    return obj


'''with open(getPath()) as json_file:
    data_vis = json.load(json_file)'''
with open('test.json') as json_file:
    data_vis = json.load(json_file)

sample_and_surface = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
for i in range(0, const.n_z):
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            #sample_and_surface[i][j][k] = np.array(data_vis['SH'][i][j][k])
            sample_and_surface[i][j][k] = np.max(data_vis['HC'][i][j][k])

'''temperature = np.array(data_vis['Temperature'])
for i in range(0, const.n_z):
    for j in range(const.n_y//2 + 1, const.n_y):
        for k in range(const.n_x//2 + 1, const.n_x):
            temperature[i][j][k] = 0
sample = plot_3D(temperature)'''

#sample = plot_3D(np.array(data_vis['Temperature'][-1]))
#print(np.sum([data_vis['Outgassing rate'][b] * const.dt for b in range(len(data_vis['Outgassing rate']))]))
'''for i in range(0, const.n_z):
    if data_vis['Temperature'][len(data_vis['Temperature'])-1][i][const.n_y//2][const.n_x//2] > 0:
        print(i-1)
        break'''
#sample = plot_3D(np.array(data_vis['Water content'][len(data_vis['Water content'])-2]))
#sample = plot_3D(np.array(data_vis['Temperature'][2]))
#sample_and_surface = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
'''sample_and_surface = np.zeros((const.n_z, const.n_y, const.n_x), dtype=np.float64)
#sample_and_surface = np.array(data_vis['gas mass'][1])
#surface = np.array(data_vis['Surface'])
for i in range(0, const.n_z):
    for j in range(0, const.n_y):
        for k in range(0, const.n_x):
            #if data_vis['SH'][i][j][k] == 1:
                #sample_and_surface[i][j][k] = 50
            #if np.sum(data_vis['Surface'][i][j][k]) != 0:
            if data_vis['Surface'][i][j][k][0] != 0 or data_vis['Surface'][i][j][k][1] != 0:
                sample_and_surface[i][j][k] = 1
            #if data_vis['Surface'][3][i][j][k] != 0:
                #sample_and_surface[i][j][k] = 100'''
'''for each in data_vis['SuS']:
    sample_and_surface[each[2]][each[1]][each[0]] = 50'''
#print(data_vis['RSurface'])
'''for each in data_vis['RSurface']:
    sample_and_surface[each[2]][each[1]][each[0]] = 100'''
    #if np.sum(data_vis['Surface'][each[2]][each[1]][each[0]]) != 0:
        #sample_and_surface[each[2]][each[1]][each[0]] = 1
sample = plot_3D(sample_and_surface)
#print(data_vis['gas mass'][3][1][50][50])
#print(data_vis['gas mass'][3][0][50][50])
'''
#sample = bar_chart_2D(data_vis['dx'], data_vis['dy'], data_vis['Lamp Power'])
#sample, x, y = bar_chart_2D(data_vis['dx'], data_vis['dy'], data_vis['Lamp Power'])
#sample = mlab.barchart(np.array(data_vis['Lamp Power'][1])*1000)
#nx, ny, nz = const.n_x * 1j, const.n_y * 1j, const.n_z * 1j
#x, y = np.mgrid[-5:5:ny, -5:5:nx]
#sample_2 = mlab.surf(y, x, twoD_gaussian(y, x, const.var_lamp_profile, 23610.30767673443), warp_scale=1000)
'''
'''sample_and_surface = np.abs(np.array(data_vis['temp_exp']) - np.array(data_vis['MSI_30']))
print(np.array(data_vis['MSI_30'][2][5][2]))
print(np.array(data_vis['temp_exp'][2][5][2]))
#sample_and_surface = np.array(data_vis['MSI_30'])
#sample_and_surface = np.array(data_vis['temp_exp'])
#print(np.array(data_vis['MSI_input'][10]))
print(np.array(data_vis['MSI_input'][2][5][2]))
#sample_and_surface = np.array(data_vis['MSI_input'])
sample = plot_3D(sample_and_surface)
#sample = plot_3D(np.array(data_vis['Temperature'][len(data_vis['Temperature'])-1]))'''

'''Lambda_dat = np.array(data_vis['HC'])
Lambda = np.zeros((const.n_z, const.n_y, const.n_x))
for i in range(1, const.n_z - 1):
    for j in range(1, const.n_y - 1):
        for k in range(1, const.n_x - 1):
            Lambda[i][j][k] = np.max(np.abs(Lambda_dat[i][j][k]))

sample = plot_3D(Lambda)
sample_2 = plot_3D(np.array(data_vis['SH'])*50)'''
#sample = plot_3D(np.array(data_vis['Sample holder']))
'''surface = np.zeros((const.n_z, const.n_y, const.n_z))
for i in range(1, const.n_z - 1):
    for j in range(1, const.n_y - 1):
        for k in range(1, const.n_x - 1):
            surface[i][j][k] = np.max(data_vis['Surface'][i][j][k])
sample = plot_3D(surface)'''

@mlab.animate(delay=10)
def animate():
    for i in range(len(data_vis['Temperature'])):
        time.sleep(2)
        sample.mlab_source.scalars = np.array(data_vis['Temperature'][i])
        yield


@mlab.animate(delay=10)
def animate_rotate():
    while True:
        #time.sleep(0.1)
        mlab.pitch(1)
        yield
#animate_rotate()
mlab.show()

#slice_3D(data_vis['Temperature'][0])

def polygon_3D(polygon_list, file):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    non_normal_1 = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    non_normal_2 = np.array([-0.5, 0.5, 0.5, -0.5], dtype=np.float64)
    normal = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    max_z = 0
    min_z = 0
    for i in tqdm(range(len(polygon_list))):
        if polygon_list[i][1][0] > 0:
            x = polygon_list[i][0][0] + normal
            y = polygon_list[i][0][1] + non_normal_1
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][0] < 0:
            x = polygon_list[i][0][0] - normal
            y = polygon_list[i][0][1] + non_normal_1
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][1] > 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + normal
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][1] < 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] - normal
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][2] > 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + non_normal_2
            z = polygon_list[i][0][2] + normal
        elif polygon_list[i][1][2] < 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + non_normal_2
            z = polygon_list[i][0][2] - normal
        face = Poly3DCollection([list(zip(-x, -y, -z))])
        face.set_color('#999999')
        face.set_alpha(0.4)
        if polygon_list[i][0][2] != 1:
            face.set_color('#000000')
            face.set_alpha(1.0)
        if np.max(np.abs(polygon_list[i][1][0:2])) != 0:
            face.set_color('#5e5e5e')
            face.set_alpha(0.4)
        ax.add_collection3d(face)
        if np.max(-x) > max_x:
            max_x = np.max(-x)
        if np.min(-x) < min_x:
            min_x = np.min(-x)
        if np.max(-y) > max_y:
            max_y = np.max(-y)
        if np.min(-y) < min_y:
            min_y = np.min(-y)
        if np.max(-z) > max_z:
            max_z = np.max(-z)
        if np.min(-z) < min_z:
            min_z = np.min(-z)
    max = np.max([max_x, max_y, max_z, np.abs(min_y), np.abs(min_y), np.abs(min_z)])
    '''ax.set_xlim(-max-0.5, max+0.5)
    ax.set_ylim(-max-0.5, max+0.5)
    ax.set_zlim(-max-0.5, max+0.5)'''
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    ax.set_zlim(min_z - 0.5, max_z + 0.5)
    plt.savefig(file + '.png', dpi=600)
    #plt.show()


def polygon_3D_ray_traced(polygon_list, view_factor_matrix, target_polygon, elev, azim, file):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False, elev=elev, azim=azim)
    fig.add_axes(ax)
    non_normal_1 = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    non_normal_2 = np.array([-0.5, 0.5, 0.5, -0.5], dtype=np.float64)
    normal = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    max_z = 0
    min_z = 0
    view_factor_min = np.infty
    view_factor_max = - np.infty
    for i in range(len(view_factor_matrix[target_polygon])):
        if view_factor_matrix[target_polygon][i] > 0 and view_factor_matrix[target_polygon][i] < view_factor_min:
            view_factor_min = view_factor_matrix[target_polygon][i]
        if view_factor_matrix[target_polygon][i] > 0 and view_factor_matrix[target_polygon][i] > view_factor_max:
            view_factor_max = view_factor_matrix[target_polygon][i]
    for i in tqdm(range(len(polygon_list))):
        if polygon_list[i][1][0] > 0:
            x = polygon_list[i][0][0] + normal
            y = polygon_list[i][0][1] + non_normal_1
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][0] < 0:
            x = polygon_list[i][0][0] - normal
            y = polygon_list[i][0][1] + non_normal_1
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][1] > 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + normal
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][1] < 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] - normal
            z = polygon_list[i][0][2] + non_normal_2
        elif polygon_list[i][1][2] > 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + non_normal_2
            z = polygon_list[i][0][2] + normal
        elif polygon_list[i][1][2] < 0:
            x = polygon_list[i][0][0] + non_normal_1
            y = polygon_list[i][0][1] + non_normal_2
            z = polygon_list[i][0][2] - normal
        face = Poly3DCollection([list(zip(-x, -y, -z))])
        face.set_color('#cfcfcf') #face.set_color('#999999')
        face.set_alpha(0.4)
        #face.set_hatch('+')
        if i == target_polygon:
            face.set_color('#000000')#face.set_color('#ff0000')
            face.set_alpha(1.0)
            #face.set_hatch('')
        if view_factor_matrix[target_polygon][i] == 0:
            ax.add_collection3d(face)
        else:
            #red = hex(int((view_factor_matrix[target_polygon][i] - view_factor_min) / view_factor_max * 191))
            #green = hex(int(-(view_factor_matrix[target_polygon][i] - view_factor_min) / view_factor_max * (226 - 140) + 226))
            green = str(hex(int(-(view_factor_matrix[target_polygon][i] - view_factor_min) / view_factor_max * (100 - 50) + 100)))
            red, blue = green, green
            color = '#000000'
            if len(red) == 3:
                color = '#0' + red[2:3] + '0' + green[2:3] + '0' +blue[2:3]
            else:
                color = '#' + red[2:4] + green[2:4] + blue[2:4]
            face.set_color(color)
            #face.set_hatch('')
            '''if len(red) == 3:
                face.set_color('#0' + str(red)[2:3] + str(green)[2:4] + 'ff' )
            else:
                face.set_color('#' + str(red)[2:4] + str(green)[2:4] + 'ff' )'''
            face.set_alpha(0.4)
            ax.add_collection3d(face)
        if np.max(-x) > max_x:
            max_x = np.max(-x)
        if np.min(-x) < min_x:
            min_x = np.min(-x)
        if np.max(-y) > max_y:
            max_y = np.max(-y)
        if np.min(-y) < min_y:
            min_y = np.min(-y)
        if np.max(-z) > max_z:
            max_z = np.max(-z)
        if np.min(-z) < min_z:
            min_z = np.min(-z)
    max = np.max([max_x, max_y, max_z, np.abs(min_y), np.abs(min_y), np.abs(min_z)])
    '''xs = np.array([polygon_list[240][0][0] + 1/2 * np.sign(polygon_list[240][1][0]), polygon_list[75][0][0] + 1/2 * np.sign(polygon_list[75][1][0])])
    ys = np.array([polygon_list[240][0][1] + 1/2 * np.sign(polygon_list[240][1][1]), polygon_list[75][0][1] + 1/2 * np.sign(polygon_list[75][1][1])])
    zs = np.array([polygon_list[240][0][2] + 1/2 * np.sign(polygon_list[240][1][2]), polygon_list[75][0][2] + 1/2 * np.sign(polygon_list[75][1][2])])
    ax.plot(-xs, -ys, -zs, color='black', lw=2)'''
    '''ax.set_xlim(-max-0.5, max+0.5)
    ax.set_ylim(-max-0.5, max+0.5)
    ax.set_zlim(-max-0.5, max+0.5)'''
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 0.5)
    ax.set_zlim(min_z - 0.5, max_z + 0.5)
    plt.savefig(file + '.png', dpi=600)
    #plt.show()