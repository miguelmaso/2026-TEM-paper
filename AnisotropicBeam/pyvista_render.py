import sys
import vtk ## need to import to allow rendering math. See: https://github.com/pyvista/pyvista/discussions/2928
import pyvista as pv

pl = pv.Plotter()

path = 'results/AnisotropicBeam'

for i in range(21):
    step = 100*i
    filename = path + f'_{step:03d}.vtu'

    args = {
        'show_edges' : False,
    }
    if i <= 10:
        args['color'] = 'lightgray'
    else:
        args['color'] = 'skyblue'

    grid = pv.read(filename)
    grid.warp_by_vector('u', factor=1, inplace=True)

    pl.add_mesh(grid, **args)
    pl.enable_terrain_style()

pl.show(cpos = [(0.15,0.1,0.1), (.05,0,0), (0,0,1)])
