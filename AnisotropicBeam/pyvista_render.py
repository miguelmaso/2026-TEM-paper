import sys
import vtk ## need to import to allow rendering math. See: https://github.com/pyvista/pyvista/discussions/2928
import pyvista as pv

pl = pv.Plotter()

path = 'results - order 1 - 10 divisions/AnisotropicBeam'

for step in range(0, 1000, 50):
    filename = path + f'_{step:03d}.vtu'

    if step <= 500:
        color = 'lightgray'  # 'lightsteelblue'
    else:
        color = 'skyblue'  # 'deepskyblue'

    grid = pv.read(filename)
    grid.warp_by_vector('u', factor=1, inplace=True)

    surface = grid.extract_surface()
    surface.clean(tolerance=1e-8, inplace=True)
    surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        feature_angle=180.0,
        inplace=True
    )

    pl.add_mesh(
        surface,
        color=color,
        show_edges=False,
        smooth_shading=False,
        ambient=0.35,
        diffuse=0.55,
        specular=0.15,
        specular_power=25,
    )
    pl.enable_terrain_style()

pl.show(cpos = [(0.15,0.1,0.1), (.05,0,0), (0,0,1)])
