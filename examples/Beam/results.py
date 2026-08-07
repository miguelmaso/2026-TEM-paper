#
# /> 'C:\Program Files\Paraview x.x.x\bin\pvpython.exe' beam_geometry.py
#

import os
from paraview.simple import *

# 1. Configuración de archivos y vista
path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(path, "results", "Beam.pvd")  # O tu archivo de secuencia/serie (.pvd / .vtu)
output_image = os.path.join(path, "beam_deformed.png")

reader = OpenDataFile(filename)
view = GetActiveViewOrCreate("RenderView")

# Estilo visual limpio para publicación
view.Background = [1.0, 1.0, 1.0]
view.CameraParallelProjection = 1
view.UseFXAA = 1

steps_ramp = range(0, 1_000, 100)  # Etapa 1: Incremento de voltaje
steps_relax = range(1_000, 20_000, 5_000)  # Etapa 2: Relajación

dt = 1e-4
steps_ramp  = [i*dt for i in steps_ramp]
steps_relax = [i*dt for i in steps_relax]

# Paletas de color RGB [0, 1]
COLOR_GRAY = [0.85, 0.85, 0.85]  # Gris claro (Incremento)
COLOR_BLUE = [0.35, 0.65, 0.90]  # Azul claro (Relajación)


def add_deformed_step(time_value, color, scale_factor=1.0, opacity=0.6):
    """Congela un timestep específico, aplica deformación por vector 'u' y lo añade a la vista."""
    ft = ForceTime(Input=reader)
    ft.ForcedTime = time_value

    warp = WarpByVector(Input=ft)
    warp.Vectors = ["POINTS", "u"]
    warp.ScaleFactor = scale_factor

    disp = Show(warp, view)
    disp.Representation = "Surface With Edges"
    disp.AmbientColor = color
    disp.DiffuseColor = color
    disp.EdgeColor = [0.3, 0.3, 0.3]
    disp.LineWidth = 1.0
    disp.Opacity = opacity
    return disp


# 3. Superposición de geometrías
# Etapa 1: Incremento de voltaje (Gris)
for t in steps_ramp:
    add_deformed_step(t, COLOR_GRAY, scale_factor=1.0, opacity=0.5)

# Etapa 2: Relajación (Azul)
for t in steps_relax:
    add_deformed_step(t, COLOR_BLUE, scale_factor=1.0, opacity=0.7)

# 4. Renderizado y exportación
view.ResetCamera()
Render()

SaveScreenshot(output_image, view, ImageResolution=[2400, 1500])
