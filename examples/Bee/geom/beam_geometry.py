#
# /> 'C:\Program Files\Paraview x.x.x\bin\pvpython.exe' beam_geometry.py
#


from paraview.simple import *
import sys, os

# ============================================================
# CONFIGURACIÓN
# ============================================================

path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(path, "geometry_2.vtu")
result_name = "mid"
output_image = os.path.join(path, "beam_geometry.png")

# ============================================================
# CARGA DE GEOMETRÍA
# ============================================================

reader = OpenDataFile(filename)
view = GetActiveViewOrCreate("RenderView")
display = Show(reader, view)

# ============================================================
# VISUALIZACIÓN: Surface With Edges
# ============================================================

display.Representation = "Surface With Edges"
display.EdgeColor = [0.45, 0.45, 0.45]
display.LineWidth = 1.5

# ============================================================
# VISUALIZAR RESULTADO
# ============================================================

ColorBy(display, ("CELLS", result_name))
display.SetScalarBarVisibility(view, False)

# ============================================================
# COLORMAP X Ray
# ============================================================

lut = GetColorTransferFunction(result_name)
lut.ApplyPreset("X Ray", True)
display.RescaleTransferFunctionToDataRange(True, False)

# ============================================================
# ESTILO VISUAL PAPER-LIKE
# ============================================================

view.Background = [1.0, 1.0, 1.0]
view.CameraParallelProjection = 1
view.UseFXAA = 1

view.KeyLightWarmth = 0.5
view.FillLightWarmth = 0.5
view.HeadLightWarmth = 0.5

# ============================================================
# DESACTIVAR ORIENTATION AXES
# ============================================================

view.OrientationAxesVisibility = 1
view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
view.OrientationAxesInteractivity = 0

# ============================================================
# AJUSTAR VISTA
# ============================================================

view.CameraPosition = [-0.6, -1.6, 1.0] 
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 0, 1]
view.ResetCamera()
Render()

# ============================================================
# GUARDAR FIGURA
# ============================================================

SaveScreenshot(
    output_image,
    view,
    ImageResolution=[2400, 1800]
)


# ============================================================
# ACTUALIZA LOS EJES
# ============================================================

# sys.path.insert(0, path)
# from orientation_axes import add_orientation_axes

# add_orientation_axes(output_image, output_image)
