from paraview.simple import *
import sys, os

# ============================================================
# CONFIGURACIÓN
# ============================================================

path = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(path, "results", "geometry_2.vtu")
result_name = "top_electrode"
output_image = os.path.join(path, "results", "membrane_geometry.png")

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
display.LineWidth = 1.0

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

view.OrientationAxesVisibility = 0
view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
view.OrientationAxesInteractivity = 0

# ============================================================
# AJUSTAR VISTA
# ============================================================

view.ResetCamera()
view.CameraPosition = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 1, 0]
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

sys.path.insert(0, path)
from orientation_axes import add_orientation_axes

add_orientation_axes(output_image, output_image)
