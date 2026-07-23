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
display.Ambient = 0.4
display.Diffuse = 0.6

# ============================================================
# VISUALIZAR RESULTADO
# ============================================================

# ColorBy(display, ("CELLS", result_name))
# display.SetScalarBarVisibility(view, False)

# ============================================================
# COLORMAP X Ray
# ============================================================

# lut = GetColorTransferFunction(result_name)
# lut.ApplyPreset("X Ray", True)
# display.RescaleTransferFunctionToDataRange(True, False)

# ============================================================
# RESALTAR ELECTRODOS
# ============================================================

L = 0.025
W = 0.003
T = 0.0005

def add_technical_tube(p1, p2, radius=0.00002, color=[0.12, 0.12, 0.12]):
    """Crea un tubo geométrico 3D entre dos puntos con un color gris oscuro/negro."""
    line = Line()
    line.Point1 = p1
    line.Point2 = p2
    
    tube = Tube(Input=line)
    tube.NumberofSides = 12
    tube.Radius = radius
    
    disp = Show(tube, view)
    disp.Representation = "Surface"
    disp.AmbientColor = color
    disp.DiffuseColor = color
    # Un toque sutil de brillo especular define mejor la curvatura del tubo
    disp.Specular = 0.2
    disp.SpecularColor = [1.0, 1.0, 1.0]
    return tube

add_technical_tube([0.0, 0.0, T], [L, 0.0, T])
add_technical_tube([L, 0.0, T],   [L, W, T])
add_technical_tube([0.0, 0.0, 0.0], [L, 0.0, 0.0])
add_technical_tube([L, 0.0, 0.0],   [L, W, 0.0])

# ============================================================
# RESALTAR EMPOTRAMIENTO
# ============================================================

fixed_plane = Plane()
fixed_plane.Origin = [0.0, -.3*W, -1.*T]
fixed_plane.Point1 = [0.0, 1.3*W,   0.0]
fixed_plane.Point2 = [0.0, -.3*W, 3.0*T]

display_plane = Show(fixed_plane, view)
display_plane.Representation = "Surface"
display_plane.AmbientColor = [0.9, 0.9, 0.9] # Gris industrial claro
display_plane.DiffuseColor = [0.9, 0.9, 0.9]
display_plane.Opacity = 0.65                 # Sólido pero con leve transparencia estética

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

view.ViewSize = [2400, 1500]
view.CameraPosition = [0.6, -1.6, 1.0] 
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 0, 1]
# view.ResetCamera()
view.CameraParallelScale = view.CameraParallelScale * 0.18
Render()

# ============================================================
# GUARDAR FIGURA
# ============================================================

SaveScreenshot(
    output_image,
    view,
    ImageResolution=[2400, 1000]
)


# ============================================================
# ACTUALIZA LOS EJES
# ============================================================

# sys.path.insert(0, path)
# from orientation_axes import add_orientation_axes

# add_orientation_axes(output_image, output_image)
