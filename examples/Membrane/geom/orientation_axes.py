from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings

warnings.filterwarnings('ignore')


class ReferenceAxesJMPS:
    """
    Ejes de referencia profesionales para JMPS.
    Vista Z- (frente): X horizontal derecha, Y vertical arriba, Z punto interior.
    """
    
    def __init__(self, 
                 image_path,
                 output_path,
                 position="lower_left",
                 margin=40,
                 size=300,
                 scale_factor=0.45,
                 label_distance=2.0):
        """
        Args:
            image_path: ruta de imagen de entrada
            output_path: ruta de imagen de salida
            position: "lower_left", "lower_right", "upper_left", "upper_right"
            margin: margen desde borde (píxeles)
            size: tamaño del área de ejes (píxeles)
            scale_factor: longitud de ejes (0-1)
            label_distance: distancia de etiqueta al eje
        """
        
        self.image_path = image_path
        self.output_path = output_path
        self.position = position
        self.margin = margin
        self.size = size
        self.scale_factor = scale_factor
        self.label_distance = label_distance
        
        # Colores ISO estándar (oscurecidos)
        self.colors = {
            'X': (180, 0, 0, 255),       # Rojo oscuro
            'Y': (0, 140, 0, 255),       # Verde oscuro
            'Z': (0, 60, 180, 255)       # Azul oscuro
        }
        
        # Etiquetas LaTeX
        self.labels = {
            'X': 'X_1',
            'Y': 'X_2',
            'Z': 'Z_3'
        }
        
        # Parámetros de dibujo
        self.line_width = 6
        self.arrow_length = 32
        self.arrow_width = 18
        self.z_radius = 18
        self.z_point = 10
        
        # Cargar imagen
        try:
            self.img = Image.open(image_path)
            self.width, self.height = self.img.size
        except Exception as e:
            raise IOError(f"Error cargando imagen: {e}")
    
    def _render_latex(self, latex_text, fontsize=45):
        """Renderiza LaTeX con fuente Computer Modern/Times."""
        try:
            fig = plt.figure(figsize=(2, 2), dpi=150)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Configurar LaTeX con Computer Modern (por defecto en matplotlib)
            plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
            plt.rcParams['text.usetex'] = False
            
            # Renderizar
            ax.text(0.5, 0.5, f'${latex_text}$',
                   fontsize=fontsize,
                   ha='center', va='center',
                   family='serif')
            
            # Convertir a imagen
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            
            img = Image.frombytes("RGB", size, raw_data)
            img = img.convert('RGBA')
            
            # Hacer fondo transparente (píxeles blancos)
            img_array = np.array(img)
            white_mask = (img_array[:, :, 0] > 240) & \
                        (img_array[:, :, 1] > 240) & \
                        (img_array[:, :, 2] > 240)
            img_array[white_mask, 3] = 0
            
            img = Image.fromarray(img_array)
            plt.close(fig)
            return img
            
        except Exception as e:
            print(f"⚠ Error renderizando LaTeX '{latex_text}': {e}")
            return None
    
    def _colorize_latex(self, img, color):
        """Coloriza texto LaTeX según el color del eje."""
        img_array = np.array(img)
        
        # Máscara de píxeles oscuros (texto)
        darkness = 255 - np.mean(img_array[:, :, :3], axis=2)
        mask = darkness > 150
        
        # Aplicar color
        img_array[mask, 0] = color[0]
        img_array[mask, 1] = color[1]
        img_array[mask, 2] = color[2]
        img_array[mask, 3] = color[3]
        
        return Image.fromarray(img_array)
    
    def _draw_axis(self, draw, center, angle, color, latex_label):
        """Dibuja un eje con línea, flecha y etiqueta."""
        
        scale = self.size * self.scale_factor
        
        # Dirección del eje
        dir_vec = np.array([np.cos(angle), np.sin(angle)])
        perp_vec = np.array([-np.sin(angle), np.cos(angle)])
        
        # Punto final
        end = center + scale * dir_vec
        
        # Línea del eje
        draw.line(
            [(center[0], center[1]), (end[0], end[1])],
            fill=color,
            width=self.line_width
        )
        
        # Punta de flecha
        p1 = end
        p2 = end - dir_vec * self.arrow_length - perp_vec * self.arrow_width
        p3 = end - dir_vec * self.arrow_length + perp_vec * self.arrow_width
        
        draw.polygon([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1])], fill=color)
        
        # Etiqueta LaTeX
        label_dist = scale * self.label_distance
        text_pos = center + label_dist * dir_vec
        
        latex_img = self._render_latex(latex_label)
        
        if latex_img is not None:
            latex_img = self._colorize_latex(latex_img, color)
            
            label_x = int(text_pos[0] - latex_img.width // 2)
            label_y = int(text_pos[1] - latex_img.height // 2)
            
            # Mantener dentro de imagen
            label_x = max(0, min(label_x, self.width - latex_img.width))
            label_y = max(0, min(label_y, self.height - latex_img.height))
            
            self.img.paste(latex_img, (label_x, label_y), latex_img)
    
    def _draw_z_symbol(self, draw, center):
        """Dibuja símbolo Z (círculo con punto)."""
        
        color = self.colors['Z']
        
        # Circunferencia exterior
        draw.ellipse(
            [center[0] - self.z_radius, center[1] - self.z_radius,
             center[0] + self.z_radius, center[1] + self.z_radius],
            outline=color,
            width=5
        )
        
        # Punto en el centro
        draw.ellipse(
            [center[0] - self.z_point, center[1] - self.z_point,
             center[0] + self.z_point, center[1] + self.z_point],
            fill=color
        )
        
        # Etiqueta Z_i
        latex_z = self._render_latex(self.labels['Z'])
        
        if latex_z is not None:
            latex_z = self._colorize_latex(latex_z, color)
            
            z_label_y = int(center[1] - self.z_radius - 40)
            z_label_x = int(center[0] - latex_z.width)
            
            z_label_y = max(0, z_label_y)
            z_label_x = max(0, min(z_label_x, self.width // 2 - latex_z.width))
            
            self.img.paste(latex_z, (z_label_x, z_label_y), latex_z)
    
    def _calculate_position(self):
        """Calcula posición del centro."""
        
        positions = {
            "lower_left": (self.margin, self.height - self.size - self.margin),
            "lower_right": (self.width - self.size - self.margin, 
                           self.height - self.size - self.margin),
            "upper_left": (self.margin, self.margin),
            "upper_right": (self.width - self.size - self.margin, self.margin),
        }
        
        if self.position not in positions:
            self.position = "lower_left"
        
        x0, y0 = positions[self.position]
        cx = x0 + self.size // 2
        cy = y0 + self.size // 2
        
        return cx, cy
    
    def draw(self):
        """Dibuja los ejes en la imagen."""
        
        draw = ImageDraw.Draw(self.img, 'RGBA')
        cx, cy = self._calculate_position()
        center = np.array([cx, cy])
        
        # Eje X (horizontal derecha, 0°)
        self._draw_axis(draw, center, 0, self.colors['X'], self.labels['X'])
        
        # Eje Y (vertical arriba, 90° = π/2)
        # En PIL, Y aumenta hacia abajo, así que -π/2 hace que apunte arriba
        self._draw_axis(draw, center, -np.pi/2, self.colors['Y'], self.labels['Y'])
        
        # Eje Z (símbolo)
        self._draw_z_symbol(draw, center)
    
    def save(self, quality=95):
        """Guarda la imagen."""
        try:
            self.img.save(self.output_path, quality=quality, dpi=(300, 300))
            print(f"Imagen guardada: {self.output_path}")
            return True
        except Exception as e:
            print(f"Error guardando: {e}")
            return False


def add_orientation_axes(image_path, 
                           output_path,
                           position="lower_left",
                           margin=40,
                           size=280):
    """
    Agrega ejes de referencia JMPS a una imagen.
    
    Args:
        image_path: ruta de entrada
        output_path: ruta de salida
        position: "lower_left", "lower_right", "upper_left", "upper_right"
        margin: margen desde borde (píxeles)
        size: tamaño de ejes (píxeles)
    
    Ejemplo:
        add_orientation_axes(
            "membrane_geometry.png",
            "membrane_geometry.png",
            position="lower_left",
            size=280
        )
    """
    
    axes = ReferenceAxesJMPS(
        image_path=image_path,
        output_path=output_path,
        position=position,
        margin=margin,
        size=size
    )
    
    axes.draw()
    axes.save()