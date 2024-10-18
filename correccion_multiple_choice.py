import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(img):
    """
    Función para detectar celdas en un formulario de multiple choice.
    """

    # Umbralizar la imagen
    th = 150  # Ajusta este valor según sea necesario
    img_th = img < th

    # Sumar píxeles en cada columna y fila
    img_cols = np.sum(img_th, axis=0)  # Sumas verticales
    img_rows = np.sum(img_th, axis=1)  # Sumas horizontales

    # Definir umbrales para las líneas
    th_col = 550  # Ajusta este umbral para columnas
    th_row = 450  # Ajusta este umbral para filas

    # Detectar líneas verticales y horizontales
    img_cols_th = img_cols > th_col
    img_rows_th = img_rows > th_row

    # Encontrar posiciones de las líneas
    vertical_lines = np.where(img_cols_th)[0]
    horizontal_lines = np.where(img_rows_th)[0]

    return vertical_lines, horizontal_lines


def extraer_celdas(img, vertical_lines, horizontal_lines):
    # Extraer subimágenes basadas en las líneas detectadas
    th = 150
    sub_images = []
    for i in range(len(horizontal_lines) - 1):
        for j in range(len(vertical_lines) - 1):
            # Coordenadas de la subimagen
            x_start = vertical_lines[j]
            x_end = vertical_lines[j + 1]
            y_start = horizontal_lines[i]
            y_end = horizontal_lines[i + 1]

            # Extraer la subimagen
            sub_img = img[y_start:y_end, x_start:x_end]

            # Umbralizar la subimagen para detectar letras
            sub_img_th = sub_img < th
            
            # Detección de componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sub_img_th.astype(np.uint8), connectivity=4)

            # Filtrar componentes por área (por ejemplo, componentes con área mayor a 10)
            th_area = 5000  # Ajusta este valor según sea necesario
            ix_area = stats[:, -1] > th_area
            stats = stats[ix_area, :]

            # Si hay componentes válidos, guardar la subimagen
            if len(stats) > 0:
                sub_images.append(sub_img)

    return sub_images
    
def detectar_encabezado(imagen_path, th_area=500, height_threshold=30):
    """
    Función para detectar y validar el encabezado de un formulario.
    """
    # Leer la imagen en escala de grises
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    # Umbralizar la imagen
    th = 150  # Ajusta este valor según sea necesario
    img_th = img < th

    # Detección de componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_th.astype(np.uint8), connectivity=8)

    # Buscar el primer encabezado válido en la parte superior
    encabezado = None
    for i in range(1, num_labels):  # Comenzar en 1 para evitar el fondo
        x, y, width, height, area = stats[i]
        
        # Validar que el área y la altura estén dentro de los rangos esperados
        if area > th_area and height > height_threshold:
            # Verificar si es el componente más alto (parte superior)
            if encabezado is None or y < encabezado[1]:
                encabezado = (x, y, width, height)

    # Si se encontró un encabezado, dibujar el rectángulo en la imagen original
    if encabezado:
        x, y, width, height = encabezado
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Mostrar la imagen original con el encabezado detectado
        plt.figure(figsize=(10, 5))
        plt.imshow(img, cmap='gray')
        plt.title('Encabezado Detectado')
        plt.axis('off')
        plt.show()
    else:
        print("No se encontró un encabezado válido.")

    return encabezado

# Ejemplo de uso
encabezado = detectar_encabezado('files/examen_1.png')
print(f'Encabezado detectado: {encabezado}')