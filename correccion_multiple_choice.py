import cv2
import numpy as np
import matplotlib.pyplot as plt


# Función para cargar y mostrar la imagen original
def cargar_y_mostrar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    return img

def umbralizar_imagen(img, thresh=128, maxval=255):
    if len(img.shape) == 3:  # Si la imagen está en color, conviértela a escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, img_umbralizada = cv2.threshold(img, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)
    return img, img_umbralizada

# Función para obtener y dibujar contornos en la imagen
def obtener_y_dibujar_contornos(img_umbralizada, grosor=1):

    contornos,_ = cv2.findContours(img_umbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    return contornos_ordenados
# Función para recortar la región de una pregunta dada su índice
def recortar_pregunta_por_indice(index_pregunta, contornos_ordenados, img_umbralizada, dicc_indices):
    
    contorno_pregunta = contornos_ordenados[dicc_indices[index_pregunta]]
    x, y, w, h = cv2.boundingRect(contorno_pregunta)
    pregunta = img_umbralizada[y:y+h, x:x+w]  # Recortar la región de interés

    return pregunta

# Función para detectar la línea horizontal y recortar la región de respuesta
def detectar_linea_y_recortar_respuesta(pregunta):
    
    contornos, _ = cv2.findContours(pregunta, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    contorno_linea = contornos_ordenados[1]  # El segundo contorno es la línea horizontal
    x, y, w, h = cv2.boundingRect(contorno_linea)
    
    respuesta = pregunta[0:y, x:x+w]  # Recortamos desde la línea hacia arriba

    respuesta =  respuesta[::-1,] # Rotacion 180º

    img_zeros = respuesta==0

    img_row_zeros = img_zeros.any(axis=1)

    img_row_zeros_idxs = np.argwhere(np.logical_not(respuesta.all(axis=1))) # Tengo los indices de los renglones

    if img_row_zeros_idxs.size == 0:  # Condición si es vacío no hay respuesta
        return None
    
    if img_row_zeros_idxs[0] > 10: # respuesta vacía con texto arriba
        return None

    start_end = np.diff(img_row_zeros) # inicio y final de los textos

    renglones_indxs = np.argwhere(start_end) # indices de los mismos

    start_idx = (renglones_indxs[0]).item()

    end_idx = (renglones_indxs[1]).item()

    respuesta = respuesta[start_idx:end_idx+1, :] # cortamos el sector respuesta

    respuesta =  respuesta[::-1,] # volvemos a rotarla
    
    return respuesta

def detectar_letra(recorte_respuesta):
    # Conectar componentes
    connectivity = 8  # Conexión de 8 vecinos
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(recorte_respuesta, connectivity, cv2.CV_32S)

    # Revisar si hay más de una componente conectada (excluyendo el fondo)
    if num_labels <= 1:  # Solo hay fondo
        return None # No se detectó ninguna letra

    # Si hay más de una componente conectada, indicaría más de una letra o ruido
    if num_labels > 2:  # Excluyendo el fondo
        return None  # Hay más de una letra o ruido

    # Obtener el rectángulo delimitador de la única letra
    x, y, w, h = stats[0][:4]  # stats[1] porque stats[0] es el fondo

    # Recortar la imagen de la letra desde la imagen binaria original
    letra_recortada = recorte_respuesta[y:y+h, x:x+w]  # Aquí se realiza el recorte correcto

    return letra_recortada  # Devolvemos la letra recortada


def identificador_letra(letra_recortada):
    
    img_expand = cv2.copyMakeBorder(letra_recortada, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255) # agregamos bordes

    img_inv = img_expand==0 # invertimos para que quede fondo negro

    inv_uint8 = img_inv.astype(np.uint8) # conversión para que no quede bool

    contours,_ = cv2.findContours (inv_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # buscamos contornos

    if len(contours) == 1:
        return "C"
        
    if len(contours) == 3:
        return "B"

    if len(contours) == 2:

        kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])              # definimos un filtro horizontal para detectar líneas

        filtro_aplicado = cv2.filter2D(letra_recortada, cv2.CV_64F, kernel)     # Aplicar el filtro a la imagen de la letra

        magnitud_filtro = np.abs(filtro_aplicado)                               # Obtener la magnitud del filtro

 
        umbral = magnitud_filtro.max() * 0.8                                    # Umbralizar la imagen filtrada para obtener una imagen binaria
        imagen_binaria = magnitud_filtro >= umbral

        
        lineas_horizontales = np.any(imagen_binaria, axis=1)                    # Contar las filas con al menos un valor True
        cantidad_lineas = np.sum(lineas_horizontales)

        if cantidad_lineas == 1:
            return "A"

        else: 
            return "D"
        
def correccion_examen(examen, verbose=True):
    # Diccionario para mapear las preguntas con los índices de contornos
    dicc_indices = {1: 6, 2: 5, 3: 4, 4: 3, 5: 12, 6: 7, 7: 11, 8: 8, 9: 9, 10: 10}

    img, img_umbralizada = umbralizar_imagen(examen)

    contornos_ordenados = obtener_y_dibujar_contornos(img_umbralizada, grosor=1)

    respuestas_correctas = {1: "C", 2: "B", 3: "A", 4: "D", 5: "B", 6: "B", 7: "A", 8: "B", 9: "D", 10: "D"}

    cantidad_correctas = 0

    for i in range(1, 11):  # Iteramos por todas las preguntas
        
        pregunta = recortar_pregunta_por_indice(i, contornos_ordenados, img_umbralizada, dicc_indices)
        
        box_respuesta = detectar_linea_y_recortar_respuesta(pregunta)

        if box_respuesta is None:
            if verbose:
                print(f'Pregunta {i}: MAL')
            continue
        
        respuesta = detectar_letra(box_respuesta)

        if respuesta is None:
            if verbose:
                print(f'Pregunta {i}: MAL')
            continue

        letra_identificada = identificador_letra(respuesta)

        if letra_identificada == respuestas_correctas[i]:
            if verbose:
                print(f'Pregunta {i}: OK')
            cantidad_correctas += 1
        else:
            if verbose:
                print(f'Pregunta {i}: MAL')
            
    return cantidad_correctas


# Función para recortar el encabezado
def recortar_encabezado(img_umbralizada, contornos_ordenados):
    contorno_grande = contornos_ordenados[0]  # El primer contorno más grande
    x, y, w, h = cv2.boundingRect(contorno_grande)
    encabezado = img_umbralizada[:y, :]  # Recortar desde la parte superior hasta la línea superior del contorno
    return encabezado

# Función para quedarnos con la info del encabezado
def recortar_lineas_encabezado(encabezado):
    _, encabezado_bin = umbralizar_imagen(encabezado)
    contornos_lineas = obtener_y_dibujar_contornos(encabezado_bin)

    # Definir límites de longitud
    longitud_minima = 60  # Longitud mínima para considerar una línea
    longitud_maxima = 300  # Longitud máxima para considerar una línea

    lineas_recortadas = []
    for contorno in contornos_lineas:
        x, y, w, h = cv2.boundingRect(contorno)
        longitud = w  # Longitud de la línea
        
        if longitud_minima < longitud < longitud_maxima:  
            recorte = encabezado[y:y+h, x:x+w]  # Recorta la línea
            lineas_recortadas.append((recorte, y, x, w, h))  # Guardar posición y tamaño
            
    return lineas_recortadas

# Función para recortar desde cada línea hacia arriba
def recortar_hacia_arriba(lineas_recortadas, encabezado):
    recortes_finales = []
    
    for recorte, y, x, w, h in lineas_recortadas:
        recorte_arriba = encabezado[:y, x:x+w]  # Recortar todo desde la parte superior hasta la línea
        recortes_finales.append(recorte_arriba)
    
    return recortes_finales

# Función para obtener índices de letras en una línea
def obtener_indices(x):
    ren_col_zeros = x.any(axis=0)
    cambios = np.diff(ren_col_zeros.astype(int))
    letras_indxs = np.argwhere(cambios).flatten()
    return letras_indxs

# Función para contar espacios en una línea
def contar_espacios(indices_letras, threshold=7, threshold1=6):
    espacios = 0
    for i in range(len(indices_letras) - 1):
        distancia = indices_letras[i + 1] - indices_letras[i]
        if distancia == threshold or distancia == threshold1:
            espacios += 1
    return espacios

# Función para validar el nombre en la línea 1
def validar_nombre(indices_letras):
    cantidad_letras = len(indices_letras) // 2  # Cada par de inicio y fin representa una letra
    espacios = contar_espacios(indices_letras)
    palabras = espacios + 1  # Estimación simple de palabras separadas por espacio
    if palabras >= 2 and cantidad_letras <= 25 and espacios >= 1:
        return "OK"
    return "MAL"

# Función para validar la clase en la línea 2
def validar_clase(indices_letras):
    cantidad_letras = len(indices_letras) // 2  # Cada par de inicio y fin representa una letra
    if cantidad_letras == 1:
        return "OK"
    return "MAL"

# Función para validar la fecha en la línea 3
def validar_fecha(indices_letras):
    cantidad_letras = len(indices_letras) // 2  # Cada par de inicio y fin representa una letra
    if cantidad_letras == 8:
        return "OK"
    return "MAL"

# Función para combinar todas las imágenes en una sola de forma vertical
def combinar_imagenes(imgs, nombres, aprobados, ancho=300, alto=100):
    filas = []
    for img, nombre, aprobado in zip(imgs, nombres, aprobados):
        # Cambiar el tamaño de cada imagen a un formato estándar
        img_redimensionada = cv2.resize(img, (ancho, alto))

        # Convertir a BGR para agregar color en los bordes
        img_bgr = cv2.cvtColor(img_redimensionada, cv2.COLOR_GRAY2BGR)

        # Dibujar un borde verde si está aprobado o rojo si no
        color_borde = (0, 255, 0) if aprobado else (0, 0, 255)
        img_bordeada = cv2.copyMakeBorder(img_bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color_borde)

        # Poner el texto del nombre del alumno
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Ajustar el tamaño de la fuente automáticamente dependiendo del tamaño de la imagen
        text_size = cv2.getTextSize(nombre, font, 0.7, 2)[0]
        font_scale = min(ancho / (text_size[0] + 20), 0.7)  # Ajustar el escalado del texto

        # Centrar el texto
        text_x = (img_bordeada.shape[1] - text_size[0]) // 2
        text_y = (img_bordeada.shape[0] + text_size[1]) // 2

        # Poner el texto en la imagen
        cv2.putText(img_bordeada, nombre, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        filas.append(img_bordeada)

    # Combinar todas las imágenes en una sola de forma vertical
    imagen_final = np.vstack(filas)
    return imagen_final


# Rutas de las imágenes
rutas_imagenes = ['files/examen_1.png', 'files/examen_2.png', 'files/examen_3.png', 
                  'files/examen_4.png', 'files/examen_5.png']

# Listas para almacenar los recortes de los nombres, sus etiquetas y si aprobaron o no
imagenes_nombre = []
etiquetas_nombre = []
aprobados_examen = []

for i, ruta_imagen in enumerate(rutas_imagenes):
    print(f"Imagen {i + 1}")
    
    # 1. Cargar la imagen
    img = cargar_y_mostrar_imagen(ruta_imagen)

    # 2. Umbralizar la imagen
    im, img_umbralizada = umbralizar_imagen(img)

    # 3. Obtener contornos
    contornos = obtener_y_dibujar_contornos(img_umbralizada)

    # 4. Recortar el encabezado
    encabezado_recortado = recortar_encabezado(img_umbralizada, contornos)

    # 5. Recortar las líneas del encabezado
    lineas_recortadas = recortar_lineas_encabezado(encabezado_recortado)

    # 6. Recortar hacia arriba
    lineas_finales = recortar_hacia_arriba(lineas_recortadas, encabezado_recortado)

    # Variables para almacenar el nombre y respuestas correctas
    nombre_alumno = ""
    respuestas_correctas = 0

    # 7. Validar encabezado y obtener el nombre del alumno
    for j, linea_recortada in enumerate(lineas_finales):
        # Obtener índices de letras
        indices_letras = obtener_indices(linea_recortada)

        # Validación específica para cada línea
        if j == 0:  # Línea 1 - Name
            validacion = validar_nombre(indices_letras)
            etiqueta_validacion = "Validación Name"
            imagenes_nombre.append(linea_recortada)  # Guardar la imagen del nombre
            etiquetas_nombre.append(nombre_alumno)  # Guardar el nombre del alumno
        elif j == 1:  # Línea 2 - Class
            validacion = validar_clase(indices_letras)
            etiqueta_validacion = "Validación Class"
        elif j == 2:  # Línea 3 - Date
            validacion = validar_fecha(indices_letras)
            etiqueta_validacion = "Validación Date"

        print(f"{etiqueta_validacion}: {validacion}")
    
    # 8. Corregir examen

    respuestas_correctas = correccion_examen(img_umbralizada, contornos)

    # 9. Evaluar si el alumno aprobó o no
    aprobado = respuestas_correctas >= 6
    aprobados_examen.append(aprobado)

# 10. Generar imagen final con los recortes de los nombres
imagen_salida = combinar_imagenes(imagenes_nombre, etiquetas_nombre, aprobados_examen)

# 11. Mostrar o guardar la imagen final
plt.imshow(cv2.cvtColor(imagen_salida, cv2.COLOR_BGR2RGB))
plt.title("Resultados de los Exámenes")
plt.axis('off')
plt.show()

# Guardar la imagen en un archivo:
cv2.imwrite('resultado_examenes.png', imagen_salida)
