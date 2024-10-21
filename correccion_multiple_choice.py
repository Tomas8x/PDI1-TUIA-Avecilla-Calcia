import cv2
import numpy as np
import matplotlib.pyplot as plt



# Función para cargar y mostrar la imagen original
def cargar_y_mostrar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    return img

# Función para umbralizar la imagen
def umbralizar_imagen(img, thresh=128, maxval=255):
    _, img_umbralizada = cv2.threshold(img, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return img_umbralizada

# Función para obtener contornos
def obtener_contornos(img_umbralizada):
    contornos, _ = cv2.findContours(img_umbralizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
    return contornos_ordenados

# Función para obtener las preguntas
def obtener_preguntas(img_umbralizada, contornos_ordenados, n=2):
    preguntas = []
    for i in range(n):
        x, y, w, h = cv2.boundingRect(contornos_ordenados[i])
        pregunta = img_umbralizada[y:y+h, x:x+w]
        preguntas.append(pregunta)
    return preguntas

# Función para recortar preguntas de cada columna
def recortar_preguntas(img_umbralizada, contornos_ordenados):
    preguntas = []
    for contorno in contornos_ordenados:
        x, y, w, h = cv2.boundingRect(contorno)
        if h > 50 and h < 500:  
            pregunta = img_umbralizada[y:y+h, x:x+w]
            preguntas.append(pregunta)
    return preguntas

# Función para recortar las líneas de las preguntas
def recortar_lineas_preguntas(preguntas):
    lineas_recortadas_preguntas = []
    
    for pregunta in preguntas:
        pregunta_bin = umbralizar_imagen(pregunta)
        contornos_lineas = obtener_contornos(pregunta_bin)

        longitud_minima = 30 
        longitud_maxima = pregunta.shape[1]  # Considera el ancho de la pregunta como la longitud máxima

        lineas_pregunta = []
        for contorno in contornos_lineas:
            x, y, w, h = cv2.boundingRect(contorno)
            longitud = w
            
            if longitud_minima < longitud < longitud_maxima:
                recorte = pregunta[y:y+h, x:x+w]
                lineas_pregunta.append((recorte, y, x, w, h))
        
        lineas_recortadas_preguntas.append(lineas_pregunta)
    
    return lineas_recortadas_preguntas

def recortar_respuesta_desde_linea(pregunta, margen_bajo=5, margen_arriba=0, margen_derecho=0):
    # Umbralizamos la imagen de la pregunta para facilitar la detección de la línea
    pregunta_bin = umbralizar_imagen(pregunta)
    contornos_lineas = obtener_contornos(pregunta_bin)

    # Límites de altura para detectar una línea horizontal
    altura_minima_linea = 1  # Altura mínima para una línea delgada
    altura_maxima_linea = 10  # Altura máxima para evitar bloques grandes

    for contorno in contornos_lineas:
        x, y, w, h = cv2.boundingRect(contorno)
        # Si el contorno corresponde a una línea horizontal
        if altura_minima_linea <= h <= altura_maxima_linea:
            # Ajustar el recorte añadiendo un margen por debajo, por encima y a la derecha de la línea
            y_nuevo = min(y + margen_bajo, pregunta.shape[0])  # Evitar que se pase del borde inferior
            y_inicial = max(0, y - margen_arriba)  # Evitar que se pase del borde superior
            x_final = max(0, x + w - margen_derecho)  # Ajustar el ancho para recortar desde el lado derecho
            recorte_arriba = pregunta[y_inicial:y_nuevo, x:x_final]  # Recorte desde el inicio de la línea hasta el nuevo ancho
            return recorte_arriba

    # Si no se encuentra una línea, retornar la pregunta completa
    return pregunta

# Función para recortar el encabezado
def recortar_encabezado(img_umbralizada, contornos_ordenados):
    contorno_grande = contornos_ordenados[0]  # El primer contorno más grande
    x, y, w, h = cv2.boundingRect(contorno_grande)
    encabezado = img_umbralizada[:y, :]  # Recortar desde la parte superior hasta la línea superior del contorno
    return encabezado

# Función para quedarnos con la info del encabezado
def recortar_lineas_encabezado(encabezado):
    encabezado_bin = umbralizar_imagen(encabezado)
    contornos_lineas = obtener_contornos(encabezado_bin)

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

def identificador_letra(letra_recortada):
    
    img_expand = cv2.copyMakeBorder(letra_recortada, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255) # agregamos bordes
    

    img_inv = img_expand==0 # invertimos para que quede fondo negro

    inv_uint8 = img_inv.astype(np.uint8) # conversión para que no quede bool

    contours,_ = cv2.findContours(inv_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # buscamos contornos

    if len(contours) == 1:
        return "C" 
    elif len(contours) == 3:
        return "B"
    elif len(contours) == 2:

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

# Función para validar las preguntas
def validar_pregunta(indices_letras, recorte_respuesta):
    if len(indices_letras) == 0:
        return "Incorrecto - No hay índices"
    elif len(indices_letras) > 2:
        return "Incorrecto - Más de 2 índices"
    else:
        # Conectar componentes
        connectivity = 8  # Conexión de 8 vecinos
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(recorte_respuesta, connectivity, cv2.CV_32S)

        # Obtener el rectángulo delimitador de la única letra
        x, y, w, h = stats[0][:4]  # stats[1] porque stats[0] es el fondo

        # Recortar la imagen de la letra desde la imagen binaria original
        letra_recortada = recorte_respuesta[y:y+h, x:x+w]  # Aquí se realiza el recorte correcto
        
        letra = identificador_letra(letra_recortada)
        

        return "OK", letra

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
    img_umbralizada = umbralizar_imagen(img)

    # 3. Obtener contornos
    contornos = obtener_contornos(img_umbralizada)

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
    
    # 8. Procesar preguntas
    preguntas_recortadas = recortar_preguntas(img_umbralizada, contornos)

    """
    Las imagenes se recortan en el siguiente orden (6 - 2 - 4 - 3 - 1 - 10 - 9 - 8 - 7 - 5)
    Al saber las respuesta a cada una de antemano decidimos ordenarlas en una lista con respecto a como las recorta el script
    """
    respuestas_correjidas = {1: "B",
                             2: "B",
                             3: "D",
                             4: "A",
                             5: "C",
                             6: "D",
                             7: "D",
                             8: "B",
                             9: "A",
                             10: "B"}

    # Validar preguntas
    for k, pregunta_recortada in enumerate(preguntas_recortadas):
        
        # Recortar la respuesta desde la línea horizontal hacia arriba
        respuesta_recortada = recortar_respuesta_desde_linea(pregunta_recortada, margen_bajo=0, margen_arriba=13, margen_derecho=10)

        # Obtener los índices de letras en la respuesta recortada
        indices_letras_respuesta = obtener_indices(respuesta_recortada)
        
    # Validar la respuesta
        validacion_pregunta = validar_pregunta(indices_letras_respuesta, respuesta_recortada)
        
        print(f"Pregunta {k + 1}: {validacion_pregunta}")
        if validacion_pregunta[0] == "OK":
            respuestas_correctas += 1
            letra = validacion_pregunta[1]
            if letra == respuestas_correjidas[k + 1]:
                pass
            
        
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
