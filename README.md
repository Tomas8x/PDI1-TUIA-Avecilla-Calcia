# PDI1-TUIA-Avecilla-Calcia
# README

## Descripción

Este script en Python utiliza OpenCV y NumPy para procesar imágenes de exámenes, extrayendo información como nombres de alumnos, clases, fechas y respuestas a preguntas. A través de técnicas de umbralización y detección de contornos, el código identifica y valida las respuestas, generando una imagen final con los resultados.

## Requisitos

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Instalación de paquetes requeridos:
```bash
pip install opencv-python numpy matplotlib
```

## Estructura del Código

1. **Carga de Imágenes**: Las imágenes de los exámenes se cargan desde rutas específicas.
2. **Umbralización**: Las imágenes se convierten a escala de grises y se aplican técnicas de umbralización para facilitar el procesamiento.
3. **Detección de Contornos**: Se detectan los contornos en la imagen umbralizada para identificar diferentes secciones (encabezado y preguntas).
4. **Recorte de Información**: Se recortan las secciones relevantes de la imagen, incluyendo el encabezado y las respuestas a las preguntas.
5. **Validación de Respuestas**: Se validan los nombres, clases, fechas y respuestas utilizando métodos específicos que comprueban la cantidad de letras y espacios.
6. **Generación de Resultados**: Se combinan las imágenes recortadas en una sola imagen que muestra el nombre del alumno y si ha aprobado o no.

## Uso

1. **Modificar Rutas**: Asegúrate de que las rutas de las imágenes en la lista `rutas_imagenes` son correctas y accesibles.
2. **Ejecutar el Script**: Ejecuta el script en un entorno Python. Esto procesará las imágenes y generará una salida visual con los resultados.

