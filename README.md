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

```bash
python nombre_del_script.py
```

3. **Visualizar Resultados**: La imagen final se mostrará utilizando Matplotlib. También puedes descomentar la línea para guardar la imagen generada.

## Funciones Principales

- `cargar_y_mostrar_imagen(ruta_imagen)`: Carga y devuelve una imagen en escala de grises.
- `umbralizar_imagen(img, thresh=128, maxval=255)`: Umbraliza la imagen para binarizarla.
- `obtener_contornos(img_umbralizada)`: Obtiene y ordena los contornos de la imagen.
- `recortar_preguntas(img_umbralizada, contornos_ordenados)`: Recorta las preguntas de la imagen utilizando los contornos detectados.
- `validar_nombre(indices_letras)`: Valida la línea del nombre del alumno.
- `combinar_imagenes(imgs, nombres, aprobados)`: Combina las imágenes de resultados en una sola imagen.

## Notas

- Asegúrate de que las imágenes de entrada están correctamente alineadas y son de buena calidad para obtener los mejores resultados.
- Puedes ajustar los parámetros de umbralización y recorte según sea necesario para adaptarse a diferentes formatos de examen.

## Contribuciones

Si deseas contribuir a este proyecto, no dudes en hacer un fork del repositorio y enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.
