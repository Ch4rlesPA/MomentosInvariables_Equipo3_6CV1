# García Islas Asael.
# Ponce Anaya Carlos.
# Rivera Torres Iván Gilberto.
# 6CV1.

# Librerías.
import os, random, shutil
import matplotlib.pyplot as plt
import cv2
import mahotas
import pandas as pd
import numpy as np

# Importación del banco de datos 'shapes'.
ruta = "C:/Users/asael/OneDrive/Documentos/ESCUELA/7_SEMESTRE/IA/Programas/Proyecto2/shapes"
# Listado del contenido de 'shapes'.
listado_archivos = os.listdir(ruta)
print(listado_archivos)

# Función para contar el total de imágenes de cada clase.
def conteo_imagenes(datos_directorio):
    for nombre_clases in os.listdir(datos_directorio):
        clases_directorio = os.path.join(datos_directorio, nombre_clases)
        patrones = os.listdir(clases_directorio)
        print(f"La clase {nombre_clases} tiene {len(patrones)} imágenes.")
datos_directorio = ruta
conteo_imagenes(datos_directorio)

# Creación de muestra de 1000 patrones
def crear_muestra(datos_directorio):
    muestra = {}
    for nombre_clases in os.listdir(datos_directorio):
        clases_directorio = os.path.join(datos_directorio, nombre_clases)
        patrones = os.listdir(clases_directorio)
        muestra[nombre_clases] = random.sample(patrones, 250)
    return muestra
datos_directorio = ruta
muestra = crear_muestra(datos_directorio)

total_muestreo = 0
for nombre_clases, patrones in muestra.items():
    print(f"Clase: {nombre_clases} (elegidas aleatoriamente: {len(patrones)})")
    total_muestreo += len(patrones)
    for i in range(2):
        image_path = os.path.join(datos_directorio, nombre_clases, patrones[i])
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(f"Imagen: {patrones[i]}")
        plt.show()
print(f'\nTotal de muestras creadas por todas las clases: {total_muestreo}')

# Función para procesar la muestra y obtener contornos.
def procesar_muestra(datos_directorio, muestra):
    contornos_dict = {}
    for nombre_clases, patrones in muestra.items():
        contornos_clase = []
        for i in range(len(patrones)):
            image_path = os.path.join(datos_directorio, nombre_clases, patrones[i])
            # Imagen en escala de grises.
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Suiavizado de la imagen.
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Binarización de la imagen.
            ret, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Encontrar los bordes de Canny.
            edged = cv2.Canny(binary_image, 30, 200)
            # Encontrando los contornos.
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Almacenamiento de los contornos en la lista de contornos de la clase.
            contornos_clase.append(contours)
        # Almacenamiento de la lista de contornos en el diccionario.
        contornos_dict[nombre_clases] = contornos_clase
    return contornos_dict

# Procesar la muestra y obtener contornos.
contornos_dict = procesar_muestra(datos_directorio, muestra)

# Visualización de contornos.
def visualizar_contornos(datos_directorio, muestra, contornos_dict):
    for nombre_clases, patrones in muestra.items():
        for i in range(2):
            image_path = os.path.join(datos_directorio, nombre_clases, patrones[i])
            image = plt.imread(image_path)
            # Imagen en escala de grises.
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Suiavizado de la imagen.
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Binarización de la imagen.
            ret, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Encontrar los bordes de Canny.
            edged = cv2.Canny(binary_image, 30, 200)
            # Encontrando los contornos.
            contours = contornos_dict[nombre_clases][i]

            # Ploteo de la imagen original.
            plt.figure(figsize=(12, 4))
            plt.subplot(131), plt.imshow(image)
            plt.title(f"Imagen: {patrones[i]}")

            # Ploteo de bordes Canny.
            plt.subplot(132), plt.imshow(edged, cmap='gray')
            plt.title("Detección de Contornos tras Búsqueda de Bordes Canny")

            # Ploteo del contorno de la figura.
            plt.subplot(133)
            plt.imshow(image, cmap='gray')
            plt.title("Contours")
            for contour in contours:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='lime', linewidth=3)
            plt.show()

# Visualización de contornos.
visualizar_contornos(datos_directorio, muestra, contornos_dict)

# Función para el cálculo de los momentos de Hu de los contornos.
def calcular_momentos_hu(contornos_dict):
    momentos_hu_dict = {}
    for nombre_clases, contornos_clase in contornos_dict.items():
        momentos_hu_clase = []
        for contorno in contornos_clase:
            # Conversión del contorno a un formato aceptado por cv2.moments().
            contorno_array = np.vstack(contorno).squeeze()
            # Cálculo de los momentos geométricos y momentos de Hu.
            momentos_hu = cv2.HuMoments(cv2.moments(contorno_array)).flatten()
            momentos_hu_clase.append(momentos_hu)
        momentos_hu_dict[nombre_clases] = momentos_hu_clase
    return momentos_hu_dict

# Llamada a la función.
momentos_hu_dict = calcular_momentos_hu(contornos_dict)

# Visualización de los momentos de Hu.
def visualizar_momentos_hu(momentos_hu_dict):
    for nombre_clases, momentos_hu_clase in momentos_hu_dict.items():
        for i, momentos_hu in enumerate(momentos_hu_clase):
            print(f"Clase: {nombre_clases}, Contorno: {i + 1}")
            print("Momentos de Hu:")
            print(momentos_hu)
            print("\n")

# Muestra los momentos de Hu calculados para cada contorno.
#visualizar_momentos_hu(momentos_hu_dict)

# Función para el cálculo de los momentos de Zernike de los contornos.
def calcular_momentos_zernike(contornos_dict, orden_momento):
    momentos_zernike_dict = {}
    for nombre_clases, contornos_clase in contornos_dict.items():
        momentos_zernike_clase = []
        for contorno in contornos_clase:
            # Conversión del contorno a un formato aceptado por mahotas.
            contorno_array = np.vstack(contorno).squeeze()
            # Cálculo de los momentos de Zernike.
            momentos_zernike = mahotas.features.zernike_moments(contorno_array, orden_momento)
            momentos_zernike_clase.append(momentos_zernike)
        momentos_zernike_dict[nombre_clases] = momentos_zernike_clase
    return momentos_zernike_dict

# Llamada a la función.
orden_momento = 50  # radio de los momentos de Zernike.
momentos_zernike_dict = calcular_momentos_zernike(contornos_dict, orden_momento)

# Visualización de los momentos de Zernike.
def visualizar_momentos_zernike(momentos_zernike_dict):
    for nombre_clases, momentos_zernike_clase in momentos_zernike_dict.items():
        for i, momentos_zernike in enumerate(momentos_zernike_clase):
            print(f"Clase: {nombre_clases}, Contorno: {i + 1}")
            print("Momentos de Zernike:")
            print(momentos_zernike)
            print("\n")

# Muestra los momentos de Zernike calculados para cada contorno.
#visualizar_momentos_zernike(momentos_zernike_dict)
