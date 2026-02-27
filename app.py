import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Detector Base de CVC", layout="wide")

st.title("🔬 Módulo de Prueba: Detección Pura de Símbolos")
st.markdown("Si este módulo no detecta los cuadrados correctamente, no avanzaremos al cálculo legal. Sube una imagen para auditar el motor de visión.")

def detectar_simbolos(image_bytes):
    # 1. Cargar imagen y convertir a escala de grises
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_debug = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Binarización (Tinta negra = Blanco 255, Papel = Negro 0)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    alto, ancho = thresh.shape
    
    # --- A. ENCONTRAR EL CENTRO EXACTO (MÉTODO DE PROYECCIÓN) ---
    # Ignoramos el 25% superior e inferior para que el texto/leyenda no interfiera
    zona_media_y = thresh[int(alto*0.25):int(alto*0.75), :]
    suma_filas = np.sum(zona_media_y, axis=1)
    cy = np.argmax(suma_filas) + int(alto*0.25) # La fila con más tinta es el eje X
    
    zona_media_x = thresh[:, int(ancho*0.25):int(ancho*0.75)]
    suma_columnas = np.sum(zona_media_x, axis=0)
    cx = np.argmax(suma_columnas) + int(ancho*0.25) # La columna con más tinta es el eje Y
    
    # Dibujar cruz AZUL para demostrar que encontramos el centro real
    cv2.line(img_debug, (0, cy), (ancho, cy), (255, 0, 0), 1)
    cv2.line(img_debug, (cx, 0), (cx, alto), (255, 0, 0), 1)

    # --- B. AISLAR EL ÁREA DEL CAMPO VISUAL ---
    # Encontramos hasta dónde llega el eje horizontal para saber el radio del campo
    pixeles_eje_x = np.where(thresh[cy, :] > 0)[0]
    if len(pixeles_eje_x) > 0:
        radio_campo = int(max(cx - pixeles_eje_x[0], pixeles_eje_x[-1] - cx) * 1.05)
    else:
        radio_campo = int(min(ancho, alto) * 0.4)

    # Crear una máscara circular para borrar los textos de afuera
    mascara_circular = np.zeros_like(thresh)
    cv2.circle(mascara_circular, (cx, cy), radio_campo, 255, -1)
    campo_limpio = cv2.bitwise_and(thresh, mascara_circular)

    # --- C. DETECCIÓN DE SÍMBOLOS (MUESTREO DE NÚCLEO) ---
    
    # Borramos temporalmente las líneas rectas de la grilla para separar los símbolos
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.03), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.03)))
    lineas_h = cv2.morphologyEx(campo_limpio, cv2.MORPH_OPEN, kernel_h)
    lineas_v = cv2.morphologyEx(campo_limpio, cv2.MORPH_OPEN, kernel_v)
    grilla = cv2.add(lineas_h, lineas_v)
    grilla_engrosada = cv2.dilate(grilla, np.ones((3,3), np.uint8))
    
    # Restamos la grilla
    simbolos_separados = cv2.subtract(campo_limpio, grilla_engrosada)
    
    # Pequeña dilatación para unir símbolos que hayan sido cortados por la resta
    simbolos_unidos = cv2.dilate(simbolos_separados, np.ones((2,2), np.uint8))
    
    contornos, _ = cv2.findContours(simbolos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtros de tamaño dinámicos basados en la resolución de la imagen
    area_min = (ancho * 0.002) ** 2
    area_max = (ancho * 0.02) ** 2
    
    cuadrados_encontrados = 0
    circulos_encontrados = 0
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Si tiene el tamaño de un símbolo...
        if area_min < area < area_max:
            # MAGIA: Miramos el recuadro en la imagen ORIGINAL binarizada (campo_limpio)
            roi = campo_limpio[y:y+h, x:x+w]
            
            # Extraemos el "corazón" del símbolo (el 40% del centro exacto)
            y1_core, y2_core = int(h*0.3), int(h*0.7)
            x1_core, x2_core = int(w*0.3), int(w*0.7)
            
            if y2_core > y1_core and x2_core > x1_core:
                corazon = roi[y1_core:y2_core, x1_core:x2_core]
                
                # ¿Qué porcentaje del corazón es tinta negra?
                porcentaje_tinta_corazon = np.sum(corazon > 0) / float(corazon.size)
                
                # Centro del símbolo para dibujar
                px, py = x + w//2, y + h//2
                
                # Si el corazón es macizo (más del 55% de tinta)...
                if porcentaje_tinta_corazon > 0.55:
                    cuadrados_encontrados += 1
                    cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 0, 255), 2) # Caja Roja
                else:
                    # Si el corazón está vacío (papel blanco)...
                    circulos_encontrados += 1
                    cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 1) # Caja Verde

    return img_debug, cuadrados_encontrados, circulos_encontrados

# --- INTERFAZ ---
archivo = st.file_uploader("Sube un estudio de CVC para testear la detección", type=["jpg", "jpeg", "png"])

if archivo is not None:
    img_res, total_cuadrados, total_circulos = detectar_simbolos(archivo.getvalue())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        st.image(Image.fromarray(img_rgb), caption="Auditoría Visual (Cruz Azul = Centro detectado)", use_container_width=True)
    with col2:
        st.metric("Cuadrados (Rojos)", total_cuadrados)
        st.metric("Círculos/Huecos (Verdes)", total_circulos)
        st.write("---")
        st.write("**Instrucciones de Auditoría:**")
        st.write("1. Verifica que la cruz azul esté en el centro exacto.")
        st.write("2. Verifica que los cuadraditos negros tengan una caja **ROJA**.")
        st.write("3. Si esto es perfecto, avísame y rearmamos el cálculo de incapacidad.")
