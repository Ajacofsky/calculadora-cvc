import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# ==========================================
# 1. MOTOR DE VISIÓN COMPUTARIZADA Y LÓGICA
# ==========================================

def procesar_campo_visual(image_bytes):
    if not image_bytes:
        return None, 0, 0

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, 0, 0

    img_heatmap = img.copy()
    overlay = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alto, ancho = gray.shape
    
    # Binarización automática (OTSU) para adaptar el contraste
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- A. CALIBRACIÓN GEOMÉTRICA BLINDADA ---
    # Enmascaramos el 25% inferior para que la tabla no confunda a los ejes
    mask_ejes = thresh.copy()
    limite_inferior = int(alto * 0.75)
    mask_ejes[limite_inferior:, :] = 0 
    
    kernel_h_axis = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho * 0.2), 1))
    lineas_h = cv2.morphologyEx(mask_ejes, cv2.MORPH_OPEN, kernel_h_axis)
    
    kernel_v_axis = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto * 0.2)))
    lineas_v = cv2.morphologyEx(mask_ejes, cv2.MORPH_OPEN, kernel_v_axis)
    
    interseccion = cv2.bitwise_and(lineas_h, lineas_v)
    y_coords, x_coords = np.where(interseccion > 0)
    
    if len(x_coords) > 0:
        cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
    else:
        cx, cy = int(ancho / 2), int(alto / 2)
        
    fila_eje_h = lineas_h[cy-5 : cy+5, cx:] 
    _, x_h = np.where(fila_eje_h > 0)
    if len(x_h) > 0:
        dist_60 = np.max(x_h) 
        pixels_por_10_grados = dist_60 / 6.0
    else:
        dist_60 = int((ancho - cx) * 0.75)
        pixels_por_10_grados = dist_60 / 6.0
    
    # --- B. DETECCIÓN DE SÍMBOLOS CON EVALUACIÓN EN IMAGEN ORIGINAL ---
    
    # Restamos las líneas SOLO para encontrar las coordenadas sin que se peguen los símbolos
    thresh_sin_ejes = cv2.subtract(thresh, cv2.bitwise_or(lineas_h, lineas_v))
    contornos, _ = cv2.findContours(thresh_sin_ejes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos_totales = []
    
    max_area_esperada = (pixels_por_10_grados * 0.8) ** 2
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area_caja = w * h
        aspect_ratio = float(w)/h if h > 0 else 0
        
        if 8 < area_caja < max_area_esperada and 0.4 < aspect_ratio < 2.5:
            
            # MAGIA: Evaluamos el contenido mirando la imagen ORIGINAL (thresh)
            # Así los cuadrados centrales no pierden su "tinta" por culpa de los ejes
            roi_original = thresh[y:y+h, x:x+w]
            pixeles_tinta = cv2.countNonZero(roi_original)
            indice_relleno = pixeles_
# ==========================================
# 2. INTERFAZ DE USUARIO (WEB APP)
# ==========================================

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("""
**Método de Análisis:** Detección morfológica por tamaño y solidez.
- **Puntos Rojos:** Cuadrados negros detectados (Fallados).
- **Puntos Verdes:** Círculos detectados (Vistos).
- **Regla:** Densidad ≥ 70% = 10° (Celeste) | > 0% = 5° (Amarillo).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_modo_v3")

col1, col2 = st.columns(2)

incap_OD = 0.0
incap_OI = 0.0

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando imagen con algoritmo morfológico..."):
                img_res, grados, incap = procesar_campo_visual(file.getvalue())
            
            if img_res is not None:
                # Conversión segura a PIL para Streamlit Cloud
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                st.image(pil_img, caption=f"Auditoría y Mapa de Calor - {titulo}", use_container_width=True)
                st.success(f"**Grados No Vistos:** {grados}° / 320°")
                st.metric(label=f"Incapacidad {titulo}", value=f"{incap:.2f}%")
                return incap
            else:
                st.error("Error al procesar. Verifique el formato de la imagen.")
    return 0.0

incap_OD = mostrar_resultado(col1, "Ojo Derecho (OD)", "file_od")

if modo_evaluacion == "Bilateral (Ambos ojos)":
    incap_OI = mostrar_resultado(col2, "Ojo Izquierdo (OI)", "file_oi")

# ==========================================
# 3. RESULTADO FINAL LEGAL
# ==========================================
st.divider()
st.header("📊 Informe Final de Incapacidad Visual")

if modo_evaluacion == "Bilateral (Ambos ojos)":
    if incap_OD > 0 and incap_OI > 0:
        suma_aritmetica = incap_OD + incap_OI
        incapacidad_bilateral = suma_aritmetica * 1.5
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Suma Aritmética", f"{suma_aritmetica:.2f}%")
        col_b.metric("Factor Bilateralidad", "x 1.5")
        col_c.metric("Incapacidad Total", f"{incapacidad_bilateral:.2f}%", delta="Final Legal")
    else:
        st.info("Suba ambas imágenes para el cálculo bilateral.")
