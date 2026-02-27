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
    
    # --- A. CALIBRACIÓN GEOMÉTRICA (Usando OTSU global para buscar los ejes) ---
    _, thresh_global = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_ejes = thresh_global.copy()
    limite_inferior = int(alto * 0.75)
    mask_ejes[limite_inferior:, :] = 0 # Ocultar tabla inferior
    
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
    
    # --- B. DETECCIÓN DE SÍMBOLOS (VISIÓN ADAPTATIVA) ---
    
    # MAGIA: Adaptive Threshold se ajusta a las "zonas grises" y variaciones de luz del centro
    thresh_local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    
    # Engrosamos los ejes un poco para asegurar que "corten" bien los símbolos al restarlos
    ejes_unidos = cv2.bitwise_or(lineas_h, lineas_v)
    kernel_corte = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ejes_gruesos = cv2.dilate(ejes_unidos, kernel_corte)
    
    # Restamos los ejes para dejar los símbolos sueltos (aunque tengan un "hueco" de la línea restada)
    thresh_sin_ejes = cv2.subtract(thresh_local, ejes_gruesos)
    contornos, _ = cv2.findContours(thresh_sin_ejes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos_totales = []
    
    max_area_esperada = (pixels_por_10_grados * 0.8) ** 2
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area_caja = w * h
        aspect_ratio = float(w)/h if h > 0 else 0
        
        # Filtramos ruido minúsculo
        if 8 < area_caja < max_area_esperada and 0.4 < aspect_ratio < 2.5:
            
            # Calculamos cuánta tinta quedó en esa cajita
            roi = thresh_sin_ejes[y:y+h, x:x+w]
            pixeles_tinta = cv2.countNonZero(roi)
            indice_relleno = pixeles_tinta / float(area_caja)
            
            px, py = x + w//2, y + h//2
            dx, dy = px - cx, py - cy
            radio_pixel = math.hypot(dx, dy)
            grados_fisicos = (radio_pixel / pixels_por_10_grados) * 10
            
            if 2 < grados_fisicos <= 41:
                angulo = math.degrees(math.atan2(dy, dx))
                if angulo < 0: angulo += 360
                
                # REGLA: Un círculo hueco ronda el 20-30% de relleno. 
                # Un cuadrado, aún con la cruz del eje borrada, supera el 45-50%.
                if indice_relleno >= 0.45: 
                    tipo = 'fallado'
                    cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1) # Punto Rojo
                else:                     
                    tipo = 'visto'
                    cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1) # Punto Verde
                    
                puntos_totales.append({'r': grados_fisicos, 'ang': angulo, 'tipo': tipo})

    # --- C. ANÁLISIS POR ZONAS Y MAPA DE CALOR ---
    grados_no_vistos_total = 0
    
    for i in range(1, 5):
        radio_dibujo = int(i * pixels_por_10_grados)
        cv2.circle(img_heatmap, (cx, cy), radio_dibujo, (0, 0, 255), 1)
        
    for anillo in range(4):
        limite_inf = anillo * 10
        limite_sup = (anillo + 1) * 10
        
        for octante in range(8):
            ang_inf = octante * 45
            ang_sup = (octante + 1) * 45
            
            puntos_zona = [p for p in puntos_totales if limite_inf <= p['r'] < limite_sup and ang_inf <= p['ang'] < ang_sup]
            
            total_pts = len(puntos_zona)
            fallados = sum(1 for p in puntos_zona if p['tipo'] == 'fallado')
            
            color_zona = None
            grados_perdidos = 0
            
            if total_pts > 0:
                densidad = (fallados / total_pts) * 100
                
                if densidad >= 70:
                    grados_perdidos = 10
                    color_zona = (255, 200, 0) # Celeste
                elif 0 < densidad < 70:
                    grados_perdidos = 5
                    color_zona = (0, 255, 255) # Amarillo
            
            grados_no_vistos_total += grados_perdidos
            
            if color_zona:
                r_in = int(limite_inf * (pixels_por_10_grados/10))
                r_out = int(limite_sup * (pixels_por_10_grados/10))
                cv2.ellipse(overlay, (cx, cy), (r_out, r_out), 0, ang_inf, ang_sup, color_zona, -1)
                cv2.ellipse(overlay, (cx, cy), (r_in, r_in), 0, ang_inf, ang_sup, (255, 255, 255), -1)

    cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
    
    for angulo_linea in range(0, 360, 45):
        rad = math.radians(angulo_linea)
        x2 = int(cx + (4.2 * pixels_por_10_grados) * math.cos(rad))
        y2 = int(cy + (4.2 * pixels_por_10_grados) * math.sin(rad))
        cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

    porcentaje_perdida_cv = (grados_no_vistos_total / 320.0) * 100
    incapacidad_ojo = porcentaje_perdida_cv * 0.25

    return img_heatmap, grados_no_vistos_total, incapacidad_ojo

# ==========================================
# 2. INTERFAZ DE USUARIO (WEB APP)
# ==========================================

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("""
**Método de Análisis:** Detección morfológica con evaluación original.
- **Puntos Rojos:** Cuadrados negros detectados (Fallados).
- **Puntos Verdes:** Círculos detectados (Vistos).
- **Regla:** Densidad ≥ 70% = 10° (Celeste) | > 0% = 5° (Amarillo).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_modo_v4")

col1, col2 = st.columns(2)

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando imagen con algoritmo avanzado..."):
                img_res, grados, incap = procesar_campo_visual(file.getvalue())
            
            if img_res is not None:
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
incap_OI = 0.0

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
