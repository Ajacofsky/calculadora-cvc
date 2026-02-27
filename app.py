import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import traceback

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

# ==========================================
# MOTOR DE VISIÓN COMPUTARIZADA Y LÓGICA
# ==========================================

def procesar_campo_visual(image_bytes):
    try:
        if not image_bytes:
            return None, 0, 0, "No hay imagen"

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, 0, 0, "Formato de imagen inválido."

        img_heatmap = img.copy()
        overlay = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alto, ancho = gray.shape
        
        # 1. Binarización
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # --- A. ENCONTRAR EL CENTRO (PROYECCIÓN FÍSICA) ---
        zona_media_y = thresh[int(alto*0.25):int(alto*0.75), :]
        suma_filas = np.sum(zona_media_y, axis=1)
        cy = np.argmax(suma_filas) + int(alto*0.25) 
        
        zona_media_x = thresh[:, int(ancho*0.25):int(ancho*0.75)]
        suma_columnas = np.sum(zona_media_x, axis=0)
        cx = np.argmax(suma_columnas) + int(ancho*0.25) 
        
        # Escala: Buscamos hasta dónde llega la cruz horizontal
        fila_eje_h = thresh[cy-2 : cy+2, cx:]
        _, x_h = np.where(fila_eje_h > 0)
        dist_60 = np.max(x_h) if len(x_h) > 0 else int((ancho - cx) * 0.75)
        pixels_por_10_grados = float(dist_60 / 6.0)

        if pixels_por_10_grados <= 0:
            pixels_por_10_grados = 1.0 

        # --- B. DETECCIÓN DE SÍMBOLOS (MUESTREO DE NÚCLEO) ---
        # Borramos líneas rectas temporalmente
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.03), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.03)))
        lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        grilla = cv2.add(lineas_h, lineas_v)
        grilla_engrosada = cv2.dilate(grilla, np.ones((3,3), np.uint8))
        
        simbolos_separados = cv2.subtract(thresh, grilla_engrosada)
        simbolos_unidos = cv2.dilate(simbolos_separados, np.ones((2,2), np.uint8))
        
        contornos, _ = cv2.findContours(simbolos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        puntos_totales = []
        
        area_min = (ancho * 0.002) ** 2
        area_max = (ancho * 0.02) ** 2
        
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            if area_min < area < area_max:
                roi = thresh[y:y+h, x:x+w]
                
                # Corazón del símbolo (40% central)
                y1, y2 = int(h*0.3), int(h*0.7)
                x1, x2 = int(w*0.3), int(w*0.7)
                
                if y2 > y1 and x2 > x1:
                    corazon = roi[y1:y2, x1:x2]
                    porcentaje_tinta = np.sum(corazon > 0) / float(corazon.size)
                    
                    px, py = x + w//2, y + h//2
                    dx, dy = px - cx, py - cy
                    r_pixel = math.hypot(dx, dy)
                    g_fisicos = (r_pixel / pixels_por_10_grados) * 10
                    
                    # FILTRO DE ORO: Solo consideramos lo que está entre 2 y 41 grados
                    if 2 < g_fisicos <= 41:
                        ang = math.degrees(math.atan2(dy, dx))
                        if ang < 0: ang += 360
                        
                        if porcentaje_tinta > 0.55:
                            tipo = 'fallado'
                            cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1) # Rojo
                        else:
                            tipo = 'visto'
                            cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1) # Verde
                            
                        puntos_totales.append({'r': g_fisicos, 'ang': ang, 'tipo': tipo})

        # --- C. ANÁLISIS DE DENSIDAD POR OCTANTES Y MAPA DE CALOR ---
        grados_no_vistos_total = 0
        
        # Dibujar anillos guía finos
        for i in range(1, 5):
            r_dibujo = int(i * pixels_por_10_grados)
            cv2.circle(img_heatmap, (cx, cy), r_dibujo, (0, 0, 255), 1)
            
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
                
                # REGLA DEL 70%
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
                    r_in = limite_inf * (pixels_por_10_grados/10.0)
                    r_out = limite_sup * (pixels_por_10_grados/10.0)
                    r_center = int((r_in + r_out) / 2.0)
                    grosor = int(r_out - r_in)
                    
                    if grosor > 0:
                        cv2.ellipse(overlay, (cx, cy), (r_center, r_center), 0, ang_inf, ang_sup, color_zona, grosor + 1)

        cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
        
        # Dibujar cruz central final
        for angulo_linea in range(0, 360, 45):
            rad = math.radians(angulo_linea)
            x2 = int(cx + (4.2 * pixels_por_10_grados) * math.cos(rad))
            y2 = int(cy + (4.2 * pixels_por_10_grados) * math.sin(rad))
            cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

        # Cálculo de incapacidad
        porcentaje_perdida_cv = (grados_no_vistos_total / 320.0) * 100
        incapacidad_ojo = porcentaje_perdida_cv * 0.25

        return img_heatmap, grados_no_vistos_total, incapacidad_ojo, None

    except Exception as e:
        return None, 0, 0, traceback.format_exc()

# ==========================================
# INTERFAZ WEB
# ==========================================

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("""
**Análisis Activo:** Proyección de Ejes, Muestreo de Núcleo y Regla del 70%.
- **Puntos Rojos:** Cuadrados (Fallados).
- **Puntos Verdes:** Círculos (Vistos).
- **Celeste:** Densidad ≥ 70% (10°). **Amarillo:** > 0% (5°).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_final_1")

col1, col2 = st.columns(2)

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando densidades..."):
                img_res, grados, incap, error_msg = procesar_campo_visual(file.getvalue())
            
            if error_msg:
                st.error("Error interno del sistema:")
                st.code(error_msg)
                return 0.0
            elif img_res is not None:
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(img_rgb), caption=f"Mapa de Calor - {titulo}", use_container_width=True)
                st.success(f"**Grados No Vistos:** {grados}° / 320°")
                st.metric(label=f"Incapacidad {titulo}", value=f"{incap:.2f}%")
                return incap
            else:
                st.error("Error desconocido.")
    return 0.0

incap_OD = mostrar_resultado(col1, "Ojo Derecho (OD)", "od_file")
incap_OI = 0.0

if modo_evaluacion == "Bilateral (Ambos ojos)":
    incap_OI = mostrar_resultado(col2, "Ojo Izquierdo (OI)", "oi_file")

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
