import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import traceback

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

# ==========================================
# MOTOR DE VISIÓN COMPUTARIZADA (VERSIÓN ORO)
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
        overlay = np.zeros_like(img, dtype=np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alto, ancho = gray.shape
        
        # 1. Binarización de Alto Contraste
        gray_contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        thresh = cv2.adaptiveThreshold(gray_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
        
        # --- A. ENCONTRAR EL CENTRO EXACTO ---
        roi_y = thresh[int(alto*0.3):int(alto*0.7), :]
        cy = np.argmax(np.sum(roi_y, axis=1)) + int(alto*0.3)
        roi_x = thresh[:, int(ancho*0.3):int(ancho*0.7)]
        cx = np.argmax(np.sum(roi_x, axis=0)) + int(ancho*0.3)

        # --- B. DETECCIÓN DE SÍMBOLOS (Restando Grilla) ---
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.05), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.05)))
        lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        
        grilla = cv2.bitwise_or(lineas_h, lineas_v)
        grilla_dilatada = cv2.dilate(grilla, np.ones((3,3), np.uint8))
        
        simbolos_aislados = cv2.subtract(thresh, grilla_dilatada)
        simbolos_aislados = cv2.morphologyEx(simbolos_aislados, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        contornos, _ = cv2.findContours(simbolos_aislados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        simbolos_validos = []
        area_min = (ancho * 0.0025) ** 2
        area_max = (ancho * 0.025) ** 2
        
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if area_min < area < area_max and 0.5 < aspect_ratio < 2.0:
                px, py = x + w//2, y + h//2
                
                # Muestreo del Corazón en la imagen original binarizada
                y1, y2 = int(h*0.3), int(h*0.7)
                x1, x2 = int(w*0.3), int(w*0.7)
                
                if y2 > y1 and x2 > x1:
                    corazon = thresh[y+y1:y+y2, x+x1:x+x2]
                    # CORRECCIÓN DE ERROR APLICADA AQUÍ: cv2.countNonZero
                    densidad_tinta = cv2.countNonZero(corazon) / float(corazon.size)
                    
                    tipo = 'fallado' if densidad_tinta > 0.45 else 'visto'
                    simbolos_validos.append({'px': px, 'py': py, 'tipo': tipo})

        # --- C. CALIBRACIÓN DE ESCALA INFALIBLE (Distancia Vecina) ---
        if len(simbolos_validos) < 15:
            return None, 0, 0, "No se detectaron suficientes puntos para calibrar."
            
        distancias = []
        for i, s1 in enumerate(simbolos_validos):
            min_d = float('inf')
            for j, s2 in enumerate(simbolos_validos):
                if i != j:
                    d = math.hypot(s1['px'] - s2['px'], s1['py'] - s2['py'])
                    if d > (ancho * 0.005): # Ignorar si detectó dos pedazos del mismo símbolo
                        if d < min_d: min_d = d
            if min_d != float('inf'):
                distancias.append(min_d)
                
        # La mediana de las distancias equivale exactamente a 6 grados físicos
        pixeles_por_6_grados = np.median(distancias)
        pixels_por_10_grados = float((pixeles_por_6_grados / 6.0) * 10.0)

        # --- D. FILTRADO LEGAL (40 Grados) Y CONTEO ---
        puntos_zona = {anillo: {octante: {'vistos':0, 'fallados':0} for octante in range(8)} for anillo in range(4)}
        
        for sim in simbolos_validos:
            dx, dy = sim['px'] - cx, sim['py'] - cy
            r_deg = (math.hypot(dx, dy) / pixels_por_10_grados) * 10.0
            
            if 1 <= r_deg <= 41: # Zona clínica
                ang = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                
                anillo = min(3, int(r_deg // 10))
                octante = min(7, int(ang // 45))
                
                if sim['tipo'] == 'fallado':
                    puntos_zona[anillo][octante]['fallados'] += 1
                    cv2.circle(img_heatmap, (sim['px'], sim['py']), 4, (0, 0, 255), -1) # Rojo
                else:
                    puntos_zona[anillo][octante]['vistos'] += 1
                    cv2.circle(img_heatmap, (sim['px'], sim['py']), 2, (0, 255, 0), -1) # Verde

        # --- E. PINTURA VECTORIAL DE CUADRANTES ---
        grados_no_vistos_total = 0
        
        for anillo in range(4):
            r_in = int(anillo * pixels_por_10_grados)
            r_out = int((anillo + 1) * pixels_por_10_grados)
            
            for octante in range(8):
                ang_in = octante * 45
                ang_out = (octante + 1) * 45
                
                f = puntos_zona[anillo][octante]['fallados']
                v = puntos_zona[anillo][octante]['vistos']
                total = f + v
                
                if total > 0:
                    densidad = (f / float(total)) * 100
                    color = None
                    
                    if densidad >= 70:
                        grados_no_vistos_total += 10
                        color = (255, 200, 0) # Celeste BGR
                    elif densidad > 0:
                        grados_no_vistos_total += 5
                        color = (0, 255, 255) # Amarillo BGR
                        
                    if color:
                        # Dibujo perfecto usando máscaras de sustracción
                        mask = np.zeros((alto, ancho), dtype=np.uint8)
                        cv2.ellipse(mask, (cx, cy), (r_out, r_out), 0, ang_in, ang_out, 255, -1)
                        if r_in > 0:
                            cv2.ellipse(mask, (cx, cy), (r_in, r_in), 0, ang_in, ang_out, 0, -1)
                        overlay[mask == 255] = color

        cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
        
        # Dibujar Grilla Guía Final
        for i in range(1, 5):
            cv2.circle(img_heatmap, (cx, cy), int(i * pixels_por_10_grados), (0, 0, 255), 1)
        for i in range(8):
            ang_rad = math.radians(i * 45)
            x2 = int(cx + 4.2 * pixels_por_10_grados * math.cos(ang_rad))
            y2 = int(cy + 4.2 * pixels_por_10_grados * math.sin(ang_rad))
            cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

        incapacidad_ojo = (grados_no_vistos_total / 320.0) * 100 * 0.25

        return img_heatmap, grados_no_vistos_total, incapacidad_ojo, None

    except Exception as e:
        return None, 0, 0, traceback.format_exc()

# ==========================================
# INTERFAZ WEB
# ==========================================

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("""
**Arquitectura Final:** Escala Biométrica (Distancia Vectorial 6°) y Pintura Geométrica de Precisión.
- **Puntos Rojos:** Cuadrados (Fallados).
- **Puntos Verdes:** Círculos (Vistos).
- **Celeste:** Densidad ≥ 70% (10°). **Amarillo:** > 0% (5°).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_final_pro")

col1, col2 = st.columns(2)

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando auditoría espacial..."):
                img_res, grados, incap, error_msg = procesar_campo_visual(file.getvalue())
            
            if error_msg:
                st.error("Error del sistema:")
                st.code(error_msg)
                return 0.0
            elif img_res is not None:
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(img_rgb), caption=f"Mapa de Calor Clínico - {titulo}", use_container_width=True)
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
