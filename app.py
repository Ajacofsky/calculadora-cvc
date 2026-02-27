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

        # --- B. DETECCIÓN DE SÍMBOLOS ---
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.03), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.03)))
        lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        grilla = cv2.add(lineas_h, lineas_v)
        grilla_engrosada = cv2.dilate(grilla, np.ones((3,3), np.uint8))
        
        simbolos_separados = cv2.subtract(thresh, grilla_engrosada)
        simbolos_unidos = cv2.dilate(simbolos_separados, np.ones((2,2), np.uint8))
        
        contornos, _ = cv2.findContours(simbolos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area_min = (ancho * 0.002) ** 2
        area_max = (ancho * 0.02) ** 2
        
        centros_raw = []
        datos_simbolos = []
        
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            if area_min < area < area_max:
                px, py = x + w//2, y + h//2
                centros_raw.append((px, py))
                
                roi = thresh[y:y+h, x:x+w]
                y1, y2 = int(h*0.3), int(h*0.7)
                x1, x2 = int(w*0.3), int(w*0.7)
                
                if y2 > y1 and x2 > x1:
                    corazon = roi[y1:y2, x1:x2]
                    porcentaje_tinta = np.sum(corazon > 0) / float(corazon.size)
                    tipo = 'fallado' if porcentaje_tinta > 0.55 else 'visto'
                    datos_simbolos.append({'px': px, 'py': py, 'tipo': tipo})

        # --- C. AUTO-CALIBRACIÓN PERFECTA (GRILLA DE 6 GRADOS) ---
        pixels_por_10_grados = 1.0 # Valor seguro por defecto
        
        if len(centros_raw) > 10:
            min_dists = []
            for i in range(len(centros_raw)):
                dists = []
                for j in range(len(centros_raw)):
                    if i != j:
                        d = math.hypot(centros_raw[i][0] - centros_raw[j][0], centros_raw[i][1] - centros_raw[j][1])
                        if d > 5: # Ignorar ruidos pegados
                            dists.append(d)
                if dists:
                    min_dists.append(min(dists))
            
            if min_dists:
                espaciado_6_grados = np.median(min_dists)
                pixels_por_10_grados = float(espaciado_6_grados / 6.0) * 10.0

        # --- D. FILTRADO Y MAPEO (40 GRADOS CENTRALES) ---
        puntos_totales = []
        for sim in datos_simbolos:
            px, py, tipo = sim['px'], sim['py'], sim['tipo']
            dx, dy = px - cx, py - cy
            r_pixel = math.hypot(dx, dy)
            g_fisicos = (r_pixel / pixels_por_10_grados) * 10
            
            # FILTRO LEGAL DE ORO: Solo los 40 grados centrales
            if 2 < g_fisicos <= 41:
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                puntos_totales.append({'r': g_fisicos, 'ang': ang, 'tipo': tipo})
                
                if tipo == 'fallado':
                    cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1)
                else:
                    cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1)

        # --- E. CÁLCULO DE DENSIDAD POR OCTANTES ---
        grados_no_vistos_total = 0
        
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
                        color_zona = (255, 200, 0) # Celeste (BGR)
                    elif 0 < densidad < 70:
                        grados_perdidos = 5
                        color_zona = (0, 255, 255) # Amarillo (BGR)
                
                grados_no_vistos_total += grados_perdidos
                
                # PINTURA VECTORIAL EXACTA (Máscara Geométrica)
                if color_zona:
                    r_in = int(limite_inf * (pixels_por_10_grados/10.0))
                    r_out = int(limite_sup * (pixels_por_10_grados/10.0))
                    
                    mask_sector = np.zeros((alto, ancho), dtype=np.uint8)
                    cv2.ellipse(mask_sector, (cx, cy), (r_out, r_out), 0, ang_inf, ang_sup, 255, -1)
                    if r_in > 0:
                        cv2.ellipse(mask_sector, (cx, cy), (r_in, r_in), 0, ang_inf, ang_sup, 0, -1)
                        
                    overlay[mask_sector == 255] = color_zona

        cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
        
        for angulo_linea in range(0, 360, 45):
            rad = math.radians(angulo_linea)
            x2 = int(cx + (4.2 * pixels_por_10_grados) * math.cos(rad))
            y2 = int(cy + (4.2 * pixels_por_10_grados) * math.sin(rad))
            cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

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
**Sistema Activo:** Auto-Calibración Biométrica y Mapeo Vectorial Exacto.
- **Puntos Rojos:** Cuadrados (Fallados).
- **Puntos Verdes:** Círculos (Vistos).
- **Celeste:** Densidad ≥ 70% (10°). **Amarillo:** > 0% (5°).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_final_2")

col1, col2 = st.columns(2)

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando densidades matemáticas..."):
                img_res, grados, incap, error_msg = procesar_campo_visual(file.getvalue())
            
            if error_msg:
                st.error("Error interno del sistema:")
                st.code(error_msg)
                return 0.0
            elif img_res is not None:
                img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(img_rgb), caption=f"Mapa de Calor y Auditoría - {titulo}", use_container_width=True)
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
