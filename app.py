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
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # --- A. ENCONTRAR EL CENTRO EXACTO (PROYECCIÓN) ---
        kernel_cross_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.2), 1))
        kernel_cross_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.2)))
        lines_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_cross_h)
        lines_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_cross_v)
        
        inter = cv2.bitwise_and(lines_h, lines_v)
        y_pts, x_pts = np.where(inter > 0)
        
        if len(x_pts) > 0 and len(y_pts) > 0:
            cx, cy = int(np.mean(x_pts)), int(np.mean(y_pts))
        else:
            cx, cy = int(ancho / 2), int(alto / 2)

        # --- B. DETECCIÓN DE SÍMBOLOS (Motor Núcleo-Físico) ---
        # Borrar ejes para liberar símbolos
        kernel_grid_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.02), 1))
        kernel_grid_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.02)))
        grid_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_grid_h)
        grid_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_grid_v)
        grid = cv2.bitwise_or(grid_h, grid_v)
        grid_dilated = cv2.dilate(grid, np.ones((3,3), np.uint8))
        
        # Máscara para ignorar textos periféricos
        mask_circular = np.zeros_like(thresh)
        cv2.circle(mask_circular, (cx, cy), int(ancho*0.42), 255, -1)
        thresh_clean = cv2.bitwise_and(thresh, mask_circular)
        
        simbolos_img = cv2.subtract(thresh_clean, grid_dilated)
        simbolos_img = cv2.morphologyEx(simbolos_img, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        contornos, _ = cv2.findContours(simbolos_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centros_validos = []
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filtro dinámico de tamaño
            if 10 < area < 400:
                roi = thresh[y:y+h, x:x+w]
                y1, y2 = int(h*0.3), int(h*0.7)
                x1, x2 = int(w*0.3), int(w*0.7)
                
                if y2 > y1 and x2 > x1:
                    corazon = roi[y1:y2, x1:x2]
                    porcentaje_tinta = np.sum(corazon > 0) / float(corazon.size)
                    
                    tipo = 'fallado' if porcentaje_tinta > 0.55 else 'visto'
                    px, py = x + w//2, y + h//2
                    centros_validos.append({'px': px, 'py': py, 'tipo': tipo})

        # --- C. ESCALA BIOMÉTRICA (La distancia entre puntos es SIEMPRE 6°) ---
        if len(centros_validos) < 10:
            return None, 0, 0, "No se detectaron suficientes puntos para calibrar la escala."

        min_dists = []
        for i, c1 in enumerate(centros_validos):
            dists = []
            for j, c2 in enumerate(centros_validos):
                if i != j:
                    d = math.hypot(c1['px'] - c2['px'], c1['py'] - c2['py'])
                    dists.append(d)
            if dists:
                min_dist = min(dists)
                if min_dist > 5: # Evitar contar puntos rotos pegados
                    min_dists.append(min_dist)

        espaciado_6_grados = np.median(min_dists)
        pixels_por_10_grados = float((espaciado_6_grados / 6.0) * 10.0)

        # --- D. FILTRADO LEGAL A 40° Y CONTEO POR SECTOR ---
        puntos_zona = {i: {j: {'visto':0, 'fallado':0} for j in range(8)} for i in range(4)}
        
        for p in centros_validos:
            dx = p['px'] - cx
            dy = p['py'] - cy
            r_pixel = math.hypot(dx, dy)
            g_fisicos = (r_pixel / pixels_por_10_grados) * 10.0
            
            # Filtro de los 40 grados
            if 1 <= g_fisicos <= 41: 
                ang = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                
                anillo = int(g_fisicos // 10)
                if anillo == 4: anillo = 3 # Limite exacto de 40.0
                octante = int(ang // 45)
                if octante == 8: octante = 7 # Limite exacto de 360
                
                puntos_zona[anillo][octante][p['tipo']] += 1
                
                # Auditoría Visual
                color = (0, 0, 255) if p['tipo'] == 'fallado' else (0, 255, 0)
                cv2.circle(img_heatmap, (p['px'], p['py']), 3, color, -1)

        # --- E. PINTURA VECTORIAL PERFECTA ---
        # Creamos una matriz matemática de coordenadas para evitar las manchas
        Y, X = np.ogrid[:alto, :ancho]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        angle_from_center = (np.degrees(np.arctan2(Y - cy, X - cx)) + 360) % 360
        r_deg_matrix = (dist_from_center / pixels_por_10_grados) * 10.0

        grados_no_vistos_total = 0

        for anillo in range(4):
            r_min = anillo * 10
            r_max = (anillo + 1) * 10
            
            for octante in range(8):
                ang_min = octante * 45
                ang_max = (octante + 1) * 45
                
                vistos = puntos_zona[anillo][octante]['visto']
                fallados = puntos_zona[anillo][octante]['fallado']
                total = vistos + fallados
                
                if total > 0:
                    densidad_fallo = (fallados / total) * 100
                    
                    if densidad_fallo >= 70:
                        grados_no_vistos_total += 10
                        color_rgb = (255, 200, 0) # Celeste en BGR
                    elif densidad_fallo > 0:
                        grados_no_vistos_total += 5
                        color_rgb = (0, 255, 255) # Amarillo en BGR
                    else:
                        continue
                        
                    # Aplicamos la pintura usando la matriz exacta (Inquebrantable)
                    sector_mask = (r_deg_matrix >= r_min) & (r_deg_matrix < r_max) & (angle_from_center >= ang_min) & (angle_from_center < ang_max)
                    overlay[sector_mask] = color_rgb

        cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)

        # Dibujar Grilla Roja Final
        for i in range(1, 5):
            cv2.circle(img_heatmap, (cx, cy), int(i * pixels_por_10_grados), (0, 0, 255), 1)
            
        for i in range(8):
            ang = math.radians(i * 45)
            x2 = int(cx + 4.2 * pixels_por_10_grados * math.cos(ang))
            y2 = int(cy + 4.2 * pixels_por_10_grados * math.sin(ang))
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
**Análisis Activo:** Detección de Núcleo Puro + Escala Biométrica Adaptativa + Mapeo Vectorial.
- **Puntos Rojos:** Cuadrados (Fallados).
- **Puntos Verdes:** Círculos (Vistos).
- **Celeste:** Densidad ≥ 70% (10°). **Amarillo:** > 0% (5°).
""")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_final_11")

col1, col2 = st.columns(2)

def mostrar_resultado(columna, titulo, key_uploader):
    with columna:
        st.subheader(titulo)
        file = st.file_uploader(f"Subir imagen {titulo}", type=["jpg", "jpeg", "png"], key=key_uploader)
        if file is not None:
            with st.spinner("Procesando matriz vectorial de densidades..."):
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
