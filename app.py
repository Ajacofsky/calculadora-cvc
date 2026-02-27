import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image # <-- NUEVA LIBRERÍA IMPORTADA AQUÍ

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
    
    # --- A. CALIBRACIÓN GEOMÉTRICA DE EJES ---
    _, thresh_ejes = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    alto, ancho = gray.shape
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho * 0.15), 1))
    lineas_h = cv2.morphologyEx(thresh_ejes, cv2.MORPH_OPEN, kernel_h)
    
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto * 0.15)))
    lineas_v = cv2.morphologyEx(thresh_ejes, cv2.MORPH_OPEN, kernel_v)
    
    interseccion = cv2.bitwise_and(lineas_h, lineas_v)
    y_coords, x_coords = np.where(interseccion > 0)
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
    else:
        cx, cy = int(ancho / 2), int(alto / 2) 
        
    fila_eje_h = lineas_h[cy-10 : cy+10, :]
    _, x_h = np.where(fila_eje_h > 0)
    
    if len(x_h) > 0:
        extremo_derecho = np.max(x_h)
        distancia_60_grados = extremo_derecho - cx
    else:
        distancia_60_grados = int((ancho - cx) * 0.75) 
        
    pixels_por_10_grados = distancia_60_grados / 6.0
    
    # --- B. DETECCIÓN DE PUNTOS (Filtro OTSU + Análisis de Núcleo) ---
    
    # OTSU: Calcula automáticamente el mejor contraste sin importar si la foto es clara u oscura
    _, thresh_simbolos = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contornos, _ = cv2.findContours(thresh_simbolos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos_totales = []
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w)/h if h > 0 else 0
        
        # Filtros de tamaño muy tolerantes (acepta símbolos desde 6 hasta 900 píxeles)
        if 6 < area < 900 and 0.4 < aspect_ratio < 2.5:
            px = x + w // 2
            py = y + h // 2
            dx, dy = px - cx, py - cy
            radio_pixel = math.hypot(dx, dy)
            grados_fisicos = (radio_pixel / pixels_por_10_grados) * 10
            
            # Solo analizamos dentro de los 41 grados (ignorando el centro exacto)
            if 2 < grados_fisicos <= 41:
                
                # LA MAGIA: Extraemos el "Tercio Central" del símbolo
                y_start, y_end = y + h//3, y + 2*h//3
                x_start, x_end = x + w//3, x + 2*w//3
                
                if y_end > y_start and x_end > x_start:
                    nucleo = thresh_simbolos[y_start:y_end, x_start:x_end]
                    pixeles_tinta = cv2.countNonZero(nucleo)
                    area_nucleo = nucleo.size
                    
                    # Si el núcleo está más del 40% lleno de tinta, es un cuadrado macizo ■
                    if (pixeles_tinta / area_nucleo) > 0.4:
                        tipo = 'fallado'
                        cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1) # Puntito Rojo de Auditoría
                    else:
                        tipo = 'visto' # Centro vacío = Círculo ○
                        cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1) # Puntito Verde de Auditoría
                        
                    angulo = math.degrees(math.atan2(dy, dx))
                    if angulo < 0: angulo += 360
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
        x2 = int(cx + (4 * pixels_por_10_grados) * math.cos(rad))
        y2 = int(cy + (4 * pixels_por_10_grados) * math.sin(rad))
        cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

    porcentaje_perdida_cv = (grados_no_vistos_total / 320.0) * 100
    incapacidad_ojo = porcentaje_perdida_cv * 0.25

    return img_heatmap, grados_no_vistos_total, incapacidad_ojo
# 2. INTERFAZ DE USUARIO (WEB APP)
# ==========================================

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("Basado en baremo legal y método de densidad de ocupación por octantes.")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"], key="radio_modo") # Solucion error key
col1, col2 = st.columns(2)

incap_OD = 0.0
incap_OI = 0.0

with col1:
    st.subheader("Ojo Derecho (OD)")
    file_od = st.file_uploader("Subir imagen CVC Ojo Derecho", type=["jpg", "jpeg", "png"])
    if file_od is not None:
        st.info("Procesando imagen...")
        img_res_od, grados_od, incap_OD = procesar_campo_visual(file_od.getvalue())
        
        if img_res_od is not None:
            # --- SOLUCIÓN A PRUEBA DE FALLOS: CONVERSIÓN A PIL ---
            img_rgb_od = cv2.cvtColor(img_res_od, cv2.COLOR_BGR2RGB)
            pil_img_od = Image.fromarray(img_rgb_od)
            
            st.image(pil_img_od, caption="Mapa de Calor Generado - OD", use_container_width=True)
            st.success(f"**Grados No Vistos:** {grados_od}° / 320°")
            st.success(f"**Incapacidad OD:** {incap_OD:.2f}%")
        else:
            st.error("Hubo un problema procesando la imagen. Asegúrate de que sea un CVC válido.")

with col2:
    if modo_evaluacion == "Bilateral (Ambos ojos)":
        st.subheader("Ojo Izquierdo (OI)")
        file_oi = st.file_uploader("Subir imagen CVC Ojo Izquierdo", type=["jpg", "jpeg", "png"])
        if file_oi is not None:
            st.info("Procesando imagen...")
            img_res_oi, grados_oi, incap_OI = procesar_campo_visual(file_oi.getvalue())
            
            if img_res_oi is not None:
                # --- SOLUCIÓN A PRUEBA DE FALLOS: CONVERSIÓN A PIL ---
                img_rgb_oi = cv2.cvtColor(img_res_oi, cv2.COLOR_BGR2RGB)
                pil_img_oi = Image.fromarray(img_rgb_oi)
                
                st.image(pil_img_oi, caption="Mapa de Calor Generado - OI", use_container_width=True)
                st.success(f"**Grados No Vistos:** {grados_oi}° / 320°")
                st.success(f"**Incapacidad OI:** {incap_OI:.2f}%")
            else:
                st.error("Hubo un problema procesando la imagen.")

# ==========================================
# 3. RESULTADO FINAL LEGAL
# ==========================================
st.divider()
st.header("📊 Informe Final de Incapacidad Visual")

if modo_evaluacion == "Unilateral (Un solo ojo)":
    if file_od is not None and incap_OD > 0:
        st.metric(label="Incapacidad Laboral Total (OD)", value=f"{incap_OD:.2f}%")
elif modo_evaluacion == "Bilateral (Ambos ojos)":
    if file_od is not None and file_oi is not None:
        incapacidad_bilateral = (incap_OD + incap_OI) * 1.5
        st.write(f"Incapacidad Ojo Derecho: {incap_OD:.2f}%")
        st.write(f"Incapacidad Ojo Izquierdo: {incap_OI:.2f}%")
        st.write(f"Suma aritmética: {(incap_OD + incap_OI):.2f}%")
        st.write("**Índice de Bilateralidad aplicado:** x 1.5")
        st.metric(label="Incapacidad Laboral Total (Bilateral)", value=f"{incapacidad_bilateral:.2f}%")
    else:
        st.warning("Por favor, suba las imágenes de ambos ojos para calcular la incapacidad bilateral.")
