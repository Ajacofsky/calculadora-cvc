import streamlit as st
import cv2
import numpy as np
import math

# ==========================================
# 1. MOTOR DE VISIÓN COMPUTARIZADA Y LÓGICA
# ==========================================

def procesar_campo_visual(image_bytes):
    # Verificación de seguridad para Streamlit Cloud
    if not image_bytes:
        return None, 0, 0

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, 0, 0

    img_heatmap = img.copy()
    overlay = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- A. CALIBRACIÓN GEOMÉTRICA ---
    alto, ancho = gray.shape
    cx, cy = int(ancho / 2), int(alto / 2) 
    
    # Estimación de la marca de 60 grados (ajustado a las imágenes subidas)
    distancia_60_grados = int((ancho - cx) * 0.8) 
    pixels_por_10_grados = distancia_60_grados / 6.0
    
    # --- B. DETECCIÓN DE PUNTOS (SÍMBOLOS) ---
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    puntos_totales = []
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area_caja = w * h
        
        # Filtrar por tamaño para ignorar texto y ejes (ajustado a los CVC de Humphrey)
        if 15 < area_caja < 250:
            # Nuevo método de OCR: Densidad de píxeles reales
            roi = thresh[y:y+h, x:x+w]
            pixeles_activos = cv2.countNonZero(roi)
            densidad_forma = pixeles_activos / float(area_caja)
            
            # Centroide exacto
            px = x + (w // 2)
            py = y + (h // 2)
            
            # Coordenadas polares
            dx, dy = px - cx, py - cy
            radio_pixel = math.sqrt(dx**2 + dy**2)
            grados_fisicos = (radio_pixel / pixels_por_10_grados) * 10
            
            angulo = math.degrees(math.atan2(dy, dx))
            if angulo < 0: angulo += 360
            
            if grados_fisicos <= 40:
                # El cuadrado ■ es macizo (densidad > 0.6). El círculo ○ es hueco.
                tipo = 'fallado' if densidad_forma > 0.6 else 'visto'
                puntos_totales.append({'r': grados_fisicos, 'ang': angulo, 'tipo': tipo})

    # --- C. ANÁLISIS POR ZONAS (Densidad) ---
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

# ==========================================
# 2. INTERFAZ DE USUARIO (WEB APP)
# ==========================================

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("Basado en baremo legal y método de densidad de ocupación por octantes.")

modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"])

col1, col2 = st.columns(2)

incap_OD = 0.0
incap_OI = 0.0

with col1:
    st.subheader("Ojo Derecho (OD)")
    file_od = st.file_uploader("Subir imagen CVC Ojo Derecho", type=["jpg", "jpeg", "png"])
    if file_od is not None:
        st.info("Procesando imagen...")
        # Usamos getvalue() para proteger el buffer en la nube
        img_res_od, grados_od, incap_OD = procesar_campo_visual(file_od.getvalue())
        
        if img_res_od is not None:
            # Quitamos cvtColor y usamos channels="BGR" nativo de Streamlit
            st.image(img_res_od, channels="BGR", caption="Mapa de Calor Generado - OD", use_container_width=True)
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
                st.image(img_res_oi, channels="BGR", caption="Mapa de Calor Generado - OI", use_container_width=True)
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
