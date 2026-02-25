import streamlit as st
import cv2
import numpy as np
import math

# ==========================================
# 1. MOTOR DE VISIÓN COMPUTARIZADA Y LÓGICA
# ==========================================

def procesar_campo_visual(image_bytes):
    """
    Procesa la imagen del CVC, detecta la escala, los puntos, 
    genera el mapa de calor y calcula los grados perdidos.
    """
    # Convertir los bytes subidos a una imagen de OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_heatmap = img.copy()
    overlay = img.copy() # Capa transparente para el mapa de calor
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- A. CALIBRACIÓN GEOMÉTRICA ---
    # En un entorno real, aquí usamos detección de líneas (HoughLines) para hallar los ejes x,y.
    # Para este código, simularemos la detección del centro y la marca de 60°.
    alto, ancho = gray.shape
    cx, cy = int(ancho / 2), int(alto / 2) # Asumimos el centro de la imagen (ajustable)
    
    # Asumimos que la marca de 60 grados está a un 80% del centro hacia el borde derecho
    distancia_60_grados = int((ancho - cx) * 0.8) 
    
    # TU LÓGICA DE ESCALA: Dividimos en 6 para saber cuánto vale 10 grados en píxeles
    pixels_por_10_grados = distancia_60_grados / 6.0
    
    # --- B. DETECCIÓN DE PUNTOS (SÍMBOLOS) ---
    # Binarizar para resaltar objetos oscuros
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Detectar contornos (cuadraditos y círculos)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    puntos_totales = []  # Lista de dicts: {'x': x, 'y': y, 'tipo': 'visto' o 'fallado', 'r': radio, 'angulo': angulo}
    
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if 15 < area < 200: # Filtrar ruido y texto
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])
                
                # Calcular solidez para diferenciar cuadrado (■) de círculo (○)
                x, y, w, h = cv2.boundingRect(cnt)
                solidez = area / float(w * h)
                
                # Calcular coordenadas polares respecto al centro (cx, cy)
                dx, dy = px - cx, py - cy
                radio_pixel = math.sqrt(dx**2 + dy**2)
                grados_fisicos = (radio_pixel / pixels_por_10_grados) * 10
                
                # Calcular ángulo (0 a 360 grados)
                angulo = math.degrees(math.atan2(dy, dx))
                if angulo < 0: angulo += 360
                
                # Si está dentro de los 40 grados de interés
                if grados_fisicos <= 40:
                    tipo = 'fallado' if solidez > 0.7 else 'visto' # >0.7 suele ser un cuadrado sólido
                    puntos_totales.append({'r': grados_fisicos, 'ang': angulo, 'tipo': tipo})

    # --- C. ANÁLISIS POR ZONAS (32 Zonas de 10° cada una) ---
    grados_no_vistos_total = 0
    
    # Dibujar anillos de referencia (10, 20, 30, 40)
    for i in range(1, 5):
        radio_dibujo = int(i * pixels_por_10_grados)
        cv2.circle(img_heatmap, (cx, cy), radio_dibujo, (0, 0, 255), 1) # Anillos rojos
        
    # Analizar cada uno de los 4 anillos (0-10, 10-20, 20-30, 30-40)
    for anillo in range(4):
        limite_inf = anillo * 10
        limite_sup = (anillo + 1) * 10
        
        # Analizar cada uno de los 8 octantes (45 grados cada uno)
        for octante in range(8):
            ang_inf = octante * 45
            ang_sup = (octante + 1) * 45
            
            # Filtrar los puntos que caen exactamente en esta porción de la "tarta"
            puntos_zona = [p for p in puntos_totales if limite_inf <= p['r'] < limite_sup and ang_inf <= p['ang'] < ang_sup]
            
            total_pts = len(puntos_zona)
            fallados = sum(1 for p in puntos_zona if p['tipo'] == 'fallado')
            
            color_zona = None
            grados_perdidos = 0
            
            if total_pts > 0:
                densidad = (fallados / total_pts) * 100
                
                # TU REGLA DE ORO DE DENSIDAD
                if densidad >= 70:
                    grados_perdidos = 10
                    color_zona = (255, 200, 0) # Celeste en BGR
                elif 0 < densidad < 70:
                    grados_perdidos = 5
                    color_zona = (0, 255, 255) # Amarillo en BGR
                else:
                    grados_perdidos = 0
                    color_zona = None # Transparente
            
            grados_no_vistos_total += grados_perdidos
            
            # Dibujar la porción del mapa de calor si tiene color
            if color_zona:
                r_in = int(limite_inf * (pixels_por_10_grados/10))
                r_out = int(limite_sup * (pixels_por_10_grados/10))
                cv2.ellipse(overlay, (cx, cy), (r_out, r_out), 0, ang_inf, ang_sup, color_zona, -1)
                cv2.ellipse(overlay, (cx, cy), (r_in, r_in), 0, ang_inf, ang_sup, (255,255,255), -1) # "Agujero" central

    # Fusionar el mapa de calor transparente (40% opacidad) con la imagen original
    cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
    
    # Dibujar ejes cartesianos y bisectrices
    for angulo_linea in range(0, 360, 45):
        rad = math.radians(angulo_linea)
        x2 = int(cx + (4 * pixels_por_10_grados) * math.cos(rad))
        y2 = int(cy + (4 * pixels_por_10_grados) * math.sin(rad))
        cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

    # Calcular Incapacidad del Ojo
    # Fórmula: (Grados No Vistos / 320) * 100 * 0.25
    porcentaje_perdida_cv = (grados_no_vistos_total / 320.0) * 100
    incapacidad_ojo = porcentaje_perdida_cv * 0.25

    return img_heatmap, grados_no_vistos_total, incapacidad_ojo

# ==========================================
# 2. INTERFAZ DE USUARIO (WEB APP)
# ==========================================

st.set_page_config(page_title="Calculadora Pericial de Campo Visual", layout="wide")

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("Basado en baremo legal y método de densidad de ocupación por octantes.")

# Opciones de Evaluación
modo_evaluacion = st.radio("Seleccione el tipo de evaluación:", ["Unilateral (Un solo ojo)", "Bilateral (Ambos ojos)"])

col1, col2 = st.columns(2)

incap_OD = 0.0
incap_OI = 0.0

with col1:
    st.subheader("Ojo Derecho (OD)")
    file_od = st.file_uploader("Subir imagen CVC Ojo Derecho (JPG)", type=["jpg", "jpeg"])
    if file_od is not None:
        st.info("Procesando imagen...")
        img_res_od, grados_od, incap_OD = procesar_campo_visual(file_od.read())
        # Convertir BGR a RGB para mostrar en Streamlit
        st.image(cv2.cvtColor(img_res_od, cv2.COLOR_BGR2RGB), caption="Mapa de Calor Generado - OD", use_container_width=True)
        st.success(f"**Grados No Vistos:** {grados_od}° / 320°")
        st.success(f"**Incapacidad OD:** {incap_OD:.2f}%")

with col2:
    if modo_evaluacion == "Bilateral (Ambos ojos)":
        st.subheader("Ojo Izquierdo (OI)")
        file_oi = st.file_uploader("Subir imagen CVC Ojo Izquierdo (JPG)", type=["jpg", "jpeg"])
        if file_oi is not None:
            st.info("Procesando imagen...")
            img_res_oi, grados_oi, incap_OI = procesar_campo_visual(file_oi.read())
            st.image(cv2.cvtColor(img_res_oi, cv2.COLOR_BGR2RGB), caption="Mapa de Calor Generado - OI", use_container_width=True)
            st.success(f"**Grados No Vistos:** {grados_oi}° / 320°")
            st.success(f"**Incapacidad OI:** {incap_OI:.2f}%")

# ==========================================
# 3. RESULTADO FINAL LEGAL
# ==========================================
st.divider()
st.header("📊 Informe Final de Incapacidad Visual")

if modo_evaluacion == "Unilateral (Un solo ojo)":
    if file_od is not None:
        st.metric(label="Incapacidad Laboral Total (OD)", value=f"{incap_OD:.2f}%")
elif modo_evaluacion == "Bilateral (Ambos ojos)":
    if file_od is not None and file_oi is not None:
        # Sumar incapacidades de ambos ojos y aplicar índice de bilateralidad (1,5)
        incapacidad_bilateral = (incap_OD + incap_OI) * 1.5
        
        st.write(f"Incapacidad Ojo Derecho: {incap_OD:.2f}%")
        st.write(f"Incapacidad Ojo Izquierdo: {incap_OI:.2f}%")
        st.write(f"Suma aritmética: {(incap_OD + incap_OI):.2f}%")
        st.write("**Índice de Bilateralidad aplicado:** x 1.5")
        st.metric(label="Incapacidad Laboral Total (Bilateral)", value=f"{incapacidad_bilateral:.2f}%")
    else:
        st.warning("Por favor, suba las imágenes de ambos ojos para calcular la incapacidad bilateral.")
