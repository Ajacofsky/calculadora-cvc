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
    
    # Binarización automática (OTSU) para adaptarse a cualquier contraste
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- A. CALIBRACIÓN GEOMÉTRICA ---
    # Usamos kernels muy grandes para encontrar solo los ejes principales
    kernel_h_axis = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho * 0.2), 2))
    lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h_axis)
    
    kernel_v_axis = cv2.getStructuringElement(cv2.MORPH_RECT, (2, int(alto * 0.2)))
    lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v_axis)
    
    interseccion = cv2.bitwise_and(lineas_h, lineas_v)
    y_coords, x_coords = np.where(interseccion > 0)
    
    if len(x_coords) > 0:
        cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
    else:
        cx, cy = int(ancho / 2), int(alto / 2)
        
    # Calcular escala
    fila_eje_h = lineas_h[cy-5 : cy+5, :]
    _, x_h = np.where(fila_eje_h > 0)
    if len(x_h) > 0:
        dist_60 = np.max(x_h) - cx
    else:
        dist_60 = int((ancho - cx) * 0.8)
    
    pixels_por_10_grados = dist_60 / 6.0
    
    # --- B. DETECCIÓN ROBUSTA POR TAMAÑO MORFOLÓGICO ---
    puntos_totales = []
    
    # 1. Definimos el tamaño esperado de un cuadrado (aprox. 1/6 de un sector de 10°)
    tamano_simbolo_estimado = int(pixels_por_10_grados / 6.0)
    # Creamos un kernel de erosión que sea un poco más chico que un cuadrado, 
    # pero definitivamente más grande que un círculo o una línea.
    kernel_size = max(3, int(tamano_simbolo_estimado * 0.8))
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 2. APLICAMOS EROSIÓN AGRESIVA: Solo sobreviven los objetos grandes y macizos (■)
    thresh_solo_cuadrados = cv2.erode(thresh, kernel_erosion, iterations=1)
    
    # Encontramos los centros de los cuadrados sobrevivientes
    contornos_cuadrados, _ = cv2.findContours(thresh_solo_cuadrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros_cuadrados = []
    for cnt in contornos_cuadrados:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            px, py = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            centros_cuadrados.append((px, py))

    # 3. Analizamos TODOS los objetos en la imagen original
    contornos_todos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contornos_todos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w)/h if h > 0 else 0
        
        # Filtro laxo para descartar ruido muy pequeño o líneas muy largas
        if 5 < area < (pixels_por_10_grados**2) and 0.3 < aspect_ratio < 3.0:
            px, py = x + w//2, y + h//2
            
            # Coordenadas polares
            dx, dy = px - cx, py - cy
            radio_pixel = math.sqrt(dx**2 + dy**2)
            grados_fisicos = (radio_pixel / pixels_por_10_grados) * 10
            
            if 2 < grados_fisicos <= 41: # Ignoramos el centro exacto
                angulo = math.degrees(math.atan2(dy, dx))
                if angulo < 0: angulo += 360
                
                # DETERMINACIÓN DEL TIPO:
                # ¿Este punto coincide con la ubicación de un cuadrado que sobrevivió a la erosión?
                es_cuadrado_macizo = False
                for (sq_x, sq_y) in centros_cuadrados:
                    distancia = math.hypot(px - sq_x, py - sq_y)
                    if distancia < (kernel_size * 2): # Si está muy cerca de un sobreviviente
                        es_cuadrado_macizo = True
                        break
                
                if es_cuadrado_macizo:
                    tipo = 'fallado'
                    # Auditoría: Círculo ROJO relleno grande
                    cv2.circle(img_heatmap, (px, py), int(kernel_size), (0, 0, 255), -1)
                else:
                    tipo = 'visto'
                    # Auditoría: Círculo VERDE relleno pequeño
                    cv2.circle(img_heatmap, (px, py), int(kernel_size/2), (0, 255, 0), -1)
                    
                puntos_totales.append({'r': grados_fisicos, 'ang': angulo, 'tipo': tipo})

    # --- C. ANÁLISIS POR ZONAS Y MAPA DE CALOR ---
    grados_no_vistos_total = 0
    
    # Dibujar anillos de referencia
    for i in range(1, 5):
        radio_dibujo = int(i * pixels_por_10_grados)
        cv2.circle(img_heatmap, (cx, cy), radio_dibujo, (0, 0, 255), 2) # Anillos más gruesos
        
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
                    color_zona = (255, 200, 0) # Celeste (BGR)
                elif 0 < densidad < 70:
                    grados_perdidos = 5
                    color_zona = (0, 255, 255) # Amarillo (BGR)
            
            grados_no_vistos_total += grados_perdidos
            
            if color_zona:
                r_in = int(limite_inf * (pixels_por_10_grados/10))
                r_out = int(limite_sup * (pixels_por_10_grados/10))
                # Dibujar la zona coloreada
                cv2.ellipse(overlay, (cx, cy), (r_out, r_out), 0, ang_inf, ang_sup, color_zona, -1)
                # "Borrar" el centro para que quede el anillo
                cv2.ellipse(overlay, (cx, cy), (r_in, r_in), 0, ang_inf, ang_sup, (0, 0, 0), -1)

    # Crear máscara para el overlay (donde no es negro, aplicamos color)
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    img_heatmap_masked = cv2.bitwise_and(img_heatmap, img_heatmap, mask=cv2.bitwise_not(mask))
    overlay_masked = cv2.bitwise_and(overlay, overlay, mask=mask)
    final_composition = cv2.add(img_heatmap_masked, overlay_masked)
    # Mezclar con un poco de transparencia sobre la imagen original marcada
    cv2.addWeighted(final_composition, 0.5, img_heatmap, 0.5, 0, img_heatmap)

    # Dibujar ejes finales
    cv2.line(img_heatmap, (cx, 0), (cx, alto), (0, 0, 255), 2)
    cv2.line(img_heatmap, (0, cy), (ancho, cy), (0, 0, 255), 2)
    for angulo_linea in range(45, 360, 90):
        rad = math.radians(angulo_linea)
        x2 = int(cx + (4.2 * pixels_por_10_grados) * math.cos(rad))
        y2 = int(cy + (4.2 * pixels_por_10_grados) * math.sin(rad))
        cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 2)

    porcentaje_perdida_cv = (grados_no_vistos_total / 320.0) * 100
    incapacidad_ojo = porcentaje_perdida_cv * 0.25

    return img_heatmap, grados_no_vistos_total, incapacidad_ojo

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
