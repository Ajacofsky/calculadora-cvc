import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import traceback

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

# ==========================================
# MOTOR DE VISIÓN (GRILLA ORIGINAL + FILTRO FANTASMA)
# ==========================================

def procesar_campo_visual(image_bytes):
    try:
        if not image_bytes:
            return None, 0, 0, "No hay imagen"

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None, 0, 0, "Formato inválido."

        img_heatmap = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alto, ancho = gray.shape
        
        # 1. BINARIZACIÓN
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # 2. CENTRO Y ESCALA (LA GRILLA EXACTA QUE FUNCIONABA)
        mask_ejes = thresh.copy()
        mask_ejes[int(alto*0.75):, :] = 0 # Ocultar tabla inferior
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(ancho*0.15), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(alto*0.15)))
        lineas_h = cv2.morphologyEx(mask_ejes, cv2.MORPH_OPEN, kernel_h)
        lineas_v = cv2.morphologyEx(mask_ejes, cv2.MORPH_OPEN, kernel_v)
        
        inter = cv2.bitwise_and(lineas_h, lineas_v)
        y_coords, x_coords = np.where(inter > 0)
        
        if len(x_coords) > 0:
            cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
        else:
            cx, cy = ancho//2, alto//2
            
        eje_derecho = lineas_h[cy-5:cy+5, cx:]
        _, x_h = np.where(eje_derecho > 0)
        dist_60 = np.max(x_h) if len(x_h) > 0 else (ancho - cx)*0.75
        
        # EL CÁLCULO ORIGINAL QUE NO DEBÍ TOCAR
        pixels_por_10_grados = float(dist_60 / 6.0)

        # Margen físico para ignorar la cruz y sus marquitas
        margen_eje = pixels_por_10_grados * 0.15

        # 3. DETECCIÓN (Motor de Núcleo del 40%)
        grilla = cv2.bitwise_or(lineas_h, lineas_v)
        grilla_dilatada = cv2.dilate(grilla, np.ones((3,3), np.uint8))
        
        simbolos_aislados = cv2.subtract(mask_ejes, grilla_dilatada)
        simbolos_aislados = cv2.morphologyEx(simbolos_aislados, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        contornos, _ = cv2.findContours(simbolos_aislados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        puntos_zona = {a: {o: {'v':0, 'f':0} for o in range(8)} for a in range(4)}
        
        area_min = (ancho * 0.002) ** 2
        area_max = (ancho * 0.025) ** 2
        
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if area_min < area < area_max and 0.4 < aspect_ratio < 2.5:
                px, py = x + w//2, y + h//2
                dx, dy = px - cx, py - cy
                
                # FILTRO FANTASMA: Si toca el eje central, se ignora
                if abs(dx) < margen_eje or abs(dy) < margen_eje:
                    continue
                
                roi = thresh[y:y+h, x:x+w]
                y1, y2 = int(h*0.3), int(h*0.7)
                x1, x2 = int(w*0.3), int(w*0.7)
                
                if y2 > y1 and x2 > x1:
                    corazon = roi[y1:y2, x1:x2]
                    densidad_tinta = cv2.countNonZero(corazon) / float(corazon.size)
                    tipo = 'fallado' if densidad_tinta > 0.45 else 'visto'
                    
                    r_deg = (math.hypot(dx, dy) / pixels_por_10_grados) * 10.0
                    
                    if 1 <= r_deg <= 41:
                        ang = (math.degrees(math.atan2(dy, dx)) + 360.001) % 360
                        anillo = min(3, int(r_deg // 10))
                        octante = min(7, int(ang // 45))
                        
                        if tipo == 'fallado':
                            puntos_zona[anillo][octante]['f'] += 1
                            cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1)
                        else:
                            puntos_zona[anillo][octante]['v'] += 1
                            cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1)

        # 4. PINTURA DE OCTANTES (Regla del 70%)
        overlay = np.zeros_like(img, dtype=np.uint8)
        grados_no_vistos = 0
        
        for a in range(4):
            r_in = int(a * pixels_por_10_grados)
            r_out = int((a + 1) * pixels_por_10_grados)
            
            for o in range(8):
                ang_in, ang_out = o * 45, (o + 1) * 45
                f = puntos_zona[a][o]['f']
                v = puntos_zona[a][o]['v']
                
                if (f + v) > 0:
                    pct = (f / float(f + v)) * 100
                    color = None
                    if pct >= 70:
                        color = (255, 200, 0) # Celeste BGR
                        grados_no_vistos += 10
                    elif pct > 0:
                        color = (0, 255, 255) # Amarillo BGR
                        grados_no_vistos += 5
                        
                    if color:
                        mask = np.zeros((alto, ancho), dtype=np.uint8)
                        cv2.ellipse(mask, (cx, cy), (r_out, r_out), 0, ang_in, ang_out, 255, -1)
                        if r_in > 0:
                            cv2.ellipse(mask, (cx, cy), (r_in, r_in), 0, ang_in, ang_out, 0, -1)
                        overlay[mask == 255] = color

        gray_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        alpha = 0.5
        for c in range(3):
            img_heatmap[:,:,c] = np.where(gray_mask > 0, 
                                          img_heatmap[:,:,c] * (1 - alpha) + overlay[:,:,c] * alpha, 
                                          img_heatmap[:,:,c])

        # Dibujar grilla final
        for i in range(1, 5):
            cv2.circle(img_heatmap, (cx, cy), int(i * pixels_por_10_grados), (0, 0, 255), 1)
        for i in range(8):
            rad = math.radians(i * 45)
            x2 = int(cx + 4.2 * pixels_por_10_grados * math.cos(rad))
            y2 = int(cy + 4.2 * pixels_por_10_grados * math.sin(rad))
            cv2.line(img_heatmap, (cx, cy), (x2, y2), (0, 0, 255), 1)

        incapacidad = (grados_no_vistos / 320.0) * 100 * 0.25
        return img_heatmap, grados_no_vistos, incapacidad, None

    except Exception as e:
        return None, 0, 0, traceback.format_exc()

# ==========================================
# INTERFAZ WEB
# ==========================================

st.title("👁️ Evaluación Legal de Campo Visual Computarizado")
st.markdown("""
**Versión Activa:** Grilla Original Restaurada + Regla 70% + Filtro Anti-Marcas.
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
            with st.spinner("Procesando matriz definitiva..."):
                img_res, grados, incap, error_msg = procesar_campo_visual(file.getvalue())
            
            if error_msg:
                st.error("Error del sistema:")
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
