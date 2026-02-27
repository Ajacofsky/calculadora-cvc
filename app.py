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
    
    # 1. Binarización Adaptativa
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    
    # --- A. CALIBRACIÓN GEOMÉTRICA BLINDADA ---
    mask_ejes = thresh.copy()
    limite_inferior = int(alto * 0.75)
    mask_ejes[limite_inferior:, :] = 0 
    
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

    # --- B. DETECCIÓN POR FUSIÓN Y DENSIDAD ---
    
    # Aislar ejes y dilatarlos suavemente
    ejes_unidos = cv2.bitwise_or(lineas_h, lineas_v)
    ejes_dilated = cv2.dilate(ejes_unidos, np.ones((3,3), np.uint8))
    
    # Restar ejes para aislar símbolos
    thresh_sin_ejes = cv2.subtract(thresh, ejes_dilated)
    
    # MAGIA 1: Fusión. Cerramos los huecos que dejaron los ejes al cortar los cuadrados
    thresh_fusion = cv2.morphologyEx(thresh_sin_ejes, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    contornos, _ = cv2.findContours(thresh_fusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    puntos_totales = []
    
    max_area_esperada = (pixels_por_10_grados * 0.8) ** 2
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtrar ruido minúsculo
        if 5 < w * h < max_area_esperada:
            # Evaluar el contenido exacto dentro de la caja
            roi_eval = thresh_sin_ejes[y:y+h, x:x+w]
            
            # Prueba de robustez: Erosión 2x2. (Destruye líneas finas, sobreviven bloques gruesos)
            eroded = cv2.erode(roi_eval, np.ones((2,2), np.uint8), iterations=1)
            
            # Prueba de Densidad
            densidad_caja = cv2.countNonZero(roi_eval) / float(w * h)
            
            px, py = x + w//2, y + h//2
            dx, dy = px - cx, py - cy
            r_pixel = math.hypot(dx, dy)
            g_fisicos = (r_pixel / pixels_por_10_grados) * 10
            
            if 2 < g_fisicos <= 41:
                ang = math.degrees(math.atan2(dy, dx))
                if ang < 0: ang += 360
                
                # REGLA DEFINITIVA: Si sobrevivió a la erosión (bloque) o si está lleno más del 40%
                if cv2.countNonZero(eroded) > 0 or densidad_caja > 0.40:
                    tipo = 'fallado'
                    cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1) # Rojo
                else:
                    tipo = 'visto'
                    cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1) # Verde
                    
                puntos_totales.append({'r': g_fisicos, 'ang': ang, 'tipo': tipo})

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
            
            # MAGIA 2: Dibujo de trazo grueso sin sobrescritura blanca
            if color_zona:
                r_in = limite_inf * (pixels_por_10_grados/10)
                r_out = limite_sup * (pixels_por_10_grados/10)
                r_center = int((r_in + r_out) / 2.0)
                grosor = int(r_out - r_in)
                
                # Se dibuja la línea gruesa que encaja exactamente en el anillo correspondiente
                cv2.ellipse(overlay, (cx, cy), (r_center, r_center), 0, ang_inf, ang_sup, color_zona, grosor + 1)

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
