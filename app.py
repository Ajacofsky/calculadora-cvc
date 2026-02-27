import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import traceback

# Configuración inicial de la página
st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

# ==========================================
# 1. MOTOR DE VISIÓN COMPUTARIZADA Y LÓGICA
# ==========================================

def procesar_campo_visual(image_bytes):
    try:
        if not image_bytes:
            return None, 0, 0, "No hay imagen"

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, 0, 0, "No se pudo decodificar la imagen."

        img_heatmap = img.copy()
        overlay = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alto, ancho = gray.shape
        
        # Binarización Adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
        
        # --- A. CALIBRACIÓN GEOMÉTRICA ---
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
            pixels_por_10_grados = float(dist_60 / 6.0)
        else:
            dist_60 = int((ancho - cx) * 0.75)
            pixels_por_10_grados = float(dist_60 / 6.0)

        if pixels_por_10_grados <= 0:
            pixels_por_10_grados = 1.0 

        puntos_totales = []

        # --- B. EXTRACCIÓN ESTRUCTURAL PURA ---
        
        # 1. ATRApar SÓLO LOS CUADRADOS (Erosión destructiva)
        # Adaptamos el tamaño de la "lija" a la resolución de la foto
        k_size = max(3, int(pixels_por_10_grados * 0.08))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        eroded_squares = cv2.erode(thresh, kernel_erode, iterations=1)
        
        contornos_sq, _ = cv2.findContours(eroded_squares, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centros_cuadrados = []
        
        for cnt in contornos_sq:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 1 and h > 1: # Ignorar ruido diminuto
                px, py = x + w//2, y + h//2
                centros_cuadrados.append((px, py))
                
                dx, dy = px - cx, py - cy
                r_pixel = math.hypot(dx, dy)
                g_fisicos = (r_pixel / pixels_por_10_grados) * 10
                if 2 < g_fisicos <= 41:
                    ang = math.degrees(math.atan2(dy, dx))
                    if ang < 0: ang += 360
                    puntos_totales.append({'r': g_fisicos, 'ang': ang, 'tipo': 'fallado'})
                    cv2.circle(img_heatmap, (px, py), 4, (0, 0, 255), -1) # Rojo

        # 2. ATRAPAR SÓLO LOS CÍRCULOS (Por descarte)
        ejes_unidos = cv2.bitwise_or(lineas_h, lineas_v)
        ejes_dilated = cv2.dilate(ejes_unidos, np.ones((3,3), np.uint8))
        thresh_sin_ejes = cv2.subtract(thresh, ejes_dilated)
        
        # Ocultar los cuadrados que ya atrapamos para no confundirlos
        mask_cuadrados = np.zeros_like(thresh)
        for px, py in centros_cuadrados:
            cv2.circle(mask_cuadrados, (px, py), int(k_size * 2), 255, -1)
            
        thresh_solo_circulos = cv2.subtract(thresh_sin_ejes, mask_cuadrados)
        contornos_circ, _ = cv2.findContours(thresh_solo_circulos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centros_circulos = []
        max_area_esperada = (pixels_por_10_grados * 0.8) ** 2
        
        for cnt in contornos_circ:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if 4 < area < max_area_esperada:
                px, py = x + w//2, y + h//2
                
                # Agrupar fragmentos del mismo círculo si fue cortado por la cruz
                es_nuevo = True
                for cx_circ, cy_circ in centros_circulos:
                    if math.hypot(px - cx_circ, py - cy_circ) < (pixels_por_10_grados * 0.3):
                        es_nuevo = False
                        break
                
                if es_nuevo:
                    centros_circulos.append((px, py))
                    dx, dy = px - cx, py - cy
                    r_pixel = math.hypot(dx, dy)
                    g_fisicos = (r_pixel / pixels_por_10_grados) * 10
                    if 2 < g_fisicos <= 41:
                        ang = math.degrees(math.atan2(dy, dx))
                        if ang < 0: ang += 360
                        puntos_totales.append({'r': g_fisicos, 'ang': ang, 'tipo': 'visto'})
                        cv2.circle(img_heatmap, (px, py), 2, (0, 255, 0), -1) # Verde

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
                    r_in = limite_inf * (pixels_por_10_grados/10.0)
                    r_out = limite_sup * (pixels_por_10_grados/10.0)
                    r_center = int((r_in + r_out) / 2.0)
                    grosor = int(r_out - r_in)
                    
                    if grosor > 0:
                        cv2.ellipse(overlay, (cx, cy), (r_center, r_center), 0, ang_inf, ang_sup, color_zona, grosor + 1)

        cv2.addWeighted(overlay, 0.4, img_heatmap, 0.6, 0, img_heatmap)
        
        for angulo_linea in range(0, 360, 45):
            rad = math.radians(angulo_linea)
            x2 = int(cx + (4.2 * pixels_por_10_grados) * math.cos(rad))
            y2 =
