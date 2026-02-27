import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import traceback

st.set_page_config(page_title="Calculadora Pericial de CVC", layout="wide")

# ==========================================
# MOTOR DE VISIÓN (MOTOR ORO + FILTRO FANTASMA)
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
        
        # 2. CENTRO Y ESCALA
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
        pixels_por_10_grados = float(dist_60 / 6.0)

        # Margen físico para ignorar la cruz y sus marquitas
        margen_eje = pixels_por_10_grados * 0.15

        # 3. DETECCIÓN (Motor Original Restaurado a Dinámico)
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
                
                if y2 > y1 and x2 > x
