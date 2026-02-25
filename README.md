# 👁️ Calculadora Pericial de Campo Visual Computarizado (CVC)

Una aplicación web desarrollada en Python diseñada para automatizar, evaluar y calcular la incapacidad visual basada en la planimetría de campos visuales computarizados. Utiliza algoritmos de Visión Computarizada (Computer Vision) para garantizar precisión algorítmica y eliminar el sesgo de la estimación visual humana.

## 📋 Descripción del Proyecto

En el ámbito médico-legal y pericial, el cálculo de la incapacidad por déficit campimétrico requiere un análisis minucioso de la densidad de escotomas (puntos no vistos) dentro de los 40 grados centrales de visión. 

Este software procesa imágenes de CVC (específicamente formatos que utilizan simbología de cuadrados negros `■` para puntos fallados y círculos `○` para puntos vistos), calibra la escala espacial automáticamente y aplica las reglas de baremación vigentes para generar un mapa de calor y el porcentaje exacto de incapacidad laboral (unilateral o bilateral).

## ⚖️ Base Teórica y Matemática (Método de Valoración)

El algoritmo divide el área de estudio (los 40° centrales) en **32 zonas de evaluación**. Estas zonas se generan intersectando 8 octantes (cuadrantes divididos por bisectrices) con 4 anillos concéntricos equidistantes (10°, 20°, 30° y 40°). Cada zona representa un valor potencial de 10 grados.

### 1. Evaluación de Densidad de Ocupación por Zona
Para cada una de las 32 zonas, el sistema detecta los puntos de estímulo y calcula la densidad matemática:

$$Densidad = \left( \frac{\text{Cuadraditos Negros}}{\
