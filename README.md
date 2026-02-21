# Análisis Bayesiano en Turismo: Dashboard Interactivo
**Universidad Externado - Pregrado en Ciencia de Datos**

Este es un proyecto autocontenible que modela e interactúa con un problema real de toma de decisiones en turismo (Villa de Leyva) usando Inferencia Bayesiana.

## 🚀 ¿Cómo ejecutar este proyecto? (Fácil y Rápido)

Este proyecto está configurado para instalar todo lo que necesita de forma automática en un entorno virtual aislado, para que no ensucies tu PC. 

Solo necesitas tener **Python 3** instalado en tu computadora.

### Si estás en Windows:
1. Abre la carpeta del proyecto.
2. Haz doble clic en el archivo `iniciar.bat`.
3. ¡La instalación tomará un par de minutos la primera vez, y luego se abrirá tu navegador automáticamente!

### Si estás en Mac / Linux:
1. Abre una terminal en la carpeta del proyecto.
2. Ejecuta el shell script:
   ```bash
   ./iniciar.sh
   ```
3. (Si te pide permisos de ejecución, corre `chmod +x iniciar.sh` primero).

---

## 📊 ¿Qué hace esta herramienta?

El dashboard web interactivo (desarrollado en Streamlit) permite jugar de forma visual con los dos problemas del caso de estudio:

1. **Parte A: Gamma-Poisson (Conteos en la Plaza Mayor)**
   - ¿Cuántos tamales debemos cocinar basados en los turistas que llegaron hoy ($y=55$)?
   - *Impacto Visual:* Compara cómo cambia la decisión de porciones ("eficiente" vs "prudente") al enfrentar el Prior Histórico ($\mu=50$) vs el Prior Débil de Festival ($\mu=80, \sigma^2 \to \infty$).
2. **Parte B: Beta-Binomial (Ubicación en el Centro Histórico)**
   - ¿Instalamos el puesto de venta en el centro o a las afueras? (Dato real: 42 de 100 turistas se quedaron en el centro).
   - *Impacto Visual:* Muestra el choque de trenes entre la "Experiencia Local" ($p \approx 0.40$), la prior "No Informativa" y el "Hostalero Sesgado" ($p \approx 0.80$), respondiendo visualmente la pregunta: *¿quién manda, el dato o la creencia previa?*

### 🛠️ Tecnologías Usadas
*   `Python 3`
*   `Streamlit` (Interfaz Web)
*   `Scipy.stats` (Cálculos de probabilidades y funciones de densidad)
*   `Plotly` y `Matplotlib` (Gráficos interactivos)
