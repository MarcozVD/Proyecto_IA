## Proyecto de IA
# Marcos Valera - Daniel Villamizar
## Proyecto de IA - Sistema de Predicción de Fallas (MetroPT3)

**Proyecto:** Sistema para detección y predicción de fallas en compresores de aire usando modelos de Machine Learning.

**Autores:**
- **Marcos Valera**
- **Daniel Villamizar**

**Resumen:**
- Este repositorio contiene una aplicación web (Streamlit) y una aplicación móvil (Expo / React Native) para predecir fallas en compresores de aire usando un modelo XGBoost entrenado y optimizado. Incluye herramientas para predicción individual, procesamiento por lotes, análisis de datos y comparación de modelos (curva ROC / AUC).

**Datos:**
- Archivo principal: `MetroPT3(AirCompressor).csv`
- Registros: ~220,320
- Fallas etiquetadas: 3,168 (≈ 1.4%)

**Estructura principal del repositorio**
- `steamlit.py` — Interfaz web (Streamlit) para predicción, visualización y evaluación del modelo.
- `app.py`, `a.py`, `asd.py`, `otro.py`, `sdf.py` — scripts auxiliares (varios experimentos / utilidades).
- `copia_de_preprocesamiento_metro_p3.py` — script de preprocesamiento de datos.
- `MetroPT3(AirCompressor).csv` — dataset principal.
- `models/` — carpeta destinada a almacenar modelos y artefactos (p. ej. pesos, scalers).
- `expogo/proyecto_ia/` — aplicación móvil Expo (React Native):
	- `App.tsx` — interfaz principal de la app móvil.
	- `package.json`, `tsconfig.json`, `App.tsx`, `index.ts` y `assets/` — recursos y configuración del proyecto móvil.
- `metropt3_data/failures.csv` — listado de fallas / etiquetas derivadas.
- `requirements.txt` — dependencias Python para la parte de ML / Streamlit.
- `package.json` — dependencias del front-end / herramientas JS (cuando aplica).

**Características principales**
- Interfaz Streamlit para: entrada de sensores (sliders y texto), análisis de outliers, visualización de ROC/AUC, comparación de modelos y explicación del proceso de entrenamiento.
- Aplicación móvil Expo que replica la funcionalidad de predicción individual con validación de entradas y conexión a un servidor de predicción (endpoint: `/predict`).
- Modelo seleccionado: **XGBoost**, optimizado con **GridSearchCV**, balanceado con **SMOTE** y datos escalados con **StandardScaler**.
- Umbral óptimo de decisión usado en la app: **0.3847** (calculado mediante análisis Precision-Recall).

**Sensores (variables) y rangos usados**
Las 12 variables usadas por el modelo son (nombres tal como aparecen en código/UX):
- `TP2` — Temperatura Punto 2 (rango: -0.032 a 10.676)
- `TP3` — Temperatura Punto 3 (rango: 0.730 a 10.302)
- `H1` — Humedad Relativa (rango: -0.036 a 10.288)
- `DV_pressure` — Presión Válvula (rango: -0.032 a 9.844)
- `Reservoirs` — Nivel Reservorios (rango: 0.712 a 10.300)
- `Oil_temperature` — Temperatura Aceite (rango: 15.400 a 89.050)
- `Motor_current` — Corriente Motor (rango: 0.020 a 9.295)
- `COMP` — Estado Compresor (binario 0/1)
- `DV_eletric` — Válvula Eléctrica (binario 0/1)
- `Towers` — Torres Activas (binario 0/1)
- `MPG` — Motor Principal (binario 0/1)
- `LPS` — Sensor Baja Presión (binario 0/1)

**Instalación y ejecución**

Recomendado: usar entornos virtuales separados para Python y Node.

- Python / Streamlit (PowerShell):

```powershell
# Crear y activar entorno (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ejecutar la app Streamlit
streamlit run steamlit.py
```

- Aplicación móvil (Expo):

```powershell
# En la carpeta del cliente móvil
cd expogo\proyecto_ia
npx expo start -c
# Abrir en Expo Go (móvil) o emulador
```

**Conexión entre app móvil y servidor**
- La app móvil incluye una pantalla de conexión donde debes introducir la IP:puerto del servidor (ej.: `192.168.1.100:5000`).
- Endpoint esperado: `POST http://<IP>:<PORT>/predict` con payload JSON conteniendo las 12 variables de sensores.

**Uso rápido**
- Web: Ajusta los sensores en la interfaz (`steamlit.py`) y pulsa para predecir, analizar outliers o ver la curva ROC y comparación de modelos.
- Móvil: Conecta al servidor desde la pantalla de conexión, completa los valores de sensores y pulsa "Realizar Predicción".

**Desarrollo y notas técnicas**
- Modelo y entrenamiento: XGBoost con GridSearchCV para optimizar hiperparámetros; SMOTE para balanceo; StandardScaler para escalado.
- Umbral óptimo: calculado con criterio de Precision-Recall para maximizar F1 / trade-off entre precisión y recall.
- Si trabajas en la app móvil y ves comportamientos de teclado/perdida de foco, revisa `App.tsx` — implementaciones con `TextInput`, `useRef` y uso de `defaultValue`/drafts para evitar re-renders frecuentes.

**Contribuir**
- Forkea el repo y crea un branch por feature/bugfix.
- Asegúrate de mantener el estilo y actualizar `requirements.txt` o `package.json` si añades dependencias.



