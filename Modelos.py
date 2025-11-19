# -*- coding: utf-8 -*-



# --- Librerías de preprocesamiento y modelado ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import numpy as np
# --- Paso 1: Descargar el archivo ZIP del dataset ---


# --- Paso 4: Cargar los datos relevantes en un DataFrame de pandas ---
# Aquí tienes que ajustar el nombre del archivo CSV o txt que corresponda al dataset de sensores
# Por ejemplo:
sensor_file = 'MetroPT3(AirCompressor).csv'  # AJUSTA según el nombre real
print("Cargando sensor data desde:", sensor_file)
df = pd.read_csv(sensor_file, parse_dates=['timestamp'])
print("Dimensiones del DataFrame:", df.shape)
print(df.head())

# --- Paso 5: Inspección inicial y limpieza básica ---
print("Tipos de columna:\n", df.dtypes)
print("Valores nulos por columna:\n", df.isna().sum())

"""|### 📄 Información sobre fallos del conjunto de datos METROP3

El conjunto de datos no está etiquetado, pero los **informes de fallos proporcionados por la empresa** están disponibles en la siguiente tabla.  
Estos reportes permiten evaluar la eficacia de los **algoritmos de detección de anomalías**, **predicción de fallos** y **estimación de vida útil restante (RUL)**.

| Nº | Hora de inicio      | Hora de finalización | Tipo de falla | Severidad | Informe |
|:--:|:--------------------|:---------------------|:--------------|:-----------|:---------|
| 1 | 18/04/2020 00:00 | 18/04/2020 23:59 | Fuga de aire | Alta tensión | — |
| 2 | 29/05/2020 23:30 | 30/05/2020 06:00 | Fuga de aire | Alta tensión | Mantenimiento el 30 de abril a las 12:00 |
| 3 | 06/06/2020 10:00 | 07/06/2020 14:30 | Fuga de aire | Alta tensión | Mantenimiento el 8 de junio a las 16:00 |
| 4 | 15/07/2020 14:30 | 15/07/2020 19:00 | Fuga de aire | Alta tensión | Mantenimiento el 16 de julio a las 00:00 |

"""

# --- Paso 6: Crear una etiqueta supervisada (ejemplo: falla vs normal) ---
import io

# Data from the markdown cell
failures_data = """Nº|Hora de inicio|Hora de finalización|Tipo de falla|Severidad|Informe
1|18/04/2020 00:00|18/04/2020 23:59|Fuga de aire|Alta tensión|—
2|29/05/2020 23:30|30/05/2020 06:00|Fuga de aire|Alta tensión|Mantenimiento el 30 de abril a las 12:00
3|06/06/2020 10:00|07/06/2020 14:30|Fuga de aire|Alta tensión|Mantenimiento el 8 de junio a las 16:00
4|15/07/2020 14:30|15/07/2020 19:00|Fuga de aire|Alta tensión|Mantenimiento el 16 de julio a las 00:00
"""

# Read the string data into a DataFrame
failures = pd.read_csv(io.StringIO(failures_data), sep='|', skipinitialspace=True)
failures.columns = failures.columns.str.strip() # Remove any leading/trailing spaces from column names
# Drop the columns that are not needed for failure start and end times
failures = failures.drop(columns=['Nº', 'Tipo de falla', 'Severidad', 'Informe'])
failures.columns = ['start_time', 'end_time']

# Convert to datetime objects
failures['start_time'] = pd.to_datetime(failures['start_time'], format='%d/%m/%Y %H:%M')
failures['end_time'] = pd.to_datetime(failures['end_time'], format='%d/%m/%Y %H:%M')

# Ensure the directory exists
output_dir = '/content/metropt3_data'
os.makedirs(output_dir, exist_ok=True)

# Save to failures.csv
failures_file = os.path.join(output_dir, 'failures.csv')
failures.to_csv(failures_file, index=False)

print("Cargando informes de fallos desde:", failures_file)
failures = pd.read_csv(failures_file, parse_dates=['start_time','end_time'])
print("Informes de fallos:\n", failures.head())

# Añadir columna 'falla' al DataFrame principal
df['falla'] = 0
for _, row in failures.iterrows():
    mask = (df['timestamp'] >= row['start_time']) & (df['timestamp'] <= row['end_time'])
    df.loc[mask, 'falla'] = 1
print("Distribución de la etiqueta 'falla':\n", df['falla'].value_counts())

df.head(100000)

# ==========================================================
#  4. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================================================

# --- Distribución de la variable objetivo ---
plt.figure(figsize=(4,4))
df['falla'].value_counts().plot(kind='bar', color=['steelblue','orange'])
plt.title("Distribución de la variable objetivo (Falla vs Normal)")
plt.xticks(ticks=[0,1], labels=['Normal (0)', 'Falla (1)'])
plt.show()

# --- Estadísticas descriptivas ---
print("\nResumen estadístico:\n", df.describe())

import seaborn as sns
# --- Matriz de correlación ---
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True)
plt.title("Mapa de calor de correlaciones entre variables numéricas")
plt.show()

import math

# --- Detección visual de outliers (boxplot ejemplo) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Unnamed: 0', 'falla'])

# Calculate number of rows needed (2 plots per row)
num_plots = len(numeric_cols)
num_rows = math.ceil(num_plots / 2)

plt.figure(figsize=(10, 5 * num_rows))

for i, col in enumerate(numeric_cols):
    plt.subplot(num_rows, 2, i + 1) # Create subplot for each column
    sns.boxplot(y=df[col])
    plt.title(f'Distribución de {col}')
    plt.ylabel('') # Remove y-label to avoid clutter

plt.tight_layout()
plt.suptitle("Distribución de variables numéricas y detección de valores atípicos", y=1.02, fontsize=16)
plt.show()

"""El dataset no está etiquetado originalmente, pero tú ya generaste etiquetas de falla (1) y normal (0) a partir de los informes.
Entonces:

Si los outliers aparecen en los periodos con etiqueta de falla (1), probablemente son parte del patrón anómalo que tu modelo debe aprender.
 No los elimines; son indicadores de condición anómala.

Si los outliers aparecen fuera de esos periodos (etiqueta 0), podrían ser ruido o errores de medición, y podrías:

Reemplazarlos con interpolación,

O truncarlos (winsorizing),

O aplicar detección de outliers local (como Isolation Forest) para limpiarlos solo en la clase “normal”.
"""

df.columns



# Detección de outliers basada en rango intercuartílico (IQR)
def detectar_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

# Ejemplo con una variable
variable = 'DV_pressure'  # Cambia por una variable relevante
outliers = detectar_outliers(df, variable)

print(f"Outliers detectados en '{variable}':", len(outliers))
print("Distribución por tipo de condición:")
print(outliers['falla'].value_counts())

# Visualizar
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='falla', y=variable)
plt.title(f"Distribución de {variable} por condición (0=Normal, 1=Falla)")
plt.show()

# --- Seleccionar solo columnas numéricas (excluyendo la etiqueta y timestamp) ---
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['falla']]

# --- Crear DataFrame resumen de outliers ---
outlier_summary = []
for col in num_cols:
    outliers = detectar_outliers(df, col)
    total = len(outliers)
    if total > 0:
        fallas = outliers['falla'].sum()
        normales = total - fallas
        outlier_summary.append({
            'Variable': col,
            'Outliers totales': total,
            'En fallas (1)': fallas,
            'En normalidad (0)': normales,
            '% en fallas': round((fallas / total) * 100, 2)
        })

outlier_df = pd.DataFrame(outlier_summary).sort_values(by='% en fallas', ascending=False)

print("Resumen de outliers por variable (ordenado por % en fallas):")
print(outlier_df)

# ==========================================================
#  Visualización general de presencia de outliers
# ==========================================================
plt.figure(figsize=(10,5))
sns.barplot(data=outlier_df, x='Variable', y='% en fallas', hue='Variable', legend=False, palette='coolwarm')
plt.xticks(rotation=45, ha='right')
plt.title("Porcentaje de outliers asociados a fallas (por variable)")
plt.ylabel("% de outliers durante fallas")
plt.xlabel("Variable del sensor")
plt.show()

"""| Variable                                              | % en fallas   | Interpretación                                                                                                                          | Recomendación                                                               |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Oil_temperature (22.58%)**                          | Bastante alto | La temperatura del aceite muestra picos fuera de rango en fallas. Puede ser un **indicador temprano de sobrecarga o fricción interna.** | ✅ Conservar los outliers. Analizar tendencia temporal.                      |
| **DV_pressure (18.70%)**                              | Moderado      | La presión diferencial cambia fuertemente en fallas. Posible **fuga o bloqueo parcial.**                                                | ✅ Conservar; revisar correlación con “Towers” y “COMP”.                     |
| **LPS, H1, Towers, DV_eletric, COMP, MPG** (8–14%)    | Bajo-medio    | Muestran cierta relación con fallas, pero también valores atípicos en normalidad.                                                       | ⚠️ Analizar individualmente. Pueden requerir **normalización o suavizado**. |
| **Reservoirs, TP3, TP2 (~8%)**                        | Bajo          | No parecen determinantes, pero podrían reflejar condiciones ambientales o de mantenimiento.                                             | ⚙️ Revisar si su rango natural es amplio.                                   |
| **Pressure_switch, Caudal_impulses, Oil_level (<1%)** | Muy bajo      | Sus outliers no se asocian con fallas. Posiblemente **ruido de sensores o valores saturados.**                                          | 🚫 Pueden suavizarse o incluso descartarse.                                 |

Qué hacer con cada grupo de variables?

** Mantener los outliers (no quitarlos):**

Estas variables aportan información crítica sobre fallas:

* Oil_temperature

* DV_pressure

* LPS

Sus picos o desviaciones pueden ser síntomas reales de fallas.

El modelo debe ver esos valores extremos para aprender el patrón anómalo.


**Revisar o suavizar (según comportamiento temporal):**

Estas variables tienen outliers, pero no están tan concentrados en fallas:

H1, Towers, DV_eletric, COMP, MPG, Reservoirs, TP3, TP2

**Recomendación:**

- No los elimines completamente, pero puedes aplicar Winsorizing o clipping (limitar los valores extremos al percentil 1 y 99).

- Esto mantiene su información pero evita que los outliers exageren su peso en el modelo.
"""

for col in ['H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2']:
    p1, p99 = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], p1, p99)

"""**Eliminar o suavizar agresivamente:**

Estas variables no aportan información sobre fallas y sus outliers solo agregan ruido:

- Pressure_switch

- Caudal_impulses

- Oil_level

Puedes hacer una de estas dos cosas:

Eliminar completamente las variables del modelo si no son relevantes físicamente.
"""

df.drop(columns=['Pressure_switch', 'Caudal_impulses', 'Oil_level'], inplace=True)

df

# ==========================================================
#  5. SEPARACIÓN Y BALANCEO DE CLASES (SMOTE)
# ==========================================================
X = df.drop(columns=['falla','timestamp','Unnamed: 0'])
y = df['falla']

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Aplicar escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar SMOTE solo sobre el conjunto de entrenamiento
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("\nAntes del balanceo:", y_train.value_counts().to_dict())
print("Después del balanceo:", pd.Series(y_train_res).value_counts().to_dict())

# ==========================================================
#  6. ENTRENAR Y COMPARAR MODELOS
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier

# Diccionario de modelos a evaluar
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "LinearSVM": LinearSVC(max_iter=3000, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        scale_pos_weight=1  # ya usaste SMOTE → NO usar balanceo interno
    )
}

results = []

print("\n==================== INICIANDO ENTRENAMIENTO ====================\n")

for name, model in models.items():
    print(f"\n🔵 Entrenando modelo: {name} ...")

    # Entrenamiento
    model.fit(X_train_res, y_train_res)

    # Predicción
    y_pred = model.predict(X_test_scaled)

    # Métricas principales
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)       # muy importante para fallas
    f1 = f1_score(y_test, y_pred)

    print(f"\nResultados — {name}:")
    print("Accuracy:", round(acc, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar resultados para comparar después
    results.append({
        "Modelo": name,
        "Accuracy": acc,
        "Recall": rec,
        "F1 Score": f1
    })

# Convertir resultados en DataFrame
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)

print("\n==================== COMPARACIÓN FINAL ====================\n")
print(results_df)

best_model_name = results_df.iloc[0]["Modelo"]
print(f"\n🏆 MEJOR MODELO: **{best_model_name}** segun F1 Score\n")
