# -*- coding: utf-8 -*-

# --- Librerías de preprocesamiento y modelado ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import seaborn as sns
import io
import math
import joblib   # ← necesario para guardar scaler y modelo

# --- Carga del dataset ---
sensor_file = 'MetroPT3(AirCompressor).csv'
print("Cargando sensor data desde:", sensor_file)
df = pd.read_csv(sensor_file, parse_dates=['timestamp'])
print("Dimensiones del DataFrame:", df.shape)
print(df.head())

# --- Limpieza ---
print("Tipos de columna:\n", df.dtypes)
print("Valores nulos por columna:\n", df.isna().sum())

# ============================
# CREACIÓN DE ETIQUETA FALLA
# ============================
failures_data = """Nº|Hora de inicio|Hora de finalización|Tipo de falla|Severidad|Informe
1|18/04/2020 00:00|18/04/2020 23:59|Fuga de aire|Alta tensión|—
2|29/05/2020 23:30|30/05/2020 06:00|Fuga de aire|Alta tensión|Mantenimiento el 30 de abril a las 12:00
3|06/06/2020 10:00|07/06/2020 14:30|Fuga de aire|Alta tensión|Mantenimiento el 8 de junio a las 16:00
4|15/07/2020 14:30|15/07/2020 19:00|Fuga de aire|Alta tensión|Mantenimiento el 16 de julio a las 00:00
"""

failures = pd.read_csv(io.StringIO(failures_data), sep='|', skipinitialspace=True)
failures = failures.drop(columns=['Nº','Tipo de falla','Severidad','Informe'])
failures.columns = ['start_time','end_time']
failures['start_time'] = pd.to_datetime(failures['start_time'], format='%d/%m/%Y %H:%M')
failures['end_time'] = pd.to_datetime(failures['end_time'], format='%d/%m/%Y %H:%M')

df['falla'] = 0
for _, row in failures.iterrows():
    mask = (df['timestamp'] >= row['start_time']) & (df['timestamp'] <= row['end_time'])
    df.loc[mask, 'falla'] = 1

print("Distribución de fallas:\n", df['falla'].value_counts())

# ==========================================================
# TRATAMIENTO DE OUTLIERS Y PREPROCESAMIENTO
# ==========================================================
def detectar_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5*IQR
    high = Q3 + 1.5*IQR
    return df[(df[col] < low) | (df[col] > high)]

for col in ['H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2']:
    p1, p99 = np.percentile(df[col],[1,99])
    df[col] = np.clip(df[col],p1,p99)

df.drop(columns=['Pressure_switch','Caudal_impulses','Oil_level'], inplace=True)

# ==========================================================
# SEPARACIÓN Y BALANCEO DE CLASES
# ==========================================================
X = df.drop(columns=['falla','timestamp','Unnamed: 0'])
y = df['falla']
print('hola', X.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanceo
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("Antes SMOTE:", y_train.value_counts().to_dict())
print("Después SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ==========================================================
#  SOLO KNN + HIPERPARAMETRIZACIÓN
# ==========================================================
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

print("\n🔵 Buscando mejores hiperparámetros para KNN...\n")

grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_res, y_train_res)

print("\n🎯 MEJORES PARÁMETROS ENCONTRADOS:")
print(grid.best_params_)

# Entrenar con el mejor modelo
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

# ==========================================================
# MÉTRICAS FINALES
# ==========================================================
print("\n================ RESULTADOS FINALES KNN ================\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nClasification Report:")
print(classification_report(y_test, y_pred))

print("\n🏆 Modelo final entrenado: KNN optimizado con GridSearchCV")

# ==========================================================
#  GUARDAR SCALER Y MODELO ENTRENADO
# ==========================================================
os.makedirs("models", exist_ok=True)

scaler_path = "models/scaler_knn.pkl"
model_path = "models/modelo_knn.pkl"

joblib.dump(scaler, scaler_path)
joblib.dump(best_knn, model_path)

print(f"\n💾 Scaler guardado en: {scaler_path}")
print(f"💾 Modelo KNN guardado en: {model_path}")
