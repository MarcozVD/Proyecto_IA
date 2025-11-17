# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import io
import joblib
import os

# ==========================================================
# CARGA DEL DATASET
# ==========================================================
sensor_file = 'MetroPT3(AirCompressor).csv'
df = pd.read_csv(sensor_file, parse_dates=['timestamp'])

# ==========================================================
# CREACIÓN DE ETIQUETA DE FALLA
# ==========================================================
failures_data = """Nº|Hora de inicio|Hora de finalización|Tipo de falla|Severidad|Informe
1|18/04/2020 00:00|18/04/2020 23:59|Fuga de aire|Alta tensión|—
2|29/05/2020 23:30|30/05/2020 06:00|Fuga de aire|Alta tensión|Mantenimiento el 30 de abril a las 12:00
3|06/06/2020 10:00|07/06/2020 14:30|Fuga de aire|Alta tensión|Mantenimiento el 8 de junio a las 16:00
4|15/07/2020 14:30|15/07/2020 19:00|Fuga de aire|Alta tensión|Mantenimiento el 16 de julio a las 00:00
"""
failures = pd.read_csv(io.StringIO(failures_data), sep='|')
failures = failures[['Hora de inicio', 'Hora de finalización']]
failures.columns = ['start', 'end']
failures['start'] = pd.to_datetime(failures['start'], format='%d/%m/%Y %H:%M')
failures['end'] = pd.to_datetime(failures['end'], format='%d/%m/%Y %H:%M')

df['falla'] = 0
for _, row in failures.iterrows():
    df.loc[(df['timestamp'] >= row['start']) & (df['timestamp'] <= row['end']), 'falla'] = 1

# ==========================================================
# LIMPIEZA Y OUTLIERS
# ==========================================================
cols_to_clip = ['H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2']
for col in cols_to_clip:
    p1, p99 = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], p1, p99)

df.drop(columns=['Pressure_switch', 'Caudal_impulses', 'Oil_level'], inplace=True)

# ==========================================================
# SEPARACIÓN Y BALANCEO
# ==========================================================
X = df.drop(columns=['falla','timestamp','Unnamed: 0'], errors='ignore')
y = df['falla']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# ==========================================================
# KNN + HIPERPARAMETRIZACIÓN
# ==========================================================
param_grid = {
    "n_neighbors": [15, 25, 35],  # vecinos más grandes para no sobreajustar
    "weights": ["distance"],       # pesos por distancia
    "metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1
)
grid.fit(X_train_res, y_train_res)
best_knn = grid.best_estimator_

# ==========================================================
# PREDICCIÓN EN TEST DESBALANCEADO
# ==========================================================
probs_test = best_knn.predict_proba(X_test_scaled)[:,1]

# Ajustamos el umbral para la clase minoritaria
threshold = 0.1  # puedes ajustar entre 0.05 y 0.2
y_pred_test = (probs_test > threshold).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, probs_test))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_test))

# ==========================================================
# GUARDADO DE MODELO
# ==========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler_knn.pkl")
joblib.dump(best_knn, "models/modelo_knn.pkl")
