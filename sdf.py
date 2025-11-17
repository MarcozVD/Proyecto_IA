# -*- coding: utf-8 -*-

# --- Librerías ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import io
import joblib

# ==========================================================
# CARGA DEL DATASET
# ==========================================================
sensor_file = 'MetroPT3(AirCompressor).csv'
print("Cargando:", sensor_file)

df = pd.read_csv(sensor_file, parse_dates=['timestamp'])
print("Shape:", df.shape)
print(df.head())

print("Nulos:\n", df.isna().sum())

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

print("\nDistribución de fallas:")
print(df['falla'].value_counts())

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
X = df.drop(columns=['falla','timestamp','Unnamed: 0'])
y = df['falla']

print("\nPrimeras filas de X:\n", X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balanceo SMOTE - SOLO TRAIN
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("\nAntes de SMOTE:", y_train.value_counts().to_dict())
print("Después de SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ==========================================================
# KNN + HIPERPARAMETRIZACIÓN
# ==========================================================
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

print("\n🔍 Buscando mejores hiperparámetros...\n")

grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train_res, y_train_res)

best_knn = grid.best_estimator_
print("Mejores parámetros:", grid.best_params_)

# ==========================================================
# PREDICCIÓN EN TEST NORMAL (desbalanceado)
# ==========================================================
y_pred_test = best_knn.predict(X_test_scaled)
probs_test = best_knn.predict_proba(X_test_scaled)[:,1]

print("\n=========== RESULTADOS EN TEST DESBALANCEADO ===========\n")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1 Score:", f1_score(y_test, y_pred_test))
print("ROC-AUC:", roc_auc_score(y_test, probs_test))

print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred_test))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_test))

# ==========================================================
# BALANCEAR TEST SOLO PARA EVALUACIÓN
# ==========================================================
sm_test = SMOTE(random_state=42)
X_test_res, y_test_res = sm_test.fit_resample(X_test_scaled, y_test)

y_pred_bal = best_knn.predict(X_test_res)

print("\n=========== RESULTADOS EN TEST BALANCEADO ===========\n")
print("Accuracy:", accuracy_score(y_test_res, y_pred_bal))
print("Recall:", recall_score(y_test_res, y_pred_bal))
print("F1 Score:", f1_score(y_test_res, y_pred_bal))

print("\nMatriz de confusión:\n", confusion_matrix(y_test_res, y_pred_bal))
print("\nReporte de clasificación (BALANCEADO):\n", classification_report(y_test_res, y_pred_bal))

# ==========================================================
# GUARDADO DE MODELO
# ==========================================================
os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler_knn.pkl")
joblib.dump(best_knn, "models/modelo_knn.pkl")

print("\nModelo y scaler guardados en /models/")
