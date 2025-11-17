# -*- coding: utf-8 -*-
# KNN con split temporal correcto

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import io
import joblib

# ==========================================================
# 1. CARGAR DATA
# ==========================================================
df = pd.read_csv("MetroPT3(AirCompressor).csv", parse_dates=["timestamp"])

# ==========================================================
# 2. CREAR LA ETIQUETA DE FALLA
# ==========================================================
failures_data = """Nº|Hora de inicio|Hora de finalización|Tipo|Severidad|Informe
1|18/04/2020 00:00|18/04/2020 23:59|Fuga|Alta|—
2|29/05/2020 23:30|30/05/2020 06:00|Fuga|Alta|—
3|06/06/2020 10:00|07/06/2020 14:30|Fuga|Alta|—
4|15/07/2020 14:30|15/07/2020 19:00|Fuga|Alta|—
"""

failures = pd.read_csv(io.StringIO(failures_data), sep="|")[["Hora de inicio", "Hora de finalización"]]
failures.columns = ["start", "end"]

failures["start"] = pd.to_datetime(failures["start"], format="%d/%m/%Y %H:%M")
failures["end"]   = pd.to_datetime(failures["end"],   format="%d/%m/%Y %H:%M")

df["falla"] = 0

for _, r in failures.iterrows():
    df.loc[(df["timestamp"] >= r["start"]) & (df["timestamp"] <= r["end"]), "falla"] = 1

# ==========================================================
# 3. LIMPIEZA Y OUTLIERS
# ==========================================================
cols_to_clip = ["H1","Towers","DV_eletric","COMP","MPG","Reservoirs","TP3","TP2"]
for col in cols_to_clip:
    p1, p99 = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], p1, p99)

df.drop(columns=["Pressure_switch","Caudal_impulses","Oil_level"], inplace=True)

# ==========================================================
# 4. SPLIT TEMPORAL (NO RANDOM)
# ==========================================================
df = df.sort_values("timestamp")

split = int(len(df) * 0.80)
train = df.iloc[:split]
test  = df.iloc[split:]

X_train = train.drop(columns=["falla","timestamp","Unnamed: 0"])
y_train = train["falla"]

X_test = test.drop(columns=["falla","timestamp","Unnamed: 0"])
y_test = test["falla"]

# ==========================================================
# 5. ESCALADO
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==========================================================
# 6. KNN + HIPERPARAMETRIZACIÓN
# ==========================================================
param_grid = {
    "n_neighbors": [3,5,7,9,11],
    "weights": ["uniform","distance"],
    "metric": ["euclidean","manhattan"]
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

best_knn = grid.best_estimator_
print("\nMejores parámetros:", grid.best_params_)

# ==========================================================
# 7. EVALUACIÓN
# ==========================================================
y_pred = best_knn.predict(X_test_scaled)

print("\n=== RESULTADOS KNN ===\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte:\n", classification_report(y_test, y_pred))

# ==========================================================
# 8. GUARDADO
# ==========================================================
import os
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler_knn.pkl")
joblib.dump(best_knn, "models/modelo_knn.pkl")
