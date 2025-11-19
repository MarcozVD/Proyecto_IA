# -*- coding: utf-8 -*-

# --- Librerías de preprocesamiento y modelado ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import io
import xgboost as xgb
import joblib
from datetime import timedelta

# ==========================================================
# 1. CARGA DEL DATASET
# ==========================================================
sensor_file = 'MetroPT3(AirCompressor).csv'  # AJUSTA según tu archivo
print("Cargando sensor data desde:", sensor_file)
df = pd.read_csv(sensor_file, parse_dates=['timestamp'])
numeric_cols = df.select_dtypes(include=[np.number]).columns

min_max_df = pd.DataFrame({
    'Min': df[numeric_cols].min(),
    'Max': df[numeric_cols].max()
})
print(min_max_df)

print("Dimensiones del DataFrame:", df.shape)
print(df.head())

print("Valores nulos por columna:\n", df.isna().sum())

# ==========================================================
# 2. CREACIÓN DE ETIQUETA DE FALLA (CON BUFFER)
# ==========================================================
failures_data = """Nº|Hora de inicio|Hora de finalización|Tipo de falla|Severidad|Informe
1|18/04/2020 00:00|18/04/2020 23:59|Fuga de aire|Alta tensión|—
2|29/05/2020 23:30|30/05/2020 06:00|Fuga de aire|Alta tensión|Mantenimiento el 30 de abril a las 12:00
3|06/06/2020 10:00|07/06/2020 14:30|Fuga de aire|Alta tensión|Mantenimiento el 8 de junio a las 16:00
4|15/07/2020 14:30|15/07/2020 19:00|Fuga de aire|Alta tensión|Mantenimiento el 16 de julio a las 00:00
"""

failures = pd.read_csv(io.StringIO(failures_data), sep='|')
failures = failures.drop(columns=['Nº','Tipo de falla','Severidad','Informe'])
failures.columns = ['start_time','end_time']

failures['start_time'] = pd.to_datetime(failures['start_time'], format='%d/%m/%Y %H:%M')
failures['end_time']   = pd.to_datetime(failures['end_time'], format='%d/%m/%Y %H:%M')

# ----------- BUFFER PARA ETIQUETAR FALLAS ----------
PRE_FAIL_HOURS = 6   # horas antes
POST_FAIL_HOURS = 1  # horas después

df['falla'] = 0
for _, row in failures.iterrows():
    start = row['start_time'] - timedelta(hours=PRE_FAIL_HOURS)
    end   = row['end_time']   + timedelta(hours=POST_FAIL_HOURS)
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'falla'] = 1

print("\nDistribución de la etiqueta 'falla' con buffer:")
print(df['falla'].value_counts())

# ==========================================================
# 3. PREPROCESAMIENTO Y LIMPIEZA DE OUTLIERS
# ==========================================================
for col in ['H1','Towers','DV_eletric','COMP','MPG','Reservoirs','TP3','TP2']:
    p1, p99 = np.percentile(df[col], [1, 99])
    df[col] = np.clip(df[col], p1, p99)

df.drop(columns=['Pressure_switch','Caudal_impulses','Oil_level'], inplace=True)

# ==========================================================
# 4. SEPARACIÓN Y BALANCEO DE CLASES (SMOTE)
# ==========================================================
X = df.drop(columns=['falla','timestamp','Unnamed: 0'], errors='ignore')
y = df['falla']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("\nAntes del balanceo:", y_train.value_counts().to_dict())
print("Después del balanceo:", pd.Series(y_train_res).value_counts().to_dict())

# ==========================================================
# 5. ENTRENAMIENTO XGBOOST CON GRIDSEARCHCV
# ==========================================================
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [150, 300],
    'min_child_weight': [3, 5],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

grid_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_xgb.fit(X_train_res, y_train_res)
best_xgb = grid_xgb.best_estimator_
print("\nMejores parámetros:", grid_xgb.best_params_)

# ==========================================================
# 6. CÁLCULO DEL MEJOR UMBRAL
# ==========================================================
y_probs = best_xgb.predict_proba(X_test_scaled)[:,1]
prec, rec, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\n🔥 Mejor umbral encontrado: {best_threshold:.4f}")

# ==========================================================
# 7. EVALUACIÓN CON EL UMBRAL OPTIMIZADO
# ==========================================================
y_pred_adj = (y_probs >= best_threshold).astype(int)

print("\nReporte de clasificación usando el umbral optimizado:")
print(classification_report(y_test, y_pred_adj))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_adj))

# ==========================================================
# 8. GUARDADO DE MODELO, SCALER Y UMBRAL
# ==========================================================
os.makedirs("models", exist_ok=True)
joblib.dump(best_xgb, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(best_threshold, "models/best_threshold.pkl")

print("\nModelo, scaler y umbral guardados en /models/")
