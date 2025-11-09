import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ====== 1. Carga de datos ======
df = pd.read_csv('MetroPT3(AirCompressor).csv')
df = df.rename(columns={"Unnamed: 0": "Nr"})
df = df.drop(columns=['timestamp', 'Nr'])

# ====== 2. Correlación y reducción ======
corr_threshold = 0.8
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
df_reduced = df.drop(columns=to_drop)
print("Columnas eliminadas por alta correlación:", to_drop)

# ====== 3. Creación de etiqueta de fallos ======
col_stats = df_reduced.describe().loc[['min', 'max']]

def detectar_fallo(row):
    fallos = []

    if 'Motor_current' in row.index:
        if row['Motor_current'] < 0.5:
            fallos.append("Apagado")
        elif row['Motor_current'] > 9:
            fallos.append("Sobrecarga")

    if 'Oil_temperature' in row.index:
        min_temp, max_temp = col_stats.loc['min', 'Oil_temperature'], col_stats.loc['max', 'Oil_temperature']
        if row['Oil_temperature'] < min_temp * 0.8 or row['Oil_temperature'] > max_temp * 1.2:
            fallos.append("Temperatura Anómala")

    if 'Reservoirs' in row.index and row['Reservoirs'] < 7.0:
        fallos.append("Presión Baja en Reservorio")

    if 'DV_pressure' in row.index and row['DV_pressure'] == 0:
        fallos.append("Compresor Bajo Carga")

    if 'Oil_level' in row.index and row['Oil_level'] == 1:
        fallos.append("Nivel Bajo de Aceite")

    if not fallos:
        fallos.append("Normal")

    return ", ".join(fallos)

df_reduced["FaultType"] = df_reduced.apply(detectar_fallo, axis=1)

print("\nDistribución de tipos de fallos detectados:")
print(df_reduced["FaultType"].value_counts())

# ====== 4. Outliers ======
numeric_cols = df_reduced.select_dtypes(include=[np.number]).columns

def reemplazar_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mediana = data[col].median()
    data[col] = np.where((data[col] < lower) | (data[col] > upper), mediana, data[col])
    return data[col]

for col in numeric_cols:
    df_reduced[col] = reemplazar_outliers_iqr(df_reduced, col)

print("\n✅ Outliers reemplazados por la mediana.\n")

# ====== 5. Preparación de datos ======
label_candidate = "FaultType"
X = df_reduced.drop(columns=[label_candidate])
y = df_reduced[label_candidate]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

joblib.dump(le, 'encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("💾 Codificador y scaler guardados.\n")

# ====== 6. División de datos ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ====== 7. Modelo base XGBoost ======
xgb_base = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss")
xgb_base.fit(X_train, y_train)

y_pred_base = xgb_base.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average="macro")

print("🔹 Modelo base XGBoost:")
print(f"Accuracy: {acc_base:.4f}")
print(f"F1-score macro: {f1_base:.4f}")

# ====== 8. Hiperparametrización con RandomizedSearchCV ======
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5],
}

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss")

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=25,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n✅ Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# ====== 9. Evaluación del modelo optimizado ======
best_xgb = random_search.best_estimator_
y_pred_opt = best_xgb.predict(X_test)

acc_opt = accuracy_score(y_test, y_pred_opt)
f1_opt = f1_score(y_test, y_pred_opt, average="macro")

print("\n🔹 Modelo optimizado:")
print(f"Accuracy: {acc_opt:.4f}")
print(f"F1-score macro: {f1_opt:.4f}")

# ====== 10. Comparación ======
print("\n🔸 Comparación de desempeño:")
print(f"Accuracy base: {acc_base:.4f} → Optimizado: {acc_opt:.4f}")
print(f"F1 base: {f1_base:.4f} → Optimizado: {f1_opt:.4f}")

# ====== 11. Matriz de confusión ======
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión - XGBoost Optimizado")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# ====== 12. Guardar modelo final ======
joblib.dump(best_xgb, 'xgboost_optimizado.joblib')
print("\n💾 Modelo optimizado guardado como 'xgboost_optimizado.joblib'")
