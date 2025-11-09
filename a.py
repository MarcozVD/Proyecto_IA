# =====================================
# 🔹 ENTRENAMIENTO DE KNN CON HIPERPARAMETRIZACIÓN
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ====== 1. Carga de datos ======
df_0 = pd.read_csv('MetroPT3(AirCompressor).csv')
df = df_0.head(10000)
df = df_0.rename(columns={"Unnamed: 0": "Nr"})
df = df.drop(columns=['timestamp', 'Nr'])

# ====== 2. Estadística descriptiva ======
print(df.describe())

# ====== 3. Correlación ======
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Mapa de Correlacion")
plt.show()

# Eliminar columnas altamente correlacionadas
corr_threshold = 0.8
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
df_reduced = df.drop(columns=to_drop)
print("Columnas eliminadas por alta correlación:", to_drop)

# ====== 3.5 Creación de etiqueta de fallos ======
col_stats = df_reduced.describe().loc[['min', 'max']]
def detectar_fallo(row):
    if 'Motor_current' in row.index:
        if row['Motor_current'] < 0.5:
            return "Apagado"
        elif row['Motor_current'] > 9:
            return "Sobrecarga"

    if 'Oil_temperature' in row.index:
        min_temp, max_temp = col_stats.loc['min', 'Oil_temperature'], col_stats.loc['max', 'Oil_temperature']
        if row['Oil_temperature'] < min_temp * 0.8 or row['Oil_temperature'] > max_temp * 1.2:
            return "Temperatura Anómala"

    if 'Reservoirs' in row.index and row['Reservoirs'] < 7.0:
        return "Presión Baja en Reservorio"

    if 'DV_pressure' in row.index and row['DV_pressure'] == 0:
        return "Compresor Bajo Carga"

    if 'Oil_level' in row.index and row['Oil_level'] == 1:
        return "Nivel Bajo de Aceite"

    return "Normal"

df_reduced["FaultType"] = df_reduced.apply(detectar_fallo, axis=1)
print("\nDistribución de tipos de fallos detectados:")
print(df_reduced["FaultType"].value_counts())

# ====== 4. Detección y reemplazo de outliers ======
numeric_cols = df.select_dtypes(include=[np.number]).columns

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
    df[col] = reemplazar_outliers_iqr(df, col)

print("\n✅ Outliers reemplazados por la mediana en las columnas numéricas.\n")

# ====== 5. Valores faltantes ======
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(pd.DataFrame({'Missing': missing, '%': missing_pct}))

# ====== 6. Normalización y codificación ======
label_candidate = "FaultType"
X = df_reduced.drop(columns=[label_candidate])
y = df_reduced[label_candidate]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ====== 7. División de datos ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ====== 8. Hiperparametrización del modelo KNN ======
from sklearn.model_selection import RandomizedSearchCV

knn_base = KNeighborsClassifier()

param_dist = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_dist,
    n_iter=15,
    scoring='f1_macro',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("\n🔍 Buscando los mejores hiperparámetros para KNN...")
random_search.fit(X_train, y_train)

print("\n✅ Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# ====== 9. Evaluación del modelo optimizado ======
best_knn = random_search.best_estimator_
y_pred = best_knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Resultados del modelo KNN optimizado:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score (macro): {f1m:.4f}")
print("Matriz de confusión:\n", cm)

# ====== 10. Visualización ======
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - KNN Optimizado")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# ====== 11. Guardar el modelo entrenado ======
joblib.dump(best_knn, "modelo_KNN_optimizado.pkl")
joblib.dump(y_encoded, "encoder_labels.pkl")
print("\n💾 Modelo KNN optimizado guardado como 'modelo_KNN_optimizado.pkl'")
