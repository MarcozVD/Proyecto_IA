import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC  # 🔹 Versión más rápida del SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ====== 1. Carga de datos ======
df = pd.read_csv('MetroPT3(AirCompressor).csv')
df = df.rename(columns={"Unnamed: 0": "Nr"})
df = df.drop(columns=['timestamp', 'Nr'])

# ====== 2. Estadística descriptiva ======
print(df.describe())

# ====== 3. Correlación ======
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Mapa de Correlación")
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

# ====== 5. Normalización ======
label_candidate = "FaultType"
X = df_reduced.drop(columns=[label_candidate])
y = df_reduced[label_candidate]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(df_scaled.head())

# ====== 6. Codificación de etiquetas ======
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Guardar codificador
joblib.dump(le, 'encoder.joblib')
print("\n💾 Codificador guardado como 'encoder.joblib'")

# ====== 7. División de datos ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nDistribución original del conjunto de entrenamiento:", Counter(y_train))

# ====== 8. Balanceo de clases con SMOTE ======
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("✅ Balanceo con SMOTE completado.")
print("Distribución después de SMOTE:", Counter(y_train_res))

# ====== 9. Entrenamiento de modelos ======
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "LinearSVM": LinearSVC(max_iter=2000, class_weight='balanced', random_state=42),  # ⚡ rápido y eficiente
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss', random_state=42,
        scale_pos_weight=1
    )
}

results = []
best_model = None
best_score = -1

for name, model in models.items():
    print(f"\n🚀 Entrenando modelo: {name} ...")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    results.append((name, acc, f1m))
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1-score (macro): {f1m:.4f}")
    print("📊 Matriz de confusión:\n", cm)

    if f1m > best_score:
        best_score = f1m
        best_model = model
        best_name = name

# ====== 10. Comparación de resultados ======
df_results = pd.DataFrame(results, columns=["Modelo", "Accuracy", "F1_macro"])
print("\n🏁 Resultados comparativos:")
print(df_results.sort_values(by="F1_macro", ascending=False))

# ====== 11. Guardar mejor modelo ======
joblib.dump(best_model, 'best_model.joblib')
print(f"\n💾 Mejor modelo guardado: {best_name} con F1={best_score:.4f}")
