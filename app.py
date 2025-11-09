import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
plt.title("Mapa de Correlacion")
plt.show()

# Eliminar columnas altamente correlacionadas
corr_threshold = 0.8
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
df_reduced = df.drop(columns=to_drop)
print("Columnas eliminadas por alta correlación:", to_drop)

# ====== 3.5 Creación de etiqueta de fallos (Opción 1: etiquetas combinadas) ======
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

# ====== 4. Detección de outliers ======
numeric_cols = df.select_dtypes(include=[np.number]).columns

def detectar_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((data[col] < lower) | (data[col] > upper)).sum()

outliers = {col: detectar_outliers_iqr(df, col) for col in numeric_cols}
outliers_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Outliers'])
outliers_df['%_Outliers'] = (outliers_df['Outliers'] / len(df)) * 100
print(outliers_df.sort_values('%_Outliers', ascending=False))

# ====== 5. Reemplazo de outliers ======
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

# ====== 6. Valores faltantes ======
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(pd.DataFrame({'Missing': missing, '%': missing_pct}))

# ====== 7. Normalización ======
label_candidate = "FaultType"
X = df_reduced.drop(columns=[label_candidate])
y = df_reduced[label_candidate]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(df_scaled.head())

# ====== 8. Codificación de etiquetas ======
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Guardar el codificador
joblib.dump(le, 'encoder.joblib')
print("\n💾 Codificador guardado como 'encoder.joblib'")

# ====== 9. División de datos ======
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ====== 10. Entrenamiento de modelos ======
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(max_iter=10000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
best_model = None
best_score = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    results.append((name, acc, f1m))
    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1m:.4f}")
    print("Matriz de confusión:\n", cm)

    if f1m > best_score:
        best_score = f1m
        best_model = model
        best_name = name

# ====== 11. Comparación de resultados ======
df_results = pd.DataFrame(results, columns=["Modelo", "Accuracy", "F1_macro"])
print("\nResultados comparativos:")
print(df_results.sort_values(by="F1_macro", ascending=False))

# ====== 12. Guardar el mejor modelo ======
joblib.dump(best_model, 'best_model.joblib')
print(f"\n💾 Mejor modelo guardado: {best_name} con F1={best_score:.4f}")
