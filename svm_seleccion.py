import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np 
import warnings
import urllib3 
import matplotlib.pyplot as plt 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

warnings.filterwarnings("ignore", message="Connecting to https://localhost:9200 using SSL with verify_certs=False is insecure.")
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.svm._base')
warnings.filterwarnings("ignore", category=FutureWarning)


try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "123456"), 
        verify_certs=False 
    )
    if not es.ping():
        raise ValueError("Falló la conexión a Elasticsearch. Verifica tus credenciales y la URL.")

    query = {
        "query": {
            "range": {
                "ANIO": {
                    "gte": 2020,
                    "lte": 2023
                }
            }
        }
    }

    print("Extrayendo datos de Elasticsearch...")
    datos_crudos = []
    for doc in scan(es, index="accidentes_con_fecha", query=query, size=1000):
        datos_crudos.append(doc["_source"])

    if not datos_crudos:
        print("No se encontraron datos para el rango de años especificado.")
        exit()
        
    df = pd.DataFrame(datos_crudos)
    print(f"Datos extraídos: {df.shape[0]} filas, {df.shape[1]} columnas.")

except Exception as e:
    print(f"Error durante la conexión o extracción de datos: {e}")
    exit()

if 'DIASEMANA' in df.columns:
    print("Homogenizando valores en la columna 'DIASEMANA'...")
    df['DIASEMANA'] = df['DIASEMANA'].astype(str).str.lower()
    corrections_map = {
        'sabado': 'sábado',
        'miercoles': 'miércoles',
        'nan': 'Desconocido' 
    }
    df['DIASEMANA'] = df['DIASEMANA'].replace(corrections_map)
    df['DIASEMANA'] = df['DIASEMANA'].str.capitalize()
    df.loc[df['DIASEMANA'] == 'Desconocido'.capitalize(), 'DIASEMANA'] = 'Desconocido'
print("\nIniciando preparación de datos...")
target_col = 'CLASACC'
valid_clasacc = ['Fatal', 'No fatal', 'Sólo Daños']
df = df[df[target_col].isin(valid_clasacc)]

if df.empty:
    print(f"No quedan datos después de filtrar '{target_col}' por valores válidos. Verifica tus datos.")
    exit()

numerical_features = [
    'ANIO', 'MES', 'ID_HORA', 'ID_MINUTO',
    'AUTOMOVIL', 'CAMPASAJ', 'MICROBUS', 'PASCAMION', 'OMNIBUS', 'TRANVIA',
    'CAMIONETA', 'CAMION', 'TRACTOR', 'FERROCARRI', 'MOTOCICLET', 'BICICLETA', 'OTROVEHIC'
]
categorical_features = [
    'ID_ENTIDAD', 'ID_MUNICIPIO', 
    'DIASEMANA', 'URBANA', 'SUBURBANA', 'TIPACCID', 'CAUSAACCI',
    'CAPAROD', 'SEXO', 'ALIENTO', 'CINTURON'
]

all_selected_features = numerical_features + categorical_features + ['ID_EDAD']
original_numerical_features = [col for col in numerical_features if col in df.columns]
original_categorical_features = [col for col in categorical_features if col in df.columns]
id_edad_present = 'ID_EDAD' in df.columns

missing_features_check = [col for col in all_selected_features if col not in df.columns]
if missing_features_check:
    print(f"Advertencia: Las siguientes columnas no se encontraron y serán omitidas del set inicial: {missing_features_check}")

if id_edad_present:
    try:
        df['ID_EDAD'] = df['ID_EDAD'].astype(str) 
        df['ID_EDAD'] = pd.to_numeric(df['ID_EDAD'], errors='coerce')
        if 'ID_EDAD' not in original_numerical_features:
             original_numerical_features.append('ID_EDAD')
    except Exception as e:
        print(f"Error procesando ID_EDAD: {e}. Se excluirá esta columna.")
        if 'ID_EDAD' in original_numerical_features: original_numerical_features.remove('ID_EDAD')

for col in original_numerical_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

for col in original_categorical_features:
    if col in df.columns:
        if col != 'DIASEMANA':
            df[col] = df[col].astype(str) 
            df[col].fillna('Desconocido', inplace=True)
        else: 
            df[col] = df[col].astype(str) 
            if df[col].isnull().any():
                 df[col].fillna('Desconocido', inplace=True)

X = df[original_numerical_features + original_categorical_features]
y = df[target_col]

if X.empty:
    print("No hay características (X) después del preprocesamiento.")
    exit()

print(f"Dimensiones de X: {X.shape}, Dimensiones de y: {y.shape}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), original_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), original_categorical_features)
    ],
    remainder='drop' 
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y      
)
print(f"Tamaño del set de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del set de prueba: {X_test.shape[0]}")

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(C=1.0, class_weight='balanced', random_state=42, dual="auto", max_iter=5000))
])

print("\nEntrenando el modelo SVM principal...")
try:
    svm_pipeline.fit(X_train, y_train)
    print("Entrenamiento del modelo principal completado.")
except Exception as e:
    print(f"Error durante el entrenamiento del modelo principal: {e}")
    exit()

print("\nRealizando predicciones con el modelo principal...")
y_pred = svm_pipeline.predict(X_test)
print("\n Resultados de la Clasificación del Modelo Principal ")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Exactitud): {accuracy:.4f}")
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred, labels=svm_pipeline.classes_) 
cm_df = pd.DataFrame(cm, index=svm_pipeline.classes_, columns=svm_pipeline.classes_)
print(cm_df)
print("\n Visualización Interactiva de Fronteras de Decisión SVM 2D ")

def plot_decision_boundary(X_plot, y_plot, model, title, feature_names, class_names_legend):
    h = .05 
    x_min, x_max = X_plot[:, 0].min() - .5, X_plot[:, 0].max() + .5
    y_min, y_max = X_plot[:, 1].min() - .5, X_plot[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, s=20, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel(f"{feature_names[0]} (escalado)")
    plt.ylabel(f"{feature_names[1]} (escalado)")
    plt.title(title)
    if class_names_legend is not None and len(np.unique(y_plot)) <= len(class_names_legend):
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                              markerfacecolor=scatter.cmap(scatter.norm(idx)))
                   for idx, cls in enumerate(class_names_legend[:len(np.unique(y_plot))]) ] 
        plt.legend(title="Clases", handles=handles)
    return plt

label_encoder_plot = LabelEncoder()
y_train_encoded_for_plot = label_encoder_plot.fit_transform(y_train) 
plot_class_names = label_encoder_plot.classes_

available_numeric_for_plot = [col for col in original_numerical_features if col in X_train.columns]

if not available_numeric_for_plot or len(available_numeric_for_plot) < 2:
    print("No hay suficientes características numéricas disponibles para la visualización 2D.")
else:
    print("\nCaracterísticas numéricas disponibles para graficar:")
    for i, fname in enumerate(available_numeric_for_plot):
        print(f"{i+1}. {fname}")

    selected_indices = []
    selected_features_names = []

    while len(selected_features_names) < 2:
        try:
            if len(selected_features_names) == 0:
                choice = int(input(f"Selecciona el número de la PRIMERA característica (1-{len(available_numeric_for_plot)}): ")) - 1
            else:
                choice = int(input(f"Selecciona el número de la SEGUNDA característica (1-{len(available_numeric_for_plot)}), diferente a la primera: ")) - 1

            if 0 <= choice < len(available_numeric_for_plot):
                if choice not in selected_indices:
                    selected_indices.append(choice)
                    selected_features_names.append(available_numeric_for_plot[choice])
                else:
                    print("Error: Ya has seleccionado esa característica. Elige una diferente.")
            else:
                print(f"Error: Selección inválida. Introduce un número entre 1 y {len(available_numeric_for_plot)}.")
        except ValueError:
            print("Error: Entrada no válida. Introduce un número.")

    print(f"Características seleccionadas para graficar: {selected_features_names[0]} y {selected_features_names[1]}")

    X_train_2d = X_train[selected_features_names].copy() 
    
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)

    print("\n Información de Escala para las Características Seleccionadas (StandardScaler) ")
    for i, feature_name in enumerate(selected_features_names):
        print(f"- Característica '{feature_name}':")
        print(f"  - Media original (μ) calculada de X_train: {scaler_2d.mean_[i]:.4f}")
        print(f"  - Desviación Estándar original (σ) calculada de X_train: {scaler_2d.scale_[i]:.4f}")
        print(f"  (Transformación: Valor Escalado = (Valor Original - {scaler_2d.mean_[i]:.2f}) / {scaler_2d.scale_[i]:.2f})")

    print(f"\nEntrenando LinearSVC en 2D con '{selected_features_names[0]}' y '{selected_features_names[1]}'...")
    svm_linear_2d = LinearSVC(C=1.0, class_weight='balanced', random_state=42, dual="auto", max_iter=5000)
    svm_linear_2d.fit(X_train_2d_scaled, y_train_encoded_for_plot)
    plot_linear = plot_decision_boundary(X_train_2d_scaled, y_train_encoded_for_plot, svm_linear_2d,
                           f'LinearSVC con {selected_features_names[0]} vs {selected_features_names[1]}',
                           feature_names=selected_features_names,
                           class_names_legend=plot_class_names)

    print(f"\nEntrenando SVC (kernel RBF) en 2D con '{selected_features_names[0]}' y '{selected_features_names[1]}'...")
    svm_rbf_2d = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    svm_rbf_2d.fit(X_train_2d_scaled, y_train_encoded_for_plot)
    plot_rbf = plot_decision_boundary(X_train_2d_scaled, y_train_encoded_for_plot, svm_rbf_2d,
                           f'SVC (RBF) con {selected_features_names[0]} vs {selected_features_names[1]}',
                           feature_names=selected_features_names,
                           class_names_legend=plot_class_names)

    plt.show()


print("\nProceso finalizado.")