import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import urllib3 
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
        df[col] = df[col].astype(str)
        df[col].fillna('Desconocido', inplace=True)

X = df[original_numerical_features + original_categorical_features]
y = df[target_col]

if X.empty:
    print("No hay características (X) después del preprocesamiento. Revisa la selección de columnas y los datos.")
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

print("\nEntrenando el modelo SVM...")
svm_pipeline.fit(X_train, y_train)
print("Entrenamiento completado.")


print("\nRealizando predicciones en el conjunto de prueba...")
y_pred = svm_pipeline.predict(X_test)
print("\n Resultados de la Clasificación ")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (Exactitud): {accuracy:.4f}")
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred, labels=svm_pipeline.classes_) 
cm_df = pd.DataFrame(cm, index=svm_pipeline.classes_, columns=svm_pipeline.classes_)
print(cm_df)
