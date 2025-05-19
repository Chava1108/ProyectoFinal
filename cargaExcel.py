import pandas as pd
import glob
import os
from elasticsearch import Elasticsearch, helpers
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ruta_datos = 'conjunto_de_datos'
url_elasticsearch = 'https://localhost:9200'
es = Elasticsearch(
    url_elasticsearch,
    basic_auth=("elastic", "123456"),
    verify_certs=False
)


archivos = sorted(glob.glob(os.path.join(ruta_datos, 'atus_anual_*.csv')))


columnas_necesarias = [
    "COBERTURA", "ID_ENTIDAD", "ID_MUNICIPIO", "ANIO", "MES", 
    "ID_HORA", "ID_MINUTO", "ID_DIA", "DIASEMANA", "URBANA", 
    "SUBURBANA", "TIPACCID", "AUTOMOVIL", "CAMPASAJ", "MICROBUS", 
    "PASCAMION", "OMNIBUS", "TRANVIA", "CAMIONETA", "CAMION", 
    "TRACTOR", "FERROCARRI", "MOTOCICLET", "BICICLETA", "OTROVEHIC", 
    "CAUSAACCI", "CAPAROD", "SEXO", "ALIENTO", "CINTURON", 
    "ID_EDAD", "CONDMUERTO", "CONDHERIDO", "PASAMUERTO", "PASAHERIDO",
    "PEATMUERTO", "PEATHERIDO", "CICLMUERTO", "CICLHERIDO", "OTROMUERTO", 
    "OTROHERIDO", "NEMUERTO", "NEHERIDO", "CLASACC", "ESTATUS"
]

for archivo in archivos:
    print(f"Procesando archivo: {archivo}")
    df = pd.read_csv(archivo, dtype=str, usecols=columnas_necesarias)
    
    acciones = [
        {
            "_index": "accidentes",   
            "_source": row.to_dict()
        }
        for _, row in df.iterrows()
    ]
    
    # Ejecutar la inserción en bulk
    helpers.bulk(es, acciones)
    print(f"Se insertaron {len(acciones)} documentos desde {archivo}")
