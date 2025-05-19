from elasticsearch import Elasticsearch, helpers
import json

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "123456"),
    verify_certs=False 
)

index_name = "accidentes_mexico"

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

def cargar_json_en_elasticsearch(ruta_json, tamano_bloque=5000):
    with open(ruta_json, 'r', encoding='utf-8') as f:
        bloque = []
        for i, linea in enumerate(f, 1):
            doc = json.loads(linea)
            bloque.append({"_index": index_name, "_source": doc})
            if i % tamano_bloque == 0:
                helpers.bulk(es, bloque)
                print(f"Cargados {i} documentos...")
                bloque = []
        if bloque:
            helpers.bulk(es, bloque)
            print(f"Cargados {i} documentos finales.")

cargar_json_en_elasticsearch("accidentes_1997_2023.json")
