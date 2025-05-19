from elasticsearch import Elasticsearch

import requests
"""
url = "https://localhost:9200/accidentes_con_fecha"
auth = ('elastic', '123456')  # Reemplaza con tu contraseña real

response = requests.delete(url, auth=auth, verify=False)
print(response.status_code, response.text)

"""

es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "123456"),
    verify_certs=False  
)

response = es.search(index="accidentes_con_fecha", size=10)


for hit in response["hits"]["hits"]:
   print(hit["_source"])

#response = es.count(index="accidentes_con_fecha")
#total_documentos = response["count"]

#print("Número de registros en el índice 'accidentes':", total_documentos)
