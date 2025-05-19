from elasticsearch import Elasticsearch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url_elasticsearch = 'https://localhost:9200'

es = Elasticsearch(
    url_elasticsearch,
    basic_auth=("elastic", "123456"),
    verify_certs=False
)

response = es.search(index="catalogo_estados", body={"query": {"match_all": {}}})
for hit in response['hits']['hits']:
    print(hit['_source'])
