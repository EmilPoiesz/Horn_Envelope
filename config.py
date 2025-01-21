URL = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
HEADERS = {'User-Agent' : 'HornEnvelopeLearnerOccupationRetrivalQueryBot (emilpo@uio.no)'} 

SPARQL_QUERIES = {
    'occupation_query':  """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?individual ?gender ?birth ?nationality ?nationalityID WHERE {{
            ?id wdt:P106 wd:{occupationID} ;
                wdt:P27 ?nationalityID ;
                wdt:P21 ?gid ;
                wdt:P569 ?birth .

            OPTIONAL {{
                ?nationalityID rdfs:label ?nationality filter (lang(?nationality) = "en") .
            }}
            OPTIONAL {{
                ?id rdfs:label ?individual filter (lang(?individual) = "en") .
            }}
            OPTIONAL {{
                ?gid rdfs:label ?gender filter (lang(?gender) = "en") .
            }}
        }}
        LIMIT 50000""",
    'verify_nationality_query': """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        ASK {{
            {{wd:{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q6256}}
            UNION
            {{wd:{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q7275}}
            UNION
            {{wd:{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q3024240}}
        }}""",
    'get_continent_query': """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?continent ?cid WHERE {{
            wd:{nid} wdt:P30 ?cid .

            OPTIONAL{{
                ?cid rdfs:label ?continent filter (lang(?continent) = "en") .
            }}
        }}""",
    'get_continent_extensive_query': """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?continent ?cid WHERE {{
            wd:{nid} wdt:P17/wdt:P30 | wdt:P1366/wdt:P30 | wdt:P30 ?cid .

            OPTIONAL{{
                ?cid rdfs:label ?continent filter (lang(?continent) = "en") .
            }}
        }}"""
}

AGE_CONTAINERS = [1875, 1925, 1951, 1970]

EPSILON = 0.2
DELTA = 0.1