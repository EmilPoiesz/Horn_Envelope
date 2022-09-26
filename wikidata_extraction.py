import requests 
import json
import pandas as pd
from helper_functions import *

def buildquery(occupation):
    #function to insert the occupation id properly into the correct query
    basequery = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?individual ?gender ?birth ?nationality ?nid WHERE {{
        ?id wdt:P106 {occ} .
        ?id wdt:P27 ?nid .
        ?id wdt:P21 ?gid .

        OPTIONAL{{
        ?id wdt:P569 ?birth .
        }}
        OPTIONAL {{
            ?nid rdfs:label ?nationality filter (lang(?nationality) = "en") .
        }}
        OPTIONAL {{
            ?id rdfs:label ?individual filter (lang(?individual) = "en") .
        }}
        OPTIONAL {{
            ?gid rdfs:label ?gender filter (lang(?gender) = "en") .
        }}
    }}
    """
    occ_id = "wd:" + occupation
    return basequery.format(occ = occ_id)

#MAIN PART:
#   import the occupations and query wikidata for each of them
#   save results for each occupation in separate json file in dataset folder

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
filename = "occupations.csv"
occ_list = import_occupations(filename)
occ_list_new = []

#query all occupations
for occupation in occ_list:
    id = occupation[1]
    name = occupation[0]
    # get the data from wikidata
    if name == 'actor':
        data = requests.get(url ,params={'query': buildquery(id), 'format': 'json'}).json()
        occ_file = 'dataset_raw/' + name + '.json'
        print("Queried for " + name + " and found " + str(len(data['results']['bindings'])) + " results in wikidata.")
        #if there are results, continue execution
        if len(data['results']['bindings']) > 0:
            occ_data = []
            for item in data['results']['bindings']:
            #just keep entries that have a nationality AND name AND gender AND birthdate
                if 'individual' in item and 'nationality' in item and 'gender' in item and 'birth' in item and 'nid' in item:
                    birthyear = get_birth(item['birth'])
                    #remove the ones with birthyear = unknown!
                    if birthyear != 'unknown':
                        occ_data.append({
                        'name': item['individual']['value'],
                        'gender': item['gender']['value'],
                        'birth': birthyear,
                        'nationality': item['nationality']['value'],
                        'nid': get_nid(item['nid'])
                        })
            #if there are results after the cleanup, continue execution
            if len(occ_data) > 0:
                #save cleaned dataframe 
                file = 'data_dataframes/' + name + '.csv'
                df = pd.DataFrame(occ_data)
                #df = df.drop_duplicates(subset='name', keep='first')
                print("\tAfter cleanup: " + str(len(df)) + " datapoints left")
                df.to_csv(file, index=False)
                #save uncleaned raw data seperately
                with open(occ_file, 'w') as outfile:
                    json.dump(data, outfile)
                #keep track of occupations that are actually saved as datapoints
                #occ_list_new.append(occupation)

#occ_df = pd.DataFrame(occ_list_new)
#occ_df.to_csv('occupations_updated.csv', index=None, header=None)