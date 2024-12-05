import requests 
import json
import pandas as pd
from helper_functions import *

def buildquery(occupation):
    # Read more on how to make a SPARQL query: https://ramiro.org/notebook/us-presidents-causes-of-death/
    # Read more on the use and need or User-Agent: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
    #
    # Optional means to take information if present but not exclude those 
    # without the information. 
    #
    # PREFIX helps shorten uri's

    # wdt:P106 - occupation
    # wdt:P27  - country of citizenship
    # wdt:P21  - sex and gender
    #
    # OPTIONAL:
    # wdt:P569 - date of birth
    # filter [country of citizenship | individual | gender] to use english language names
    
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

if __name__ == "__main__":
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'

    # running automated queries we need to add 'bot' to the name of the agent. Also provide a way to be contacted (like an email).
    headers  = {'User-Agent' : 'HornEnvelopeLearnerOccupationRetrivalQueryBot (emilpo@uio.no)'} 
    filename = "occupations.csv"


    # import occupations from csv.
    occ_list = pd.read_csv(filename, header=None).to_numpy()
    occ_list_new = []

    #query all occupations
    for occupation in occ_list:
        id   = occupation[1]
        name = occupation[0]

        #if name == 'actor' : #Why only when name is 'actor'?
            
        # Send a request to Wikidata
        data = requests.get(url ,params={'query': buildquery(id), 'format': 'json'}, headers=headers).json()
        print("Queried for " + name + " and found " + str(len(data['results']['bindings'])) + " results in wikidata.")
        
        #if there are results, continue execution
        if len(data['results']['bindings']) > 0:
            occ_data = []
            
            for item in data['results']['bindings']:
            #just keep entries that have a gender for classification result comparison
                if 'gender' in item:
                    birthyear = get_birth(item)
                    occ_data.append({
                    'name': return_if_exists(item, 'individual'),
                    'gender': return_if_exists(item, 'gender'),
                    'birth': birthyear,
                    'nationality': return_if_exists(item, 'nationality'),
                    'nid': get_nid(return_if_exists(item, 'nid'))
                    })

            #if there are results after the cleanup, continue execution
            if len(occ_data) > 0:
                #save cleaned dataframe 
                file = 'data/data_dataframes/' + name + '.csv'
                df = pd.DataFrame(occ_data)
                #df = df.drop_duplicates(subset='name', keep='first')
                print("\tAfter cleanup: " + str(len(df)) + " datapoints left")
                df.to_csv(file, index=False)
                #save uncleaned raw data seperately
                occ_file = 'data/dataset_raw/' + name + '.json'
                with open(occ_file, 'w') as outfile:
                    json.dump(data, outfile)
                #keep track of occupations that are actually saved as datapoints
                occ_list_new.append(occupation)

    occ_df = pd.DataFrame(occ_list_new)
    occ_df.to_csv('data/occupations_updated.csv', index=None, header=None)