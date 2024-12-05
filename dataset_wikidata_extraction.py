import requests 
import json
import sys
import pandas as pd
from helper_functions import *

def buildquery(occupation, limit=1_000_000, offset=0):
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
    # wdt:P569 - date of birth
    #
    # filter non-english name for gender and nationality
    #
    # OPTIONAL:
    # filter individual to use english language names
    
    basequery = """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?individual ?gender ?birth ?nationality ?nationalityID WHERE {{
        ?personID wdt:P106 wd:{occupationID} ;
                  wdt:P27 ?nationalityID ;
                  wdt:P21 ?genderID ;
                  wdt:P569 ?birth .
        
        ?nationalityID rdfs:label ?nationality filter (lang(?nationality) = "en") .
        ?genderID rdfs:label ?gender filter (lang(?gender) = "en") .
        
        OPTIONAL {{
            ?personID rdfs:label ?individual filter (lang(?individual) = "en") .
        }}
    }}
    LIMIT {limit} 
    OFFSET {offset}
    """
    return basequery.format(occupationID = occupation, limit=limit, offset=offset)

import requests

url = "https://query.wikidata.org/sparql"
headers = {"Accept": "application/json"}

def fetch_data(occupation_id, occupation_name, limit=1_000):
    offset = 0
    all_results = []
    
    while True:
        # Build the query with the current offset
        query = buildquery(occupation_id, limit=limit, offset=offset)
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
        
        try:
            # Decode JSON response
            data = response.json()
            bindings = data.get('results', {}).get('bindings', [])
            
            # Stop the loop if no more results are returned
            if not bindings:
                break
            
            all_results.extend(bindings)
            offset += limit  # Increase the offset for the next batch
            
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON for {occupation_name}. Response status code:")
            print(response.status_code)
            print(response.text)  # Print raw response for debugging
            break

    return data, all_results


if __name__ == "__main__":
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'

    # running automated queries we need to add 'bot' to the name of the agent. Also provide a way to be contacted (like an email).
    headers  = {'User-Agent' : 'HornEnvelopeLearnerOccupationRetrivalQueryBot (emilpo@uio.no)'} 
    occ_lst_filename = "occupations.csv"

    # import occupations from csv.
    occupation_list = pd.read_csv(occ_lst_filename, header=None).to_numpy()
    occupation_list_clean = []

    #query all occupations
    for occupation in occupation_list:
        occupation_id   = occupation[1]
        occupation_name = occupation[0]
            
        # Send a request to Wikidata
        #response = requests.get(url, params={'query': buildquery(occupation_id), 'format': 'json'}, headers=headers)
        #try:
        #    data = response.json()
        #except requests.exceptions.JSONDecodeError:
        #    print(f"Failed to decode JSON for {occupation_name}. Response status code:")
        #    print(response.status_code)
        #    continue
        
        #results = data['results']['bindings']
        data, results = fetch_data(occupation_id, occupation_name)
        print("Queried for " + occupation_name + " and found " + str(len(results)) + " results in wikidata.")
        
        #if there are results, continue execution
        if len(results) > 0:
            occupation_data = []
            
            for item in results:

                occupation_data.append({
                    'name': return_if_exists(item, 'individual'),
                    'gender': return_if_exists(item, 'gender'),
                    'birth': get_birth(item),
                    'nationality': return_if_exists(item, 'nationality'),
                    'nationalityID': get_nid(return_if_exists(item, 'nationalityID'))
                })

            #if there are results after the cleanup, continue execution
            if len(occupation_data) > 0:
                #save cleaned dataframe 
                file = 'data/data_dataframes/' + occupation_name + '.csv'
                df = pd.DataFrame(occupation_data)
                #df = df.drop_duplicates(subset='name', keep='first')
                print("\tAfter cleanup: " + str(len(df)) + " datapoints left")
                df.to_csv(file, index=False)
                #save uncleaned raw data seperately
                occ_file = 'data/dataset_raw/' + occupation_name + '.json'
                with open(occ_file, 'w') as outfile:
                    json.dump(data, outfile)
                #keep track of occupations that are actually saved as datapoints
                occupation_list_clean.append(occupation)

    df_occupations = pd.DataFrame(occupation_list_clean)
    df_occupations.to_csv('data/occupations_updated.csv', index=None, header=None)