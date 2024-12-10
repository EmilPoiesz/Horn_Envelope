import requests 
import json
import os
import pandas as pd
from helper_functions import *

def build_occupation_query(occupation, limit, offset):
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
    
    query = """
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
    return query.format(occupationID = occupation, limit=limit, offset=offset)

def build_verification_query(nid):
    
    # Q6256 - Country
    # Q7275 - State
    # Q3024240 - Historical Country

    # P31 - Instance of
    # P17 - country
    # P279 - Subclass of

    # {nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q6256}
    # means that nid is matched with instance of (wdt:P31) zero or more (*) followed by (/) country (wdt:P17)
    # zero or more optional (?) follwed by (/) one instance of (P31) subclass of (P279) zero or more (*) country (wd:Q6256)

    # The full query asks (true/false) if {nid} is is matched with any Country, State, or Historical Country. 

    query =  """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        ASK {{
            {{{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q6256}}
            UNION
            {{{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q7275}}
            UNION
            {{{nid} wdt:P31*/wdt:P17?/wdt:P31/wdt:P279* wd:Q3024240}}
        }}
        """
    return query.format(nid='wd:' + nid)

def fetch_data(occupation_id, occupation_name, limit=10_000):
    # For very large queries it is good practice to break it up 
    # into smaller queries to prevent overloading the service.

    offset = 0
    all_results = []
    
    while True:
        # Build the query with the current offset
        query = build_occupation_query(occupation_id, limit=limit, offset=offset)
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
        
        try:
            data = response.json()
            bindings = data['results']['bindings']
            
            # Stop the loop if no more results are returned
            if not bindings: break
            
            all_results.extend(bindings)
            offset += limit  # Increase the offset for the next batch
            
        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON for {occupation_name}. Response status code:")
            print(response.status_code)
            print(response.text) 
            break

    return all_results

def query_wikidata(occupation_list):

    occupation_list_clean = []
    # Query wikidata for all occupations
    for occupation in occupation_list:
        occupation_id   = occupation[1]
        occupation_name = occupation[0]
        
        # Make the query
        print(f'We are querying for {occupation_name}.')
        results = fetch_data(occupation_id, occupation_name)
        print("Queried for " + occupation_name + " and found " + str(len(results)) + " results in wikidata.")
        
        # Process the results
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

            # Save the dataframe as CSV
            df = pd.DataFrame(occupation_data)
            df.to_csv(f'data/csv/{occupation_name}.csv', index=False)

            # Save raw JSON data
            with open(f'data/json/{occupation_name}.json', 'w') as outfile:
                json.dump(results, outfile)

            # Keep track of extracted occupations
            occupation_list_clean.append(occupation)

    if os.path.exists('data/occupations_extracted.csv'):
        df_occupations = pd.read_csv('data/occupations_extracted.csv', header=None)
        df_occupations = pd.concat([df_occupations, pd.DataFrame(occupation_list_clean)])
    else:
        df_occupations = pd.DataFrame(occupation_list_clean)
    df_occupations.to_csv('data/occupations_extracted.csv', index=None, header=None)

def remove_invalid_nID_and_standardize_gender(occupations_extracted, nIDs_valid):
    amounts = {}
    for occupation in occupations_extracted.values:
        idx_to_delete = []
        occupation_name = occupation[0]
        occupation_df = pd.read_csv(f'data/csv/{occupation_name}.csv')
        total_pre_cleanup = len(occupation_df)

        for idx, row in occupation_df.iterrows():
            # Check if row should be deleted
            nID = row['nationalityID']            
            if not nIDs_valid[nID]: 
                idx_to_delete.append(idx)
                continue

            # Standardize gender if row is kept
            occupation_df.at[idx,'gender'] = standardize_gender(row['gender'])
            
        # Drop duplicate rows and rows with invalid countries  
        occupation_df = occupation_df.drop(idx_to_delete)
        occupation_df = occupation_df.drop_duplicates(subset='name', keep='first')
        occupation_df = occupation_df.drop_duplicates(keep='first')
        total_post_cleanup = len(occupation_df)

        # Save clean data to file
        occupation_df.to_csv(f'data/csv_clean/{occupation_name}.csv', index=False)
        amounts[occupation_name] = (total_pre_cleanup, total_post_cleanup)
        print(f'Cleaned up {occupation_name}')
    print(amounts)

def standardize_gender(gender):
    is_female = gender == 'female' or gender == 'transgender female' or gender == 'cisgender female' or gender == 'trans woman'
    is_male   = gender == 'male'   or gender == 'transgender male'   or gender == 'cisgender male'   or gender == 'trans man'
    return 'female' if is_female else 'male' if is_male else 'other'

if __name__ == "__main__":

    # When running automated queries we need to add 'bot' to the name of the agent. 
    # Also provide a way to be contacted (like an email).
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    headers  = {'User-Agent' : 'HornEnvelopeLearnerOccupationRetrivalQueryBot (emilpo@uio.no)'} 

    # Get list of new occupations
    if os.path.exists('data/occupations_extracted.csv'):
        # Merges the list of occupations with those we have already extracted
        # making sure we only query for new occupations to limit the amount of queries.
        occupations = pd.read_csv("occupations.csv", header=None)
        occupations_extracted = pd.read_csv("data/occupations_extracted.csv", header=None)
        occupations_new = occupations[~occupations[0].isin(occupations_extracted[0])]
    else: 
        occupations_new = pd.read_csv("occupations.csv", header=None)
    
    # If new occupations were discovered we query wikidata.
    if len(occupations_new) > 0: 
        query_wikidata(occupations_new.to_numpy())
        occupations_extracted = pd.read_csv("data/occupations_extracted.csv", header=None)

    # Load list of known countries.
    if os.path.exists('data/nIDs_valid.csv'):
        nIDs_valid_df = pd.read_csv('data/nIDs_valid.csv')
    else:
        nIDs_valid_df = pd.DataFrame(columns=['nID', 'nationality', 'is_valid'])
    nIDs_valid = nIDs_valid_df.set_index('nID').to_dict(orient='index')

    # Collect all unique nationalities from the new occupations list.
    nIDs = {}
    for idx, occupation in occupations_extracted.iterrows():
        occupation_name = occupation[0]
        occupation_df = pd.read_csv(f'data/csv/{occupation_name}.csv')

        for _, row in occupation_df.iterrows():
            nID = row['nationalityID']
            nationality = row['nationality']
            # Sets the nationality if not already in the dictionary.
            nIDs.setdefault(nID, nationality)

    # For each unique nID check if it is instance (and subclass) of "country" or "state" in wikidata.
    print('Querying wikidata to verify new nationalities')
    for nID, nationality in nIDs.items():
        if nID in nIDs_valid: continue  # Skip if its already a known country
        query = build_verification_query(nid=nID)
        response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers).json()
        
        nID_is_valid = response['boolean']
        nIDs_valid[nID] = {'nationality': nationality, 'is_valid': nID_is_valid}

    # Convert dictionary to DataFrame and save updated list of valid and invalid countries.
    nIDs_valid_df = pd.DataFrame.from_dict(nIDs_valid, orient='index').reset_index()
    nIDs_valid_df.rename(columns={'index': 'nID'}, inplace=True)
    nIDs_valid_df.to_csv('data/nIDs_valid.csv', index=False)

    # Remove entries with invalid countires from each occupations list.
    # Refactor gender if entry is kept.
    remove_invalid_nID_and_standardize_gender(occupations_extracted, nIDs_valid)


    

    