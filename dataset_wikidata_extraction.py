import requests 
import json
import os
import sys
import pandas as pd

from config import SPARQL_QUERIES, URL, HEADERS
from helper_functions import *

def send_query(query):
    """
    Queries wikidata with error handeling. 

    This ensures that we continue to query Wikidata if one query throws an error.
    """
    try:
        response = requests.get(URL, params={'query': query, 'format': 'json'}, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\nNetwork error occurred: {e}\n")
        return {}
    
    try: 
        result = response.json()
    except requests.JSONDecodeError as e:
        print(f"\nCould not parse JSON error: {e}\n")
        return {}
    
    return result

def extract_unique_nationalities(occupations_extracted):
    """
    Loads all occupation files and creates a combined dict of all unique nationalities

    Returns:
        dict: nID as index, nationality as value
    """
    nIDs = {}
    for idx, occupation in occupations_extracted.iterrows():
        occupation_name = occupation[0]
        occupation_df = pd.read_csv(f'data/csv/{occupation_name}.csv')

        for _, row in occupation_df.iterrows():
            nID = row['nationalityID']
            nationality = row['nationality']
            
            # Sets the nationality if not already in the dictionary.
            nIDs.setdefault(nID, nationality)
    return nIDs

def extract_new_occupations():
    """
    Calculates the list of occupations that have not been extracted yet.

    Returns:
        List: New occupations we need to query
    """
    if os.path.exists('data/occupations_extracted.csv'):
        occupations = pd.read_csv("occupations.csv", header=None)
        occupations_extracted = pd.read_csv("data/occupations_extracted.csv", header=None)
        return occupations[~occupations[0].isin(occupations_extracted[0])].to_numpy()
    return pd.read_csv("occupations.csv", header=None).to_numpy()

def load_known_countries():
    """
    Loads the file with known countries with their names and validity.

    Returns:
        dict: nID as index, nationality and is_valid as values.
    """
    if os.path.exists('data/known_countries.csv'):
        known_countries_df = pd.read_csv('data/known_countries.csv')
    else:
        known_countries_df = pd.DataFrame(columns=['nID', 'nationality', 'continent', 'is_valid'])
    known_countries = known_countries_df.set_index('nID').to_dict(orient='index')
    return known_countries

def remove_invalid_nID_and_standardize_gender(occupations_extracted, nIDs_valid):
    """
    Removes invalid nIDs (countries that don't exists, errors in wikidata, etc) and standardizes the gender of the remaining entries.

    Saves the results as a csv in 'data/csv_clean/..'
    """
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
    """
    Standardizes gender.

    Returns: 
        string: 'female', 'male', or 'other'
    """
    is_female = gender == 'female' or gender == 'transgender female' or gender == 'cisgender female' or gender == 'trans woman'
    is_male   = gender == 'male'   or gender == 'transgender male'   or gender == 'cisgender male'   or gender == 'trans man'
    return 'female' if is_female else 'male' if is_male else 'other'

def datafield_in_entry(entry, datafield):
    """
    Checks if the datafield is in the entry then returns that value.

    If datafield not in the entry then return '?'
    """
    return entry[datafield]['value'] if datafield in entry else '?'

if __name__ == "__main__":

    new_occupations = extract_new_occupations()
    if len(new_occupations) <= 0: 
        sys.exit("No new occupations")

    # Query wikidata for all new occupations
    new_occupations_extracted = []
    for occupation in new_occupations:
        occupation_id   = occupation[1]
        occupation_name = occupation[0]
        
        print(f'We are querying Wikidata for {occupation_name}.')
        query = SPARQL_QUERIES['occupation_query'].format(occupationID=occupation_id)
        response = send_query(query)
        results = response.get('results', {}).get('bindings', [])
        
        if len(results) == 0:
            print("We encountered an error and found no results.")
            continue
        print("Queried for " + occupation_name + " and found " + str(len(results)) + " results in wikidata.")
        
        occupation_data = []
        for entry in results:
            
            birth = '?'
            if entry['birth']['type'] == 'literal':
                date_of_birth = entry['birth']['value'].split('-')
                birth = date_of_birth[0]
                if len(birth) == 0:
                    birth = '-' + date_of_birth[1]

            occupation_data.append({
                'name': datafield_in_entry(entry, 'individual'),
                'gender': datafield_in_entry(entry, 'gender'),
                'birth': birth,
                'nationality': datafield_in_entry(entry,'nationality'),
                'nationalityID': datafield_in_entry(entry,'nationalityID').split('/')[-1]
            }) 

        # Save the dataframe as CSV
        pd.DataFrame(occupation_data).to_csv(f'data/csv/{occupation_name}.csv', index=False)

        # Save raw JSON data
        with open(f'data/json/{occupation_name}.json', 'w') as outfile:
            json.dump(results, outfile)

        # Keep track of extracted occupations
        new_occupations_extracted.append(occupation)

    # Update extracted occupations list
    if os.path.exists('data/occupations_extracted.csv'):
        df_occupations = pd.read_csv('data/occupations_extracted.csv', header=None)
        df_occupations = pd.concat([df_occupations, pd.DataFrame(new_occupations_extracted)])\
            .to_csv('data/occupations_extracted.csv', index=None, header=None)
    else:
        df_occupations = pd.DataFrame(new_occupations_extracted)\
            .to_csv('data/occupations_extracted.csv', index=None, header=None)
    

    occupations_extracted = pd.read_csv("data/occupations_extracted.csv", header=None)
    unique_nIDs = extract_unique_nationalities(occupations_extracted)
    known_countries = load_known_countries()
    
    # Save the validity and continent of each unique nID
    print('Querying wikidata to verify new nationalities')
    for nID, nationality in unique_nIDs.items():
        if nID in known_countries: continue  # Skip if its already a known country
        
        # Validity of country
        query = SPARQL_QUERIES['verify_nationality_query'].format(nid=nID)
        response = send_query(query)
        nID_is_valid = response['boolean']

        # Continent of country
        continent = '?'
        query = SPARQL_QUERIES['get_continent_query'].format(nid=nID)
        response = send_query(query)
        continent_result = response.get('results', {}).get('bindings', [])
        
        if continent_result:
            continent = continent_result[0].get('continent', {}).get('value',[])
        else:
            query = SPARQL_QUERIES['get_continent_extensive_query'].format(nid=nID)
            response = send_query(query)
            continent_result = response.get('results', {}).get('bindings', [])
            if continent_result:
                continent = continent_result[0].get('continent', {}).get('value',[])
        

        known_countries[nID] = {
            'nationality': nationality, 
            'continent': continent,
            'is_valid': nID_is_valid
        }

    # Convert dictionary to DataFrame and save updated list of valid and invalid countries.
    df = pd.DataFrame.from_dict(known_countries, orient='index').reset_index()
    df.rename(columns={'index': 'nID'}, inplace=True)
    df.to_csv('data/known_countries.csv', index=False)

    remove_invalid_nID_and_standardize_gender(occupations_extracted, known_countries)


    

    