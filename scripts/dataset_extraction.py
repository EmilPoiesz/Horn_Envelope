import requests 
import json
import os
import pandas as pd

from config import SPARQL_QUERIES, URL, HEADERS
from helper_functions import *

def send_query(query):
    """
    Queries wikidata with error handling. 

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
    except json.JSONDecodeError as e:
        print(f"\nCould not parse JSON error: {e}\n")
        return {}
    
    return result

def extract_unique_nationalities(occupations:pd.DataFrame):
    """
    Loads all occupation files and creates a combined dict of all unique nationalities

    Returns:
        dict: nID as index, nationality as value
    """
    nIDs = {}
    for idx, occupation in occupations.iterrows():
        occupation_name = occupation.occupation
        occupation_df = pd.read_csv(f'data/csv/{occupation_name}.csv')

        for _, row in occupation_df.iterrows():
            nID = row['nationalityID']
            nationality = row['nationality']
            
            # Add nationality to dictionary if not already present.
            nIDs.setdefault(nID, nationality)
    return nIDs

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
            if not nID in nIDs_valid: 
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

def safe_get(result, attribute):
    """
    Checks if the attribute exists in the result then returns that value.
    If the attribute is not in the entry then return '?'.
    """
    return result[attribute]['value'] if attribute in result else '?'

if __name__ == "__main__":

    occupations_df = pd.read_csv("data/occupations.csv", header=0)
    new_occupations = occupations_df[occupations_df['extracted'] == False] 

    # Query wikidata for all new occupations
    new_occupations_extracted = []
    for _, occupation in new_occupations.iterrows():
        occupation_id   = occupation.occupation_id
        occupation_name = occupation.occupation
        
        print(f'We are querying Wikidata for {occupation_name}.')
        query = SPARQL_QUERIES['occupation_query'].format(occupationID=occupation_id)
        response = send_query(query)
        results = response.get('results', {}).get('bindings', [])
        
        if len(results) == 0:
            print("We encountered an error and found no results.")
            continue
        print("Queried for " + occupation_name + " and found " + str(len(results)) + " results in wikidata.")
        
        occupation_data = []
        for result in results:
            
            birth = '?'
            if result['birth']['type'] == 'literal':
                date_of_birth = result['birth']['value'].split('-')
                birth = date_of_birth[0]
                if len(birth) == 0:
                    birth = '-' + date_of_birth[1]

            occupation_data.append({
                'name': safe_get(result,'individual'),
                'gender': safe_get(result,'gender'),
                'nationality': safe_get(result,'nationality'),
                'nationalityID': safe_get(result,'nationalityID').split('/')[-1], #extract id from url
                'birth': birth
            }) 

        # Save the dataframe as CSV
        pd.DataFrame(occupation_data).to_csv(f'data/csv/{occupation_name}.csv', index=False)

        # Save raw JSON data
        with open(f'data/json/{occupation_name}.json', 'w') as outfile:
            json.dump(results, outfile)

        # Keep track of extracted occupations
        new_occupations_extracted.append(occupation)

    # Update the 'extracted' column in occupations.csv
    for occupation in new_occupations_extracted:
        occupation_name = occupation[0]
        occupations_df.loc[occupations_df['occupation'] == occupation_name, 'extracted'] = True
    occupations_df.to_csv("data/occupations.csv", index=False)

    unique_nIDs = extract_unique_nationalities(occupations_df)
    known_countries = load_known_countries()
    
    # Save the validity and continent of each unique nID
    print('Querying wikidata to verify new nationalities')
    for nID, nationality in unique_nIDs.items():
        if nID in known_countries: continue  # Skip if its already a known country
        
        # Validity of country
        query = SPARQL_QUERIES['verify_nationality_query'].format(nid=nID)
        response = send_query(query)
        nID_is_valid = response.get('boolean', False)

        # Continent of country
        continent = '?'
        query = SPARQL_QUERIES['get_continent_query'].format(nid=nID)
        response = send_query(query)
        continent_result = response.get('results', {}).get('bindings', [])
        
        if not continent_result:
            query = SPARQL_QUERIES['get_continent_extensive_query'].format(nid=nID)
            response = send_query(query)
            continent_result = response.get('results', {}).get('bindings', [])
        
        if continent_result:
            continent = continent_result[0].get('continent', {}).get('value', '?')
        
        if continent == '?': continue # If nation is not in a continent, skip 

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


    

    