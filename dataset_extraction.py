import requests 
import json
import os
import pandas as pd

from config import SPARQL_QUERIES, URL, HEADERS


def send_query(query):
    """
    Sends a query to Wikidata and handles potential errors.
    This function sends a GET request to the Wikidata API with the provided query.
    It includes error handling to manage network issues and JSON parsing errors,
    ensuring that the querying process continues even if an error occurs.
    
    Args:
        query (str): The SPARQL query string to be sent to the Wikidata API.
    Returns:
        dict: The JSON response from the Wikidata API if the request is successful and the response is valid JSON.
              Returns an empty dictionary if a network error or JSON parsing error occurs.
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

def extract_unique_nationalities(occupations):
    """
    Extracts unique nationalities from a list of occupations by loading CSV files.

    Args:
        occupations (list of str): List of occupation names corresponding to CSV filenames.
    Returns:
        dict: A dictionary with nationality IDs as keys and nationalities as values.
    """

    nIDs = {}
    for occupation in occupations:
        occupation_df = pd.read_csv(f'data/csv/{occupation}.csv')

        for row in occupation_df.itertuples():
            nID = row.nationalityID
            nationality = row.nationality
            nIDs.setdefault(nID, nationality)
    return nIDs

def load_known_countries():
    """
    Load the file containing known countries with their names, continents, and validity status.

    If the file 'data/known_countries.csv' exists, it reads the data from the file.
    Otherwise, it creates an empty DataFrame with columns 'nID', 'nationality', 'continent', and 'is_valid'.

    Returns:
        dict: A dictionary where the keys are 'nID' and the values are dictionaries containing
              'nationality', 'continent', and 'is_valid' as keys.
    """
    if os.path.exists('data/known_countries.csv'):
        known_countries_df = pd.read_csv('data/known_countries.csv')
    else:
        known_countries_df = pd.DataFrame(columns=['nID', 'nationality', 'continent', 'is_valid'])
    known_countries = known_countries_df.set_index('nID').to_dict(orient='index')
    return known_countries

def clean_occupation_data(occupation_name, nIDs_valid):
    """
    Cleans the data for a single occupation by removing invalid nationality IDs and standardizing gender.
    Args:
        occupation_name (str): The name of the occupation corresponding to the CSV filename in 'data/csv/'.
        nIDs_valid (set): Set of valid nationality IDs to be used for filtering entries.
    Returns:
        pd.DataFrame: The cleaned DataFrame for the occupation.
    """
    occupation_df = pd.read_csv(f'data/csv/{occupation_name}.csv')
    total_pre_cleanup = len(occupation_df)

    # Filter out rows with invalid nationality IDs
    occupation_df = occupation_df[occupation_df['nationalityID'].isin(nIDs_valid)]

    # Standardize gender
    occupation_df['gender'] = occupation_df['gender'].apply(standardize_gender)

    # Drop duplicate rows
    occupation_df = occupation_df.drop_duplicates(subset='name', keep='first')
    occupation_df = occupation_df.drop_duplicates(keep='first')
    total_post_cleanup = len(occupation_df)

    print(f'Cleaned up {occupation_name}: {total_pre_cleanup} -> {total_post_cleanup}')
    return occupation_df

def standardize_gender(gender):
    """
    Standardizes various gender identifiers into three categories: 'female', 'male', or 'other'.

    Args:
        gender (str): The gender identifier to be standardized. This can include terms like 'female', 
                      'transgender female', 'cisgender female', 'trans woman', 'male', 'transgender male', 
                      'cisgender male', 'trans man', or any other gender identifier.

        str: A standardized gender category, which will be 'female' for any female-related identifiers, 
             'male' for any male-related identifiers, and 'other' for any identifiers that do not match 
             the specified female or male terms.
    """
    is_female = gender == 'female' or gender == 'transgender female' or gender == 'cisgender female' or gender == 'trans woman'
    is_male   = gender == 'male'   or gender == 'transgender male'   or gender == 'cisgender male'   or gender == 'trans man'
    return 'female' if is_female else 'male' if is_male else 'other'

def safe_get(result, attribute):
    """
    Retrieve the value of a specified attribute from a result dictionary.

    This function checks if the given attribute exists in the result dictionary.
    If the attribute exists, it returns the associated value. If the attribute
    does not exist, it returns a default value of '?'.

    Args:
        result (dict): The dictionary from which to retrieve the attribute value.
        attribute (str): The key of the attribute to retrieve from the result dictionary.

    Returns:
        The value associated with the attribute if it exists, otherwise '?'.
    """
    return result[attribute]['value'] if attribute in result else '?'

def parse_occupation_data(results):
    """
    Parses occupation data from a list of results.
    Args:
        results (list): A list of dictionaries containing occupation data. Each dictionary is expected to have the keys:
            - 'birth': The birthdate of the individial.
            - 'individual': The name of the individual.
            - 'gender': The gender of the individual.
            - 'nationality': The nationality of the individual.
            - 'nationalityID': The URL containing the nationality ID of the individual.
    Returns:
        list: A list of dictionaries, each containing the following keys:
            - 'name': The name of the individual.
            - 'gender': The gender of the individual.
            - 'nationality': The nationality of the individual.
            - 'nationalityID': The extracted ID from the nationality URL.
            - 'birth': The birth year of the individual, or '?' if not available.
    """
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
        
    return occupation_data

if __name__ == "__main__":

    occupations_df = pd.read_csv("data/occupations.csv", header=0)
    new_occupations = occupations_df[occupations_df['extracted'] == False] 

    # Query wikidata for all new occupations
    new_occupations_extracted = []
    for occupation in new_occupations.itertuples():
        occupation_id   = occupation.occupation_id
        occupation_name = occupation.occupation
        
        print(f'We are querying Wikidata for {occupation_name}.')
        query = SPARQL_QUERIES['occupation_query'].format(occupationID=occupation_id)
        response = send_query(query)
        results = response.get('results', {}).get('bindings', [])
        
        if len(results) == 0: print("We encountered an error and found no results."); continue
        new_occupations_extracted.append(occupation)

        print("Queried for " + occupation_name + " and found " + str(len(results)) + " results in wikidata.")
        with open(f'data/json/{occupation_name}.json', 'w') as outfile:
            json.dump(results, outfile)

        occupation_data = parse_occupation_data(results) 
        pd.DataFrame(occupation_data).to_csv(f'data/csv/{occupation_name}.csv', index=False)
        

    # Update the 'extracted' column in occupations.csv
    for occupation in new_occupations_extracted:
        occupation_name = occupation.occupation
        occupations_df.loc[occupations_df['occupation'] == occupation_name, 'extracted'] = True
    occupations_df.to_csv("data/occupations.csv", index=False)
    extracted_occupations = occupations_df[occupations_df['extracted'] == True]['occupation'].values

    unique_nIDs = extract_unique_nationalities(extracted_occupations)
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
        
        # If continent is not found, query the countries that the nation is part of
        if not continent_result:
            query = SPARQL_QUERIES['get_continent_extensive_query'].format(nid=nID)
            response = send_query(query)
            continent_result = response.get('results', {}).get('bindings', [])
        
        if continent_result:
            continent = continent_result[0].get('continent', {}).get('value', '?')
        
        if continent == '?': continue # If nation doesn't have a continent then skip 

        known_countries[nID] = {
            'nationality': nationality, 
            'continent': continent,
            'is_valid': nID_is_valid
        }

    # Convert dictionary to DataFrame and save updated list of valid and invalid countries.
    know_countries_df = pd.DataFrame.from_dict(known_countries, orient='index').reset_index()
    know_countries_df.rename(columns={'index': 'nID'}, inplace=True)
    know_countries_df.to_csv('data/known_countries.csv', index=False)

    # Remove invalid country IDs and standardize gender for each occupation
    for occupation_name in extracted_occupations:
        cleaned_df = clean_occupation_data(occupation_name, known_countries)
        cleaned_df.to_csv(f'data/csv_clean/{occupation_name}.csv', index=False)
        


    

    