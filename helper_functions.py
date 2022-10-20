import pandas as pd
import numpy as np
import requests

def import_occupations(filename):
    #import all occupations and their ids from a given csv file
    occ_list = pd.read_csv(filename, header=None).to_numpy()
    return occ_list

def get_pronoun(gender):
    if gender == 'female':
        return 'she'
    elif gender == 'male':
        return 'he'
    else:
        return 'they'

def get_birth(birth_data):
    if birth_data['type'] == 'uri':
        return "unknown"
    elif birth_data['type'] == 'literal':
        value = birth_data['value'].split('-')
        if len(value[0]) > 0:
            #birthyear AD
            return value[0]
        else:
            #birthyear BC
            return str(value[1]) + " BC"

def get_nid(nid_data):
    value = nid_data['value'].split('/')
    return value[-1]

def replace_gender(gender):
    if gender == 'female' or gender == 'transgender female' or gender == 'cisgender female':
        return 'female'
    elif gender == 'male' or gender == 'transgender male' or gender == 'cisgender male':
        return 'male'
    else:
        return 'diverse'

def load_pred_df(model, occupation, new=False):
    if new:
        path = model + '_predictions_new/' + occupation + '.csv'
        df = pd.read_csv(path, names=['sentence', 'label', 'he', 'she', 'they'])
    else:
        path = model + '_predictions/' + occupation + '.csv'
        df = pd.read_csv(path, names=['sentence', 'label', 'prediction1', 'score1', 'prediction2', 'score2'])
    return df

def evaluate_predictions(model, occ_names, path=None):
    she_p = []
    he_p = []
    ppbs = []
    for occupation in occ_names:
        data = load_pred_df(model, occupation, new=True)
        data['she_p'] = data['she'] / (data['she'] + data['he'])
        data['he_p'] = data['he'] / (data['she'] + data['he'])
        data['ppbs'] = data['he_p'] - data['she_p']
        she_p.append(data['she_p'].mean())
        he_p.append(data['he_p'].mean())
        ppbs.append(data['ppbs'].mean())
    arr = np.array([occ_names, she_p, he_p, ppbs]).transpose()
    occupation_data = pd.DataFrame(arr, columns=['occupation', 'she_p', 'he_p', 'ppbs'])
    if path != None:
        occupation_data.to_csv(path)
    return occupation_data

def get_matrix(model, occ_list):
    occupation_matrices = {} #holds all confusion matrices

    for occ in occ_list:
        matrix = pd.DataFrame(np.zeros((3,3)), index=['he', 'she', 'they'], columns=['He', 'She', 'They'])
        occ_name = occ[0]
        df = load_pred_df(model, occ_name)
        grouped = df.groupby(['label', 'prediction1']).size()
        for index in grouped.keys():
            matrix.at[index] = grouped[index]
        occupation_matrices[occ_name] = matrix.fillna(0).transpose()
    return occupation_matrices

def get_continent(countryid, original=False):
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    headers = {'User-Agent' : 'MasterThesisQueryBot (sbl009@uib.no)'}
    if original:
        query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?continent ?cid WHERE {{
            wd:{country} wdt:P30 ?cid .

            OPTIONAL{{
                ?cid rdfs:label ?continent filter (lang(?continent) = "en") .
            }}
        }}
        """.format(country = countryid)
    else:
        query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?continent ?cid WHERE {{
            wd:{country} wdt:P17/wdt:P30 | wdt:P1366/wdt:P30 | wdt:P30 ?cid .

            OPTIONAL{{
                ?cid rdfs:label ?continent filter (lang(?continent) = "en") .
            }}
        }}
        """.format(country = countryid)
    data = requests.get(url ,params={'query': query, 'format': 'json'}, headers=headers).json()
    return data

