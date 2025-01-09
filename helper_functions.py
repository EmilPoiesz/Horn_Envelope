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

def get_birth(dataitem):
    if 'birth' in dataitem:
        if dataitem['birth']['type'] == 'uri':
            return '?'
        elif dataitem['birth']['type'] == 'literal':
            value = dataitem['birth']['value'].split('-')
            if len(value[0]) > 0:
                #birthyear AD
                return value[0]
            else:
                #birthyear BC
                return str(value[1]) + " BC"
    else:
        return '?'
def get_nid(nid_data):
    if nid_data == '?':
        return '?'
    else:
        value = nid_data.split('/')
        return value[-1]

def replace_gender(gender):
    if gender == 'female' or gender == 'transgender female' or gender == 'cisgender female':
        return 'female'
    elif gender == 'male' or gender == 'transgender male' or gender == 'cisgender male':
        return 'male'
    else:
        return 'diverse' 

def safe_get(dataitem, field):
    if field in dataitem:
        return dataitem[field]['value']
    else:
        return '?'

def load_pred_df(model, occupation, short=False):
    if short:
        path = 'data/probing_' + model + '/' + occupation + '.csv'
        df = pd.read_csv(path)
    else:
        path = 'data/probing_' + model + '/' + occupation + '_format.csv'
        df = pd.read_csv(path)
    return df

def load_pred_df_new(model, occupation):
    path = 'data/probing_' + model + '/' + occupation + '_format.csv'
    df = pd.read_csv(path, index_col = 0, low_memory=False)
    return df

def evaluate_predictions(model, occ_names, path=None):
    she_p = []
    he_p = []
    ppbs = []
    for occupation in occ_names:
        data = load_pred_df(model, occupation, short=True)
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

def get_matrix_subset(model, occ_list):
    occupation_matrices = {} #holds all confusion matrices

    for occ_name in occ_list:
        matrix = pd.DataFrame(np.zeros((3,3)), index=['he', 'she', 'they'], columns=['He', 'She', 'They'])
        df = load_pred_df_new(model, occ_name)
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

def lm_inference(unmasker, sentence, model = 'roberta-base'):
    if model.split('-')[0] == 'bert' :
        sentence = sentence.replace('<mask>', '[MASK]')
    return unmasker(sentence)

def get_prediction(result, binary = False):
    if binary:
        if result[0]['token_str'] == 'She':
            return 0
        elif result[0]['token_str'] == 'He':
            return 1
        else:
            del result[0]
            print('Recursion')
            return get_prediction(result, binary = True)
    else:    
        return result[0]['token_str']