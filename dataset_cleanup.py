from helper_functions import *
import pandas as pd
import requests
from binarize_features import *

occ_list = import_occupations('data/occupations_updated.csv')

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
headers = {'User-Agent' : 'MasterThesisQueryBot (sbl009@uib.no)'}
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

#The first part is to seperate out those countries, that are not actual countries according to wikidata (their nid is not a country)

"""
# Q6256 is wikidata country
occ_list = import_occupations('data/occupations_updated.csv')
amounts = {}
nids = []
#collect all unique nids from dataset
for occupation in occ_list:
        name = occupation[0]
        occ_file = 'data/data_dataframes/' + name + '.csv'
        df = pd.read_csv(occ_file)
        for index, row in df.iterrows():
            nid = row['nid']
            if nid not in nids and 'Q' in nid:
                nids.append(nid)

#for each unique nid check, if its a country (instanceof (and subclass) of "country" or "state"
delete_nids = []
for nid in nids:
    q = query.format(nid = 'wd:' + nid)
    data = requests.get(url ,params={'query': q, 'format': 'json'}, headers=headers).json()
    if not data['boolean']:
        print('Delete country \t' + nid)
        delete_nids.append(nid)
print(delete_nids)
with open('data/country_nids_del.csv', 'w') as f:
    f.write('\n'.join(delete_nids))
"""

"""
Cleanup thats done on the data:
    Get all those countries that are supposed to be deleted and remove their datapoint from the data
    ort gender into 3 categories
    delete duplicate names (multiple datapoints for the same entity)
"""

amounts = {}
delete_nids = pd.read_csv('data/country_nids_del.csv', header=None).to_numpy()
for occupation in occ_list:
    to_delete = []
    name = occupation[0]
    occ_file = 'data/data_dataframes/' + name + '.csv'
    df = pd.read_csv(occ_file)
    old_total = len(df)
    for index, row in df.iterrows():
        nationality = row['nationality']
        nid = row['nid']
        if not 'Q' in nid:
            df.at[index, 'nid'] = '?'
        if nid in delete_nids:
            to_delete.append(index)
        df.at[index,'gender'] = replace_gender(row['gender'])
    df = df.drop(to_delete)
    df = df.drop_duplicates(subset='name', keep='first')
    df = df.drop_duplicates(keep='first')
    total_tuple = (old_total, len(df))
    df.to_csv('data/dataframes_cleaned/' + name + '.csv', index=False)
    amounts[name] = total_tuple
    print('Finished with ' + name)
print(amounts)
