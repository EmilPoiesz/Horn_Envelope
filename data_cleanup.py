from helper_functions import *
import pandas as pd
import requests

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
headers = {'User-Agent' : 'MasterThesisQueryBot (sbl009@uib.no)'}
query =  """
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    ASK {{
        {nid} wdt:P31/wdt:P279* wd:Q56061 .
    }}
    """
occ_list = import_occupations('occupations_updated.csv')
amounts = {}
"""
nids = []
for occupation in occ_list:
        name = occupation[0]
        occ_file = 'data_dataframes/' + name + '.csv'
        df = pd.read_csv(occ_file)
        for index, row in df.iterrows():
            nid = row['nid']
            if nid not in nids:
                nids.append(nid)

delete_nids = []
for nid in nids:
    q = query.format(nid = 'wd:' + nid)
    data = requests.get(url ,params={'query': q, 'format': 'json'}, headers=headers).json()
    if not data['boolean']:
        print('Delete country \t' + nid)
        delete_nids.append(nid)
print(delete_nids)
with open('country_nids_del.csv', 'w') as f:
    f.write('\n'.join(delete_nids))
"""
delete_nids = pd.read_csv('country_nids_del.csv', header=None).to_numpy()
for occupation in occ_list:
    to_delete = []
    name = occupation[0]
    occ_file = 'data_dataframes/' + name + '.csv'
    df = pd.read_csv(occ_file)
    old_total = len(df)
    for index, row in df.iterrows():
        nationality = row['nationality']
        nid = row['nid']
        if nid in delete_nids:
            to_delete.append(index)
        df.at[index,'gender'] = replace_gender(row['gender'])
    df = df.drop(to_delete)
    df = df.drop_duplicates(subset='name', keep='first')
    total_tuple = (old_total, len(df))
    df.to_csv('dataframes_cleaned/' + name + '.csv', index=False)
    amounts[name] = total_tuple
    print('Finished with ' + name)
print(amounts)
