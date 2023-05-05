import pandas as pd
from helper_functions import *
from binarize_features import *

def get_container(birthyear, age_containers):
    if birthyear == '?':
        return "in an unknown timeperiod"
    if isinstance(birthyear, str) and 'BC' in birthyear:
        return "before " + str(age_containers[0])
    for i in range(len(age_containers)):
        if i == 0:
            if int(birthyear) < age_containers[i]:
                    return "before " + str(age_containers[i])
        else:
            if int(birthyear) <= age_containers[i] and int(birthyear) > age_containers[i-1]:
                return "between " + str(age_containers[i-1]) + " and " + str(age_containers[i])
    return "after " + str(age_containers[i-1])

occ_list = import_occupations('data_new/occupations_updated.csv')

# Make list of all countries in the new dataset and compare to the list of the old dataset
# hand-annotate missing countries with their continent
"""
old_countries = pd.read_csv('data/country_list_continents_new.csv', names=['cid', ' country', 'continent']).set_index('cid').fillna('?')
old_cids = old_countries.index.values.tolist()
nids = []
country_pairs = []
for occupation in occ_list:
        name = occupation[0]
        occ_file = 'data_new/dataframes_cleaned/' + name + '.csv'
        df = pd.read_csv(occ_file)
        for index, row in df.iterrows():
            nationality = row['nationality']
            nid = row['nid']
            if nid not in old_cids and nid not in nids:
                pair = [nid, nationality]
                nids.append(nid)
                country_pairs.append(pair)

df = pd.DataFrame(country_pairs)
df.to_csv('data_new/country_list_extra.csv', index=False, header=False)
"""
old_countries = pd.read_csv('data/country_list_continents_new.csv', names=['cid', ' country', 'continent']).fillna('?')
new_countries = pd.read_csv('data_new/country_list_continents.csv', names=['cid', ' country', 'continent']).fillna('?')
total_countries = pd.concat([old_countries, new_countries], ignore_index=True).set_index('cid')
total_countries.to_csv('data_new/country_list_continents_total.csv', header=False)

#binarizer = Binarizer('data_new/country_list_continents_total.csv', 5, 'data_new/occupations_updated.csv')

age_containers = pd.read_csv('data/age_containers' + str(5) + '.csv', header=None).to_numpy().flatten()

sentence = "<mask> was born {birthyear} in {nationality} and is a {occupation}."

for occupation in occ_list:
    name = occupation[0]
    occ_file = 'data_new/dataframes_cleaned/' + name + '.csv'
    df = pd.read_csv(occ_file)
    sentence_list = []
    for index, row in df.iterrows():
        birthtext = get_container(row['birth'], age_containers)
        continent = total_countries.at[row['nid'], 'continent']
        if continent == '?':
            continent = 'an unknown place'
        s = sentence.format(birthyear=birthtext, nationality=continent, occupation=name.replace('_', ' '))
        sentence_list.append([s, get_pronoun(row['gender'])])
    pd.DataFrame(sentence_list).to_csv('data_new/dataset_refac/' + name + '.csv',index=False, header=False)