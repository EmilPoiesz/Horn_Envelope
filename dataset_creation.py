import pandas as pd
from helper_functions import *

sentence = "<mask> was born in {birthyear} in {nationality} and is a {occupation}."
occ_list = import_occupations("occupations_updated.csv")
"""
id = occ_list[0][1]
name = occ_list[0][0]
occ_file = 'data_dataframes/' + name + '.csv'
df = pd.read_csv(occ_file)
for index,row in df.iterrows():
    s = sentence.format(pronoun= get_pronoun(row['gender']), birthyear=row['birth'], nationality=row['nationality'], occupation=name.replace('_', ' '))
"""
country_list = pd.read_csv('countries_with_nids_grammar.csv', header=None).to_numpy()
country_dict= {row[0]: row[1] for row in country_list}

for occupation in occ_list:
    id = occupation[1]
    name = occupation[0]
    occ_file = 'dataframes_cleaned/' + name + '.csv'
    df = pd.read_csv(occ_file)
    sentence_list = []
    for index,row in df.iterrows():
        s = sentence.format(birthyear=row['birth'], nationality=country_dict[row['nid']], occupation=name.replace('_', ' '))
        sentence_list.append([s, get_pronoun(row['gender'])] )
    sentence_df = pd.DataFrame(sentence_list)
    sentence_df.to_csv('dataset.csv', mode='a',index=False, header=False)