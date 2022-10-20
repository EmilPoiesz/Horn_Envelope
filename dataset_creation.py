import pandas as pd
from helper_functions import *

sentence = "<mask> was born in {birthyear} in {nationality} and is a {occupation}."
occ_list = import_occupations("data/occupations_updated.csv")

country_list = pd.read_csv('data/countries_with_nids_grammar.csv', header=None).to_numpy()
country_dict= {row[0]: row[1] for row in country_list}

for occupation in occ_list:
    id = occupation[1]
    name = occupation[0]
    occ_file = 'data/dataframes_cleaned/' + name + '.csv'
    df = pd.read_csv(occ_file)
    sentence_list = []
    for index,row in df.iterrows():
        s = sentence.format(birthyear=row['birth'], nationality=country_dict[row['nid']], occupation=name.replace('_', ' '))
        sentence_list.append([s, get_pronoun(row['gender'])] )
    sentence_df = pd.DataFrame(sentence_list)
    sentence_df.to_csv('data/dataset.csv', mode='a',index=False, header=False)