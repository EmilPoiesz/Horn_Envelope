import pandas as pd
import os
import sys
from helper_functions import *

"""
    Goes through all the data files and creates sentence/label pairs
    This is, what can be fed into the language model
"""

def load_occupations():
    # If dataset_wikidata_extraction.py has not been run before this file then we need to throw an error.
    if os.path.exists("data/occupations_extracted.csv"):
        return pd.read_csv("data/occupations_extracted.csv", header=None)
    
    print('No occupations have been extracted yet, you first need to run the file dataset_wikidata_extraction.py.')
    sys.exit(0)
    

if __name__ == "__main__":

    sentence = "<mask> was born in {birthyear} in {nationality} and is a {occupation}."
    
    occupations = load_occupations()

    
    # Load list of known countries.
    if os.path.exists('data/nIDs_valid.csv'):
        nIDs_valid = pd.read_csv('data/nIDs_valid.csv').set_index('nID')['is_valid'].to_dict()
    else:
        nIDs_valid = {}

    country_list = pd.read_csv('data/countries_with_nids_grammar.csv', header=None).to_numpy()
    country_dict= {row[0]: row[1] for row in country_list}

    for occupation in occupations:
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