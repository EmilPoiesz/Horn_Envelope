######################################################################################
#                                                                                    #
#    Generates sentence/label pairs for every datapoint extrtacted from wikidata.    #
#                                                                                    #
######################################################################################

import pandas as pd
import os
import sys

def load_occupations():
    # If dataset_wikidata_extraction.py has not been run before this file then we need to throw an error.
    if os.path.exists("data/occupations_extracted.csv"):
        return pd.read_csv("data/occupations_extracted.csv", header=None)
    
    print('No occupations have been extracted yet, you first need to run the file dataset_wikidata_extraction.py.')
    sys.exit(0)

def get_pronoun(gender):
    is_female = gender == 'female' or gender == 'transgender female' or gender == 'cisgender female' or gender == 'trans woman'
    is_male   = gender == 'male'   or gender == 'transgender male'   or gender == 'cisgender male'   or gender == 'trans man'
    return 'female' if is_female else 'male' if is_male else 'other'
    
if __name__ == "__main__":

    query_sentence = "<mask> was born in {birthyear} in {nationality} and is a {occupation}."
    occupations   = load_occupations()

    for idx, occupation in occupations.iterrows():
        name = occupation[0]
        
        df = pd.read_csv(f'data/csv_clean/{name}.csv')
        sentences = []
        for index,row in df.iterrows():
            sentence = query_sentence.format(birthyear=row['birth'], nationality=row['nationality'], occupation=name.replace('_', ' '))
            pronoun = get_pronoun(row['gender'])
            
            # Store sentence with correct pronoun as a traning example
            sentences.append([sentence, pronoun] )

        sentence_df = pd.DataFrame(sentences)
        sentence_df.to_csv('data/dataset.csv', mode='a',index=False, header=False)