######################################################################################
#                                                                                    #
#    Generates sentence/label pairs for every datapoint extrtacted from wikidata.    #
#                                                                                    #
######################################################################################

import pandas as pd
    
def get_birthyear_container(age_containers, birthyear):

    if birthyear == '?': return 'in an unknown timeperiod'
    birthyear = int(birthyear)
    if birthyear < age_containers[0]: return f'before {age_containers[0]}'
    
    for i in range(len(age_containers) - 1):
        if age_containers[i] <= birthyear <= age_containers[i + 1]:
            return f'between {age_containers[i]} and {age_containers[i + 1]}'
        
    return f'after {age_containers[-1]}'
    

if __name__ == "__main__":
    age_containers = [1875, 1925, 1951, 1970]
    query_sentence = "<mask> was born in {birthyear} in {nationality} and is a {occupation}."
    occupations_df = pd.read_csv("data/occupations.csv", header=0)
    extracted_occupations = occupations_df[occupations_df['extracted'] == True]['occupation'].values

    for occupation in extracted_occupations:
        df = pd.read_csv(f'data/csv_clean/{occupation}.csv')
        
        sentences = []
        for row in df.itertuples():

            birthyear = get_birthyear_container(age_containers, row.birth)
            sentence = query_sentence.format(birthyear=birthyear, nationality=row.nationality, occupation=occupation.replace('_', ' '))
            pronoun = row.gender
            
            # Store sentence with correct pronoun as a traning example
            sentences.append([sentence, pronoun] )

        sentence_df = pd.DataFrame(sentences)
        sentence_df.to_csv('data/dataset.csv', mode='a',index=False, header=False)