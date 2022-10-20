import pandas as pd
import numpy as np

class Binarizer:
    """
        Takes care of converting a datapoint of the given dataset to a binary vector and vice-versa. The conversion is 
        a necessary step in using the rule-extractor with a language model. The relevant direction is here: converting a binary vector 
        into a sentence for the language model.

        Binary senquences represent: [BIRTH -- CONTINENT -- OCCUPATION] int this order.
    """
    def __init__(self, country_file : str, occ_file : str, amount_containers: int):
        self.country_list = pd.read_csv(country_file, names=['cid', ' country', 'continent']).set_index('cid').fillna('?')
        continents = self.country_list.continent.unique()
        self.continent_lookup = {continents[i] : i for i in range(len(continents))}
        self.occupation_list = pd.read_csv(occ_file, header=None).to_numpy()
        occupations = self.occupation_list[:,:1].flatten()
        self.occupation_lookup = {occupations[i] : i for i in range(len(occupations))}
        self.age_containers = pd.read_csv('age_containers' + str(amount_containers) + '.csv', header=None).to_numpy().flatten()
        

    def sentence_to_binary(self, sentence : str):
        pass

    def data_to_binary(self, data : list):
        """ Data comes in the form of 3 attributes: birthyear, country-id and occupation (as a list)"""
        birth = self.get_container(data[0])
        continent = self.binarize_string(self.get_continent(data[1]), kind = 'continent')
        occupation = self.binarize_string(data[2], kind = 'occupation')
        return np.concatenate([np.concatenate([birth,continent]), occupation])

    def get_container(self, birthyear : int):
        vector = np.zeros(len(self.age_containers) + 1, dtype=np.int8)
        for i in range(len(self.age_containers)):
            if i == 0:
                if birthyear < self.age_containers[i]:
                    vector[i] = 1
                    return vector
            else:
                if birthyear <= self.age_containers[i] and birthyear > self.age_containers[i-1]:
                    vector[i] = 1
                    return vector
        vector[len(self.age_containers)] = 1
        return vector

    def reverse_container(self, vector):
        index = np.nonzero(vector)[0][0]
        if index == 0:
            return "before " + str(self.age_containers[index])
        elif index == len(self.age_containers):
            return "after " + str(self.age_containers[index-1])
        else:
            return "between " + str(self.age_containers[index-1]) + " and " + str(self.age_containers[index])


    def get_continent(self, cid : str):
        return self.country_list.at[cid, 'continent']

    def binarize_string(self, input : str, kind = 'continent'):
        lookup = self.get_lookup(kind)
        vector = np.zeros(len(lookup), dtype=np.int8)
        vector[lookup[input]] = 1
        return vector
    
    def binary_to_string(self, vector, kind = 'continent'):
        lookup = self.get_lookup(kind)
        inv_lookup = {v: k for k, v in lookup.items()}
        index = np.nonzero(vector)[0][0]
        return inv_lookup[index]
    
    def get_lookup(self, label : str):
        if label == 'continent':
            return self.continent_lookup
        elif label == 'occupation':
            return self.occupation_lookup
        else:
            return np.nan

    def sentence_from_binary(self, bin):
        bin = np.array(bin)
        i_birth = len(self.age_containers)+1
        i_continent = i_birth + len(self.continent_lookup)
        splitted = np.split(bin, [i_birth, i_continent])
        continent = self.binary_to_string(splitted[1], kind = 'continent')
        occupation = self.binary_to_string(splitted[2], kind = 'occupation')
        birth = self.reverse_container(splitted[0])
        sentence = "<mask> was born {birth} in {continent} and is a {occupation}."
        return sentence.format(birth=birth, continent=continent, occupation=occupation)

"""
binarizer = Binarizer('data/country_list_continents.csv', 'data/occupations_updated.csv', 5)
occ = 'warden'
df = pd.read_csv('data/dataframes_cleaned/' + occ + '.csv')
"""
data = []
for index, row in df.iterrows():
    list_row = [row['birth'], row['nid'], occ]
    data.append(list_row)
"""
p = [df.at[0,'birth'], df.at[0,'nid'], occ]
binary = binarizer.data_to_binary(p)
print(p)
print(binary)
print(binarizer.sentence_from_binary(binary))"""