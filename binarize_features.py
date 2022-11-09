import pandas as pd
import numpy as np
import random

class Binarizer:
    """
        Takes care of converting a datapoint of the given dataset to a binary vector and vice-versa. The conversion is 
        a necessary step in using the rule-extractor with a language model. The relevant direction is here: converting a binary vector 
        into a sentence for the language model.

        Binary senquences represent: [BIRTH -- CONTINENT -- OCCUPATION] int this order.

        Parameters: country_file : str
                        The file-path to import the countries with their corresponding continents
                    amount_containers : int
                        The amount of containers to be used to initialize the age containers. The containers are initialized
                        according to the total list of birth-dates divided into the given amount of intervals.
                    occupation : int or str
                        Given that occ_file=True, this expects a file-path to directly import a list of occupations to binarize from a .csv
                        that has each occupation written in a seperate line. With occ_file=False this expect an integer value to determine the 
                        threshold above which an occupation is included based on the dataset total amounts.
    """
    def __init__(self, country_file : str, amount_containers: int, occupation):
        self.continent_lookup = self.initialize_countries(country_file)
        self.occupation_lookup = self.initialize_occupations(occupation)
        self.age_containers = pd.read_csv('data/age_containers' + str(amount_containers) + '.csv', header=None).to_numpy().flatten()
        self.lengths = {'birth' : amount_containers, 'continent' : len(self.continent_lookup), 'occupation' : len(self.occupation_lookup)}

    def initialize_countries(self, country_file : str):
        country_list = pd.read_csv(country_file, names=['cid', ' country', 'continent']).set_index('cid').fillna('?')
        continents = country_list.continent.unique()
        continent_lookup = {continents[i] : i for i in range(len(continents))}
        return continent_lookup
    
    def initialize_occupations(self, occupation):
        if isinstance(occupation, str):
            occupations = pd.read_csv(occupation).to_numpy().flatten()
        elif isinstance(occupation, int):
            df = pd.read_csv('data/occupations_total.csv', index_col=0)
            sorted = df.sort_values(by=['total'], ignore_index=True)
            reduced = sorted[sorted['total'] >= occupation]
            occupations = reduced[['occupation']].to_numpy().flatten()
        else :
            occupations = []
        return {occupations[i] : i for i in range(len(occupations))}

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
        nonzero = np.nonzero(vector)
        if len(nonzero[0]) > 0:
            index = nonzero[0][0]
            if index == 0:
                return "before " + str(self.age_containers[index])
            elif index == len(self.age_containers):
                return "after " + str(self.age_containers[index-1])
            else:
                return "between " + str(self.age_containers[index-1]) + " and " + str(self.age_containers[index])
        else:
            return "in an unknown timeperiod"

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
        nonzero = np.nonzero(vector)
        if len(nonzero[0]) > 0:
            index = nonzero[0][0]
            return inv_lookup[index]
        else:
            if kind == 'continent':
                return 'an unknown place'
            else:
                return 'not known occupation'
    
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

    def is_assignment_valid(self, assignment):
        bin = np.array(assignment)
        i_birth = len(self.age_containers)+1
        i_continent = i_birth + len(self.continent_lookup)
        splitted = np.split(bin, [i_birth, i_continent])
        verb = False
        for i,arr in enumerate(splitted):
            if len(np.nonzero(arr)[0]) == 0:
                # no one in vector: take random
                verb = True
                vec = np.zeros(len(arr), dtype=np.int8)
                idx = random.sample(range(len(arr)), k=1)
                vec[idx] = 1
                splitted[i] = vec
            elif len(np.nonzero(arr)[0]) > 1:
                # multiple 1 in vector: take first one
                verb = True
                vec = np.zeros(len(arr), dtype=np.int8)
                idx = np.nonzero(arr)[0][0]
                vec[idx] = 1
                splitted[i] = vec
        if verb:
            print("Wrong assignment: ", assignment)
            print("Swap out with adjusted.")
            return np.concatenate(splitted)
        else:
            return assignment


"""
binarizer = Binarizer('data/country_list_continents.csv', 'data/occupations_updated.csv', 5, new_occ_file=False)
data = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(binarizer.is_assignment_valid(data))

data = []
for index, row in df.iterrows():
    list_row = [row['birth'], row['nid'], occ]
    data.append(list_row)

p = [df.at[0,'birth'], df.at[0,'nid'], occ]
binary = binarizer.data_to_binary(p)
print(p)
print(binary)
print(binarizer.sentence_from_binary(binary))"""