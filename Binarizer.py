import pandas as pd
import numpy as np
import random

from config import AGE_CONTAINERS

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
    def __init__(self, country_file : str, occupation):
        self.continent_lookup  = self.initialize_countries(country_file)
        self.occupation_lookup = self.initialize_occupations(occupation)
        self.age_containers    = AGE_CONTAINERS
        self.lengths = {'birth': len(AGE_CONTAINERS), 'continent': len(self.continent_lookup), 'occupation': len(self.occupation_lookup), 'gender': 2}

    def initialize_countries(self, country_file : str):
        countries = pd.read_csv(country_file, names=['nID', 'nationality', 'continent', 'is_valid']).set_index('nID').fillna('?')
        countries = countries[countries['is_valid'] == 'True'].drop(columns=['is_valid'])
        continents = countries.continent.unique()
        
        return {continents[i] : i for i in range(len(continents))}

    
    def initialize_occupations(self, occupation):
        if isinstance(occupation, str):
            occupations_df = pd.read_csv(occupation)
            occupations = occupations_df[occupations_df['extracted'] == True]['occupation'].values
        elif isinstance(occupation, int):
            df = pd.read_csv('data/occupations_total.csv', index_col=0)
            sorted = df.sort_values(by=['total'], ignore_index=True)
            reduced = sorted[sorted['total'] >= occupation]
            occupations = reduced[['occupation']].to_numpy().flatten()
        else :
            occupations = []
        return {occupations[i] : i for i in range(len(occupations))}

    def data_to_binary(self, data : list):
        """ 
            Data comes in the form of 3 attributes: birthyear, country-id and occupation (as a list)
            Continent must be a known value! -> no '?' allowed
        """
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
        #doesn't deal with unknown continent!
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

    def sentence_from_binary(self, bin, has_gender=False):
        bin = np.array(bin)
        i_birth = len(self.age_containers)+1
        i_continent = i_birth + len(self.continent_lookup)
        splitted = np.split(bin, [i_birth, i_continent])
        continent = self.binary_to_string(splitted[1], kind = 'continent')
        occupation = self.binary_to_string(splitted[2], kind = 'occupation')
        birth = self.reverse_container(splitted[0])
        if has_gender:
            gender = 'She' if bin[-2] == 1 else 'He'
            return f"{gender} was born {birth} in {continent} and is a {occupation}."
        return f"<mask> was born {birth} in {continent} and is a {occupation}."


