import pandas as pd
import numpy as np

from . import config

class Binary_parser:
    """
        Converts a datapoint in the extracted dataset to a binary vector and vice-versa. 
        Binary senquences represent: [BIRTH -- CONTINENT -- OCCUPATION] int this order.

        Parameters: 
        country_filepath (str):
            The filepath to import the countries with their corresponding continents
        occupation_filepath (str):
            The filepath to import the occupations.
    """
    def __init__(self):
        self.features = config.FEATURES
        self.lengths = {'birth': len(self.features['age_containers'])+1, 
                        'continent': len(self.features['continents']), 
                        'occupation': len(self.features['occupations']),
                        'gender': 2
                        }
        self.total_length = sum(self.lengths.values())
    
    def parse_birth(self, index):
        if index == 0: return "before " + str(self.features['age_containers'][index])
        if index == len(self.features['age_containers']): return "after " + str(self.features['age_containers'][index-1])
        return "between " + str(self.features['age_containers'][index-1]) + " and " + str(self.features['age_containers'][index])
    
    def binary_to_sentence(self, binary_str):
        """
        Creates a readable sentence from the binary string representing the input to the LLM using lookup tables. 

        Args:
            binary_str: The binary string we want to convert to a readable sentence.
        Returns:
            A readable string in the format '<gender> was born <time period> in <continent> and is a <occupation>'
        """
        birth_str, continent_str, occupation_str, gender_str = [
            array.tolist() for array in np.split(np.array(binary_str), 
                [
                    self.lengths['birth'], 
                    self.lengths['birth'] + self.lengths['continent'], 
                    self.lengths['birth'] + self.lengths['continent'] + self.lengths['occupation']
                ]
            )
        ]
        
        if 1 not in birth_str: birth = 'an unknown time period'
        else: birth = self.parse_birth(birth_str.index(1))
        
        if 1 not in continent_str: continent = 'an unknown place'
        else: continent = self.features['continents'][continent_str.index(1)]
        
        if 1 not in occupation_str: occupation = 'an unknown occupation'
        else: occupation = self.features['occupations'][occupation_str.index(1)]

        if len(gender_str) == 0: return f"<mask> was born {birth} in {continent} and is a {occupation}."
        gender = 'She' if gender_str[0] == 1 else 'He'
        
        return f"{gender} was born {birth} in {continent} and is a {occupation}."
        
    def index_to_name(self, index):
        """
        Converts the index of a feature to its name.

        Args:
            index (int): The index of the feature.
        Returns:
            The name of the variable.
        """
        if index <= self.lengths['birth']: 
            lst = [0] * self.lengths['birth']
            lst[index] = 1
            parse = self.parse_birth(lst)
            return parse
        if index <= self.lengths['birth'] + self.lengths['continent']: return self.features['continents'][index - self.lengths['birth']]
        if index <= self.lengths['birth'] + self.lengths['continent'] + self.lengths['occupation']: return self.features['occupations'][index - self.lengths['birth'] - self.lengths['continent']]
        return 'He' if index == self.total_length - 1 else 'She'

