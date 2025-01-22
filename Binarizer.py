import pandas as pd
import numpy as np

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
        self.continent_lookup  = self.initialize_continent_lookup(country_file)
        self.occupation_lookup = self.initialize_occupation_lookup(occupation)
        self.age_containers    = AGE_CONTAINERS
        self.lengths = {'birth': len(AGE_CONTAINERS)+1, 'continent': len(self.continent_lookup), 'occupation': len(self.occupation_lookup), 'gender': 2}

    def initialize_continent_lookup(self, country_file : str):
        countries = pd.read_csv(country_file, names=['nID', 'nationality', 'continent', 'is_valid']).set_index('nID').fillna('?')
        countries = countries[countries['is_valid'] == 'True'].drop(columns=['is_valid'])
        continents = countries.continent.unique()
        return { i: continents[i] for i in range(len(continents))}

    def initialize_occupation_lookup(self, occupation):
        occupations_df = pd.read_csv(occupation)
        occupations = occupations_df['occupation'].values
        return { i:occupations[i] for i in range(len(occupations))}

    def parse_birth_interval(self, binary_string):
        time_period = np.where(binary_string == 1)[0]
        if len(time_period) == 0: return "in an unknown timeperiod"

        #Can only be born once
        assert len(time_period) == 1

        time_period = time_period[0]
        if time_period == 0: return "before " + str(self.age_containers[time_period])
        if time_period == len(self.age_containers): return "after " + str(self.age_containers[time_period-1])
        return "between " + str(self.age_containers[time_period-1]) + " and " + str(self.age_containers[time_period])

    def sentence_from_binary(self, binary_string, has_gender=False):
        
        binary_string = np.array(binary_string)
        birth_string, continent_string, occupation_string, gender_string = np.split(binary_string, [len(self.age_containers)+ 1, 
                                                                                                    len(self.age_containers)+ 1 + len(self.continent_lookup), 
                                                                                                    len(self.age_containers)+ 1 + len(self.continent_lookup) + len(self.occupation_lookup)])

        birth = self.parse_birth_interval(birth_string)
        if not np.where(continent_string==1)[0]: continent = 'an unknown place'
        else: continent = self.continent_lookup[np.where(continent_string == 1)[0][0]]
        if not np.where(occupation_string==1)[0]: occupation = 'an unknown occupation'
        else: occupation = self.occupation_lookup[np.where(occupation_string == 1)[0][0]]

        if gender_string:
            gender = 'She' if gender_string[0] == 1 else 'He'
            return f"{gender} was born {birth} in {continent} and is a {occupation}."
        return f"<mask> was born {birth} in {continent} and is a {occupation}."


