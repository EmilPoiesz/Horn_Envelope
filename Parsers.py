import numpy as np
import sympy

from config import FEATURES

class BinaryParser:
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
        self.features = FEATURES
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

class EquationParser:
    """
    Parses sympy equations to a human readable format.

    Args:
        binary_parser: BinaryParser 
            Contains all binary features we are solving for.
        V: list              
            All the sympy variables used.

    """
    
    def __init__(self, binary_parser: BinaryParser, V: list):
        
        variable_values = ['Born ' + binary_parser.parse_birth(i) for i in range(binary_parser.lengths['birth'])]
        variable_values.extend(binary_parser.features['continents'])
        variable_values.extend(binary_parser.features['occupations'])
        variable_values.extend(['She', 'He'])
        
        self.mapping = {f'{V[i]}': str(variable_values[i]).replace(' ', '_') for i in range(len(V))}


    def parse(self, equation):
        if type(equation) == sympy.Implies:
            antecedent, consequent = equation.args
            consequent = list(set(consequent.args).difference(set(antecedent.args)))
            equation = sympy.Implies(antecedent, sympy.And(*consequent))
        return sympy.pretty(equation.subs(self.mapping))