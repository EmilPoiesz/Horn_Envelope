import sympy

from . import binary_parser

class EquationParser:
    """
    Parses sympy equations to a human readable format.

    Args:
        binary_parser: BinaryParser 
            Contains all binary features we are solving for.
        V: list              
            All the sympy variables used.

    """
    
    def __init__(self, binary_parser: binary_parser.BinaryParser, V: list):
        
        variable_values = ['Born ' + binary_parser.parse_birth(i) for i in range(binary_parser.lengths['birth'])]
        variable_values.extend(binary_parser.features['continents'])
        variable_values.extend(binary_parser.features['occupations'])
        variable_values.extend(['She', 'He'])
        
        self.mapping = {f'{self.V[i]}': str(variable_values[i]).replace(' ', '_') for i in range(len(V))}


    def parse(self, equation):
        if type(equation) == sympy.Implies:
            antecedent, consequent = equation.args
            consequent = list(set(consequent.args).difference(set(antecedent.args)))
            equation = sympy.Implies(antecedent, sympy.And(*consequent))
        return sympy.pretty(equation.subs(self.mapping))
    