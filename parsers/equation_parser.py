import re
import sympy

from . import binary_parser

class EquationParser:
    
    def __init__(self, binary_parser: binary_parser.Binary_parser, V: list):
        self.binary_parser = binary_parser
        self.V = V
        variable_values = [self.binary_parser.parse_birth(i) for i in range(self.binary_parser.lengths['birth'])]
        variable_values.extend(self.binary_parser.features['continents'])
        variable_values.extend(self.binary_parser.features['occupations'])
        variable_values.extend(['She', 'He'])
        self.mapping = {f'{self.V[i]}': str(variable_values[i]).replace(' ', '_') for i in range(len(V))}


    def parse(self, equation):
        return sympy.pretty(equation.subs(self.mapping))
    
if __name__ == '__main__':

    b = binary_parser.Binary_parser()
    variable_string = ','.join(f'v{i}' for i in range(b.total_length))
    V = list(sympy.symbols(variable_string))

    
    e = EquationParser(b, V)
    matches = e.parse()
    print(type(matches))
    print(matches)