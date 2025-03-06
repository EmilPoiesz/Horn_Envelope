import re
import sympy

from . import binary_parser

class EquationParser:
    
    def __init__(self, binary_parser: binary_parser.Binary_parser, V: list):
        self.binary_parser = binary_parser
        self.V = V
        variable_values = ['Born ' + self.binary_parser.parse_birth(i) for i in range(self.binary_parser.lengths['birth'])]
        variable_values.extend(self.binary_parser.features['continents'])
        variable_values.extend(self.binary_parser.features['occupations'])
        variable_values.extend(['She', 'He'])
        self.mapping = {f'{self.V[i]}': str(variable_values[i]).replace(' ', '_') for i in range(len(V))}


    def parse(self, equation):
        if type(equation) == sympy.Implies:
            antecedent, consequent = equation.args
            consequent = list(set(consequent.args).difference(set(antecedent.args)))
            equation = sympy.Implies(antecedent, sympy.And(*consequent))
        return sympy.pretty(equation.subs(self.mapping))
    
if __name__ == '__main__':

    b = binary_parser.Binary_parser()
    variable_string = ','.join(f'v{i}' for i in range(b.total_length))
    V = list(sympy.symbols(variable_string))

    
    e = EquationParser(b, V)
    eq = sympy.Implies(sympy.And(V[0], V[1], V[3]), sympy.And(V[0], V[2]))
    print(sympy.pretty(eq))

    antecedent, consequent = eq.args
    consequent = list(set(consequent.args).difference(set(antecedent.args)))
    print(sympy.pretty(sympy.Implies(antecedent, sympy.And(*consequent))))