import random
import timeit
import pickle
import json
import functools
import Binary_parser
import sympy

from Horn import evaluate, learn_horn_envelope
from transformers import pipeline
from config import EPSILON, DELTA

def get_PAC_hypothesis_space(lengths):
    """
    Calculate the equivalent sample size needed for PAC learning.
    This function computes the number of samples required to learn with 
    probability (1 - delta) and accuracy epsilon using the PAC (Probably 
    Approximately Correct) learning framework. The formula used here 
    combines the logarithms of the original PAC learning formula into a 
    single logarithm, which is valid when the number of hypotheses is finite.
    
    Args:
        lengths (dict): A dictionary where the keys are feature names and 
                        the values are the number of possible values for 
                        each feature.
    Returns:
        int: The number of samples needed to achieve the desired learning 
             accuracy and confidence.
    """
    
    total_clauses = 1
    for length in lengths.values():
        total_clauses *= length
    return int ( (1/EPSILON) * sympy.log( (sympy.Pow(2,total_clauses) / DELTA), 2))

def get_attribute_vector(length, allow_zero=True):
    """
    Generate a one hot encoded sample vector of a given length. If allow_zero=True then the zero
    vector is possible.

    Args:
        length (int): The length of the vector to be generated.
        allow_zero (bool): If True, the vector can be all zeros. If False, one element will be 1.

    Returns:
        list: A list of integers (0 or 1) of the specified length. If allow_zero is True, the list can be all zeros.
              If allow_zero is False, one element in the list will be 1.
    """

    if allow_zero:
        zero_vec_prob = 1/(length+1)
        if random.choices([True, False], weights=[zero_vec_prob, 1-zero_vec_prob])[0]: 
            return [0]*length
    
    attribute_vector = [0]*length
    attribute_vector[random.randint(0, length-1)] = 1
    return attribute_vector

def create_sample(binary_parser:Binary_parser, unmasking_model, verbose=False):
    
    # Generate sample vector
    sample_vector = []
    for att in attributes:
        sample_vector = [*sample_vector, *get_attribute_vector(binary_parser.lengths[att], allow_zero=True)]
    gender_vector = get_attribute_vector(2, allow_zero=False)
    
    assert 1 in gender_vector
    
    # Query model and get label (positive/negative counterexample?)
    sentence = binary_parser.sentence_from_binary(sample_vector)
    prediction = get_prediction(unmasking_model, sentence)
    label = (prediction in ['She', 'she'] and gender_vector[0] == 1) or (prediction in ['He', 'he'] and gender_vector[1] == 1)
    
    # Combine the sample
    sample_vector = [*sample_vector, *gender_vector]
    if verbose: print(sentence); print((sample_vector, prediction, gender_vector, label))
    
    return (sample_vector,label)

def equivalence_oracle(hypothesis, unmasking_model, V, hypothesis_space, binary_parser):
    
    assert len(hypothesis) > 0

    # Reduce the hypothesis to a conjunction of clauses
    hypothesis = functools.reduce(lambda x,y: x & y, hypothesis)

    for i in range(hypothesis_space):
        (assignment, label) = create_sample(binary_parser, unmasking_model)
        if not (bool(label) == evaluate(hypothesis, assignment, V)): 
            return (assignment, i+1)

    # No counterexample was found, the hypothesis is true.
    return True

def membership_oracle(assignment, unmasking_model, binary_parser:Binary_parser):
    sample_vector = assignment[:-2]
    gender_vector = assignment[-2:]

    # TODO: When the intersection is of two counterexamples with different gender,
    # then the assignment is without gender and is interpreted as always wrong.
    #
    # Example: 
    # neg_example:    "He was born in timeperiodA in Europe and is a Blacksmith".
    # counterexample: "She was born in timeperiodB in Europe and is a Blacksmith".
    # 
    # The intersection of counterexample and negative example is just:
    # "<mask> was born in unknown timeperiod in Europe and is a Blacksmith".
    #
    # The true gender value we check against is unkown since the intersection
    # was of two examples with different gender. This always evaluate as no since
    # our check is if the model correctly predicts "He" or "She".
    #
    # The result of this is that the negative example "He was born in timeperiodA in Europe and is a Blacksmith"
    # gets replaces with "<?> was born in unknown timeperiod in Europe and is a Blacksmith"
    # TODO: I don't think the binary_parser has an option to parse unknown gender.

    # TODO: think about this
    # If gender is not present then either gender is possible.
    if 1 not in gender_vector: return True

    sentence = binary_parser.sentence_from_binary(sample_vector)
    prediction = get_prediction(unmasking_model, sentence)
    label = (prediction in ['She', 'she'] and gender_vector[0] == 1) or (prediction in ['He', 'he'] and gender_vector[1] == 1)
    
    return label
    
def extract_horn_with_queries(language_model, V, iterations, binary_parser, background, hypothesis_space, verbose = 0):

    unmasking_model = pipeline('fill-mask', model=language_model)
    
    # Create lambda functions for asking the membership and equivalence oracles.
    ask_membership_oracle  = lambda assignment : membership_oracle(assignment, unmasking_model, binary_parser)
    ask_equivalence_oracle = lambda hypothesis : equivalence_oracle(hypothesis, unmasking_model, V, hypothesis_space, binary_parser) 

    start = timeit.default_timer()
    terminated, metadata, H, Q = learn_horn_envelope(V, ask_membership_oracle, ask_equivalence_oracle, binary_parser, 
                                                     background=background, iterations=iterations, verbose=verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    return (H, Q, runtime, terminated, metadata)


def get_prediction(unmasking_model, sentence, gender_preferred=True):
    """
    Gets the prediction of the unmasking model. If binary is set to
    True then it returns 0 or 1 for only 'He/he' and 'She/she' pronouns
    else returns the most probable token.

    Args:
        unmasking_model:  The language model
        sentence:         The masked sentence
        gender_preferred: Returns gender pronoun first
    
    Returns:
        A string with the best prediction of the model.
    """

    sentence = sentence.replace('<mask>', unmasking_model.tokenizer.mask_token)
    predictions = unmasking_model(sentence)

    if gender_preferred:
        tokens = [pred['token_str'] for pred in predictions]
        for token in tokens:
            if token in ['She', 'she', 'He', 'he']: return token

    # Return best guess
    return predictions[0]['token_str']

def make_disjoint(V):
    """
    Create a disjoint set of variables from the list V.
    
    Parameters:
    V (list): A list of variables to be made disjoint.
    
    Returns:
    set: A set of disjoint variables.
    """
    disjoint_set = set()
    for i in range(len(V)-1):
        for j in range(i+1, len(V)):
            disjoint_set.add(~(V[i] & V[j]))
    return disjoint_set

def create_background(lengths, V):
    """
    Create background knowledge as a set of disjoint variables for birth, continent, occupation, and gender.
    The variables need to be disjoint since the groups are one-hot encoded.

    Parameters:
    lengths (dict): A dictionary containing the lengths of each attribute.
    V (list): A list of variables.

    Returns:
    set: A set of disjoint variables representing the background knowledge.
    """
    birth_end      = lengths['birth']
    continent_end  = birth_end + lengths['continent']
    occupation_end = continent_end + lengths['occupation']
    gender_end     = occupation_end + lengths['gender']

    birth      = make_disjoint(V[:birth_end])
    continent  = make_disjoint(V[birth_end:continent_end])
    occupation = make_disjoint(V[continent_end:occupation_end])
    gender     = make_disjoint(V[occupation_end:gender_end])
    return birth | continent | occupation | gender

if __name__ == '__main__':
    
    # The binary parser is used to convert the data into a binary format that can be used by the Horn algorithm.
    binary_parser = Binary_parser.Binary_parser('data/known_countries.csv', 'data/occupations.csv')
    attributes = ['birth', 'continent', 'occupation']

    # Define variables
    number_of_variables = sum(binary_parser.lengths.values())
    variable_string = ','.join(f'v{i}' for i in range(number_of_variables))
    V = list(sympy.symbols(variable_string))

    background = create_background(binary_parser.lengths, V)
    pac_hypothesis_space = get_PAC_hypothesis_space(binary_parser.lengths)
    
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
    models = ['roberta-base']

    iterations = 5000
    r=0
    for language_model in models:
        #for eq in eq_amounts:
        (H, Q, runtime, terminated, average_samples) = extract_horn_with_queries(language_model, V, iterations, binary_parser, background, pac_hypothesis_space, verbose=2)
        metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
        with open('data/rule_extraction/' + language_model + '_metadata_' + str(iterations) + "_" + str(r+1) + '.json', 'w') as outfile:
            json.dump(metadata, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + str(iterations) + "_" + str(r+1) + '.txt', 'wb') as f:
            pickle.dump(H, f)