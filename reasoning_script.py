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

def get_hypothesis_space(lengths):
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

def create_sample(binary_parser:Binary_parser, unmasker, verbose=False):
    
    binary_sample = []
    for att in attributes:
        binary_sample = [*binary_sample, *get_attribute_vector(binary_parser.lengths[att], allow_zero=True)]
    sentence = binary_parser.sentence_from_binary(binary_sample)
    
    if verbose: print(sentence)

    # Binary = True forces the result to be either 'He' or 'She'. If False then give best guess.
    # 0 = female, 1 = male
    classification = get_prediction(unmasker, sentence, binary = True)
    
    # Generate the gender of the person and append to the vector ([female, male])
    # TODO: does this represent the real world? We are only looking to see what occupations are more skewed.
    # Maybe here we should also include a 'they' option? Or generate while considering historical data?
    gender_vec = get_attribute_vector(2, allow_zero=False)
    binary_sample = [*binary_sample, *gender_vec]
    
    # 1 if sampled gender and classification match       (correctly classified)
    # 0 if sampled gender and classification don't match (wrongly classified)
    label = gender_vec[classification]
    if verbose: print((binary_sample, classification, gender_vec, label))

    return (binary_sample,label)

def equivalence_oracle(hypothesis, unmasker, V, hypothesis_space, binary_parser):
    
    assert len(hypothesis) > 0

    # Reduce the hypothesis to a conjunction of clauses
    hypothesis = functools.reduce(lambda x,y: x & y, hypothesis)
    
    for i in range(hypothesis_space):
        (assignment, label) = create_sample(binary_parser, unmasker)
        if not (bool(label) == evaluate(hypothesis, assignment, V)): return (assignment, i+1)

    return True

def membership_oracle(assignment, unmasker, binary_parser:Binary_parser):
    vec = assignment[:-2]
    gender_vec = assignment[-2:]
    sentence = binary_parser.sentence_from_binary(vec)
    classification = get_prediction(unmasker, sentence, binary = True)
    label = gender_vec[classification]
    return bool(label)
    
def extract_horn_with_queries(language_model, V, iterations, binary_parser, background, hypothesis_space, verbose = 0):

    unmasking_model = pipeline('fill-mask', model=language_model)
    
    # Create lambda functions for asking the membership and equivalence oracles.
    ask_membership_oracle  = lambda assignment : membership_oracle(assignment, unmasking_model, binary_parser)
    #TODO: Should H and Q be given individually such that we can assess them seperately?
    #Checking H is fine, how to check Q?
    ask_equivalence_oracle = lambda hypothesis : equivalence_oracle(hypothesis, unmasking_model, V, hypothesis_space, binary_parser) 

    start = timeit.default_timer()
    terminated, metadata, H, Q = learn_horn_envelope(V, ask_membership_oracle, ask_equivalence_oracle, binary_parser, 
                                                     background=background, iterations=iterations, verbose=verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    return (H, Q, runtime, terminated, metadata)


def get_prediction(unmasking_model, sentence, binary=False):
    """
    Gets the prediction of the unmasking model. If binary is set to
    True then it returns 0 or 1 for only 'He/he' and 'She/she' pronouns
    else returns the most probable token.

    Args:
        unmasking_model: The language model
        sentence:        The masked sentence
        binary:          Return binary or string
    
    Returns:
        0 or 1 if binary is True, else string with most probable token.
    """

    sentence = sentence.replace('<mask>', unmasking_model.tokenizer.mask_token)
    predictions = unmasking_model(sentence)

    # TODO: This is a forceful way to ensure 'He' or 'She'. Can we think of something better?
    if binary:
        tokens = [pred['token_str'] for pred in predictions]
        for token in tokens:
            if token in ['She', 'she']: return 0
            if token in ['He', 'he']:   return 1
        #TODO: what if we get here?

    return predictions[0]['token_str'] #TODO: Not comfortable returning different types of data, should be refactored.

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
    binary_parser = Binary_parser('data/known_countries.csv', 'data/occupations.csv')
    attributes = ['birth', 'continent', 'occupation']

    # Define variables
    number_of_variables = sum(binary_parser.lengths.values())
    variable_string = ','.join(f'v{i}' for i in range(number_of_variables))
    V = list(sympy.symbols(variable_string))

    background = create_background(binary_parser.lengths, V)
    hypothesis_space = get_hypothesis_space(binary_parser.lengths)
    
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
    models = ['bert-base-cased']

    iterations = 5000
    r=0
    for language_model in models:
        #for eq in eq_amounts:
        (H, Q, runtime, terminated, average_samples) = extract_horn_with_queries(language_model, V, iterations, binary_parser, background, hypothesis_space, verbose=2)
        metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
        with open('data/rule_extraction/' + language_model + '_metadata_' + str(iterations) + "_" + str(r+1) + '.json', 'w') as outfile:
            json.dump(metadata, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + str(iterations) + "_" + str(r+1) + '.txt', 'wb') as f:
            pickle.dump(H, f)