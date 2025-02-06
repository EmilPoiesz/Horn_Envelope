import random
import timeit
from Horn import *
from Binary_parser import *
from transformers import pipeline
from helper_functions import *
import pickle
import json
from config import EPSILON, DELTA

def define_variables(number):
    """
    Creates a list of symbolic variables named 'v0', 'v1', ..., 'v(number-1)'.

    Parameters:
    number (int): The number of symbolic variables to generate.

    Returns:
    list: A list of symbolic variables.
    """
    return list(symbols("".join(['v'+str(i)+',' for i in range(number)])))

def from_set_to_hypothesis(set):
    """
    Converts a set of boolean clauses to a single boolean expression using logical AND.

    Args:
        set (iterable): An iterable of boolean clauses.

    Returns:
        Expr: The result of performing a logical AND operation on all elements in the set.
    """

    return functools.reduce(lambda x,y: x & y, set)

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
    return int ( (1/EPSILON) * log( (Pow(2,total_clauses) / DELTA), 2))

def get_random_sample(length, allow_zero=True):
    """
    Generate a random sample vector of a given length with binary values (0 or 1).

    Args:
        length (int): The length of the vector to be generated.
        allow_zero (bool): If True, the vector can be all zeros. If False, one element will be 1.

    Returns:
        list: A list of integers (0 or 1) of the specified length. If allow_zero is True, the list can be all zeros.
              If allow_zero is False, one element in the list will be 1.
    """

    vec = [0]*length
    if allow_zero:
        zero_vec_prob = 1/(length+1)
        if random.choices([True, False], weights=[zero_vec_prob, 1-zero_vec_prob])[0]: 
            return vec
    
    vec[random.randint(0, length-1)] = 1
    return vec

def create_single_sample(binary_parser:Binary_parser, unmasker, verbose=False):
    
    vec = []
    for att in attributes:
        vec = [*vec, *get_random_sample(binary_parser.lengths[att], allow_zero=True)]
    
    sentence = binary_parser.sentence_from_binary(vec)
    if verbose: print(sentence)

    # Binary = True forces the result to be either 'He' or 'She'. If False then give best guess.
    # 0 = female, 1 = male
    classification = get_prediction(unmasker, sentence, binary = True)
    
    # Generate the gender of the person and append to the vector ([female, male])
    # TODO: does this represent the real world? We are only looking to see what occupations are more skewed.
    # Maybe here we should also include a 'they' option? Or generate while considering historical data?
    gender_vec = get_random_sample(2, allow_zero=False)
    vec = [*vec, *gender_vec]
    
    # 1 if sampled gender and classification match       (correctly classified)
    # 0 if sampled gender and classification don't match (wrongly classified)
    label = gender_vec[classification]
    if verbose: print((vec, classification, gender_vec, label))

    return (vec,label)

def equivalence_oracle(hypothesis, unmasker, V, hypothesis_space, binary_parser):
    
    assert len(hypothesis) > 0
    hypothesis = from_set_to_hypothesis(hypothesis)

    for i in range(hypothesis_space):
        (assignment, label) = create_single_sample(binary_parser, unmasker)
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

    sentence = sentence.replace('<mask>', unmasking_model.tokenizer.mask_token)
    result = unmasking_model(sentence)

    # TODO: This is a forceful way to ensure 'He' or 'She'. Can we think of something better?
    if binary:
        if result[0]['token_str'] == 'She':
            return 0
        elif result[0]['token_str'] == 'He':
            return 1
        else:
            del result[0]
            print('Recursion')
            return get_prediction(result, binary = True)
    return result[0]['token_str']

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

    V = define_variables(sum(binary_parser.lengths.values()))
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