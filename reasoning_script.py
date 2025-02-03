import random
import timeit
from Horn import *
from Binarizer import *
from transformers import pipeline
from helper_functions import *
import pickle
import json
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

    return int ( (1/EPSILON) * log( (Pow(2,total_clauses) / DELTA), 2)) #pow(2, total_clauses) gives total hypothesis space

def get_random_sample(length, allow_zero=True):

    vec = [0]*length
    if allow_zero:
        zero_vec_prob = 1/(length+1)
        if random.choices([True, False], weights=[zero_vec_prob, 1-zero_vec_prob])[0]: 
            return vec
    
    vec[random.randint(0, length-1)] = 1
    return vec

def create_single_sample(lm:str, binarizer:Binarizer, unmasker, verbose=False):
    
    vec = []
    for att in attributes:
        vec = [*vec, *get_random_sample(binarizer.lengths[att], allow_zero=True)]
    
    s = binarizer.sentence_from_binary(vec)
    if verbose: print(s)

    # Binary = True forces the result to be either 'He' or 'She'. If False then give best guess.
    # 0 = female, 1 = male
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    
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

def equivalence_oracle(hypothesis, lm, unmasker, V, hypothesis_space, binarizer):
    
    assert len(hypothesis) > 0
    hypothesis = from_set_to_theory(hypothesis)

    for i in range(hypothesis_space):
        (assignment, label) = create_single_sample(lm, binarizer, unmasker)
        if not (bool(label) == evaluate(hypothesis, assignment, V)): return (assignment, i+1)

    return True

def membership_oracle(assignment, lm, unmasker, binarizer:Binarizer):
    vec = assignment[:-2]
    gender_vec = assignment[-2:]
    s = binarizer.sentence_from_binary(vec)
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    label = gender_vec[classification]
    return bool(label)
    
def extract_horn_with_queries(lm, V, iterations, binarizer, background, hypothesis_space, verbose = 0):

    unmasker = pipeline('fill-mask', model=lm)
    
    # Create lambda functions for asking the membership and equivalence oracles.
    ask_membership_oracle  = lambda assignment : membership_oracle(assignment, lm, unmasker, binarizer)
    #TODO: Should H and Q be given individually such that we can assess them seperately?
    #Checking H is fine, how to check Q?
    ask_equivalence_oracle = lambda hypothesis : equivalence_oracle(hypothesis, lm, unmasker, V, hypothesis_space, binarizer) 

    start = timeit.default_timer()
    terminated, metadata, H, Q = learn_horn_envelope(V, ask_membership_oracle, ask_equivalence_oracle, binarizer, 
                                                     background=background, iterations=iterations, verbose=verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    return (H, Q, runtime, terminated, metadata)

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
    
    # The binarizer is used to convert the data into a binary format that can be used by the Horn algorithm.
    binarizer = Binarizer('data/known_countries.csv', 'data/occupations.csv')
    attributes = ['birth', 'continent', 'occupation']

    V = define_variables(sum(binarizer.lengths.values()))
    hypothesis_space = get_hypothesis_space(binarizer.lengths)
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
    models = ['roberta-base']
    eq_amounts = [50, 100, 150, 200]
    language_model = models[0]
    #seed = 123 #reproducability
    epsilon = 0.2
    delta = 0.1

    background = create_background(binarizer.lengths, V)

    #5000 as a placeholder for an uncapped run (it will terminate way before reaching this)
    iterations = 5000
    r=0
    for language_model in models:
        #for eq in eq_amounts:
        (H, Q,runtime,terminated, average_samples) = extract_horn_with_queries(language_model, V, iterations, binarizer, background, hypothesis_space, verbose=2)
        metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
        with open('data/rule_extraction/' + language_model + '_metadata_' + str(iterations) + "_" + str(r+1) + '.json', 'w') as outfile:
            json.dump(metadata, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + str(iterations) + "_" + str(r+1) + '.txt', 'wb') as f:
            pickle.dump(H, f)