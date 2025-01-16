import numpy as np
import random
import timeit
from Horn import *
from Binarizer import *
from transformers import pipeline
from helper_functions import *
from scipy.special import comb
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
    
    total_hypotheses = 1
    for length in lengths.values():
        total_hypotheses *= length

    # Why is this the number of possible examples?
    # H = 1080 # number of possible examples

    return int ( (1/EPSILON) * log( (Pow(2,total_hypotheses) / DELTA), 2)) #pow(x,2) gives total hypothesis space, total hypothesis= total clauses?

def get_random_sample(length, allow_zero = True, amount_of_true=1):
    vec = np.zeros(length, dtype=np.int8)

    # TODO: is it needed to allow for all zeroes? The probability of equal possibility is higher
    # when the length of the attribute is smaller. It confused me why a "all equal" vector is needed.

    # allow for all zeroes: one extra sample length and if its out of index range, use all zeroes vector (equal possibility)
    if allow_zero:
        idx = random.sample(range(length + 1), k=amount_of_true)
    else:
        idx = random.sample(range(length), k=amount_of_true)
    for i in idx:
        if i < length:
            vec[i] = 1
    return list(vec)

def get_label(classification, gender):
    """
        Does handling no 1 in vector work as a diverse attribute (neither he or she) to include they?
        -> never gets predicted so what rules result from that? Influences other attributes?
    """
    if (gender[0] == 1 and classification == 0) or (gender[1] == 1 and classification == 1):
        return 1
    else:
        return 0

def create_single_sample(lm : str, binarizer : Binarizer, unmasker, verbose = False):
    vec = []

    # The assignment is stored as a vector of binary values. The first 4 values corresond to birthyear ('before X', 'between X and Y', 'after Y'), 
    # the next 7 values correspond to the continent (one-hot-encoding) and the last 10 values correspond to the occupation (one-hot-encoding).
    # Pick at random for each attribute to generate a sample.
    for att in attributes:
        # get the appropriate vector for each attribute and tie them together in the end
        vec = [*vec, *get_random_sample(binarizer.lengths[att], allow_zero=True)]
    
    s = binarizer.sentence_from_binary(vec)
    if verbose: print(s)

    # Ask the language model to predict the gender of the person in the sentence.
    # classification: 0 = female, 1 = male

    # Binary = True forces the result to be either 'He' or 'She'. If False then give best guess.
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    
    # Generate the gender of the person and append to the vector ([female, male])
    # TODO: does this represent the real world? We are only looking to see what occupations are more skewed.
    # Maybe here we should also include a 'they' option? Or generate while considering historical data?
    gender_vec = get_random_sample(2, allow_zero=False)
    vec = [*vec, *gender_vec]
    
    # if the sampled gender is equal the classification (correctly classified) then we return 1 
    # if sampled gender and classification don't match, then we return 0 
    label = get_label(classification, gender_vec)
    if verbose: print((vec, classification, gender_vec, label))

    return (vec,label)

def ask_equivalence_oracle(H, lm, unmasker, V, bad_nc, hypothesis_space, binarizer:Binarizer):
    h = true
    if len(H):
        # Create a long AND formula from all the clauses in the hypothesis.
        h = from_set_to_theory(H)
    
    # Looking through the number of examples needed according to the PAC learning framework.
    # When do we look at the next example?
    # - When the prediction of the sample is False and the sample breaks the current hypothesis.
    # - When the prediction of the sample is False and the assignment is in the bad_nc list.
    # - When the prediction of the sample is True and the sample follows the current hypothesis.

    # Small Example:
    # The hypothesis is the background knowledge, which is a set of disjoint variables.
    # H = !(a&b) & !(a&c) & !(b&c) & !(d&e)
    #
    # All predictions follow the hypothesis, so we need to find a counterexample where the prediction is False.
    # assignment = [1,0,0,  0,1] -> label = 0
    #
    # Why do we return i+1 here? -Keeping track of the number of examples needed to find a counterexample.
    for i in range(hypothesis_space):
        (assignment,label) = create_single_sample(lm, binarizer, unmasker) # This create single sample will generate random samples. Is this why we need to keep track of the number of examples? For PAC?
        # If the assignment is false but the hypothesis says its true.
        if label == 0 and evaluate(h,assignment,V) and assignment not in bad_nc: # bad_nc is a list of bad negative counterexamples?
            return (assignment, i+1)
        # If the assignment is true but the hypothesis says its false.
        if label == 1 and not evaluate(h,assignment,V):
            return (assignment, i+1)
    return True

def ask_membership_oracle(assignment, lm, unmasker, binarizer:Binarizer):
    vec = assignment[:-2]
    gender_vec = assignment[-2:]
    s = binarizer.sentence_from_binary(vec)
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    label = get_label(classification, gender_vec)
    res  = ( True if label == 1
                else False)
    return res
    
def extract_horn_with_queries(lm, V, iterations, binarizer, background, hypothesis_space, verbose = 0):
    bad_pc = []
    bad_ne =[]
    unmasker = pipeline('fill-mask', model=lm)
    
    # Create lambda functions for asking the membership and equivalence oracles.
    ask_membership  = lambda assignment : ask_membership_oracle(assignment, lm, unmasker, binarizer)
    ask_equivalence = lambda assignment : ask_equivalence_oracle(assignment, lm, unmasker, V, bad_ne, hypothesis_space, binarizer) 

    start = timeit.default_timer()
    terminated, metadata, h = learn(V, ask_membership, ask_equivalence, bad_ne, bad_pc, binarizer, 
                                    background = background, iterations=iterations, verbose = verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    return (h,runtime, terminated, metadata)

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
        (h,runtime,terminated, average_samples) = extract_horn_with_queries(language_model, V, iterations, binarizer, background, hypothesis_space, verbose=2)
        metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
        with open('data/rule_extraction/' + language_model + '_metadata_' + str(iterations) + "_" + str(r+1) + '.json', 'w') as outfile:
            json.dump(metadata, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + str(iterations) + "_" + str(r+1) + '.txt', 'wb') as f:
            pickle.dump(h, f)