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



def get_eq_sample_size(lengths):
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
    #H = 1080 # number of possible examples

    return int ( (1/EPSILON) * log( (Pow(2,total_hypotheses) / DELTA), 2))

def get_random_sample(length, allow_zero = True, amount_of_true=1):
    vec = np.zeros(length, dtype=np.int8)
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
    
    # Create a sentence from the binary vector to use as input for the language model.
    s = binarizer.sentence_from_binary(vec)
    
    if verbose: print(s)

    # Ask the language model to predict the gender of the person in the sentence.
    # classification: 0 = female, 1 = male
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    
    # Generate the gender of the person and append to the vector ([female, male])
    # NB: do this after querying the language model to avoid biasing the sample
    # TODO: does this represent the real world? We are only looking to see what occupations are more skewed.
    # Maybe here we should also include a 'they' option? Or generate while considering historical data?
    gender_vec = get_random_sample(2, allow_zero=False)
    vec = [*vec, *gender_vec]
    
    # if the sampled gender is equal the classification (correctly classified) then we return 1 as 'is valid sentence' 
    # if sampled gender and classification don't match, the sample is not valid and we return 0 as a label
    label = get_label(classification, gender_vec)
    if verbose: print((vec,classification, gender_vec, label))

    return (vec,label)

def ask_equivalence_oracle(H, lm, unmasker, V, bad_nc, binarizer : Binarizer):
    h = true
    if len(H):
        h = set2theory(H)
    for i in range(get_eq_sample_size()):
        (a,l) = create_single_sample(lm, binarizer, unmasker)
        if l == 0 and evaluate(h,a,V) and a not in bad_nc:
            return (a, i+1)
        if l == 1 and not evaluate(h,a,V):
            return (a, i+1)
    return True

def ask_membership_oracle(assignment, lm, unmasker, binarizer : Binarizer):
    vec = assignment[:-2]
    gender_vec = assignment[-2:]
    s = binarizer.sentence_from_binary(vec)
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    label = get_label(classification, gender_vec)
    res  = ( True if label == 1
                else False)
    return res
    
def extract_horn_with_queries(lm, V, iterations, binarizer, background, verbose = 0):
    bad_pc = []
    bad_ne =[]
    unmasker = pipeline('fill-mask', model=lm)
    
    # Create lambda functions for asking the membership and equivalence oracles.
    ask_membership = lambda assignment : ask_membership_oracle(assignment, lm, unmasker, binarizer)
    ask_equivalence = lambda assignment : ask_equivalence_oracle(assignment, lm, unmasker, V, bad_ne, binarizer) 

    start = timeit.default_timer()
    terminated, metadata, h = learn(V, ask_membership, ask_equivalence, bad_ne, bad_pc, background = background, iterations=iterations, verbose = verbose)
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

    V = define_variables(binarizer.lengths.values())
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
    models = ['roberta-base']
    eq_amounts = [50, 100, 150, 200]
    language_model = models[0]
    #seed = 123 #reproducability
    epsilon = 0.2
    delta = 0.1

    background = create_background(binarizer.lengths, V)
    # background is hand-written, but could be automated as well
    # Background knowledge is a set of Horn clauses that are known to be true in the target theory.
    # In this case, it is known that only one of the variables in each group can be true at the same time.
    # This doesnt work because I have different number of continents and occupations.
    #background = {(~(V[0] & V[1])),
    #(~(V[0] & V[2])),
    #(~(V[0] & V[3])),
    #(~(V[0] & V[4])),
    #(~(V[1] & V[2])),
    #(~(V[1] & V[3])),
    #(~(V[1] & V[4])),
    #(~(V[2] & V[3])),
    #(~(V[2] & V[4])),
    #(~(V[3] & V[4])), # Only one of V0 - V4, can only be born in one time
    #(~(V[5] & V[6])),
    #(~(V[5] & V[7])),
    #(~(V[5] & V[8])),
    #(~(V[5] & V[9])),
    #(~(V[5] & V[10])),
    #(~(V[5] & V[11])),
    #(~(V[5] & V[12])),
    #(~(V[5] & V[13])),
    #(~(V[6] & V[7])),
    #(~(V[6] & V[8])),
    #(~(V[6] & V[9])),
    #(~(V[6] & V[10])),
    #(~(V[6] & V[11])),
    #(~(V[6] & V[12])),
    #(~(V[6] & V[13])),
    #(~(V[7] & V[8])),
    #(~(V[7] & V[9])),
    #(~(V[7] & V[10])),
    #(~(V[7] & V[11])),
    #(~(V[7] & V[12])),
    #(~(V[7] & V[13])),
    #(~(V[8] & V[9])),
    #(~(V[8] & V[10])),
    #(~(V[8] & V[11])),
    #(~(V[8] & V[12])),
    #(~(V[8] & V[13])),
    #(~(V[9] & V[10])),
    #(~(V[9] & V[11])),
    #(~(V[9] & V[12])),
    #(~(V[9] & V[13])),
    #(~(V[10] & V[11])),
    #(~(V[10] & V[12])),
    #(~(V[10] & V[13])),
    #(~(V[11] & V[12])),
    #(~(V[11] & V[13])),
    #(~(V[12] & V[13])), # Only one of V5 - V13, can only be born in one continent
    #(~(V[14] & V[15])),
    #(~(V[14] & V[16])),
    #(~(V[14] & V[17])),
    #(~(V[14] & V[18])),
    #(~(V[14] & V[19])),
    #(~(V[14] & V[20])),
    #(~(V[14] & V[21])),
    #(~(V[15] & V[16])),
    #(~(V[15] & V[17])),
    #(~(V[15] & V[18])),
    #(~(V[15] & V[19])),
    #(~(V[15] & V[20])),
    #(~(V[15] & V[21])),
    #(~(V[16] & V[17])),
    #(~(V[16] & V[18])),
    #(~(V[16] & V[19])),
    #(~(V[16] & V[20])),
    #(~(V[16] & V[21])),
    #(~(V[17] & V[18])),
    #(~(V[17] & V[19])),
    #(~(V[17] & V[20])),
    #(~(V[17] & V[21])),
    #(~(V[18] & V[19])),
    #(~(V[18] & V[20])),
    #(~(V[18] & V[21])),
    #(~(V[19] & V[20])),
    #(~(V[19] & V[21])),
    #(~(V[20] & V[21])), # Only one of V14 - V21, can only have one occupation
    #(~(V[22] & V[23])), # Only one of V22 and V23, can only be of one gender
    #}

    #5000 as a placeholder for an uncapped run (it will terminate way before reaching this)
    iterations = 5000
    r=0
    for language_model in models:
        #for eq in eq_amounts:
        (h,runtime,terminated, average_samples) = extract_horn_with_queries(language_model, V, iterations, binarizer, background, verbose=2)
        metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
        with open('data/rule_extraction/' + language_model + '_metadata_' + str(iterations) + "_" + str(r+1) + '.json', 'w') as outfile:
            json.dump(metadata, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + str(iterations) + "_" + str(r+1) + '.txt', 'wb') as f:
            pickle.dump(h, f)