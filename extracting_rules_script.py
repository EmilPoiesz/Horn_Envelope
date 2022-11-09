import numpy as np
import random
import timeit
from rule_extractor_original.Lenses_dataset.Horn import *
from binarize_features import *
from transformers import pipeline
from helper_functions import *
from scipy.special import comb
import pickle

country_file = 'data/country_list_continents.csv'
occ_file = 'data/occupations_subset.csv'
#occ_file = 'data/occupations_subset.csv'
binarizer = Binarizer(country_file, 5, occ_file)

# add one dimension for the gender variable (last variable in the vector)
dim = sum(binarizer.lengths.values()) + 1
V = define_variables(dim)
models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
#language_model = models[3]
#seed = 123 #reproducability
epsilon = 0.3
delta = 0.1

def get_eq_sample_size(n_variables=dim, epsilon=epsilon, delta=delta):
    return int((4/epsilon) * (log( log( Pow(2,comb(n_variables,n_variables/2)) ) )/delta) + 1)

def get_random_sample(length, amount_of_true=1):
    vec = np.zeros(length, dtype=np.int8)
    idx = random.sample(range(length), k=amount_of_true)
    for i in idx:
        vec[i] = 1
    return list(vec)

#len(V) = num_variables
attributes = ['birth', 'continent', 'occupation']
#new tactic: 
#num_variables inherited by binarizer that has to be initialized
def create_classified_sample_from_LM(lm : str, sample_size : int, binarizer : Binarizer, unmasker, verbose = False):
    dataset = []
    # create all the samples randomly by specific sampling strategy and classify them immediatley 
    # -> there is no time savings in using a batch of samples at once and therefore predictions can be done immediately
    for i in range(sample_size):
        vec = []
        for att in attributes:
            # get the appropriate vector for each attribute and tie them together in the end
            vec = [*vec, *get_random_sample(binarizer.lengths[att])]
        s = binarizer.sentence_from_binary(vec)
        if verbose:
            print(s)
        classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
        # get random gender as a fourth attribute (in the end of the vector)
        gender = random.randint(0,1)
        vec.append(gender)
        # if the sampled gender is equal the classification (correctly classified) then we return 1 as 'is valid sentence' 
        # if sampled gender and classification don't match, the sample is not valid and we return 0 as a label
        label = 1 if gender == classification else 0
        if verbose:
            print((vec,classification, gender, label))
        dataset.append((vec,label))
    return dataset

def custom_EQ(H, lm, unmasker, V, bad_nc, binarizer : Binarizer):
    sample = create_classified_sample_from_LM(lm, get_eq_sample_size(), binarizer, unmasker)
    h = true
    if len(H):
        h = set2theory(H)
    for (a,l) in sample:
        if l == 0 and evaluate(h,a,V) and a not in bad_nc:
            return a
        if l == 1 and not evaluate(h,a,V):
            sample.remove((a,l))
            return a
    return True

def custom_MQ(assignment, lm, unmasker, binarizer : Binarizer):
    vec = assignment.copy()
    gender = vec.pop()
    s = binarizer.sentence_from_binary(vec)
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    res  = ( True if classification == gender
                else False)
    return res

def extract_horn_with_queries_1(lm, V, iterations, binarizer, background, verbose = 0):
    bad_pc = []
    bad_ne =[]
    unmasker = pipeline('fill-mask', model=lm)
    mq = lambda a : custom_MQ(a, lm, unmasker, binarizer)
    eq = lambda a : custom_EQ(a, lm, unmasker, V, bad_ne, binarizer)

    start = timeit.default_timer()
    h = learn(V, mq, eq, bad_ne, bad_pc, background = background, iterations=iterations, verbose = verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    runtime_per_iteration = runtime / iterations
    return (h,runtime,runtime_per_iteration)

background = {(~(V[0] & V[1])),
(~(V[0] & V[2])),
(~(V[0] & V[3])),
(~(V[0] & V[4])),
(~(V[1] & V[2])),
(~(V[1] & V[3])),
(~(V[1] & V[4])),
(~(V[2] & V[3])),
(~(V[2] & V[4])),
(~(V[3] & V[4])),
(~(V[5] & V[6])),
(~(V[5] & V[7])),
(~(V[5] & V[8])),
(~(V[5] & V[9])),
(~(V[5] & V[10])),
(~(V[5] & V[11])),
(~(V[5] & V[12])),
(~(V[5] & V[13])),
(~(V[5] & V[14])),
(~(V[5] & V[15])),
(~(V[5] & V[16])),
(~(V[6] & V[7])),
(~(V[6] & V[8])),
(~(V[6] & V[9])),
(~(V[6] & V[10])),
(~(V[6] & V[11])),
(~(V[6] & V[12])),
(~(V[6] & V[13])),
(~(V[6] & V[14])),
(~(V[6] & V[15])),
(~(V[6] & V[16])),
(~(V[7] & V[8])),
(~(V[7] & V[9])),
(~(V[7] & V[10])),
(~(V[7] & V[11])),
(~(V[7] & V[12])),
(~(V[7] & V[13])),
(~(V[7] & V[14])),
(~(V[7] & V[15])),
(~(V[7] & V[16])),
(~(V[8] & V[9])),
(~(V[8] & V[10])),
(~(V[8] & V[11])),
(~(V[8] & V[12])),
(~(V[8] & V[13])),
(~(V[8] & V[14])),
(~(V[8] & V[15])),
(~(V[8] & V[16])),
(~(V[9] & V[10])),
(~(V[9] & V[11])),
(~(V[9] & V[12])),
(~(V[9] & V[13])),
(~(V[9] & V[14])),
(~(V[9] & V[15])),
(~(V[9] & V[16])),
(~(V[10] & V[11])),
(~(V[10] & V[12])),
(~(V[10] & V[13])),
(~(V[10] & V[14])),
(~(V[10] & V[15])),
(~(V[10] & V[16])),
(~(V[11] & V[12])),
(~(V[11] & V[13])),
(~(V[11] & V[14])),
(~(V[11] & V[15])),
(~(V[11] & V[16])),
(~(V[12] & V[13])),
(~(V[12] & V[14])),
(~(V[12] & V[15])),
(~(V[12] & V[16])),
(~(V[13] & V[14])),
(~(V[13] & V[15])),
(~(V[13] & V[16])),
(~(V[14] & V[15])),
(~(V[14] & V[16])),
(~(V[15] & V[16])),
(~(V[17] & V[18])),
(~(V[17] & V[19])),
(~(V[17] & V[20])),
(~(V[17] & V[21])),
(~(V[17] & V[22])),
(~(V[17] & V[23])),
(~(V[17] & V[24])),
(~(V[17] & V[25])),
(~(V[17] & V[26])),
(~(V[18] & V[19])),
(~(V[18] & V[20])),
(~(V[18] & V[21])),
(~(V[18] & V[22])),
(~(V[18] & V[23])),
(~(V[18] & V[24])),
(~(V[18] & V[25])),
(~(V[18] & V[26])),
(~(V[19] & V[20])),
(~(V[19] & V[21])),
(~(V[19] & V[22])),
(~(V[19] & V[23])),
(~(V[19] & V[24])),
(~(V[19] & V[25])),
(~(V[19] & V[26])),
(~(V[20] & V[21])),
(~(V[20] & V[22])),
(~(V[20] & V[23])),
(~(V[20] & V[24])),
(~(V[20] & V[25])),
(~(V[20] & V[26])),
(~(V[21] & V[22])),
(~(V[21] & V[23])),
(~(V[21] & V[24])),
(~(V[21] & V[25])),
(~(V[21] & V[26])),
(~(V[22] & V[23])),
(~(V[22] & V[24])),
(~(V[22] & V[25])),
(~(V[22] & V[26])),
(~(V[23] & V[24])),
(~(V[23] & V[25])),
(~(V[23] & V[26])),
(~(V[24] & V[25])),
(~(V[24] & V[26])),
(~(V[25] & V[26]))
}

for language_model in models:
    (h,runtime,runtime_per_iteration) = extract_horn_with_queries_1(language_model, V, 50, binarizer, background, verbose=0)
    print(runtime,runtime_per_iteration)
    with open('data/' + language_model + '_rules.txt', 'wb') as f:
        pickle.dump(h, f)