import numpy as np
import random
import timeit
from rule_extractor_original.Lenses_dataset.Horn import *
from binarize_features import *
from transformers import pipeline
from helper_functions import *
from scipy.special import comb
import pickle
import json

country_file = 'data/country_list_continents_new.csv'
occ_file = 'data/occupations_subset.csv'
#occ_file = 'data/occupations_subset.csv'
binarizer = Binarizer(country_file, 5, occ_file)

# add 2 dimensions for the gender variables (last variables in the vector)
dim = sum(binarizer.lengths.values()) + 2
V = define_variables(dim)
models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
language_model = models[0]
#seed = 123 #reproducability
epsilon = 0.3
delta = 0.1

def get_eq_sample_size_old(n_variables=dim, epsilon=epsilon, delta=delta):
    return int((4/epsilon) * (log( log( Pow(2,comb(n_variables,n_variables/2)) ) )/delta) + 1)


def get_eq_sample_size(n_variables=dim, epsilon=epsilon, delta=delta):
    samplesize = int( (1/epsilon) * log( (Pow(2, Pow(n_variables, 2.1))) / delta , 2) )
    print(samplesize)
    return samplesize

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
    if (gender[0] == 1 and classification == 0) or (gender[1] == 1 and classification == 1):
        return 1
    else:
        return 0

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
            vec = [*vec, *get_random_sample(binarizer.lengths[att], allow_zero=True)]
        s = binarizer.sentence_from_binary(vec)
        if verbose:
            print(s)
        # classification: 0 = female, 1 = male
        classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
        # get random gender as a fourth attribute (in the end of the vector) as a one-hot-encoding with two dimensions: [female, male]
        gender_vec = get_random_sample(2, allow_zero=False)
        vec = [*vec, *gender_vec]
        # if the sampled gender is equal the classification (correctly classified) then we return 1 as 'is valid sentence' 
        # if sampled gender and classification don't match, the sample is not valid and we return 0 as a label
        label = get_label(classification, gender_vec)
        if verbose:
            print((vec,classification, gender_vec, label))
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
    vec = assignment[:-2]
    gender_vec = assignment[-2:]
    s = binarizer.sentence_from_binary(vec)
    classification = get_prediction(lm_inference(unmasker, s, model=lm), binary = True)
    label = get_label(classification, gender_vec)
    res  = ( True if label == 1
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
(~(V[6] & V[7])),
(~(V[6] & V[8])),
(~(V[6] & V[9])),
(~(V[6] & V[10])),
(~(V[6] & V[11])),
(~(V[6] & V[12])),
(~(V[6] & V[13])),
(~(V[7] & V[8])),
(~(V[7] & V[9])),
(~(V[7] & V[10])),
(~(V[7] & V[11])),
(~(V[7] & V[12])),
(~(V[7] & V[13])),
(~(V[8] & V[9])),
(~(V[8] & V[10])),
(~(V[8] & V[11])),
(~(V[8] & V[12])),
(~(V[8] & V[13])),
(~(V[9] & V[10])),
(~(V[9] & V[11])),
(~(V[9] & V[12])),
(~(V[9] & V[13])),
(~(V[10] & V[11])),
(~(V[10] & V[12])),
(~(V[10] & V[13])),
(~(V[11] & V[12])),
(~(V[11] & V[13])),
(~(V[12] & V[13])),
(~(V[14] & V[15])),
(~(V[14] & V[16])),
(~(V[14] & V[17])),
(~(V[14] & V[18])),
(~(V[14] & V[19])),
(~(V[14] & V[20])),
(~(V[14] & V[21])),
(~(V[14] & V[22])),
(~(V[14] & V[23])),
(~(V[15] & V[16])),
(~(V[15] & V[17])),
(~(V[15] & V[18])),
(~(V[15] & V[19])),
(~(V[15] & V[20])),
(~(V[15] & V[21])),
(~(V[15] & V[22])),
(~(V[15] & V[23])),
(~(V[16] & V[17])),
(~(V[16] & V[18])),
(~(V[16] & V[19])),
(~(V[16] & V[20])),
(~(V[16] & V[21])),
(~(V[16] & V[22])),
(~(V[16] & V[23])),
(~(V[17] & V[18])),
(~(V[17] & V[19])),
(~(V[17] & V[20])),
(~(V[17] & V[21])),
(~(V[17] & V[22])),
(~(V[17] & V[23])),
(~(V[18] & V[19])),
(~(V[18] & V[20])),
(~(V[18] & V[21])),
(~(V[18] & V[22])),
(~(V[18] & V[23])),
(~(V[19] & V[20])),
(~(V[19] & V[21])),
(~(V[19] & V[22])),
(~(V[19] & V[23])),
(~(V[20] & V[21])),
(~(V[20] & V[22])),
(~(V[20] & V[23])),
(~(V[21] & V[22])),
(~(V[21] & V[23])),
(~(V[22] & V[23])),
(~(V[24] & V[25])),
}

for r in range(10):
    for language_model in models:
        (h,runtime,runtime_per_iteration) = extract_horn_with_queries_1(language_model, V, 5, binarizer, background, verbose=0)
        runtimes = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'runtime per iteration' : runtime_per_iteration}}
        with open('data/rule_extraction/' + language_model + '_runtime_' + r+1 + '.txt', 'w') as outfile:
            json.dump(runtimes, outfile)
        with open('data/rule_extraction/' + language_model + '_rules_' + r+1 + '.txt', 'wb') as f:
            pickle.dump(h, f)