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
binarizer = Binarizer(country_file, 5, occ_file)

# add 2 dimensions for the gender variables (last variables in the vector)
dim = sum(binarizer.lengths.values()) + 2
V = define_variables(dim)
models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
eq_amounts = [50, 100, 150, 200]
language_model = models[0]
#seed = 123 #reproducability
epsilon = 0.2
delta = 0.1

def get_eq_sample_size_old(n_variables=dim, epsilon=epsilon, delta=delta):
    return int((4/epsilon) * (log( log( Pow(2,comb(n_variables,n_variables/2)) ) )/delta) + 1)


def get_eq_sample_size(n_variables=dim, epsilon=epsilon, delta=delta):
    samplesize = int( (1/epsilon) * log( (Pow(2, Pow(n_variables, 2.1))) / delta , 2) )
    return samplesize

def get_eq_sample_size(epsilon=epsilon, delta=delta):
    H = 1080 # number of possible examples
    return int ( (1/epsilon) * log( (Pow(2,H) / delta), 2))

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

def create_single_sample(lm : str, binarizer : Binarizer, unmasker, verbose = False):
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
    return (vec,label)

def custom_EQ_old(H, lm, unmasker, V, bad_nc, binarizer : Binarizer):
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

def custom_EQ(H, lm, unmasker, V, bad_nc, binarizer : Binarizer):
    h = true
    if len(H):
        h = set2theory(H)
    for i in range(get_eq_sample_size()):
        (a,l) = create_single_sample(lm, binarizer, unmasker)
        if l == 0 and evaluate(h,a,V) and a not in bad_nc:
            #print("Sample number", i+1)
            return (a, i+1)
        if l == 1 and not evaluate(h,a,V):
            #print("Sample number ", i+1)
            return (a, i+1)
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
    terminated, metadata, h = learn(V, mq, eq, bad_ne, bad_pc, background = background, iterations=iterations, verbose = verbose)
    stop = timeit.default_timer()
    runtime = stop-start

    return (h,runtime, terminated, metadata)

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
(~(V[15] & V[16])),
(~(V[15] & V[17])),
(~(V[15] & V[18])),
(~(V[15] & V[19])),
(~(V[15] & V[20])),
(~(V[15] & V[21])),
(~(V[16] & V[17])),
(~(V[16] & V[18])),
(~(V[16] & V[19])),
(~(V[16] & V[20])),
(~(V[16] & V[21])),
(~(V[17] & V[18])),
(~(V[17] & V[19])),
(~(V[17] & V[20])),
(~(V[17] & V[21])),
(~(V[18] & V[19])),
(~(V[18] & V[20])),
(~(V[18] & V[21])),
(~(V[19] & V[20])),
(~(V[19] & V[21])),
(~(V[20] & V[21])),
(~(V[22] & V[23])),
}


eq = 5000
r=0
for language_model in models:
    (h,runtime,terminated, average_samples) = extract_horn_with_queries_1(language_model, V, eq, binarizer, background, verbose=0)
    metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : {'runtime' : runtime, 'average_sample' : average_samples, "terminated" : terminated}}
    with open('data/rule_extraction_final/' + language_model + '_metadata_' + str(eq) + "_" + str(r+1) + '.json', 'w') as outfile:
        json.dump(metadata, outfile)
    with open('data/rule_extraction_final/' + language_model + '_rules_' + str(eq) + "_" + str(r+1) + '.txt', 'wb') as f:
        pickle.dump(h, f)
"""
r=0
(h,runtime,terminated, data) = extract_horn_with_queries_1('bert-base-cased', V, 200, binarizer, background, verbose=1)
metadata = {'head' : {'model' : language_model, 'experiment' : r+1},'data' : data, 'terminated' : terminated}
with open('data/rule_extraction_final/' + language_model + '_metadata_' + str(200) + "_" + str(r+1) + '.txt', 'w') as outfile:
    json.dump(metadata, outfile)
with open('data/rule_extraction_final/' + language_model + '_rules_' + str(200) + "_" + str(r+1) + '.txt', 'wb') as f:
    pickle.dump(h, f)
"""