from helper_functions import *
from transformers import pipeline
import timeit
"""
lm = 'bert-large-cased'
sentence = "<mask> was born between 1934 and 1956 in South America and is a singer."
unmasker = pipeline('fill-mask', model=lm)

total = []
for i in range(1000):
    start = timeit.default_timer()
    classification = get_prediction(lm_inference(unmasker, sentence, model=lm))
    stop = timeit.default_timer()
    runtime = stop-start
    total.append(runtime)
total_np = np.array(total)
print("Mean: ", np.mean(total_np))
print("Variance: ", np.var(total_np))
"""
unmasker = pipeline('fill-mask', model='bert-base-cased')
