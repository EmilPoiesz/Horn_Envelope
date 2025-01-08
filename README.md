# Code Base for the Master Thesis "Extracting Occupational Bias as Logical Rules from Large Language Models"

This is also the code base for the experiments part in the article "Learning Horn Envelopes via Queries to Neural
Networks: The BERT Case".

## Dataset
Some of the experiments are based on a dataset that is created out of wikidata. The corresponding files are

``dataset_extraction.py`` extracts the desired data from wikidata and saves it.

``dataset_sentence_creation.py`` converts the datapoints into sentence/label pairs, that can be used to query the large language models

``dataset_visualization.ipynb`` is a notebook, that visualizes some features of the dataset and gives a general analysis of the structure

``dataset_refactor.py`` refactors the data according to the system that is used in the rule extraction as well (age containers etc.)

## Probing
The probing of the language models is done using the huggingface.co API. The corresponding files are

``probing_script.py``, which conducts the probing over the generated dataset and saves all the scores

``probing_ppbs.ipynb``, which calculates the ppbs scores over different occupation combinations and visualizes them

``probing_confusion_matrix.ipynb``, which visualizes the classification results and true labels in confusion matrices

## Rule Extraction
The first part of the rule extraction is the binarizer, which is supposed to binarize the features for the algorithm. The corresponding preprocessing and class are found in ``binarize_preprocessing.ipynb`` and ``binarize_features/py``.

The HORN algorithm is implemented in ``Horn.py`` and taken from the implementation for the paper "Extracting Horn Theories from Neural Networks with Queries and Counterexamples" by Johanna Jøsang, Cosima Persia and Ana Ozaki. The code is similar to the original, some changes have been made to track relevant variables.

``re_script.py`` contains the code that extracts the rules out of all language models for different #EQs and so on. The hyperparameters can be changed.

``re_eval.ipynb`` contains different kinds of evaluations for the rule extraction results and also saves the rules formatted into files ``re_eval_methods.py`` contains helper methods for this. 

``re_eval_testrun.ipynb`` is currently a prototype for evaluating uncapped runs, but is still in development and not finished (the corresponding data has not arrived yet).