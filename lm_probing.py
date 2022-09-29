from transformers import pipeline
import pandas as pd
import numpy as np

models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

for model in models:
    print('===== ' + model + ' =====')
    unmasker = pipeline('fill-mask', model=model)
    data = pd.read_csv('dataset.csv', header=None, names=['sentence', 'label'])
    data['prediction1'] = 0
    data['score1'] = 0
    data['prediction2'] = 0
    data['score2'] = 0
    for index, row in data.iterrows():
        if model == 'bert-base-cased' or model == 'bert-large-cased':
            sentence = row['sentence'].replace('<mask>', '[MASK]')
        else:
            sentence = row['sentence']
        result = unmasker(sentence)
        data.at[index, 'prediction1'] = result[0]['token_str']
        data.at[index, 'prediction2'] = result[1]['token_str']
        data.at[index, 'score1'] = result[0]['score']
        data.at[index, 'score2'] = result[1]['score']
        if index%50000 == 0:
            print('Finished with row number ' + str(index))
    data.to_csv(model + '_pred.csv')