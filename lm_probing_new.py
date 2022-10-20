from transformers import pipeline
import pandas as pd
import numpy as np

models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

for model in models:
    print('===== ' + model + ' =====')
    unmasker = pipeline('fill-mask', model=model)
    data = pd.read_csv('data/dataset.csv', header=None, names=['sentence', 'label'])
    data['he'] = 0.0
    data['she'] = 0.0
    data['they'] = 0.0
    for index, row in data.iterrows():
        if model == 'bert-base-cased' or model == 'bert-large-cased':
            sentence = row['sentence'].replace('<mask>', '[MASK]')
        else:
            sentence = row['sentence']
        result = unmasker(sentence)
        for r in result:
            if r['token_str'] == 'She':
                data.at[index, 'she'] = r['score']
            elif r['token_str'] == 'He':
                data.at[index, 'he'] = r['score']
            elif r['token_str'] == 'They':
                data.at[index, 'they'] = r['score']
        if index%50000 == 0:
            print('Finished with row number ' + str(index))
    data.to_csv('data/' + model + '_pred_new.csv')