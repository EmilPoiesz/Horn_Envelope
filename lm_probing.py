from transformers import pipeline
import pandas as pd
import numpy as np

models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

for model in models:
    print('===== ' + model + ' =====')
    unmasker = pipeline('fill-mask', model=model)
    data = pd.read_csv('dataset.csv', header=None, names=['sentence', 'label'])
    data['prediction'] = 0
    for index, row in data.iterrows():
        prediction = unmasker(row['sentence'])[0]['token_str']
        data.at[index, 'prediction'] = prediction
        if index%50000 == 0:
            print('Finished with row number ' + str(index))
    data.to_csv(model + '_pred.csv')
