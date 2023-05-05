from transformers import pipeline
import pandas as pd
import numpy as np
from helper_functions import *

models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']
occ_list = import_occupations('data/occupations_updated.csv')

for model in models:
    print('===== ' + model + ' =====')
    unmasker = pipeline('fill-mask', model=model)
    for occupation in occ_list:
        name = occupation[0]
        data = pd.read_csv('data/dataset_refac/' + name + '.csv', header=None, names=['sentence', 'label'])
        data['he'] = 0.0
        data['she'] = 0.0
        data['they'] = 0.0
        data_format = data.copy()
        data_format['prediction1'] = ''
        data_format['score1'] = 0.0
        data_format['prediction2'] = ''
        data_format['score2'] = 0.0
        for index, row in data.iterrows():
            result = lm_inference(unmasker, row['sentence'], model)
            for r in result:
                if r['token_str'] == 'She':
                    data.at[index, 'she'] = r['score']
                    data_format.at[index, 'she'] = r['score']
                elif r['token_str'] == 'He':
                    data.at[index, 'he'] = r['score']
                    data_format.at[index, 'he'] = r['score']
                elif r['token_str'] == 'They':
                    data.at[index, 'they'] = r['score']
                    data_format.at[index, 'they'] = r['score']
            data_format.at[index, 'prediction1'] = result[0]['token_str']
            data_format.at[index, 'prediction2'] = result[1]['token_str']
            data_format.at[index, 'score1'] = result[0]['score']
            data_format.at[index, 'score2'] = result[1]['score']
        data.to_csv('data/probing_' + model + '/' + name + '.csv')
        data_format.to_csv('data/probing_' + model + '/' + name + '_format.csv')
        print('Finished with {occupation}'.format(occupation=name))
        