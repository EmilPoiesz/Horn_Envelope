from transformers import pipeline
import pandas as pd
import os

if __name__ == '__main__':
    
    occupations_df = pd.read_csv("data/occupations.csv")
    occupations = occupations_df[occupations_df['extracted'] == True]['occupation'].values
    models = ['roberta-base'] # temporarily only test one model 
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

    for model in models:
        os.makedirs(f'data/probing/probing_{model}', exist_ok=True)
        print('===== ' + model + ' =====')
        unmasker = pipeline('fill-mask', model=model)

        for occupation in occupations:
            data = pd.read_csv(f'data/probing_sentences/{occupation}_dataset.csv', header=None, names=['sentence', 'label']).head(n=100) # temporary fix only test random 100 rows
            data['he'] = 0.0
            data['she'] = 0.0
            data['they'] = 0.0
            data_format = data.copy()
            data_format['prediction1'] = ''
            data_format['score1'] = 0.0
            data_format['prediction2'] = ''
            data_format['score2'] = 0.0
            
            for index, row in data.iterrows():
                
                sentence = row.sentence
                if model.split('-')[0] == 'bert' :
                    sentence = sentence.replace('<mask>', '[MASK]')
                
                result = unmasker(sentence)
                
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
                data.to_csv(f'data/probing/probing_{model}/{occupation}.csv', index=False)
                data_format.to_csv(f'data/probing/probing_{model}/{occupation}_format.csv', index=False)
                