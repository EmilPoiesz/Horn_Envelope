from transformers import pipeline
import pandas as pd

if __name__ == '__main__':
    
    models = ['roberta-base'] # temporarily only test one model 
    #models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

    for model in models:
        print('===== ' + model + ' =====')
        unmasker = pipeline('fill-mask', model=model)

        data = pd.read_csv('data/dataset.csv', header=None, names=['sentence', 'label']).sample(n=100, random_state=42) # temporary fix only test random 100 rows
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
            data.to_csv('data/probing/probing_' + model + '.csv', index=False)
            data_format.to_csv('data/probing/probing_' + model + '_format.csv', index=False)
            