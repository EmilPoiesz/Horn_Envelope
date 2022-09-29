import pandas as pd 
from helper_functions import *

occ_list = import_occupations('occupations_updated.csv')
models = ['roberta-base', 'roberta-large', 'bert-base-cased', 'bert-large-cased']

for model in models:
    print("=== " + model + " ===")
    predictions_df = pd.read_csv(model + '_pred.csv', index_col=0)

    for index, row in predictions_df.iterrows():
        s = row['sentence']
        name = s.split('is a ')[-1].split('.')[0].replace(' ', '_')
        d = {'sentence' : row['sentence'], 'label' : row['label'], 'prediction1' : row['prediction1'], 'score1' : row['score1'], 'prediction2' : row['prediction2'], 'score2' : row['score2']}
        df = pd.DataFrame(d, index=[0])
        df.to_csv(model + '_predictions/' + name + '.csv', mode='a',index=False, header=None)
        if index%50000 == 0:
            print('Finished with row number ' + str(index))