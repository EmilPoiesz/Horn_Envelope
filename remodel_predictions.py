import pandas as pd 
from helper_functions import *

models = ['roberta-base']#, 'roberta-large', 'bert-base-cased', 'bert-large-cased']

for model in models:
    print("=== " + model + " ===")
    predictions_df = pd.read_csv(model + '_pred_new.csv', index_col=0)

    for index, row in predictions_df.iterrows():
        s = row['sentence']
        name = s.split('is a ')[-1].split('.')[0].replace(' ', '_')
        d = {'sentence' : row['sentence'], 'label' : row['label'], 'he' : row['he'], 'she' : row['she'], 'they' : row['they']}
        df = pd.DataFrame(d, index=[0])
        df.to_csv(model + '_predictions_new/' + name + '.csv', mode='a',index=False, header=None)
        if index%50000 == 0:
            print('Finished with row number ' + str(index))