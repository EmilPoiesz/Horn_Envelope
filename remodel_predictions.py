import pandas as pd 
from helper_functions import *

occ_list = import_occupations('occupations_updated.csv')
predictions_df = pd.read_csv('roberta-base_pred.csv', index_col=0)

for index, row in predictions_df.iterrows():
    s = row['sentence']
    name = s.split('is a ')[-1].split('.')[0].replace(' ', '_')
    d = {'sentence' : row['sentence'], 'label' : row['label'], 'prediction' : row['prediction']}
    df = pd.DataFrame(d, index=[0])
    df.to_csv('roberta-base_predictions/' + name + '.csv', mode='a',index=False)