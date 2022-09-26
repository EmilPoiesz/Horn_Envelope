import pandas as pd
from helper_functions import *

occ_list = import_occupations("occupations_updated.csv")
nids = []
country_pairs = []
for occupation in occ_list:
        name = occupation[0]
        occ_file = 'dataframes_cleaned/' + name + '.csv'
        df = pd.read_csv(occ_file)
        for index, row in df.iterrows():
            nationality = row['nationality']
            nid = row['nid']
            if nid not in nids:
                pair = [nid, nationality]
                nids.append(nid)
                country_pairs.append(pair)

df = pd.DataFrame(country_pairs)
df.to_csv('countries_with_nids.csv', index=False, header=False)