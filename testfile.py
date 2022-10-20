from itertools import count
import requests 
import json
import pandas as pd
from helper_functions import *

country_list = pd.read_csv('data/country_list_continents.csv', names=['cid', ' country', 'continent']).set_index('cid').fillna('?')
continent_lookup = {country_list.continent.unique()[i] : i for i in range(len(country_list.continent.unique()))}
vector = np.zeros(len(continent_lookup), dtype=np.int8)
vector[continent_lookup['Asia']] = 1
print(vector)
occupation_list = pd.read_csv('data/occupations_updated.csv', header=None).to_numpy()
print(occupation_list[:,:1].flatten())