from itertools import count
from regex import F
import requests 
import json
import pandas as pd
from helper_functions import *
from transformers import pipeline

def initialize_occupations(occupation):
        if isinstance(occupation, str):
            occupations = pd.read_csv(occupation).to_numpy().flatten()
        elif isinstance(occupation, int):
            df = pd.read_csv('data/occupations_total.csv', index_col=0)
            sorted = df.sort_values(by=['total'], ignore_index=True)
            reduced = sorted[sorted['total'] >= occupation]
            occupations = reduced[['occupation']].to_numpy().flatten()
        else :
            occupations = []
        return {occupations[i] : i for i in range(len(occupations))}

print(initialize_occupations('data/occupations.csv'))