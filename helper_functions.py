import pandas as pd

def import_occupations(filename):
    #import all occupations and their ids from a given csv file
    occ_list = pd.read_csv(filename, header=None).to_numpy()
    return occ_list

def get_pronoun(gender):
    if gender == 'female':
        return 'she'
    elif gender == 'male':
        return 'he'
    else:
        return 'they'

def get_birth(birth_data):
    if birth_data['type'] == 'uri':
        return "unknown"
    elif birth_data['type'] == 'literal':
        value = birth_data['value'].split('-')
        if len(value[0]) > 0:
            #birthyear AD
            return value[0]
        else:
            #birthyear BC
            return str(value[1]) + " BC"

def get_nid(nid_data):
    value = nid_data['value'].split('/')
    return value[-1]

def replace_gender(gender):
    if gender == 'female' or gender == 'transgender female' or gender == 'cisgender female':
        return 'female'
    elif gender == 'male' or gender == 'transgender male' or gender == 'cisgender male':
        return 'male'
    else:
        return 'diverse'
    