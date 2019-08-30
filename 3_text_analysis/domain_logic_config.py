import domain_logic_tCoIR as domain_logic

current_domain = domain_logic

import os
import pandas as pd

current_domain.SUBSTITUTION_FILENAME = os.path.join(domain_logic.DATA_FOLDER, 'term_substitutions.txt')

def get_tagset():
    filepath = os.path.join(domain_logic.DATA_FOLDER, 'tagset.csv')
    if os.path.isfile(filepath):
        return pd.read_csv(filepath, sep='\t').fillna('')
    return None

current_domain.get_tagset = get_tagset

# _TERM_SUBSTITUTIONS = None

# def get_term_substitutions():

#     if _TERM_SUBSTITUTIONS is None:
#         if os.path.isfile(SUBSTITUTION_FILENAME):
#             _TERM_SUBSTITUTIONS = pd.read_csv(SUBSTITUTION_FILENAME)

#     return _TERM_SUBSTITUTIONS