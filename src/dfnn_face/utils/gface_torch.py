# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 06:44:39 2025

@author: Carles
"""

# import cvxpy as cp
import numpy as np
# from typing import Tuple, Optional, Union
# import torch
# import pandas as pd
# from collections import Counter, OrderedDict
# from utils.ineq import str_equ_out
# from utils.graph import get_eq_list_new
from .ineq import str_equ_out
from .graph import get_eq_list_new

def get_rules(model, X_train, y_train, 
                normalized_range,
                config_samples,                     
                config, decimals=4):
    
    subscript_map = '0123456789'

    def eq_in_boundaries(eq, boundaries):
        for boundary in boundaries:
            if np.allclose(eq, boundary):
                return True
        return False
    
    x_range, y_range = normalized_range 
    
    boundaries = np.array([
                        [-x_range[0],  1.,  0.],
                        [x_range[1], -1.,  0.],
                        [y_range[0],  0.,  1.],
                        [y_range[1],  0, -1.]])
    
    rule_antecedents_title = 'RULE ANTECEDENTS'
    
    eq_list, inequalities, poly_class0, poly_class1, \
        inequalities_class0, inequalities_class1, \
        poly_global, all_zeros_winning_class, neuron_eqs, signs, \
        output_contrib_eqs, output_class_eq = \
                get_eq_list_new(config,config_samples, 
                                X_train, model, 
                                boundaries, decimals, 
                                y_train=y_train)
                
    rule_antecedents_text = ""
    
    if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
        inequ = np.array(inequalities_class0.array)
        rule_antecedents_text += ' Class R (red)\n'

        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            rule_antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'

        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            rule_antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'
        
        pass
    else:
        rule_antecedents_text += ' Class R (red)\n'
        rule_antecedents_text += '  No Apply\n'


    if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
        inequ = np.array(inequalities_class1.array)
        rule_antecedents_text += '\n Class G (green)\n'

        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            rule_antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'

        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            rule_antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'
    else:
        rule_antecedents_text += ' Class G (green)\n'
        rule_antecedents_text += '  No Apply\n'
    
    
    rule_consequent_title = "RULE CONSEQUENT (Network Output)"
       
    rule_consequent_text = ""
    for index, eq in enumerate(output_contrib_eqs):
        if subscript_map[index+1] == '1':
            y_text = 'Y_r'
        else:
            y_text = 'Y_g'
        rule_consequent_text += f'  {y_text} = {str_equ_out(eq, decimals=decimals, normalize=False, web_output=True)}\n'

    # '''
    # Show Inequalities: global and per class
    # '''
    
    
    
    activation_region_title = 'ACTIVATION REGION (Classes R&G)'
    activation_region_text = ""
    
    if len(inequalities.array) > 0:
        inequ = np.array(inequalities.array)
        # activation_region_title = 'ACTIVATION REGION (Classes R&G)'
        
        activation_region_text = ""
        
        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            activation_region_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'


        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            activation_region_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'

    
    return {
        'rule_antecedents': [rule_antecedents_title, rule_antecedents_text],
        'rule_consequent': [rule_consequent_title, rule_consequent_text],
        'activation_region': [activation_region_title, activation_region_text]}
    
        
   
    
    