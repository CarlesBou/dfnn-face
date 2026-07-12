# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
# import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
import os 
# import matplotlib

# import time
import numpy as np
# import random
# import cdd
# from collections import Counter, OrderedDict

# from sympy import Symbol
# import sympy as sp


# import cvxpy as cp
# from utils.face_torch import FNNModule
# from utils.face_torch import get_face_contrib_accelerated
# from utils.face_torch import count_configurations

# from utils.graph import ImplicitEquationPlotter
# from utils.graph import get_eq_list_new

# from utils.ineq import str_equ_out

# from utils.gface_torch import get_rules

import pickle


def generate_circle(n_inputs, size, seed=None, r_circle=0.75):
    
    if seed == None:
        rng = np.random.default_rng(123)   # semilla reproducible
    else:
        rng = np.random.default_rng(seed)
        
    max_x = 1
    min_x = -1
    
    x1 = rng.uniform(min_x, max_x, size)
    x2 = rng.uniform(min_x, max_x, size)
    X = np.concatenate([[x1, x2]], axis = 1)
    
    y = (x1 * x1 + x2 * x2) < r_circle
    y = y * 1
    
    '''
    Normalize input
    '''
    X = (X + 1) / 2
    
    X = X.T
    
    o_categorical = F.one_hot(torch.tensor(y), 2)
    
    return torch.from_numpy(X).float(), o_categorical.float(), y
    
        
    
'''
plt.scatter(a[0], a[1], c=c, s=3); plt.gca().set_aspect('equal', adjustable='box')

plt.scatter(a[0], a[1], c=c, s=3); plt.axis('equal')
'''


        
def get_circle_dataset(num_inputs, random_seed, r_circle=0.75):
    circle_train_fname = f'datasets/circle_train_data_r_circle={r_circle}.pickle'
    circle_test_fname = f'datasets/circle_test_data_r_circle={r_circle}.pickle'
    
    if not os.path.isfile(circle_train_fname):
        (X_train, y_train_categorical, y_train) = generate_circle(num_inputs, 10000, seed=random_seed, r_circle=r_circle)
        
        (X_test, y_test_categorical, y_test) = generate_circle(num_inputs, 5000, seed=random_seed, r_circle=r_circle)
        
        with open(circle_train_fname, 'wb') as f:
            pickle.dump((X_train, y_train_categorical, y_train), f)

        with open(circle_test_fname, 'wb') as f:
            pickle.dump((X_test, y_test_categorical, y_test), f)
    else:
        with open(circle_train_fname, 'rb') as f:
            (X_train, y_train_categorical, y_train) = pickle.load(f)

        with open(circle_test_fname, 'rb') as f:
            (X_test, y_test_categorical, y_test) = pickle.load(f)

    return X_train, y_train_categorical, y_train, \
            X_test, y_test_categorical, y_test
            
