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


def generate_damero(n_inputs, size, seed=None):
    
    if seed == None:
        rng = np.random.default_rng(123)   # semilla reproducible
    else:
        rng = np.random.default_rng(seed)
        
    cells_per_side = 3                 # 3x3
    n_per_cell = size                  # ejemplos por celda (mismo nº en cada cluster)
    sigma = 0.05                       # desviación típica (tendencia a agruparse en el centro)
    domain = (0.0, 1.0)                # cuadrado [0,1]^2

    
    # -----------------------------
    # Construcción del damero 3x3 en [0,1]^2
    # -----------------------------
    low, high = domain
    edges = np.linspace(low, high, cells_per_side + 1)           # 0, 1/3, 2/3, 1
    centers_1d = (edges[:-1] + edges[1:]) / 2                    # centros de cada celda en 1D
    X_list, y_list = [], []
    
    for i, cx in enumerate(centers_1d):          # i = índice eje x
        for j, cy in enumerate(centers_1d):      # j = índice eje y
            # Regla de damero: (i+j) par -> clase 0; impar -> clase 1
            label = (i + j) % 2
            # Muestra gaussiana 2D centrada en (cx, cy)
            pts = rng.normal(loc=(cx, cy), scale=sigma, size=(n_per_cell, 2))
            # Recortar a [0,1]^2 para mantener todo dentro del dominio
            pts = np.clip(pts, low, high)
            X_list.append(pts)
            y_list.append(np.full(n_per_cell, label, dtype=int))
    
    X = np.vstack(X_list)       #Almacena las coordenadas de los datos
    y = np.concatenate(y_list)  #Almacena los valores de clase de los datos
    

    
    o_categorical = F.one_hot(torch.tensor(y), 2)
    
    return torch.from_numpy(X).float(), o_categorical.float(), y
    
        
    

        
def get_damero_dataset(num_inputs, random_seed):
    damero_train_fname = 'datasets/damero_train_data.pickle'
    damero_test_fname = 'datasets/damero_test_data.pickle'
    if not os.path.isfile(damero_train_fname):
        (X_train, y_train_categorical, y_train) = generate_damero(num_inputs, 150, seed=random_seed)
        
        n_train = len(X_train)
        n_test = int((n_train)*((1/(2/3))-1) / 9)
        
        (X_test, y_test_categorical, y_test) = generate_damero(num_inputs, n_test, seed=random_seed)
        
        with open(damero_train_fname, 'wb') as f:
            pickle.dump((X_train, y_train_categorical, y_train), f)

        with open(damero_test_fname, 'wb') as f:
            pickle.dump((X_test, y_test_categorical, y_test), f)
    else:
        with open(damero_train_fname, 'rb') as f:
            (X_train, y_train_categorical, y_train) = pickle.load(f)

        with open(damero_test_fname, 'rb') as f:
            (X_test, y_test_categorical, y_test) = pickle.load(f)

    return X_train, y_train_categorical, y_train, \
            X_test, y_test_categorical, y_test
            
