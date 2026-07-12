# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os 
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import math
import time
import numpy as np
import random
import cdd
from collections import Counter, OrderedDict

from sympy import Symbol
import sympy as sp

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5 import QtPrintSupport,QtSvg
from PyQt5.QtGui import QPainter, QPdfWriter, QPageSize
from PyQt5.QtCore import QSizeF, QRectF, QMarginsF
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPainterPath
from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtGui import QTextDocument

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import cvxpy as cp
from utils.ineq import get_radius, get_radius_old, analyze_displacement
from utils.ineq import get_ra_matrix, get_new_ra
from utils.ineq import get_face_contrib_accelerated

import pickle
from pprint import pprint


class BooleanVarReplica:
    def __init__(self, value=False):
        self._value = bool(value)
        self._callbacks = []

    def get(self):
        """Returns the current boolean value."""
        return self._value

    def set(self, value):
        """Sets a new value and notifies all observers if it changed."""
        new_value = bool(value)
        if self._value != new_value:
            self._value = new_value
            self._notify()

    def trace_add(self, callback):
        """Register a function to be called when the value changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _notify(self):
        """Internal method to trigger all registered callbacks."""
        for callback in self._callbacks:
            callback(self._value)

    def __bool__(self):
        """Allows usage in 'if var_instance:' statements."""
        return self._value

    def __repr__(self):
        return f"BooleanVarReplica(value={self._value})"
    
    
def get_center(A, b):
    x = cp.Variable(A.shape[1])
    r = cp.Variable()
    
    norms = np.linalg.norm(A, axis=1)
    
    constraints = [A @ x + b >= r * norms]
    
    problem = cp.Problem(cp.Maximize(r), constraints)
    problem.solve()

    if problem.status in ["optimal"]:
        if r.value > 0:
            return x.value, r.value

    return None, -np.inf    

def cvx_get_minimal_matrix(matrix):
    '''
    Ojo con este -A para pasar a Ax -b <= 0 desde Ax + b >= 0
    '''
    A = -matrix[:, 1:]
    b = matrix[:, 0]

    original_center, original_radius = get_center(A, b)
    
    if original_radius == -np.inf: 
        return np.array([])
    else:
        n_rows = matrix.shape[0]

        removal_list = []
        for i in range(n_rows):
            # Create boolean mask excluding row i
            mask = np.ones(n_rows, dtype=bool)
            mask[i] = False
        
            # Get matrix without row i
            new_matrix = matrix[mask]

            new_A = -new_matrix[:, 1:]
            new_b = new_matrix[:, 0]

            new_center, new_radius = get_center(new_A, new_b)
            
            if new_radius == -np.inf:
                return np.array([])
            else:
                tolerance_radius = 1e-6
               
                radius_change = np.abs(original_radius - new_radius)
               
                print(f'RADIUS CHANGE = {radius_change:.04e}')
                if radius_change < tolerance_radius:
                   removal_list.append(i)
                else:
                   pass
            
        if len(removal_list) > 0:
            mask = np.ones(n_rows, dtype=bool)
            for i in removal_list:
                mask[i] = False
            
            return matrix[mask]
        
        return matrix
    
    
def cvx_get_minimal_matrix2(matrix):
    A = -matrix[:, 1:]
    b = matrix[:, 0]

    original_center, original_radius = get_center(A, b)
    
    if original_radius == -np.inf: 
        return np.array([])
    else:
        n_rows = matrix.shape[0]

        removal_list = []
        for i in range(n_rows):
            # Create boolean mask excluding row i
            mask = np.ones(n_rows, dtype=bool)
            mask[i] = False
        
            # Get matrix without row i
            new_matrix = matrix[mask]

            new_A = -new_matrix[:, 1:]
            new_b = new_matrix[:, 0]

            new_center, new_radius = get_center(new_A, new_b)
            
            if new_radius == -np.inf:
                return np.array([])
            else:
                tolerance_radius = 1e-10
                tolerance_center = 1e-10
                
                radius_change = np.abs(original_radius - new_radius)
                center_distance = np.sqrt(np.sum((original_center - new_center) ** 2))
                
                
                if radius_change < tolerance_radius and center_distance < tolerance_center:
                    removal_list.append(i)
                else:
                    pass
            
        if len(removal_list) > 0:
            mask = np.ones(n_rows, dtype=bool)
            for i in removal_list:
                mask[i] = False
            
            return matrix[mask]
        
        return matrix

def cvx_get_minimal_matrix_incremental(matrix):
    current_matrix = matrix.copy()
    
    while True:
        A = current_matrix[:, 1:]
        b = current_matrix[:, 0]
        original_center, original_radius = get_center(A, b)
        
        if original_radius == -np.inf:
            break
            
        removed_any = False
        n_rows = current_matrix.shape[0]
        
        for i in range(n_rows):
            # Test removal from current reduced system
            mask = np.ones(n_rows, dtype=bool)
            mask[i] = False
            
            test_matrix = current_matrix[mask]
            test_A = test_matrix[:, 1:]
            test_b = test_matrix[:, 0]
            new_center, new_radius = get_center(test_A, test_b)
            
            tolerance_radius = 1e-4
            tolerance_center = 1e-4

            if new_radius != -np.inf:
                # Check if center and radius are essentially unchanged
                if (abs(original_radius - new_radius) < tolerance_radius and 
                    np.linalg.norm(original_center - new_center) < tolerance_center):
                    current_matrix = test_matrix
                    removed_any = True
        
        if not removed_any:
            break
    
    return current_matrix


class FNNModule(nn.Module):
    # H_list = []
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self, x):
        self.H_list = []
        for layer in self.layers:
            x = layer(x)
            self.H_list.append(x)
        return x



def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets.argmax(dim=1)).sum().item()
    return correct / total


def count_configurations(model, X, get_samples=False, include_last=True, get_cm=False, y_train=None):
    configs = []
    samples = {}
    color_count = {}
    cm = {}
    
    num_outputs = model.layers[-1].out_features
    # '''
    # Obtenemos el número de salidas desde la última capa antes de la ReLU
    # '''
   
    for sample, x in enumerate(X):
        contrib, W_list, I_vecs, _, H_list, _, _ = get_face_contrib_accelerated(x, model)
            
        config = ''
        for vec in I_vecs:
            for v in vec[1:]:
                config = config + str(int(v))
                
        if not include_last:
            config = config[:-num_outputs]
            
        if configs.count(config) == 0:
            color_count[config] = {'count_r': 0, 'count_g': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} 

            samples[config] = [sample]
        else:
            samples[config].append(sample)
                      
                    
        predicted_class = 0 if H_list[-1][1] > H_list[-1][2] else 1
        
        if predicted_class:
            color_count[config]['count_r'] += 1
        else: 
            color_count[config]['count_g'] += 1

        if predicted_class == 0 and y_train[sample] == 0:
            color_count[config]['TP'] += 1 
        elif predicted_class == 0 and y_train[sample] == 1:
            color_count[config]['FP'] += 1
        elif predicted_class == 1 and y_train[sample] == 0:
            color_count[config]['FN'] += 1
        elif predicted_class == 1 and y_train[sample] == 1:
            color_count[config]['TN'] += 1

            
        configs.append(config)

    '''
    En samples devolvemos un diccionario con la configuración y una tupla con:
        último ejemplo de la configuración (índice en X)
        cuenta total de ejemplos de la configuración
    '''
    
    if not get_samples:
        return Counter(configs)
    else:
        return OrderedDict(sorted(Counter(configs).items())), samples, color_count



def get_inequalities(H_inequalities, decimals=2):        
    '''
    Get inequalities
    H_inequlities format: [b A] --> Ax <= b
    '''
    if H_inequalities == []:
        return []
     
    inequalities = []
    
    for row in H_inequalities.array:
        row = np.array(row)
        
        row[abs(row) < 1e-10] = 0
        
        if np.all(row[1:] == 0):
            continue
        
        sign = '>='
        
        if row[1] != 0:
            if row[1] < 0:
                sign = '<='
            row = row / row[1]
        elif row[2] != 0:
            if row[2] < 0:
                sign = '<='
            row = row / row[2]
            
        b = -row[0]
        A = row[1:]
        
        terms = []
        letters = 'xyz'
        
        for i, A_i in enumerate(A):
            val = eval(f'{A_i:.{decimals}f}')
            if len(terms) > 0:
                if A_i < 0:
                    if val == -1:
                        terms.append(f" - {letters[i]}")
                    else:
                        terms.append(f" - {abs(A_i):.{decimals}f}{letters[i]}")
                else:
                    if val == 1:
                        terms.append(f" + {letters[i]}")
                    elif val == 0:
                        continue
                    else:
                        terms.append(f" + {abs(A_i):.{decimals}f}{letters[i]}")
            else:
                if i == 0:
                    if val == 1:
                        terms.append(f"{letters[i]}")
                    elif val == -1:
                        terms.append(f"-{letters[i]}")
                    elif val == 0:
                        continue
                    elif val > 0:
                        terms.append(f"{abs(A_i):.{decimals}f}{letters[i]}")
                    else:
                        terms.append(f"-{abs(A_i):.{decimals}f}{letters[i]}")
                else:
                    if A_i < 0:
                        if val == -1:
                            terms.append(f"-{letters[i]}")
                        else:
                            terms.append(f"{A_i:.{decimals}f}{letters[i]}")
                    else:
                        if val == 1:
                            terms.append(f"{letters[i]}")
                        else:
                            terms.append(f"{abs(A_i):.{decimals}f}{letters[i]}")
    
        inequality = f'{"".join(t for t in terms)} {sign} {b:.{decimals}f}'
        
        if len(terms) > 0:
            inequalities.append(inequality)
        
    return inequalities


subscript_map = '0RG3456789'



def str_equ_new(eq, decimals=2, return_full=False, orig_sign=-2, normalize=True):
    
    eq = eq.copy()
    
    eq[abs(eq) < 1e-10] = 0
    
    sign = 1
    
    if eq[1] != 0:
        if eq[1] < 0:
            sign = -1
        if normalize:
            eq = eq / eq[1]
    elif eq[2] != 0:
        if eq[2] < 0:
            sign = -1
        if normalize:
            eq = eq / eq[2]
            
    
    n_coefs = len(eq[1:])
    ret = ''
    first = True
    for i in range(n_coefs):
        val = eval(f'{eq[i+1]:.{decimals}f}')
        if val == 0:
            # first = False
            continue
        elif val == 1:
            if not first:
                ret += f' + {abs(val):.{decimals}f}x{subscript_map[i+1]} '
            else:
                ret += f'x{subscript_map[i+1]} '
        elif val == -1:
            if not first:
                ret += f'- {abs(val):.{decimals}f}x{subscript_map[i+1]} '
            else:
                ret += f'-x{subscript_map[i+1]} '
        else:
            if val < 0:
                if first:
                    ret += '-'
                else:
                    ret += '- '
            else:
                if not first:
                    ret += '+ '
            ret += f'{abs(val):.{decimals}f}x{subscript_map[i+1]} '
        first = False
    if first:
        if eq[0] < 0:
            ret += f'{eq[0]:.{decimals}f}'
        else:
            ret += f'{abs(eq[0]):.{decimals}f}'
    else:
        if eq[0] < 0:
            ret += f'- {abs(eq[0]):.{decimals}f}'
        else:
            ret += f'+ {abs(eq[0]):.{decimals}f}'
            
    if return_full:
        r = sign * orig_sign
        ret += f' {">= 0" if r > 0 else "<= 0"}'
        
    return ret
    

def get_eq_list_new(config, config_samples, X_train, model, boundaries, decimals=2, bounded=True, y_train=None):
    ''' 
    Choose a sample from the current configuration
    '''
    sample = config_samples[config][0]

    contrib, W_list, I_vecs, contrib_list, H_list, O_list, I_list =  \
        get_face_contrib_accelerated(X_train[sample].numpy(), 
                                     model, 
                                     return_weighted=False) 

    '''
    Cálculo con las contribuciones
    '''
    
    w0 = contrib_list[0][1:]
    
    for i in range(len(contrib_list) - 2):
        w0 = np.concatenate((w0, contrib_list[i + 1][1:]))
    

    pass

    '''
    Para el caso de 1 en config, el vector se mantiene [b A] para expresar Ax + b >= 0
    Para el caso de 0 en config, cambiamos a [-b -A] para expresar Ax + b <= 0 --> -Ax - b >=0
    '''
    
    sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
    sign_vec[sign_vec == 0] = -1
            
    matrix = w0 * sign_vec
    
    borders = boundaries
    
    new_matrix = np.concatenate((matrix, borders))
    
    H_bounded = cdd.matrix_from_array(new_matrix, rep_type=cdd.RepType.INEQUALITY)
    H_poly_bounded = cdd.polyhedron_from_matrix(H_bounded)
    generator_bounded = cdd.copy_generators(H_poly_bounded)

    generator_bounded_list_class0, generator_bounded_list_class1 = [], []
    H_inequalities_class0, H_inequalities_class1 = [], []

    
    all_zeros_winning_class = -1
    
    '''
    Añadidmos Y1 >= Y2 ... si existe 
    '''
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    sign = H_list[-1][1:]
    sign = 1 if sign[0] > sign[1] else -1
    
    sign = 1
    
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        half_matrix_class0 = np.vstack((new_matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class0 = cdd.matrix_from_array(half_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class0 = cdd.polyhedron_from_matrix(H_bounded_half_class0)
        generator_bounded_class0 = cdd.copy_generators(H_poly_bounded_half_class0)

        half_matrix_class1 = np.vstack((new_matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class1 = cdd.matrix_from_array(half_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class1 = cdd.polyhedron_from_matrix(H_bounded_half_class1)
        generator_bounded_class1 = cdd.copy_generators(H_poly_bounded_half_class1)
        
        half_unbounded_matrix_class0 = np.vstack((matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class0 = cdd.matrix_from_array(half_unbounded_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class0 = cdd.polyhedron_from_matrix(H_unbounded_half_class0)
        generator_unbounded_class0 = cdd.copy_generators(H_poly_unbounded_half_class0)

        half_unbounded_matrix_class1 = np.vstack((matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class1 = cdd.matrix_from_array(half_unbounded_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class1 = cdd.polyhedron_from_matrix(H_unbounded_half_class1)
        generator_unbounded_class1 = cdd.copy_generators(H_poly_unbounded_half_class1)
        
        ex_class0 = False
        if len(generator_unbounded_class0.array) > 0:
            
            if not bounded:
                V_poly_unbounded_class0 = cdd.polyhedron_from_matrix(generator_unbounded_class0) 
                H_inequalities_class0 = cdd.copy_inequalities(V_poly_unbounded_class0)
            else:
                try:
                    V_poly_bounded_class0 = cdd.polyhedron_from_matrix(generator_bounded_class0) 
                    H_inequalities_class0 = cdd.copy_inequalities(V_poly_bounded_class0)
                except:
                    ex_class0 = True
                    H_inequalities_class0 = []

        
        if len(generator_unbounded_class1.array) > 0:
            if not bounded:
                V_poly_unbounded_class1 = cdd.polyhedron_from_matrix(generator_unbounded_class1) 
                H_inequalities_class1 = cdd.copy_inequalities(V_poly_unbounded_class1)
            else:
                try:
                    V_poly_bounded_class1 = cdd.polyhedron_from_matrix(generator_bounded_class1) 
                    H_inequalities_class1 = cdd.copy_inequalities(V_poly_bounded_class1)
                except:
                    H_inequalities_class1 = []
                   
        generator_bounded_list_class0 = generator_bounded_class0.array
        generator_bounded_list_class1 = generator_bounded_class1.array
    else: 
        all_zeros_winning_class = 0 if contrib_y1_y2[0] >= 0 else 1

    
    if not bounded:
        '''
        H rep from inequations WITHOUT bounding
        '''
        H_unbounded = cdd.matrix_from_array(matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_unbounded = cdd.polyhedron_from_matrix(H_unbounded)
        generator_unbounded = cdd.copy_generators(H_poly_unbounded)
        
        print(f'2rd GENERATOR UNBOUNDED={len(generator_unbounded.array)}')
        
        V_poly_unbounded = cdd.polyhedron_from_matrix(generator_unbounded) 
        H_inequalities = cdd.copy_inequalities(V_poly_unbounded)
    else:
        '''
        H rep from inequations WITH bounding
        '''
        H_bounded = cdd.matrix_from_array(matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_unbounded = cdd.polyhedron_from_matrix(H_bounded)
        generator_bounded = cdd.copy_generators(H_poly_bounded)
        
        V_poly_bounded = cdd.polyhedron_from_matrix(generator_bounded) 
        H_inequalities = cdd.copy_inequalities(V_poly_bounded)


    eq_list = []
    
    x_list = []
    
    for i in range(len(X_train[0])):
        x = Symbol(f'x_{i+1}')
        x_list.append(x)
        
    x_list = [Symbol('x'), Symbol('y')]
    
    for i, v in enumerate(w0):
        if np.allclose(v[1:], 0.):
            s = sp.Eq(x_list[0] - 2, 0)
            continue
        else:
            lhs = sum(x_list) + 2222
            
            s = sp.Eq(lhs, 0)
                        
            for j in range(len(x_list)):
                s = s.subs(x_list[j], v[j+1] * x_list[j])

            s = s.subs(2222, v[0])
            
        expr = s.lhs
        coeff_x = abs(expr.coeff(x_list[0]))
        
        if coeff_x == 0:
            coeff_x = abs(expr.coeff(x_list[1]))
            
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)

        eq_list.append((f'Eq{i+1}', new_equation, sign_vec[i][0], generator_bounded, sign_vec[i][0]))
        


    '''
    Añadimos Y1 e Y2
    '''
    for i, v in enumerate(contrib):
        if np.allclose(v[1:], 0.):
             continue
        else:
            lhs = sum(x_list) + 2222
            
            s = sp.Eq(lhs, 0)
            
            for i in range(len(x_list)):
                s = s.subs(x_list[j], v[j+1] * x_list[j])
            
            s = s.subs(2222, v[0])
           

            
    '''
    Añadimos Y1 >= Y2
    '''
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    ret_y1_y2 = None
    
    if not np.allclose(contrib_y1_y2[1:], 0.):
        ret_y1_y2 = contrib_y1_y2
        
        lhs = sum(x_list) + 2222
            
        s = sp.Eq(lhs, 0)
            
        for i in range(len(x_list)):
            s = s.subs(x_list[i], contrib_y1_y2[i+1] * x_list[i])
         
        s = s.subs(2222, contrib_y1_y2[0])

        expr = s.lhs
        coeff_x = abs(expr.coeff(x_list[0]))
        
        if coeff_x == 0:
            coeff_x = abs(expr.coeff(x_list[1]))
        
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
        sign = np.sum(contrib, axis=1)
        
        sign = 1 if sign[0] > sign[1] else -1
        
        sign2 = 1 if H_list[-1][1] > H_list[-1][2] else -1

        eq_list.append(('Eq9', new_equation, 0, generator_bounded, sign2))
        
    
    return eq_list, \
            H_inequalities, \
            generator_bounded_list_class0, generator_bounded_list_class1, \
            H_inequalities_class0, H_inequalities_class1, \
            generator_bounded.array, \
            all_zeros_winning_class, \
            w0, sign_vec, contrib, ret_y1_y2
    


def get_eq_list(config, config_samples, X_train, model, boundaries, decimals=2, bounded=True, y_train=None):
    ''' 
    Choose a sample from the current configuration
    '''
    sample = config_samples[config][0]
    
    contrib, W_list, I_vecs, contrib_list, H_list, O_list, I_list =  get_face_contrib_accelerated(
                                                        X_train[sample].numpy(), 
                                                        model, return_weighted=False) 

 
    '''
    Cálculo con las contribuciones
    '''
    
    w0 = contrib_list[0][1:]
    
    for i in range(len(contrib_list) - 2):
        w0 = np.concatenate((w0, contrib_list[i + 1][1:]))    

    pass

    '''
    Para el caso de 1 en config, el vector se mantiene [b A] para expresar Ax + b >= 0
    Para el caso de 0 en config, cambiamos a [-b -A] para expresar Ax + b <= 0 --> -Ax - b >=0
    '''
    
    sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
    sign_vec[sign_vec == 0] = -1
       
    matrix = w0 * sign_vec
    
    borders = boundaries
    
    new_matrix = np.concatenate((matrix, borders))
    
    
    H_bounded = cdd.matrix_from_array(new_matrix, rep_type=cdd.RepType.INEQUALITY)
    H_poly_bounded = cdd.polyhedron_from_matrix(H_bounded)
    generator_bounded = cdd.copy_generators(H_poly_bounded)
    
    generator_bounded_list_class0, generator_bounded_list_class1 = [], []
    H_inequalities_class0, H_inequalities_class1 = [], []


    
    all_zeros_winning_class = -1
    
    '''
    Añadidmos Y1 >= Y2 ... si existe 
    '''
    contrib_y1_y2 = contrib[0] - contrib[1]
        
    sign = H_list[-1][1:]
    sign = 1 if sign[0] > sign[1] else -1
           
    sign = 1
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        half_matrix_class0 = np.vstack((new_matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class0 = cdd.matrix_from_array(half_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class0 = cdd.polyhedron_from_matrix(H_bounded_half_class0)
        generator_bounded_class0 = cdd.copy_generators(H_poly_bounded_half_class0)

        half_matrix_class1 = np.vstack((new_matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class1 = cdd.matrix_from_array(half_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class1 = cdd.polyhedron_from_matrix(H_bounded_half_class1)
        generator_bounded_class1 = cdd.copy_generators(H_poly_bounded_half_class1)
        
        half_unbounded_matrix_class0 = np.vstack((matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class0 = cdd.matrix_from_array(half_unbounded_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class0 = cdd.polyhedron_from_matrix(H_unbounded_half_class0)
        generator_unbounded_class0 = cdd.copy_generators(H_poly_unbounded_half_class0)

        half_unbounded_matrix_class1 = np.vstack((matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class1 = cdd.matrix_from_array(half_unbounded_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class1 = cdd.polyhedron_from_matrix(H_unbounded_half_class1)
        generator_unbounded_class1 = cdd.copy_generators(H_poly_unbounded_half_class1)
        
        ex_class0 = False
        if len(generator_unbounded_class0.array) > 0:
            
            if not bounded:
                V_poly_unbounded_class0 = cdd.polyhedron_from_matrix(generator_unbounded_class0) 
                H_inequalities_class0 = cdd.copy_inequalities(V_poly_unbounded_class0)
            else:
                try:
                    V_poly_bounded_class0 = cdd.polyhedron_from_matrix(generator_bounded_class0) 
                    H_inequalities_class0 = cdd.copy_inequalities(V_poly_bounded_class0)
                except:
                    ex_class0 = True
                    H_inequalities_class0 = []

        
        if len(generator_unbounded_class1.array) > 0:
            if not bounded:
                V_poly_unbounded_class1 = cdd.polyhedron_from_matrix(generator_unbounded_class1) 
                H_inequalities_class1 = cdd.copy_inequalities(V_poly_unbounded_class1)
            else:
                try:
                    V_poly_bounded_class1 = cdd.polyhedron_from_matrix(generator_bounded_class1) 
                    H_inequalities_class1 = cdd.copy_inequalities(V_poly_bounded_class1)
                except:
                    H_inequalities_class1 = []
                    
        generator_bounded_list_class0 = generator_bounded_class0.array
        generator_bounded_list_class1 = generator_bounded_class1.array
    else: 
        all_zeros_winning_class = 0 if contrib_y1_y2[0] >= 0 else 1

    
    if not bounded:
        '''
        H rep from inequations WITHOUT bounding
        '''
        H_unbounded = cdd.matrix_from_array(matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_unbounded = cdd.polyhedron_from_matrix(H_unbounded)
        generator_unbounded = cdd.copy_generators(H_poly_unbounded)
        
        V_poly_unbounded = cdd.polyhedron_from_matrix(generator_unbounded) 
        H_inequalities = cdd.copy_inequalities(V_poly_unbounded)
    else:
        '''
        H rep from inequations WITH bounding
        '''
        H_bounded = cdd.matrix_from_array(matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_unbounded = cdd.polyhedron_from_matrix(H_bounded)
        generator_bounded = cdd.copy_generators(H_poly_bounded)
        
        V_poly_bounded = cdd.polyhedron_from_matrix(generator_bounded) 
        H_inequalities = cdd.copy_inequalities(V_poly_bounded)


    eq_list = []
    
    x = Symbol('x')
    y = Symbol('y')
    
    for i, v in enumerate(w0):
        if v[1] == 0 and v[2] == 0:
            s = sp.Eq(x - 2, 0)
            continue
        else:
            s = sp.Eq(x + y + 222, 0)
            s = s.subs(x, v[1] * x)
            s = s.subs(y, v[2] * y)
            
            s = s.subs(222, v[0])
        
        
        expr = s.lhs
        coeff_x = abs(expr.coeff(x))
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
       
        eq_list.append((f'Eq{i+1}', new_equation, sign_vec[i][0], generator_bounded, sign_vec[i][0]))
        


    '''
    Añadimos Y1 e Y2
    '''
    for i, v in enumerate(contrib):
        if v[1] == 0 and v[2] == 0:
            continue

        else:
            s = sp.Eq(x + y + 222, 0)
            s = s.subs(222, v[0])
            s = s.subs(x, v[1] * x)
            s = s.subs(y, v[2] * y)
        
            expr = s.lhs
            coeff_x = abs(expr.coeff(x))
            new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)

            
    '''
    Añadimos Y1 >= Y2
    '''
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    ret_y1_y2 = None
    
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        print('ADDING DUMMIE 2')
        ret_y1_y2 = contrib_y1_y2
        s = sp.Eq(x + y + 222, 0)
        s = s.subs(222, contrib_y1_y2[0])
        s = s.subs(x, contrib_y1_y2[1] * x)
        s = s.subs(y, contrib_y1_y2[2] * y)
     
        expr = s.lhs
        coeff_x = abs(expr.coeff(x))
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
        sign = np.sum(contrib, axis=1)
        
        sign = 1 if sign[0] > sign[1] else -1
        
        sign2 = 1 if H_list[-1][1] > H_list[-1][2] else -1
        
        eq_list.append(('Y_1 >= Y_2', new_equation, 0, generator_bounded, sign2))
       
        
    return eq_list, \
            H_inequalities, \
            generator_bounded_list_class0, generator_bounded_list_class1, \
            H_inequalities_class0, H_inequalities_class1, \
            generator_bounded.array, \
            all_zeros_winning_class, \
            w0, sign_vec, contrib, ret_y1_y2
            
            
    

class LineStyleWidget(QtWidgets.QWidget):
    def __init__(self, color, linestyle, linewidth=2, parent=None, background_color="#f1f1f1"):
        super().__init__(parent)
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.setFixedWidth(50)
        self.setFixedHeight(20)
        self.setAutoFillBackground(True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent) 
        self.background_color = background_color
        

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QColor(self.background_color))
        
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Ensure we use a proper QColor
        pen = QtGui.QPen(QtGui.QColor(self.color))
        pen.setWidth(self.linewidth)
        
        # Explicitly handle Matplotlib's 'dash' string
        if self.linestyle in ['dashed', '--', 'dash']:
            pen.setStyle(QtCore.Qt.DashLine)
        elif self.linestyle in ['dotted', ':', 'dot']:
            pen.setStyle(QtCore.Qt.DotLine)
        elif self.linestyle in ['dashdot']:
            pen.setStyle(QtCore.Qt.DashDotLine)
        else:
            pen.setStyle(QtCore.Qt.SolidLine)
            
        painter.setPen(pen)
        # Draw the line in the vertical center of the widget
        painter.drawLine(0, 10, 50, 10)
        



def str_equ_out(eq, decimals=2, return_full=False, 
            orig_sign=-2, normalize=True, web_output=False):

    subscript_map = '0123456789'
    
    eq = eq.copy()
    
    eq[abs(eq) < 1e-10] = 0
    
    sign = 1
    
    if eq[1] != 0:
        if eq[1] < 0:
            sign = -1
        if normalize:
            eq = eq / eq[1]
    elif eq[2] != 0:
        if eq[2] < 0:
            sign = -1
        if normalize:
            eq = eq / eq[2]
           
    
    n_coefs = len(eq[1:])
    ret = ''
    first = True
    for i in range(n_coefs):
        val = eval(f'{eq[i+1]:.{decimals}f}')
        if val == 0:
            # first = False
            continue
        elif val == 1:
            if not first:
                ret += f' + {abs(val):.{decimals}f}x{subscript_map[i+1]} '
            else:
                ret += f'x{subscript_map[i+1]} '
        elif val == -1:
            if not first:
                ret += f'- {abs(val):.{decimals}f}x{subscript_map[i+1]} '
            else:
                ret += f'-x{subscript_map[i+1]} '
        else:
            if val < 0:
                if first:
                    ret += '-'
                else:
                    ret += '- '
            else:
                if not first:
                    ret += '+ '
            ret += f'{abs(val):.{decimals}f}x{subscript_map[i+1]} '
        first = False
    if first:
        if eq[0] < 0:
            ret += f'{eq[0]:.{decimals}f}'
        else:
            ret += f'{abs(eq[0]):.{decimals}f}'
    else:
        if eq[0] < 0:
            if not np.isclose(eq[0], 0.0):
                if not web_output:
                    ret += f'&nbsp;- {abs(eq[0]):.{decimals}f}'
                else:
                    ret += f'- {abs(eq[0]):.{decimals}f}'
        else:
            if not np.isclose(eq[0], 0.0):
                ret += f'+ {abs(eq[0]):.{decimals}f}'
            
    if return_full:
        r = sign * orig_sign
        ret += f' {">= 0" if r > 0 else "<= 0"}'
        
    return ret

    
def print_rules(model, X_train, y_train, x_range, y_range,
                config_samples, decimals,                     
                config):
    
    subscript_map = '0123456789'

    def eq_in_boundaries(eq, boundaries):
        for boundary in boundaries:
            if np.allclose(eq, boundary):
                return True
        return False
    
    boundaries = np.array([
                        [-x_range[0],  1.,  0.],
                        [x_range[1], -1.,  0.],
                        [y_range[0],  0.,  1.],
                        [y_range[1],  0, -1.]])
    
    antecedents_text = 'RULE ANTECEDENTS\n'
    
    eq_list, inequalities, poly_class0, poly_class1, \
        inequalities_class0, inequalities_class1, \
        poly_global, all_zeros_winning_class, neuron_eqs, signs, \
        output_contrib_eqs, output_class_eq = \
                get_eq_list_new(config,config_samples, 
                                X_train, model, 
                                boundaries, decimals, 
                                y_train=y_train)
                
    if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
        inequ = np.array(inequalities_class0.array)
        antecedents_text += ' Class R (red)\n'

        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'

        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'
        
        pass
    else:
        antecedents_text += ' Class R (red)\n'
        antecedents_text += '  No Apply\n'


    if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
        inequ = np.array(inequalities_class1.array)
        antecedents_text += ' Class G (green)\n'

        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'

        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            antecedents_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'
    else:
        antecedents_text += ' Class G (green)\n'
        antecedents_text += '  No Apply\n'
    
    
    consequent_text = "RULE CONSEQUENT (Network Output)\n"
       
    for index, eq in enumerate(output_contrib_eqs):
        if subscript_map[index+1] == '1':
            y_text = 'Y_r'
        else:
            y_text = 'Y_g'
        consequent_text += f'  {y_text} = {str_equ_out(eq, decimals=decimals, normalize=False, web_output=True)}\n'

    # '''
    # Show Inequalities: global and per class
    # '''
    
    activation_region_text = ""
    
    if len(inequalities.array) > 0:
        inequ = np.array(inequalities.array)
        activation_region_text += 'ACTIVATION REGION (Classes R&G)\n'
        
        for index, eq in enumerate(inequ):
            if not eq_in_boundaries(eq, boundaries):
                continue
            activation_region_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'


        for index, eq in enumerate(inequ):
            if eq_in_boundaries(eq, boundaries):
                continue
            activation_region_text += f'  {str_equ_out(eq, decimals=decimals, return_full=True, orig_sign=1, web_output=True)}\n'


    print(antecedents_text)
    print(consequent_text)
    print(activation_region_text)
    
    pass
        
   
class QtImplicitEquationPlotter2():
    def __init__(self, configs, config_samples, color_count,
                 X_train, y_train, X_test, mlp_train,
                 train_accuracy, test_accuracy, model, 
                 point_size=3, title='Config sin definir',
                 polygon_color='gainsboro',
                 experiment='', epochs=5000, lr=0.0001, seed=33,
                 decimals=2, show_boundaries=False,
                 x_range=(-3, 3), y_range=(-3, 3),
                 show_graph=False, color_class='magenta',
                 light_point_size=2, light_point_alpha=0.4,
                 show_normalized=None,
                 test_configs=None,
                 test_color_count=None,
                 config_index=0):
        
        self.finish = False
        self.index = config_index
        self.plotter = None
    
        self.show_graph = show_graph
        self.color_class = color_class
        
        self.configs = configs
        self.test_configs = test_configs
        self.config_samples = config_samples
        self.color_count = color_count
        self.test_color_count = test_color_count
        self.X_train = X_train
        self.X_test = X_test
        
        if show_normalized is None:
            self.x_range = x_range
            self.y_range = y_range
        else:
            self.x_range = show_normalized[0]
            self.y_range = show_normalized[1]
        
        self.config_struct = self.get_config_structure(model)

        # Set colors: green = class 0, else red
        self.y_colors = np.where(y_train, 'green', 'red')
        self.y_colors_light = np.where(y_train, 'green', 'red')
        

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.model = model
        
        self.title = title
        self.point_size = point_size
        self.light_point_size = light_point_size
        self.light_point_alpha = light_point_alpha
        self.lr = lr
        self.epochs = epochs
        self.experiment = experiment
        self.seed = seed
        self.decimals = decimals
        self.show_boundaries = show_boundaries
        self.y_train = y_train
        
        self.boundaries = np.array([
                            [-self.x_range[0],  1.,  0.],
                            [self.x_range[1], -1.,  0.],
                            [-self.x_range[0],  0.,  1.],
                            [self.y_range[1],  0, -1.]])
        
        # Define symbolic variables
        self.x, self.y = sp.symbols('x y')
        
        # Initialize empty equations dictionary and scatter points
        self.equations = {}
        self.scatter_points = None
        self.scatter_points_light = None
        self.contour_sets = {}
        self.arrows_config = {}
        self.check_rows_dict = {}
        self.poly = {}
        
        self.H_inequalities = {}
        self.H_inequalities_class0 = {}
        self.H_inequalities_class1 = {}
        
        # Store button callback
        self.button_callback = None
    
        self.polygon_color = polygon_color
        
        
        # Create all GUI elements
        # self.setWindowTitle('Interactive Activation Pattern plot')
        # self.setMinimumSize(1000, 680)
        self.create_gui()
        
        self.subscript_map = '0123456789'
        
        self.replot(ini=True)
        self.update_plot()
        self.replot()
        

    def test(self, model, X_train, y_train, x_range, y_range,
                 config_samples, decimals,                     
                 config):
        
        def eq_in_boundaries(eq, boundaries):
            for boundary in boundaries:
                if np.allclose(eq, boundary):
                    return True
            return False
        
        boundaries = np.array([
                            [-self.x_range[0],  1.,  0.],
                            [self.x_range[1], -1.,  0.],
                            [-self.y_range[0],  0.,  1.],
                            [self.y_range[1],  0, -1.]])
        
        antecedents_text = 'RULE ANTECEDENTS\n'
        
        eq_list, inequalities, poly_class0, poly_class1, \
            inequalities_class0, inequalities_class1, \
            poly_global, all_zeros_winning_class, neuron_eqs, signs, \
            output_contrib_eqs, output_class_eq = \
                    get_eq_list_new(config,config_samples, 
                                    X_train, model, 
                                    boundaries, decimals, 
                                    y_train=y_train)
                    
        if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
            inequ = np.array(inequalities_class0.array)
            antecedents_text += ' Class R (red)\n'

            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'

            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'
            
            pass
        else:
            antecedents_text += ' Class R (red)\n'
            antecedents_text += '  No Apply\n'


        if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
            inequ = np.array(inequalities_class1.array)
            antecedents_text += ' Class G (green)\n'

            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'

            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'
        else:
            antecedents_text += ' Class G (green)\n'
            antecedents_text += '  No Apply\n'
        
        
        consequent_text = "RULE CONSEQUENT (Network Output)\n"
           
        for index, eq in enumerate(output_contrib_eqs):
            if self.subscript_map[index+1] == '1':
                y_text = 'Y_r'
            else:
                y_text = 'Y_g'
            consequent_text += f'  {y_text} = {self.str_equ(eq, decimals=self.decimals, normalize=False, web_output=True)}\n'

        # '''
        # Show Inequalities: global and per class
        # '''
        
        activation_region_text = ""
        
        if len(inequalities.array) > 0:
            inequ = np.array(inequalities.array)
            activation_region_text += 'ACTIVATION REGION (Classes R&G)\n'
            
            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                activation_region_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'


            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                activation_region_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'


        print(antecedents_text)
        print(consequent_text)
        print(activation_region_text)
        
        pass
    

        
    def get_config_structure(self, model, config=None):

        struct = ''
        config_output = ''
        pos = 0 
        
        for layer in model.layers[:-1]:
            if hasattr(layer, 'out_features'):
                if config is not None:
                    if pos != 0:
                        config_output += '-'
                    config_output += config[pos:pos+layer.out_features]
                    pos += layer.out_features
                else:
                    if pos != 0:
                        struct += '-'
                    struct += str(layer.out_features)
                    pos += 1

        if config is not None:
            return config_output
        else:
            return struct
        
        
    # # Function to handle mouse button press for panning
    # def on_press(self, event):
    #     if event.inaxes != self.ax:
    #         return
        
    #     if event.button == 1:  # Left mouse button
    #         self.pan_start[0] = event.xdata
    #         self.pan_start[1] = event.ydata
    
    # # Function to handle mouse button release
    # def on_release(self, event):
    #     if event.button == 1:  # Left mouse button
    #         self.pan_start[0] = None
    #         self.pan_start[1] = None

    # # Reset view with 'r' key
    # def on_key(self, event):
    #     if event.key == 'r':
    #         self.ax.set_xlim(-2, 5)
    #         self.ax.set_ylim(0, 1.1)
    #         self.canvas.draw_idle()

    # # Function to handle mouse motion during panning
    # def on_motion(self, event):
    #     if event.inaxes != self.ax or self.pan_start[0] is None:
    #         return
        
    #     # Calculate how much we've moved
    #     dx = event.xdata - self.pan_start[0]
    #     dy = event.ydata - self.pan_start[1]
        
    #     # Get current axis limits
    #     x_min, x_max = self.ax.get_xlim()
    #     y_min, y_max = self.ax.get_ylim()
        
    #     # Set new limits
    #     self.ax.set_xlim(x_min - dx, x_max - dx)
    #     self.ax.set_ylim(y_min - dy, y_max - dy)
        
    #     # Update the figure
    #     self.canvas.draw_idle()
        
    
    # def on_scroll(self, event):
    #     if event.inaxes != self.ax or event.key != 'control':
    #         return
        
    #     x_min, x_max = self.ax.get_xlim()
    #     y_min, y_max = self.ax.get_ylim()
        
    #     scale_factor = 1.2 if event.button == 'up' else 1/1.2
        
    #     # Calculate new limits centered on mouse position
    #     new_x_range = (x_max - x_min) / scale_factor
    #     new_y_range = (y_max - y_min) / scale_factor
        
    #     self.ax.set_xlim(event.xdata - new_x_range * (event.xdata - x_min) / (x_max - x_min),
    #                      event.xdata + new_x_range * (x_max - event.xdata) / (x_max - x_min))
    #     self.ax.set_ylim(event.ydata - new_y_range * (event.ydata - y_min) / (y_max - y_min),
    #                      event.ydata + new_y_range * (y_max - event.ydata) / (y_max - y_min))
    #     self.canvas.draw_idle()

    
    def create_gui(self):
        # --- Main Window Setup ---
        # self.central_widget = QtWidgets.QWidget()
        # self.setCentralWidget(self.central_widget)
        # self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # This horizontal container holds the three main pillars
        # self.top_container = QtWidgets.QHBoxLayout()
        
        
        # --- COLUMN 1: Matplotlib Figure (Left) ---
        # self.fig = Figure(figsize=(5, 5), dpi=100)
        # self.ax = self.fig.add_subplot(111)

        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        
        # self.canvas = FigureCanvas(self.fig)
        # self.top_container.addWidget(self.canvas, stretch=2)
               
        # --- COLUMN 2: Middle Box (Equations & Navigation) ---
        # self.middle_column_layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Samples Checkbox (Anchored to top)
        # self.scatter_checkbox = QtWidgets.QCheckBox("Samples")
        # self.scatter_checkbox.setStyleSheet("font-family: monospace; font-size: 12px;")
                                    
        # self.scatter_checkbox.setChecked(True)
        # self.scatter_checkbox.stateChanged.connect(self.update_plot)
        # self.middle_column_layout.addWidget(self.scatter_checkbox, stretch=0)
        
        # # 2. Scrollable Area for Dynamic Checkboxes (The "Lines Box")
        # self.check_scroll = QtWidgets.QScrollArea()
        # self.check_widget = QtWidgets.QWidget()
        # self.check_layout = QtWidgets.QVBoxLayout(self.check_widget)
        # self.check_layout.setAlignment(QtCore.Qt.AlignTop) 
        # self.check_scroll.setWidget(self.check_widget)
        # self.check_scroll.setWidgetResizable(True)
        # self.check_scroll.setFixedWidth(140)
        # self.middle_column_layout.addWidget(self.check_scroll, stretch=0)
        
        # self.check_scroll.setStyleSheet("QScrollArea { border: 1px solid #666666; }")
        
        # self.check_widget.setStyleSheet("background-color: #f1f1f1")
        
        # # 3. Navigation Buttons (Immediately below the box)
        # self.btn_layout = QtWidgets.QHBoxLayout()
        # self.btn_back = QtWidgets.QPushButton("◀")
        # self.btn_forward = QtWidgets.QPushButton("▶")
        
        # btn_style = "background-color: #4499FF; color: white; font-weight: bold; font-size: 14pt; padding: 2px;"
        # self.btn_back.setStyleSheet(btn_style)
        # self.btn_forward.setStyleSheet(btn_style)
        
        # # 1. Create a container widget for the buttons
        # nav_container = QtWidgets.QWidget()
        # nav_layout = QtWidgets.QHBoxLayout(nav_container)
        # nav_layout.setContentsMargins(0, 3, 0, 3) # Tight vertical spacing
        # nav_layout.setSpacing(10)                # Gap between the two buttons
        
        # self.btn_back.setFixedSize(45, 25)
        # self.btn_forward.setFixedSize(45, 25)
        
        # # 3. Build the layout: Stretch - Button - Button - Stretch
        # nav_layout.addWidget(self.btn_back)
        # nav_layout.addWidget(self.btn_forward)
        
        # # 4. Add the container to your main sidebar layout
        # self.middle_column_layout.addWidget(nav_container, stretch=0)
        
        # # 4. The Stretch (Pushes the whole group above it to the top)
        # self.middle_column_layout.addStretch(1)
        
        # # Add the completed middle pillar to the top container
        # self.top_container.addLayout(self.middle_column_layout)

        # # --- COLUMN 3: Info Panel (Right) ---
        # self.info_panel = QtWidgets.QTextEdit()
        # self.info_panel.setReadOnly(True)
        # # Narrowed from 300 to 250 to remove excess right space
        # self.info_panel.setFixedWidth(312)
        # self.top_container.addWidget(self.info_panel)
        
        # # Add the top container (Plot + Middle + Right) to main layout
        # self.main_layout.addLayout(self.top_container, stretch=3)


        # # --- 4. Bottom Text Panels (Inequalities & Rules) ---
        # self.bottom_layout = QtWidgets.QHBoxLayout()
        # self.left_text = QtWidgets.QTextEdit()
        # self.middle_text = QtWidgets.QTextEdit()
        # self.right_text = QtWidgets.QTextEdit()
        

        # self.left_text.setReadOnly(True)
        # self.middle_text.setReadOnly(True)
        # self.right_text.setReadOnly(True)
        
        # min_height_bottom = 235
        
        # self.left_text.setMinimumHeight(min_height_bottom)
        # self.middle_text.setMinimumHeight(min_height_bottom)
        # self.right_text.setMinimumHeight(min_height_bottom)
        
        # # Sync scrollbars
        # self.left_text.verticalScrollBar().valueChanged.connect(
        #     self.right_text.verticalScrollBar().setValue
        # )
        # self.middle_text.verticalScrollBar().valueChanged.connect(
        #     self.middle_text.verticalScrollBar().setValue
        # )
        # self.right_text.verticalScrollBar().valueChanged.connect(
        #     self.left_text.verticalScrollBar().setValue
        # )
        
        # self.middle_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.middle_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # self.middle_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.middle_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # self.right_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.right_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # self.bottom_layout.addWidget(self.left_text)
        # self.bottom_layout.addWidget(self.middle_text)
        # self.bottom_layout.addWidget(self.right_text)
        
        # self.main_layout.addLayout(self.bottom_layout)

        # # --- Final Signal Connections ---
        # self.btn_forward.clicked.connect(lambda: self.handle_action("forward"))
        # self.btn_back.clicked.connect(lambda: self.handle_action("backward"))        
        
        # Initialize variable dictionary for checkbuttons
        self.var_dict = {}
                
    
    def clear_ui_elements(self):
        # Clear the dictionary
        self.check_dict = {}
        self.line_dict = {}
        # Clear the layout widgets
        # while self.check_layout.count():
        #     item = self.check_layout.takeAt(0)
        #     widget = item.widget()
        #     if widget:
        #         widget.deleteLater()

        
    # def _on_scrollbar(self, *args):
    #     """Handle scrollbar movement"""
    #     self.left_text.yview(*args)
    #     self.middle_text.yview(*args)
    #     self.right_text.yview(*args)

    # def _left_scroll_set(self, *args):
    #     """Handle left text scrolling"""
    #     self.scrollbar.set(*args)
    #     self.right_text.yview_moveto(args[0])
    
    # def _right_scroll_set(self, *args):
    #     """Handle right text scrolling"""
    #     self.scrollbar.set(*args)
    #     self.left_text.yview_moveto(args[0])
        
        
    def clear_equations(self):
        """Surgically remove all dynamic checkboxes from the sidebar."""
        self.equations.clear()
        self.var_dict.clear()
        self.arrows_config.clear()
        self.contour_sets.clear()
        self.poly.clear()
        
        # We must use list() to avoid iteration errors while deleting
        # if hasattr(self, 'row_widgets'):
        #     for name in list(self.row_widgets.keys()):
        #         widget = self.row_widgets.pop(name)
        #         widget.setParent(None)
        #         widget.deleteLater()
        
        self.color_index = 0 
        self.replot()
        self.update_plot()
        
            
    def str_equ(self, eq, decimals=2, return_full=False, 
                orig_sign=-2, normalize=True, web_output=False):
        
        eq = eq.copy()
        
        eq[abs(eq) < 1e-10] = 0
        
        sign = 1
        
        if eq[1] != 0:
            if eq[1] < 0:
                sign = -1
            if normalize:
                eq = eq / eq[1]
        elif eq[2] != 0:
            if eq[2] < 0:
                sign = -1
            if normalize:
                eq = eq / eq[2]
               
        
        n_coefs = len(eq[1:])
        ret = ''
        first = True
        for i in range(n_coefs):
            val = eval(f'{eq[i+1]:.{decimals}f}')
            if val == 0:
                # first = False
                continue
            elif val == 1:
                if not first:
                    ret += f' + {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'x{self.subscript_map[i+1]} '
            elif val == -1:
                if not first:
                    ret += f'- {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'-x{self.subscript_map[i+1]} '
            else:
                if val < 0:
                    if first:
                        ret += '-'
                    else:
                        ret += '- '
                else:
                    if not first:
                        ret += '+ '
                ret += f'{abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
            first = False
        if first:
            if eq[0] < 0:
                ret += f'{eq[0]:.{decimals}f}'
            else:
                ret += f'{abs(eq[0]):.{decimals}f}'
        else:
            if eq[0] < 0:
                if not np.isclose(eq[0], 0.0):
                    if not web_output:
                        ret += f'&nbsp;- {abs(eq[0]):.{decimals}f}'
                    else:
                        ret += f'- {abs(eq[0]):.{decimals}f}'
            else:
                if not np.isclose(eq[0], 0.0):
                    ret += f'+ {abs(eq[0]):.{decimals}f}'
                
        if return_full:
            r = sign * orig_sign
            ret += f' {">= 0" if r > 0 else "<= 0"}'
            
        return ret
    
    
    
    # def append_bottom_text(self, text, no_trad=False, bold=False, pos=None):
    #     """Append text to the bottom text content."""
        
        
    #     letters = 'xyz'

    #     if pos is None:
    #         # Get the number of lines in each text widget
    #         left_lines = int(self.left_text.index('end-1c').split('.')[0])
    #         # right_lines = int(self.right_text.index('end-1c').split('.')[0])
            
    #         # Calculate visible lines (approximate based on height)
    #         visible_lines = self.left_text.winfo_height() // 19  # Approximate line height
            
    #         text_zone = None
    #         if left_lines <= visible_lines:
    #             # If left column has space or has fewer lines than right, add to left
    #             text_zone = self.left_text
    #         else:
    #             # Otherwise add to right column
    #             text_zone = self.right_text
    #     else:
    #         if pos == 'left':
    #             text_zone = self.left_text
    #         elif pos == 'middle' or pos == 'center':
    #             text_zone = self.middle_text
    #         else: # pos == 'right':
    #             text_zone = self.right_text


    #     new_text = ''
        
    #     self.left_text.config(state="normal")
    #     self.middle_text.config(state="normal")
    #     self.right_text.config(state="normal")
        
    #     if no_trad:
    #         new_text = text
    #         if bold:
    #             text_zone.tag_configure('bold', font=('Arial', 11, 'bold'))
    #             text_zone.insert(tk.END, new_text + '\n', 'bold')

    #         else:
    #             text_zone.tag_configure('normalX', font=('Arial', 11))
    #             text_zone.insert(tk.END, new_text + '\n', 'normalX')

    #     else:
    #         must_continue = False
    #         for i, t in enumerate(text):
    #             if must_continue:
    #                 must_continue = False
    #                 continue 
                
    #             if t == '*':
    #                 continue
             
    #             if t == '_':
    #                 pos1 = '0123456789'.find(text[i+1])
    #                 if pos1 >= 0:
    #                     must_continue = True
    #                     # new_text += f'{subscript_map[pos1]}'
    #                     new_text += f'{letters[pos1]}'
    #             else:
    #                 pos = letters.find(t)
    #                 if pos >= 0:
    #                     new_text += f' {letters[pos]}'
    #                 else:
    #                     new_text += t

    #         text_zone.tag_configure('normalX', font=('Arial', 11))
    #         text_zone.insert(tk.END, new_text + '\n', 'normalX')
        
        
    #     self.left_text.config(state="disabled")
    #     self.middle_text.config(state="disabled")
    #     self.right_text.config(state="disabled")
        
    
    # def append_text(self, text, bold=False, reverse=False, center=False):
    #     """Append text to the existing content.
        
    #     Args:
    #         text (str): Text to append
    #     """
    #     if bold:
    #         new_text = text
    #         if center:
    #             new_text = text + '\n'
                
    #         self.text_widget.config(state="normal")
            
            
    #         if reverse:
    #             self.text_widget.tag_configure('bold-reverse', font=('Courier New', 10, 'bold'), 
    #                                            background='yellow', foreground='black')
    #         else:
    #             self.text_widget.tag_configure('bold', font=('Courier New', 10, 'bold'))

    #         if center:        
    #             self.text_widget.tag_configure('center', justify='center', 
    #                                            font=('Courier New', 10, 'bold'))
    #             self.text_widget.insert(tk.END, new_text, 'center')
    #         else:
    #             if reverse:
    #                 self.text_widget.insert(tk.END, new_text, 'bold-reverse')
    #             else:
    #                 self.text_widget.insert(tk.END, new_text, 'bold')

    #             self.text_widget.insert(tk.END, '\n')
                

    #         self.text_widget.config(state="disabled")
    #     else:
    #         self.text_widget.config(state="normal")
    #         self.text_widget.insert(tk.END, text + '\n')
    #         self.text_widget.config(state="disabled")    
        
        
        
    def set_scatter_points(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points = np.array(list(points))
        self.scatter_colors = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        # self.scatter_checkbox.setEnabled(True)
        self.update_plot()

        
    def set_scatter_points_light(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points_light = np.array(list(points))
        self.scatter_colors_light = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        # self.scatter_checkbox.setEnabled(True)
        self.update_plot()
            
        
        
    def clear_scatter_points(self):
        """Remove scatter points from data and reset the UI checkbox."""
        self.scatter_points = None
        self.scatter_colors = None
        
        # Replace self.scatter_var.set(False) with the direct widget method
        # self.scatter_checkbox.setChecked(False)
        
        # Disable the checkbox since there is no data to show
        # self.scatter_checkbox.setEnabled(False)
        self.update_plot()
        
        
    def add_equation(self, name, equation, color, linestyle, direction, linewidth=3):
        """Add a new equation to the plotter.
        
        Args:
            name (str): Name of the equation to display
            equation (sympy.Eq): Sympy equation instance
        """
        if not isinstance(equation, sp.Eq):
            raise ValueError("Equation must be a sympy.Eq instance")
            
        # Add equation to dictionary
        self.equations[name] = (equation, color, linestyle, direction)
        
        
        # row_widget = QtWidgets.QWidget()

        # row_layout = QtWidgets.QHBoxLayout(row_widget)
        
        # row_layout.setContentsMargins(3, 1, 3, 1)
        # row_layout.setSpacing(1)
        
        # checkbox = QtWidgets.QCheckBox(name.replace('Eq', 'HS'))
        # checkbox.setChecked(True)
        # checkbox.stateChanged.connect(self.update_plot)
        # checkbox.setStyleSheet("""
        #                            QCheckBox {
        #                                 font-family: monospace;
        #                                 font-size: 11px;
        #                             }
        #                        """)
        
        # # Pass the linestyle and linewidth

        # line_visual = LineStyleWidget(color, linestyle, linewidth, parent=row_widget)
     
        # row_layout.addWidget(checkbox)
        # row_layout.addStretch(1)
        # row_layout.addWidget(line_visual)
        
        # self.check_layout.addWidget(row_widget)

        # self.check_dict[name] = checkbox
        # self.line_dict[name] = line_visual
        # self.check_rows_dict[name] = row_widget
        
        
        # # Create new checkbutton variable
        var = BooleanVarReplica(value=True)
        self.var_dict[name] = var
        
    def remove_equation(self, name):
        """Remove an equation from the plotter with surgical precision."""
        if name in self.equations:
            # Remove data references
            del self.equations[name]
            del self.var_dict[name]
    
            # Retrieve the container widget we stored earlier
            # if name in self.row_widgets:
            #     widget_to_kill = self.row_widgets.pop(name)
                
            #     # This is the Qt magic: remove it from the UI and schedule it for deletion
            #     widget_to_kill.setParent(None)
            #     widget_to_kill.deleteLater()
    
            # The layout automatically shifts everything up!
            self.update_plot()
         
    
    def create_implicit_function(self, equation):
        expr = equation.lhs - equation.rhs
        
        def f(x, y):
            return sp.lambdify((self.x, self.y), expr, 'numpy')(x, y)
        
        return f
    
    def plot_implicit(self, f, x_range, y_range, resolution=100, color='black', linewidth=1, linestyles='solid'):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = f(X, Y)
        
        ret = self.ax.contour(X, Y, Z, levels=[0], colors=color, linewidths=linewidth, linestyles=linestyles)
        
        return ret
    
  
    def calculate_normal_vector(self, point, current_index, vertices, direction=1):        
        """Calculate the truly orthogonal vector using adjacent points"""
        n_points = len(vertices)
        
        # Get points from both sides (5 points away is enough)
        prev_index = max(current_index - 5, 0)
        next_index = min(current_index + 5, n_points - 1)
        
        # Get points
        prev_point = vertices[prev_index]
        next_point = vertices[next_index]
        
        # Calculate tangent using points from both sides
        tangent = next_point - prev_point
        tangent = tangent / np.linalg.norm(tangent)
        
        # Calculate normal vector (rotate tangent 90 degrees)
        normal = np.array([-tangent[1], tangent[0]]) * direction
        
        return normal
    
    def calculate_normal_vector2(self, point, current_index, vertices, direction=1):
        """Calculate orthogonal vector with specific handling for high curvature points"""
        n_points = len(vertices)
        
        # First, check if we're at a point with high curvature by analyzing local geometry
        # Get immediate neighbors
        prev_index = max(current_index - 1, 0)
        next_index = min(current_index + 1, n_points - 1)
        
        prev_point = vertices[prev_index]
        next_point = vertices[next_index]
        current_point = point  # The current point
        
        # Calculate directions of adjacent segments
        prev_dir = np.zeros(2)
        next_dir = np.zeros(2)
        
        if prev_index != current_index:
            prev_dir = current_point - prev_point
            prev_norm = np.linalg.norm(prev_dir)
            if prev_norm > 1e-10:
                prev_dir = prev_dir / prev_norm
        
        if next_index != current_index:
            next_dir = next_point - current_point
            next_norm = np.linalg.norm(next_dir)
            if next_norm > 1e-10:
                next_dir = next_dir / next_norm
        
        # Calculate dot product to measure alignment between segments
        dot_product = np.dot(prev_dir, next_dir)
        
        # Detect if we're at a corner/high curvature point
        # The threshold can be adjusted based on your specific needs
        if dot_product < 0.7:  # Less than ~45 degrees between segments
            # We're at a high curvature point - use just the local segment
            # For magenta line specifically, we can use the next segment direction
            tangent = next_dir
            
            # Ensure we have a valid tangent
            if np.linalg.norm(tangent) < 1e-10:
                # Fallback if tangent is too small
                tangent = np.array([1.0, 0.0])
        else:
            # For smoother sections, use a moderate window (not too large)
            # Using 2 points instead of 5 to stay more local
            smooth_prev_index = max(current_index - 2, 0)
            smooth_next_index = min(current_index + 2, n_points - 1)
            
            smooth_prev_point = vertices[smooth_prev_index]
            smooth_next_point = vertices[smooth_next_index]
            
            # Calculate tangent
            tangent = smooth_next_point - smooth_prev_point
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-10:
                tangent = tangent / tangent_norm
            else:
                # Fallback to a default direction
                tangent = np.array([1.0, 0.0])
        
        # Calculate the normal vector (rotate tangent 90 degrees)
        normal = np.array([-tangent[1], tangent[0]]) * direction
        
        return normal
    
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    
    def find_valid_position(self, vertices, min_distance=0.5):
        """Find a random position that's far enough from existing arrows"""
        max_attempts = 50  # Prevent infinite loop
        existing_points = []
        
        # Collect existing arrow points
        for eq_config in self.arrows_config.values():
            if 'point' in eq_config:
                existing_points.append(eq_config['point'])
        
        for _ in range(max_attempts):
            # Generate random index
            random_index = np.random.randint(0, len(vertices))
            point = vertices[random_index]
            
            # Check distance from all existing points
            is_valid = True
            for existing_point in existing_points:
                if self.calculate_distance(point, existing_point) < min_distance:
                    is_valid = False
                    break
                    
            if is_valid:
                return random_index, point
                
        # If no valid position found, take the last generated one
        return random_index, point
    
    
    def add_orthogonal_arrow(self, eq_name, arrow_length=0.5, direction=1, color='green'):
        """Add arrow configuration for a specific equation"""
        
        if eq_name in self.contour_sets:
            paths = self.contour_sets[eq_name].allsegs[0]
            if paths and len(paths[0] > 0):
                vertices = paths[0]
                random_index, point = self.find_valid_position(vertices)
                
                self.arrows_config[eq_name] = {
                    'length': arrow_length,
                    'index': random_index,
                    'point': point,
                    'direction': direction,
                    'color': color
                }
                self.update_plot()
                
                
    def add_polygon(self, poly_name, vertices, fill_color='gainsboro', 
                    edge_color='green', alpha=0.8, linewidth=1):
        if vertices == None or len(vertices) == 0: 
            return
        
        x_coords, y_coords = [], []
        for vertex in vertices:
            if vertex[0] == 1:
                x_coords.append(vertex[1])
                y_coords.append(vertex[2])
 
        self.poly[poly_name] = {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'fill_color': fill_color,
            'edge_color': edge_color,
            'alpha': alpha,
            'linewidth': linewidth
        }
        
        return
    
        

    def order_vertices(self, vertices):
        # Calculate centroid
        center = np.mean(vertices, axis=0)
        
        # Calculate angles of points with respect to center
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        
        # Sort vertices based on these angles        
        sorted_indices = np.argsort(angles)
        
        ordered = vertices[sorted_indices]
    
        return ordered
    
    def longest(self, vertices):
        longest = 0
        for v1 in vertices:
            for v2 in vertices:
                l = math.dist(v1, v2)
                if longest < l:
                    longest = l
        return longest
    
    def area(self, vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0
        return area
    
    def plot_polygon(self):
        # Separate x and y coordinates

        if self.poly == None:
            return
        
        for poly in self.poly.values():
            vertices = np.array([poly['x_coords'], poly['y_coords']]).T
            
            vertices = self.order_vertices(vertices)

            '''
            Fill polygon with transparency
            '''
            longest_distance = self.longest(vertices)
            area = self.area(vertices)

            adjust = (self.x_range[1] - self.x_range[0]) / 6

            if area / (longest_distance * adjust) > 0.15:
                self.ax.fill(vertices[:,0], 
                             vertices[:,1], 
                             facecolor=poly['fill_color'],
                             alpha=poly['alpha']
                             )
            else:        
                self.ax.fill(vertices[:,0], 
                             vertices[:,1], 
                             edgecolor=poly['fill_color'], # 'red'
                             alpha=poly['alpha'],
                             linewidth=12,
                             linestyle='-'
                             )
                
        return self.ax    


    
    def update_plot(self):
        self.ax.clear()
        self.contour_sets.clear()

        if self.show_graph:
            x_range = self.x_range
            y_range = self.y_range
    
            # Ensure equal scaling
            self.ax.set_aspect('equal')
            self.ax.set_xlim(x_range)
            self.ax.set_ylim(y_range)
            
            self.ax.tick_params(axis='both', which='major', labelsize=9) 
            self.ax.set_xlabel(r'$\mathbf{x_1}$', fontsize=14, labelpad=-8)
            self.ax.set_ylabel(r'$\mathbf{x_2}$', fontsize=14, labelpad=-15)
                               
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0)
            
            for eq_name, var in self.var_dict.items():
                if var.get():
                    equation = self.equations[eq_name][0]
                    f = self.create_implicit_function(equation)
                    
                    linestyles = 'solid'
                    linewidth = 1.6  
                    new_color = self.equations[eq_name][1]
                    direction = self.equations[eq_name][3]
                    
                    if eq_name.startswith('Eq9'):
                        linewidth = 2
                        new_color = self.color_class
                        linestyles = 'dashdot'
                    elif eq_name[2] in ['1', '2', '3', '4']:
                        linestyles = self.equations[eq_name][2]
                    else:
                        linestyles = self.equations[eq_name][2]
                        linewidth = 2
                        
                    contour = self.plot_implicit(f, x_range, y_range, 
                                               resolution=200, 
                                               color=new_color,
                                               linewidth=linewidth,
                                               linestyles=linestyles)
                    
                    self.contour_sets[eq_name] = contour
      
                    if eq_name in self.arrows_config:
                        paths = contour.allsegs[0]
                        if paths:
                            vertices = paths[0]
                            point_index = self.arrows_config[eq_name]['index']
                            point = vertices[point_index]
                            
                            normal = self.calculate_normal_vector(point, point_index, vertices,
                                        direction=self.arrows_config[eq_name].get('direction', direction))
                            
                            arrow_length = self.arrows_config[eq_name]['length']
                            dx = normal[0] * arrow_length
                            dy = normal[1] * arrow_length
                            
                            adjust = (self.x_range[1] - self.x_range[0]) / 6
                            
                            self.ax.arrow(point[0], point[1], 
                                        dx * adjust, dy * adjust,
                                        head_width=0.1 * adjust,
                                        head_length=0.1 * adjust,
                                        fc=self.equations[eq_name][1],
                                        ec=self.equations[eq_name][1],
                                        length_includes_head=True,
                                        width=0.05 * adjust,
                                        zorder=1000)
    
        
                self.plot_polygon()
                
                config = list(dict(self.configs).keys())[self.index]
        
            if self.scatter_points_light is not None:
                self.ax.scatter(self.scatter_points_light[:, 0], self.scatter_points_light[:, 1], 
                                c=self.scatter_colors_light, s=self.light_point_size, 
                                zorder=0, alpha=self.light_point_alpha)

            if self.scatter_points is not None:
                self.ax.scatter(self.scatter_points[:, 0], self.scatter_points[:, 1], 
                                c=self.scatter_colors, s=self.point_size, zorder=5)
        
            self.ax.set_title(self.title, pad=7, fontsize=10)
            
            self.fig.tight_layout()
            # self.canvas.draw()
            
            pass
        
    def remove_orthogonal_arrow(self, eq_name):
        """Remove arrow from a specific equation"""
        if eq_name in self.arrows_config:
            del self.arrows_config[eq_name]
            self.update_plot()
    
    def clear_all_arrows(self):
        """Remove all arrows"""
        self.arrows_config.clear()
        self.update_plot()
    
    
    # def set_button_callback(self, callback):
    #     """Set the callback function for both buttons"""
    #     self.button_callback = callback
    #     self.forward_button.configure(command=lambda: callback("forward"))
    #     self.backward_button.configure(command=lambda: callback("backward"))


    # def on_closing(self):
    #     """Handle window closing using the same callback"""
    #     if self.button_callback:
    #         self.button_callback("end")
            
    #     self.root.destroy()
    
    def handle_action(self, action):
        if action == "forward":
            self.index += 1
            if self.index == len(self.configs):
                self.index = 0
            
            self.clear_equations()
            self.replot()
            
            self.export_pdf() 
            
            self.fig.savefig('classification_output/classifcation_output_canvas.svg', bbox_inches='tight') #, pad_inches=0.1)

            
        elif action == "backward":
            self.index -= 1
            if self.index < 0:
                self.index = len(self.configs) - 1
    
            self.clear_equations()
            self.replot()
            
        elif action == "end":
            self.finish = True
    

    def eq_in_boundaries(self, eq):
        for boundary in self.boundaries:
            if np.allclose(eq, boundary):
                return True
        return False
    
    def replot(self, ini=False):        
        config = list(dict(self.configs).keys())[self.index]
      
        # self.right_text.clear()
        # self.middle_text.clear()
        # self.left_text.clear()
        # self.info_panel.clear()
        self.title = f'Activation Pattern {self.get_config_structure(self.model, config)}'
        
        # # --- Build one single HTML string for the sidebar ---
        # html = []
        
        # # 1. Centered Header
        # html.append("<div style='font-family: monospace; text-align: center; font-size: 10pt; margin-bottom: 10px;'>")
        # html.append("<b>TOY EXPERIMENT ON 3×3 CHECKERBOARD<br>CLASSIFICATION DATASET</b>")
        # html.append("</div>")
        
        # # 2. Left-aligned Statistics
        # html.append("<div style='text-align: left; font-family: monospace; font-size: 9pt;'>")
        # html.append(f"# Neurons = {self.config_struct}<br>")
        # html.append(f"# Activation Patterns = {len(self.configs)}<br>")
        # html.append(f"# Samples = {len(self.X_train)} (train)&nbsp;&nbsp;{len(self.X_test)} (test)<br>")
        # html.append(f"# Epochs = {self.epochs}<br>")
        # html.append(f"Learning Rate = {self.lr}<br>")
        # html.append(f"Train Accuracy&nbsp;=&nbsp;{self.train_accuracy:.04f}<br>")
        # html.append(f"Test Accuracy&nbsp;&nbsp;=&nbsp;{self.test_accuracy:.04f}<br>")
        # html.append(f"Seed = {self.seed}<br>")
        # html.append("</div>")
    

        # # 3. Helper for perfectly aligned tables
        # def get_table_html_old(title_text, config_dict, rmse_dict, active_c):
        #     # We wrap the title in a span with a larger font size (e.g., 12pt)
        #     t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
        #     # The table itself remains standard size for precision
        #     t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
        #     t_html += "<tr><th width='90' align='left'>Pattern</th><th width='70' align='right'>#Samples</th><th width='80' align='right'>RMSE</th></tr>"
            
        #     for c, count in config_dict.items():
        #         bg = "background-color: #FFFF22;" if c == active_c else ""
        #         t_html += f"<tr style='{bg}'>"
        #         t_html += f"<td>{c[:4]}-{c[4:]}</td>"
        #         t_html += f"<td align='right'>{count}</td>"
        #         t_html += "</tr>"
        #     t_html += "</table><br>"
        #     return t_html

        # def get_table_html(title_text, config_dict, color_dict, config):
        #     # We wrap the title in a span with a larger font size (e.g., 12pt)
        #     t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
        #     # The table itself remains standard size for precision
        #     t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
        #     t_html += "<tr><th width='70' align='left'>Pattern</th><th width='65' align='right'>#Samples</th><th width='45' align='right'>TP</th><th width='40' align='right'>FP</th><th width='40' align='right'>TN</th><th width='40' align='right'>FN</th></tr>"
            
        #     for c, count in config_dict.items():
        #         bg = "background-color: #FFFF22;" if c == config else ""
        #         t_html += f"<tr style='{bg}'>"
        #         t_html += f"<td>{c[:4]}-{c[4:]}</td>"
        #         t_html += f"<td align='right'>{count}</td>"
        #         t_html += f"<td align='right'>{color_dict[c]['TP']}</td>"
        #         t_html += f"<td align='right'>{color_dict[c]['FP']}</td>"
        #         t_html += f"<td align='right'>{color_dict[c]['TN']}</td>"
        #         t_html += f"<td align='right'>{color_dict[c]['FN']}</td>"
        #         t_html += "</tr>"
        #     t_html += "</table><br>"
        #     return t_html

        # html.append(get_table_html("Train samples", self.configs, self.color_count, config))
        # html.append(get_table_html("Test samples", self.test_configs, self.test_color_count, config))
        
        # # --- Set the HTML once ---
        # self.info_panel.setHtml("".join(html))       
        
        eq_list, inequalities, poly_class0, poly_class1, inequalities_class0, inequalities_class1, poly_global, all_zeros_winning_class, neuron_eqs, signs, output_contrib_eqs, output_class_eq = \
                            get_eq_list_new(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)
                            # get_eq_list(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)

        self.H_inequalities[config] = inequalities
        self.H_inequalities_class0[config] = inequalities_class0
        self.H_inequalities_class1[config] = inequalities_class1
        
        
        color_index = 0
        color_list = ['blue', 'olive', 'brown', 'orange', 'lightblue', 'pink', 'gray', 'lightseagreen']


        # --- 6. Bottom Panels: HTML Construction ---

        # A. LEFT PANEL: INEQUALITIES
        # We start with the Title
        # left_html = "<div style='font-family: monospace; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
        
        # # Line 2: The Global Domain Range (separated by <br>)
        # left_html += f"&nbsp;&nbsp;{self.x_range[0]:.0{self.decimals}f} <= x{self.subscript_map[1]} <= {self.x_range[1]:.0{self.decimals}f}<br>"
        # left_html += f"&nbsp;&nbsp;{self.y_range[0]:.0{self.decimals}f} <= x{self.subscript_map[2]} <= {self.y_range[1]:.0{self.decimals}f}<br>"
        
        # # # Line 3+: Each equation on a new line
        # for index, eq in enumerate(neuron_eqs):
        #     bit = config[index]
        #     # Use orig_sign to ensure the math string shows >= 0 or <= 0 correctly
        #     eq_str = self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=int(bit))
        #     left_html += f"&nbsp;&nbsp;N={bit}, HS{index+1}: {eq_str}<br>"

        # out_text = f'&nbsp;&nbsp;Y_r >= Y_g, HS9: {self.str_equ(eq, decimals=self.decimals)} {">= 0" if signs[index] > 0 else "<= 0"}'
        
        # left_html += out_text
        
        # left_html = left_html.replace('<=', '&le;').replace('>=', '&ge;')

        # left_html += '</div>'

        # # Set the left panel
        # self.left_text.setHtml(left_html)

        if all_zeros_winning_class >= 0:
            if all_zeros_winning_class == 0:
                poly_class0 = poly_global
                poly_global = []
                inequalities_class0 = inequalities 
            elif all_zeros_winning_class == 1:
                poly_class1 = poly_global
                poly_global = []
                inequalities_class1 = inequalities
                
        
        # --- 6.1. Middle Panels: HTML Construction ---

        # We start with the Title
        # middle_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE ANTECEDENTS</b><br>"
        
        # if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
        #     inequ = np.array(inequalities_class0.array)
        #     middle_html += '&nbsp;\u2022 <b>Class R (red)</b><br>'

        #     if self.show_boundaries:
        #         for index, eq in enumerate(inequ):
        #             if not self.eq_in_boundaries(eq):
        #                 continue
        #             middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'

        #     for index, eq in enumerate(inequ):
        #         if self.eq_in_boundaries(eq):
        #             continue
        #         middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'
            
        #     pass
        # else:
        #     middle_html += '&nbsp;\u2022 <b>Class R (red)</b><br>'
        #     middle_html += '&nbsp;&nbsp;No Apply<br>'


        # if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
        #     inequ = np.array(inequalities_class1.array)
        #     middle_html += '<br>&nbsp;\u2022 <b>Class G (green)</b><br>'

        #     if self.show_boundaries:
        #         for index, eq in enumerate(inequ):
        #             if not self.eq_in_boundaries(eq):
        #                 continue
        #             middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'

        #     for index, eq in enumerate(inequ):
        #         if self.eq_in_boundaries(eq):
        #             continue
        #         middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'
        # else:
        #     middle_html += '<br>&nbsp;\u2022 <b>Class G (green)</b><br>'
        #     middle_html += '&nbsp;&nbsp;No Apply<br>'


        # middle_html = middle_html.replace('<=', '&le;').replace('>=', '&ge;')

        # middle_html += '</div>'

        # # Set the middle panel
        # self.middle_text.setHtml("".join(middle_html))
        
        # right_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE CONSEQUENT (Network Output)</b><br>"
           
        # for index, eq in enumerate(output_contrib_eqs):
        #     if self.subscript_map[index+1] == '1':
        #         y_text = 'Y_r '
        #     else:
        #         y_text = 'Y_g'
        #     right_html += f'&nbsp;&nbsp;{y_text} = {self.str_equ(eq, decimals=self.decimals, normalize=False)}<br>'

        # # '''
        # # Show Inequalities: global and per class
        # # '''
        # if len(inequalities.array) > 0:
        #     inequ = np.array(inequalities.array)
        #     right_html += '<br><b>ACTIVATION REGION (Classes R&G)</b><br>'
            
        #     if self.show_boundaries:
        #         for index, eq in enumerate(inequ):
        #             if not self.eq_in_boundaries(eq):
        #                 continue
        #             right_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'


        #     for index, eq in enumerate(inequ):
        #         if self.eq_in_boundaries(eq):
        #             continue
        #         right_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'


        # right_html = right_html.replace('<=', '&le;').replace('>=', '&ge;')
        
        # right_html += '</div>'
        # # Set the right panel
        # self.right_text.setHtml("".join(right_html))
        
        if self.show_graph:
            
            self.clear_ui_elements()

            self.color_index = 0 
            
            '''
            Add polygons, scatter points, lines, and arrows
            '''
            for index, eq in enumerate(eq_list):
                if eq[1] == False:
                    print(f'EQ[1] {index} = FALSE!!!')
                    continue
                
                if eq[2] != -2 and eq[1] is None: 
                    # self.append_bottom_text(f'{eq[0]} = 0x + 0y + b = 0', no_trad=True)
                    pass
                else:
                    if eq[2] == 0:
                        '''
                        Para la ecuación de clase, siempre marcamos la dirección 
                        hacia la clase 0 (roja)
                        '''
                        self.add_equation(eq[0], eq[1], self.color_class, 
                                          linestyle='dashdot', direction=0)
                        self.add_orthogonal_arrow(eq[0], arrow_length=0.3, 
                                                  # direction=eq[4], # eq[4] 
                                                  direction=1, # eq[4] 
                                                  color=self.color_class)
                    elif eq[2] == -2: 
                        pass
                    else:
                        if index < self.model.layers[0].out_features:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='solid', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=0.3, direction=eq[2], 
                                                      color=color_list[color_index])
                        else:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='dashed', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=0.3, direction=eq[2], 
                                                      color=color_list[color_index])
    
    
            if inequalities_class0 != []:
                self.add_polygon('Class 1', poly_class0, fill_color='lightsalmon', alpha=0.15)
    
            if inequalities_class1 != []:
                self.add_polygon('Class 2', poly_class1, fill_color='lightgreen', alpha=0.15)
    
            self.set_scatter_points_light(self.X_train,
                                    self.y_colors_light)

            self.set_scatter_points(self.X_train[self.config_samples[config]],
                                    self.y_colors[self.config_samples[config]])
            
            # self.clear_equations()
            # self.replot()


        
    def export_pdf(self, filename='classification_output/export_classification_clean.pdf'):
        # Widgets to hide while exporting
        widgets_to_hide = [self.canvas, self.btn_forward, self.btn_back]
        # widgets_to_hide = [self.canvas]
    
        # Freeze sizes for hidden widgets
        for w in widgets_to_hide:
            w.setFixedSize(w.size())
            w.hide()
    
        # Configure printer (unchanged)
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filename)
        
        # Page rectangle in device (printer) pixels
        page_rect = printer.pageRect()  # QRect in device pixels
        
        # Compute scale to map widget coordinates to page coordinates (widget -> printer device pixels)
        widget_w = float(self.width())
        widget_h = float(self.height())
        scale = min(page_rect.width() / widget_w, page_rect.height() / widget_h)
        
        # Create painter on printer early to query device pixel ratio
        painter = QtGui.QPainter(printer)
        device_ratio = painter.device().devicePixelRatioF() \
                        if hasattr(painter.device(), "devicePixelRatioF") else 1.0
        
        # device_scale maps widget pixels -> device pixels used by the painter
        device_scale = scale * device_ratio
        
        # Create a high-res pixmap sized in device pixels
        pixmap_size = QtCore.QSize(int(widget_w * device_scale), int(widget_h * device_scale))
        
        pix = QtGui.QPixmap(pixmap_size)
        pix.fill(QtCore.Qt.transparent)
        
        # If using devicePixelRatio, set it so drawPixmap behaves correctly
        if hasattr(pix, "setDevicePixelRatio"):
            pix.setDevicePixelRatio(device_ratio)
        
        # Render the widget into the pixmap (use device_scale but render in widget coordinates)
        p_pix = QtGui.QPainter(pix)
        p_pix.setRenderHints(QtGui.QPainter.Antialiasing | 
                             QtGui.QPainter.TextAntialiasing | 
                             QtGui.QPainter.SmoothPixmapTransform)
        p_pix.scale(scale, scale)
        self.render_widget_recursive(self, p_pix)
        
        self.check_widget.adjustSize()
        self.check_widget.ensurePolished()

        # Start with the margin/spacing of your layout
        current_y = self.check_widget.layout().contentsMargins().top()
        spacing = self.check_widget.layout().spacing()
        
        offset_x = 23 #XXXX6  # Moves the line to the right
        offset_y = -195 # Moves the line up

        for name, line_widget in self.line_dict.items():
            row_widget = line_widget.parentWidget()
            
            actual_x = line_widget.x() + offset_x
            actual_y = current_y + line_widget.y() + offset_y
    
            target_pos = QtCore.QPoint(actual_x, actual_y)
            
            line_widget.render(p_pix, target_pos)
            
            # Increase the tally by the height of the row + the gap between rows
            current_y += row_widget.height() + spacing
            
        
        p_pix.end()
        
        # Draw background (optional)
        painter.save()
        
        painter.restore()
        
        # Draw the pixmap at the printable origin (device coords)
        painter.drawPixmap(page_rect.x(), page_rect.y(), pix)
        
        # End drawing
        painter.end()
        
        # Restore hidden widgets
        for w in widgets_to_hide:
            w.show()


    def render_widget_recursive(self, widget, painter):
        
        if not widget.isHidden():
            pos = widget.mapTo(self, QtCore.QPoint(0, 0))
            widget.render(painter, pos)
    
        for child in widget.findChildren(QtWidgets.QWidget, "", QtCore.Qt.FindDirectChildrenOnly):
            self.render_widget_recursive(child, painter)
            
        
   
class QtImplicitEquationPlotter(QtWidgets.QMainWindow):
    def __init__(self, configs, config_samples, color_count,
                 X_train, y_train, X_test, mlp_train,
                 train_accuracy, test_accuracy, model, 
                 point_size=3, title='Config sin definir',
                 polygon_color='gainsboro',
                 experiment='', epochs=5000, lr=0.0001, seed=33,
                 decimals=2, show_boundaries=False,
                 x_range=(-3, 3), y_range=(-3, 3),
                 show_graph=False, color_class='magenta',
                 light_point_size=2, light_point_alpha=0.4,
                 show_normalized=None,
                 test_configs=None,
                 test_color_count=None,
                 config_index=0):
        
        super().__init__()
        
        self.finish = False
        self.index = config_index
        self.plotter = None
    
        self.show_graph = show_graph
        self.color_class = color_class
        
        self.configs = configs
        self.test_configs = test_configs
        self.config_samples = config_samples
        self.color_count = color_count
        self.test_color_count = test_color_count
        self.X_train = X_train
        self.X_test = X_test
        
        if show_normalized is None:
            self.x_range = x_range
            self.y_range = y_range
        else:
            self.x_range = show_normalized[0]
            self.y_range = show_normalized[1]
        
        self.config_struct = self.get_config_structure(model)

        # Set colors: green = class 0, else red
        self.y_colors = np.where(y_train, 'green', 'red')
        self.y_colors_light = np.where(y_train, 'green', 'red')
        

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.model = model
        
        self.title = title
        self.point_size = point_size
        self.light_point_size = light_point_size
        self.light_point_alpha = light_point_alpha
        self.lr = lr
        self.epochs = epochs
        self.experiment = experiment
        self.seed = seed
        self.decimals = decimals
        self.show_boundaries = show_boundaries
        self.y_train = y_train
        
        self.boundaries = np.array([
                            [-self.x_range[0],  1.,  0.],
                            [self.x_range[1], -1.,  0.],
                            [-self.x_range[0],  0.,  1.],
                            [self.y_range[1],  0, -1.]])
        
        # Define symbolic variables
        self.x, self.y = sp.symbols('x y')
        
        # Initialize empty equations dictionary and scatter points
        self.equations = {}
        self.scatter_points = None
        self.scatter_points_light = None
        self.contour_sets = {}
        self.arrows_config = {}
        self.check_rows_dict = {}
        self.poly = {}
        
        self.H_inequalities = {}
        self.H_inequalities_class0 = {}
        self.H_inequalities_class1 = {}
        
        # Store button callback
        self.button_callback = None
    
        self.polygon_color = polygon_color
        
        
        # Create all GUI elements
        self.setWindowTitle('Interactive Activation Pattern plot')
        self.setMinimumSize(1000, 680)
        self.create_gui()
        
        self.subscript_map = '0123456789'
        
        self.replot(ini=True)
        self.update_plot()
        self.replot()
        

    def test(self, model, X_train, y_train, x_range, y_range,
                 config_samples, decimals,                     
                 config):
        
        def eq_in_boundaries(eq, boundaries):
            for boundary in boundaries:
                if np.allclose(eq, boundary):
                    return True
            return False
        
        boundaries = np.array([
                            [-self.x_range[0],  1.,  0.],
                            [self.x_range[1], -1.,  0.],
                            [-self.y_range[0],  0.,  1.],
                            [self.y_range[1],  0, -1.]])
        
        antecedents_text = 'RULE ANTECEDENTS\n'
        
        eq_list, inequalities, poly_class0, poly_class1, \
            inequalities_class0, inequalities_class1, \
            poly_global, all_zeros_winning_class, neuron_eqs, signs, \
            output_contrib_eqs, output_class_eq = \
                    get_eq_list_new(config,config_samples, 
                                    X_train, model, 
                                    boundaries, decimals, 
                                    y_train=y_train)
                    
        if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
            inequ = np.array(inequalities_class0.array)
            antecedents_text += ' Class R (red)\n'

            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'

            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'
            
            pass
        else:
            antecedents_text += ' Class R (red)\n'
            antecedents_text += '  No Apply\n'


        if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
            inequ = np.array(inequalities_class1.array)
            antecedents_text += ' Class G (green)\n'

            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'

            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                antecedents_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'
        else:
            antecedents_text += ' Class G (green)\n'
            antecedents_text += '  No Apply\n'
        
        
        consequent_text = "RULE CONSEQUENT (Network Output)\n"
           
        for index, eq in enumerate(output_contrib_eqs):
            if self.subscript_map[index+1] == '1':
                y_text = 'Y_r'
            else:
                y_text = 'Y_g'
            consequent_text += f'  {y_text} = {self.str_equ(eq, decimals=self.decimals, normalize=False, web_output=True)}\n'

        # '''
        # Show Inequalities: global and per class
        # '''
        
        activation_region_text = ""
        
        if len(inequalities.array) > 0:
            inequ = np.array(inequalities.array)
            activation_region_text += 'ACTIVATION REGION (Classes R&G)\n'
            
            for index, eq in enumerate(inequ):
                if not eq_in_boundaries(eq, boundaries):
                    continue
                activation_region_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'


            for index, eq in enumerate(inequ):
                if eq_in_boundaries(eq, boundaries):
                    continue
                activation_region_text += f'  {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1, web_output=True)}\n'


        print(antecedents_text)
        print(consequent_text)
        print(activation_region_text)
        
        pass
    

        
    def get_config_structure(self, model, config=None):

        struct = ''
        config_output = ''
        pos = 0 
        
        for layer in model.layers[:-1]:
            if hasattr(layer, 'out_features'):
                if config is not None:
                    if pos != 0:
                        config_output += '-'
                    config_output += config[pos:pos+layer.out_features]
                    pos += layer.out_features
                else:
                    if pos != 0:
                        struct += '-'
                    struct += str(layer.out_features)
                    pos += 1

        if config is not None:
            return config_output
        else:
            return struct
        
        
    # Function to handle mouse button press for panning
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left mouse button
            self.pan_start[0] = event.xdata
            self.pan_start[1] = event.ydata
    
    # Function to handle mouse button release
    def on_release(self, event):
        if event.button == 1:  # Left mouse button
            self.pan_start[0] = None
            self.pan_start[1] = None

    # Reset view with 'r' key
    def on_key(self, event):
        if event.key == 'r':
            self.ax.set_xlim(-2, 5)
            self.ax.set_ylim(0, 1.1)
            self.canvas.draw_idle()

    # Function to handle mouse motion during panning
    def on_motion(self, event):
        if event.inaxes != self.ax or self.pan_start[0] is None:
            return
        
        # Calculate how much we've moved
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        # Get current axis limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        # Set new limits
        self.ax.set_xlim(x_min - dx, x_max - dx)
        self.ax.set_ylim(y_min - dy, y_max - dy)
        
        # Update the figure
        self.canvas.draw_idle()
        
    
    def on_scroll(self, event):
        if event.inaxes != self.ax or event.key != 'control':
            return
        
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        
        scale_factor = 1.2 if event.button == 'up' else 1/1.2
        
        # Calculate new limits centered on mouse position
        new_x_range = (x_max - x_min) / scale_factor
        new_y_range = (y_max - y_min) / scale_factor
        
        self.ax.set_xlim(event.xdata - new_x_range * (event.xdata - x_min) / (x_max - x_min),
                         event.xdata + new_x_range * (x_max - event.xdata) / (x_max - x_min))
        self.ax.set_ylim(event.ydata - new_y_range * (event.ydata - y_min) / (y_max - y_min),
                         event.ydata + new_y_range * (y_max - event.ydata) / (y_max - y_min))
        self.canvas.draw_idle()

    
    def create_gui(self):
        # --- Main Window Setup ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # This horizontal container holds the three main pillars
        self.top_container = QtWidgets.QHBoxLayout()
        
        
        # --- COLUMN 1: Matplotlib Figure (Left) ---
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.top_container.addWidget(self.canvas, stretch=2)
               
        # --- COLUMN 2: Middle Box (Equations & Navigation) ---
        self.middle_column_layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Samples Checkbox (Anchored to top)
        self.scatter_checkbox = QtWidgets.QCheckBox("Samples")
        self.scatter_checkbox.setStyleSheet("font-family: monospace; font-size: 12px;")
                                    
        self.scatter_checkbox.setChecked(True)
        self.scatter_checkbox.stateChanged.connect(self.update_plot)
        self.middle_column_layout.addWidget(self.scatter_checkbox, stretch=0)
        
        # 2. Scrollable Area for Dynamic Checkboxes (The "Lines Box")
        self.check_scroll = QtWidgets.QScrollArea()
        self.check_widget = QtWidgets.QWidget()
        self.check_layout = QtWidgets.QVBoxLayout(self.check_widget)
        self.check_layout.setAlignment(QtCore.Qt.AlignTop) 
        self.check_scroll.setWidget(self.check_widget)
        self.check_scroll.setWidgetResizable(True)
        self.check_scroll.setFixedWidth(140)
        self.middle_column_layout.addWidget(self.check_scroll, stretch=0)
        
        self.check_scroll.setStyleSheet("QScrollArea { border: 1px solid #666666; }")
        
        self.check_widget.setStyleSheet("background-color: #f1f1f1")
        
        # 3. Navigation Buttons (Immediately below the box)
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("◀")
        self.btn_forward = QtWidgets.QPushButton("▶")
        
        btn_style = "background-color: #4499FF; color: white; font-weight: bold; font-size: 14pt; padding: 2px;"
        self.btn_back.setStyleSheet(btn_style)
        self.btn_forward.setStyleSheet(btn_style)
        
        # 1. Create a container widget for the buttons
        nav_container = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 3, 0, 3) # Tight vertical spacing
        nav_layout.setSpacing(10)                # Gap between the two buttons
        
        self.btn_back.setFixedSize(45, 25)
        self.btn_forward.setFixedSize(45, 25)
        
        # 3. Build the layout: Stretch - Button - Button - Stretch
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_forward)
        
        # 4. Add the container to your main sidebar layout
        self.middle_column_layout.addWidget(nav_container, stretch=0)
        
        # 4. The Stretch (Pushes the whole group above it to the top)
        self.middle_column_layout.addStretch(1)
        
        # Add the completed middle pillar to the top container
        self.top_container.addLayout(self.middle_column_layout)

        # --- COLUMN 3: Info Panel (Right) ---
        self.info_panel = QtWidgets.QTextEdit()
        self.info_panel.setReadOnly(True)
        # Narrowed from 300 to 250 to remove excess right space
        self.info_panel.setFixedWidth(312)
        self.top_container.addWidget(self.info_panel)
        
        # Add the top container (Plot + Middle + Right) to main layout
        self.main_layout.addLayout(self.top_container, stretch=3)


        # --- 4. Bottom Text Panels (Inequalities & Rules) ---
        self.bottom_layout = QtWidgets.QHBoxLayout()
        self.left_text = QtWidgets.QTextEdit()
        self.middle_text = QtWidgets.QTextEdit()
        self.right_text = QtWidgets.QTextEdit()
        

        self.left_text.setReadOnly(True)
        self.middle_text.setReadOnly(True)
        self.right_text.setReadOnly(True)
        
        min_height_bottom = 235
        
        self.left_text.setMinimumHeight(min_height_bottom)
        self.middle_text.setMinimumHeight(min_height_bottom)
        self.right_text.setMinimumHeight(min_height_bottom)
        
        # Sync scrollbars
        self.left_text.verticalScrollBar().valueChanged.connect(
            self.right_text.verticalScrollBar().setValue
        )
        self.middle_text.verticalScrollBar().valueChanged.connect(
            self.middle_text.verticalScrollBar().setValue
        )
        self.right_text.verticalScrollBar().valueChanged.connect(
            self.left_text.verticalScrollBar().setValue
        )
        
        self.middle_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.middle_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.middle_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.middle_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.right_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.bottom_layout.addWidget(self.left_text)
        self.bottom_layout.addWidget(self.middle_text)
        self.bottom_layout.addWidget(self.right_text)
        
        self.main_layout.addLayout(self.bottom_layout)

        # --- Final Signal Connections ---
        self.btn_forward.clicked.connect(lambda: self.handle_action("forward"))
        self.btn_back.clicked.connect(lambda: self.handle_action("backward"))        
        
        # Initialize variable dictionary for checkbuttons
        self.var_dict = {}
                
    
    def clear_ui_elements(self):
        # Clear the dictionary
        self.check_dict = {}
        self.line_dict = {}
        # Clear the layout widgets
        while self.check_layout.count():
            item = self.check_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        
    def _on_scrollbar(self, *args):
        """Handle scrollbar movement"""
        self.left_text.yview(*args)
        self.middle_text.yview(*args)
        self.right_text.yview(*args)

    def _left_scroll_set(self, *args):
        """Handle left text scrolling"""
        self.scrollbar.set(*args)
        self.right_text.yview_moveto(args[0])
    
    def _right_scroll_set(self, *args):
        """Handle right text scrolling"""
        self.scrollbar.set(*args)
        self.left_text.yview_moveto(args[0])
        
        
    def clear_equations(self):
        """Surgically remove all dynamic checkboxes from the sidebar."""
        self.equations.clear()
        self.var_dict.clear()
        self.arrows_config.clear()
        self.contour_sets.clear()
        self.poly.clear()
        
        # We must use list() to avoid iteration errors while deleting
        if hasattr(self, 'row_widgets'):
            for name in list(self.row_widgets.keys()):
                widget = self.row_widgets.pop(name)
                widget.setParent(None)
                widget.deleteLater()
        
        self.color_index = 0 
        self.replot()
        self.update_plot()
        
            
    def str_equ(self, eq, decimals=2, return_full=False, 
                orig_sign=-2, normalize=True, web_output=False):
        
        eq = eq.copy()
        
        eq[abs(eq) < 1e-10] = 0
        
        sign = 1
        
        if eq[1] != 0:
            if eq[1] < 0:
                sign = -1
            if normalize:
                eq = eq / eq[1]
        elif eq[2] != 0:
            if eq[2] < 0:
                sign = -1
            if normalize:
                eq = eq / eq[2]
               
        
        n_coefs = len(eq[1:])
        ret = ''
        first = True
        for i in range(n_coefs):
            val = eval(f'{eq[i+1]:.{decimals}f}')
            if val == 0:
                # first = False
                continue
            elif val == 1:
                if not first:
                    ret += f' + {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'x{self.subscript_map[i+1]} '
            elif val == -1:
                if not first:
                    ret += f'- {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'-x{self.subscript_map[i+1]} '
            else:
                if val < 0:
                    if first:
                        ret += '-'
                    else:
                        ret += '- '
                else:
                    if not first:
                        ret += '+ '
                ret += f'{abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
            first = False
        if first:
            if eq[0] < 0:
                ret += f'{eq[0]:.{decimals}f}'
            else:
                ret += f'{abs(eq[0]):.{decimals}f}'
        else:
            if eq[0] < 0:
                if not np.isclose(eq[0], 0.0):
                    if not web_output:
                        ret += f'&nbsp;- {abs(eq[0]):.{decimals}f}'
                    else:
                        ret += f'- {abs(eq[0]):.{decimals}f}'
            else:
                if not np.isclose(eq[0], 0.0):
                    ret += f'+ {abs(eq[0]):.{decimals}f}'
                
        if return_full:
            r = sign * orig_sign
            ret += f' {">= 0" if r > 0 else "<= 0"}'
            
        return ret
    
    
    
    def append_bottom_text(self, text, no_trad=False, bold=False, pos=None):
        """Append text to the bottom text content."""
        
        
        letters = 'xyz'

        if pos is None:
            # Get the number of lines in each text widget
            left_lines = int(self.left_text.index('end-1c').split('.')[0])
            # right_lines = int(self.right_text.index('end-1c').split('.')[0])
            
            # Calculate visible lines (approximate based on height)
            visible_lines = self.left_text.winfo_height() // 19  # Approximate line height
            
            text_zone = None
            if left_lines <= visible_lines:
                # If left column has space or has fewer lines than right, add to left
                text_zone = self.left_text
            else:
                # Otherwise add to right column
                text_zone = self.right_text
        else:
            if pos == 'left':
                text_zone = self.left_text
            elif pos == 'middle' or pos == 'center':
                text_zone = self.middle_text
            else: # pos == 'right':
                text_zone = self.right_text


        new_text = ''
        
        self.left_text.config(state="normal")
        self.middle_text.config(state="normal")
        self.right_text.config(state="normal")
        
        if no_trad:
            new_text = text
            if bold:
                text_zone.tag_configure('bold', font=('Arial', 11, 'bold'))
                text_zone.insert(tk.END, new_text + '\n', 'bold')

            else:
                text_zone.tag_configure('normalX', font=('Arial', 11))
                text_zone.insert(tk.END, new_text + '\n', 'normalX')

        else:
            must_continue = False
            for i, t in enumerate(text):
                if must_continue:
                    must_continue = False
                    continue 
                
                if t == '*':
                    continue
             
                if t == '_':
                    pos1 = '0123456789'.find(text[i+1])
                    if pos1 >= 0:
                        must_continue = True
                        # new_text += f'{subscript_map[pos1]}'
                        new_text += f'{letters[pos1]}'
                else:
                    pos = letters.find(t)
                    if pos >= 0:
                        new_text += f' {letters[pos]}'
                    else:
                        new_text += t

            text_zone.tag_configure('normalX', font=('Arial', 11))
            text_zone.insert(tk.END, new_text + '\n', 'normalX')
        
        
        self.left_text.config(state="disabled")
        self.middle_text.config(state="disabled")
        self.right_text.config(state="disabled")
        
    
    def append_text(self, text, bold=False, reverse=False, center=False):
        """Append text to the existing content.
        
        Args:
            text (str): Text to append
        """
        if bold:
            new_text = text
            if center:
                new_text = text + '\n'
                
            self.text_widget.config(state="normal")
            
            
            if reverse:
                self.text_widget.tag_configure('bold-reverse', font=('Courier New', 10, 'bold'), 
                                               background='yellow', foreground='black')
            else:
                self.text_widget.tag_configure('bold', font=('Courier New', 10, 'bold'))

            if center:        
                self.text_widget.tag_configure('center', justify='center', 
                                               font=('Courier New', 10, 'bold'))
                self.text_widget.insert(tk.END, new_text, 'center')
            else:
                if reverse:
                    self.text_widget.insert(tk.END, new_text, 'bold-reverse')
                else:
                    self.text_widget.insert(tk.END, new_text, 'bold')

                self.text_widget.insert(tk.END, '\n')
                

            self.text_widget.config(state="disabled")
        else:
            self.text_widget.config(state="normal")
            self.text_widget.insert(tk.END, text + '\n')
            self.text_widget.config(state="disabled")    
        
        
        
    def set_scatter_points(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points = np.array(points)
        self.scatter_colors = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        self.scatter_checkbox.setEnabled(True)
        self.update_plot()

        
    def set_scatter_points_light(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points_light = np.array(points)
        self.scatter_colors_light = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        self.scatter_checkbox.setEnabled(True)
        self.update_plot()
            
        
        
    def clear_scatter_points(self):
        """Remove scatter points from data and reset the UI checkbox."""
        self.scatter_points = None
        self.scatter_colors = None
        
        # Replace self.scatter_var.set(False) with the direct widget method
        self.scatter_checkbox.setChecked(False)
        
        # Disable the checkbox since there is no data to show
        self.scatter_checkbox.setEnabled(False)
        self.update_plot()
        
        
    def add_equation(self, name, equation, color, linestyle, direction, linewidth=3):
        """Add a new equation to the plotter.
        
        Args:
            name (str): Name of the equation to display
            equation (sympy.Eq): Sympy equation instance
        """
        if not isinstance(equation, sp.Eq):
            raise ValueError("Equation must be a sympy.Eq instance")
            
        # Add equation to dictionary
        self.equations[name] = (equation, color, linestyle, direction)
        
        
        row_widget = QtWidgets.QWidget()

        row_layout = QtWidgets.QHBoxLayout(row_widget)
        
        row_layout.setContentsMargins(3, 1, 3, 1)
        # row_layout.setSpacing(1)
        
        checkbox = QtWidgets.QCheckBox(name.replace('Eq', 'HS'))
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.update_plot)
        checkbox.setStyleSheet("""
                                   QCheckBox {
                                        font-family: monospace;
                                        font-size: 11px;
                                    }
                               """)
        
        # Pass the linestyle and linewidth

        line_visual = LineStyleWidget(color, linestyle, linewidth, parent=row_widget)
     
        row_layout.addWidget(checkbox)
        row_layout.addStretch(1)
        row_layout.addWidget(line_visual)
        
        self.check_layout.addWidget(row_widget)

        self.check_dict[name] = checkbox
        self.line_dict[name] = line_visual
        self.check_rows_dict[name] = row_widget
        
        
        # # Create new checkbutton variable
        var = BooleanVarReplica(value=True)
        self.var_dict[name] = var
        
    def remove_equation(self, name):
        """Remove an equation from the plotter with surgical precision."""
        if name in self.equations:
            # Remove data references
            del self.equations[name]
            del self.var_dict[name]
    
            # Retrieve the container widget we stored earlier
            if name in self.row_widgets:
                widget_to_kill = self.row_widgets.pop(name)
                
                # This is the Qt magic: remove it from the UI and schedule it for deletion
                widget_to_kill.setParent(None)
                widget_to_kill.deleteLater()
    
            # The layout automatically shifts everything up!
            self.update_plot()
         
    
    def create_implicit_function(self, equation):
        expr = equation.lhs - equation.rhs
        
        def f(x, y):
            return sp.lambdify((self.x, self.y), expr, 'numpy')(x, y)
        
        return f
    
    def plot_implicit(self, f, x_range, y_range, resolution=100, color='black', linewidth=1, linestyles='solid'):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = f(X, Y)
        
        ret = self.ax.contour(X, Y, Z, levels=[0], colors=color, linewidths=linewidth, linestyles=linestyles)
        
        return ret
    
  
    def calculate_normal_vector(self, point, current_index, vertices, direction=1):        
        """Calculate the truly orthogonal vector using adjacent points"""
        n_points = len(vertices)
        
        # Get points from both sides (5 points away is enough)
        prev_index = max(current_index - 5, 0)
        next_index = min(current_index + 5, n_points - 1)
        
        # Get points
        prev_point = vertices[prev_index]
        next_point = vertices[next_index]
        
        # Calculate tangent using points from both sides
        tangent = next_point - prev_point
        tangent = tangent / np.linalg.norm(tangent)
        
        # Calculate normal vector (rotate tangent 90 degrees)
        normal = np.array([-tangent[1], tangent[0]]) * direction
        
        return normal
    
    def calculate_normal_vector2(self, point, current_index, vertices, direction=1):
        """Calculate orthogonal vector with specific handling for high curvature points"""
        n_points = len(vertices)
        
        # First, check if we're at a point with high curvature by analyzing local geometry
        # Get immediate neighbors
        prev_index = max(current_index - 1, 0)
        next_index = min(current_index + 1, n_points - 1)
        
        prev_point = vertices[prev_index]
        next_point = vertices[next_index]
        current_point = point  # The current point
        
        # Calculate directions of adjacent segments
        prev_dir = np.zeros(2)
        next_dir = np.zeros(2)
        
        if prev_index != current_index:
            prev_dir = current_point - prev_point
            prev_norm = np.linalg.norm(prev_dir)
            if prev_norm > 1e-10:
                prev_dir = prev_dir / prev_norm
        
        if next_index != current_index:
            next_dir = next_point - current_point
            next_norm = np.linalg.norm(next_dir)
            if next_norm > 1e-10:
                next_dir = next_dir / next_norm
        
        # Calculate dot product to measure alignment between segments
        dot_product = np.dot(prev_dir, next_dir)
        
        # Detect if we're at a corner/high curvature point
        # The threshold can be adjusted based on your specific needs
        if dot_product < 0.7:  # Less than ~45 degrees between segments
            # We're at a high curvature point - use just the local segment
            # For magenta line specifically, we can use the next segment direction
            tangent = next_dir
            
            # Ensure we have a valid tangent
            if np.linalg.norm(tangent) < 1e-10:
                # Fallback if tangent is too small
                tangent = np.array([1.0, 0.0])
        else:
            # For smoother sections, use a moderate window (not too large)
            # Using 2 points instead of 5 to stay more local
            smooth_prev_index = max(current_index - 2, 0)
            smooth_next_index = min(current_index + 2, n_points - 1)
            
            smooth_prev_point = vertices[smooth_prev_index]
            smooth_next_point = vertices[smooth_next_index]
            
            # Calculate tangent
            tangent = smooth_next_point - smooth_prev_point
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 1e-10:
                tangent = tangent / tangent_norm
            else:
                # Fallback to a default direction
                tangent = np.array([1.0, 0.0])
        
        # Calculate the normal vector (rotate tangent 90 degrees)
        normal = np.array([-tangent[1], tangent[0]]) * direction
        
        return normal
    
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    
    def find_valid_position(self, vertices, min_distance=0.5):
        """Find a random position that's far enough from existing arrows"""
        max_attempts = 50  # Prevent infinite loop
        existing_points = []
        
        # Collect existing arrow points
        for eq_config in self.arrows_config.values():
            if 'point' in eq_config:
                existing_points.append(eq_config['point'])
        
        for _ in range(max_attempts):
            # Generate random index
            random_index = np.random.randint(0, len(vertices))
            point = vertices[random_index]
            
            # Check distance from all existing points
            is_valid = True
            for existing_point in existing_points:
                if self.calculate_distance(point, existing_point) < min_distance:
                    is_valid = False
                    break
                    
            if is_valid:
                return random_index, point
                
        # If no valid position found, take the last generated one
        return random_index, point
    
    
    def add_orthogonal_arrow(self, eq_name, arrow_length=0.5, direction=1, color='green'):
        """Add arrow configuration for a specific equation"""
        
        if eq_name in self.contour_sets:
            paths = self.contour_sets[eq_name].allsegs[0]
            if paths and len(paths[0] > 0):
                vertices = paths[0]
                random_index, point = self.find_valid_position(vertices)
                
                self.arrows_config[eq_name] = {
                    'length': arrow_length,
                    'index': random_index,
                    'point': point,
                    'direction': direction,
                    'color': color
                }
                self.update_plot()
                
                
    def add_polygon(self, poly_name, vertices, fill_color='gainsboro', 
                    edge_color='green', alpha=0.8, linewidth=1):
        if vertices == None or len(vertices) == 0: 
            return
        
        x_coords, y_coords = [], []
        for vertex in vertices:
            if vertex[0] == 1:
                x_coords.append(vertex[1])
                y_coords.append(vertex[2])
 
        self.poly[poly_name] = {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'fill_color': fill_color,
            'edge_color': edge_color,
            'alpha': alpha,
            'linewidth': linewidth
        }
        
        return
    
        

    def order_vertices(self, vertices):
        # Calculate centroid
        center = np.mean(vertices, axis=0)
        
        # Calculate angles of points with respect to center
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        
        # Sort vertices based on these angles        
        sorted_indices = np.argsort(angles)
        
        ordered = vertices[sorted_indices]
    
        return ordered
    
    def longest(self, vertices):
        longest = 0
        for v1 in vertices:
            for v2 in vertices:
                l = math.dist(v1, v2)
                if longest < l:
                    longest = l
        return longest
    
    def area(self, vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0
        return area
    
    def plot_polygon(self):
        # Separate x and y coordinates

        if self.poly == None:
            return
        
        for poly in self.poly.values():
            vertices = np.array([poly['x_coords'], poly['y_coords']]).T
            
            vertices = self.order_vertices(vertices)

            '''
            Fill polygon with transparency
            '''
            longest_distance = self.longest(vertices)
            area = self.area(vertices)

            adjust = (self.x_range[1] - self.x_range[0]) / 6

            if area / (longest_distance * adjust) > 0.15:
                self.ax.fill(vertices[:,0], 
                             vertices[:,1], 
                             facecolor=poly['fill_color'],
                             alpha=poly['alpha']
                             )
            else:        
                self.ax.fill(vertices[:,0], 
                             vertices[:,1], 
                             edgecolor=poly['fill_color'], # 'red'
                             alpha=poly['alpha'],
                             linewidth=12,
                             linestyle='-'
                             )
                
        return self.ax    


    
    def update_plot(self):
        self.ax.clear()
        self.contour_sets.clear()

        if self.show_graph:
            x_range = self.x_range
            y_range = self.y_range
    
            # Ensure equal scaling
            self.ax.set_aspect('equal')
            self.ax.set_xlim(x_range)
            self.ax.set_ylim(y_range)
            
            self.ax.tick_params(axis='both', which='major', labelsize=9) 
            self.ax.set_xlabel(r'$\mathbf{x_1}$', fontsize=14, labelpad=-8)
            self.ax.set_ylabel(r'$\mathbf{x_2}$', fontsize=14, labelpad=-15)
                               
            self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0)
            
            for eq_name, var in self.var_dict.items():
                if var.get():
                    equation = self.equations[eq_name][0]
                    f = self.create_implicit_function(equation)
                    
                    linestyles = 'solid'
                    linewidth = 1.6  
                    new_color = self.equations[eq_name][1]
                    direction = self.equations[eq_name][3]
                    
                    if eq_name.startswith('Eq9'):
                        linewidth = 2
                        new_color = self.color_class
                        linestyles = 'dashdot'
                    elif eq_name[2] in ['1', '2', '3', '4']:
                        linestyles = self.equations[eq_name][2]
                    else:
                        linestyles = self.equations[eq_name][2]
                        linewidth = 2
                        
                    contour = self.plot_implicit(f, x_range, y_range, 
                                               resolution=200, 
                                               color=new_color,
                                               linewidth=linewidth,
                                               linestyles=linestyles)
                    
                    self.contour_sets[eq_name] = contour
      
                    if eq_name in self.arrows_config:
                        paths = contour.allsegs[0]
                        if paths:
                            vertices = paths[0]
                            point_index = self.arrows_config[eq_name]['index']
                            point = vertices[point_index]
                            
                            normal = self.calculate_normal_vector(point, point_index, vertices,
                                        direction=self.arrows_config[eq_name].get('direction', direction))
                            
                            arrow_length = self.arrows_config[eq_name]['length']
                            dx = normal[0] * arrow_length
                            dy = normal[1] * arrow_length
                            
                            adjust = (self.x_range[1] - self.x_range[0]) / 6
                            
                            self.ax.arrow(point[0], point[1], 
                                        dx * adjust, dy * adjust,
                                        head_width=0.1 * adjust,
                                        head_length=0.1 * adjust,
                                        fc=self.equations[eq_name][1],
                                        ec=self.equations[eq_name][1],
                                        length_includes_head=True,
                                        width=0.05 * adjust,
                                        zorder=1000)
    
        
                self.plot_polygon()
                
                config = list(dict(self.configs).keys())[self.index]
        
            if self.scatter_points_light is not None:
                self.ax.scatter(self.scatter_points_light[:, 0], self.scatter_points_light[:, 1], 
                                c=self.scatter_colors_light, s=self.light_point_size, 
                                zorder=0, alpha=self.light_point_alpha)

            if self.scatter_points is not None:
                self.ax.scatter(self.scatter_points[:, 0], self.scatter_points[:, 1], 
                                c=self.scatter_colors, s=self.point_size, zorder=5)
        
            self.ax.set_title(self.title, pad=7, fontsize=10)
            
            self.fig.tight_layout()
            # self.canvas.draw()
            
            pass
        
    def remove_orthogonal_arrow(self, eq_name):
        """Remove arrow from a specific equation"""
        if eq_name in self.arrows_config:
            del self.arrows_config[eq_name]
            self.update_plot()
    
    def clear_all_arrows(self):
        """Remove all arrows"""
        self.arrows_config.clear()
        self.update_plot()
    
    
    def set_button_callback(self, callback):
        """Set the callback function for both buttons"""
        self.button_callback = callback
        self.forward_button.configure(command=lambda: callback("forward"))
        self.backward_button.configure(command=lambda: callback("backward"))


    def on_closing(self):
        """Handle window closing using the same callback"""
        if self.button_callback:
            self.button_callback("end")
            
        self.root.destroy()
    
    def handle_action(self, action):
        if action == "forward":
            self.index += 1
            if self.index == len(self.configs):
                self.index = 0
            
            self.clear_equations()
            self.replot()
            
            self.export_pdf() 
            
            self.fig.savefig('classification_output/classifcation_output_canvas.svg', bbox_inches='tight') #, pad_inches=0.1)

            
        elif action == "backward":
            self.index -= 1
            if self.index < 0:
                self.index = len(self.configs) - 1
    
            self.clear_equations()
            self.replot()
            
        elif action == "end":
            self.finish = True
    

    def eq_in_boundaries(self, eq):
        for boundary in self.boundaries:
            if np.allclose(eq, boundary):
                return True
        return False
    
    def replot(self, ini=False):        
        config = list(dict(self.configs).keys())[self.index]
      
        self.right_text.clear()
        self.middle_text.clear()
        self.left_text.clear()
        self.info_panel.clear()
        self.title = f'TRAIN SAMPLES: Activation Pattern {self.get_config_structure(self.model, config)}'
        
        # --- Build one single HTML string for the sidebar ---
        html = []
        
        # 1. Centered Header
        html.append("<div style='font-family: monospace; text-align: center; font-size: 10pt; margin-bottom: 10px;'>")
        html.append("<b>TOY EXPERIMENT ON 3×3 CHECKERBOARD<br>CLASSIFICATION DATASET</b>")
        html.append("</div>")
        
        # 2. Left-aligned Statistics
        html.append("<div style='text-align: left; font-family: monospace; font-size: 9pt;'>")
        html.append(f"# Neurons = {self.config_struct}<br>")
        html.append(f"# Activation Patterns = {len(self.configs)}<br>")
        html.append(f"# Samples = {len(self.X_train)} (train)&nbsp;&nbsp;{len(self.X_test)} (test)<br>")
        html.append(f"# Epochs = {self.epochs}<br>")
        html.append(f"Learning Rate = {self.lr}<br>")
        html.append(f"Train Accuracy&nbsp;=&nbsp;{self.train_accuracy:.04f}<br>")
        html.append(f"Test Accuracy&nbsp;&nbsp;=&nbsp;{self.test_accuracy:.04f}<br>")
        html.append(f"Seed = {self.seed}<br>")
        html.append("</div>")
    

        # 3. Helper for perfectly aligned tables
        def get_table_html_old(title_text, config_dict, rmse_dict, active_c):
            # We wrap the title in a span with a larger font size (e.g., 12pt)
            t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
            # The table itself remains standard size for precision
            t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
            t_html += "<tr><th width='90' align='left'>Pattern</th><th width='70' align='right'>#Samples</th><th width='80' align='right'>RMSE</th></tr>"
            
            for c, count in config_dict.items():
                bg = "background-color: #FFFF22;" if c == active_c else ""
                t_html += f"<tr style='{bg}'>"
                t_html += f"<td>{c[:4]}-{c[4:]}</td>"
                t_html += f"<td align='right'>{count}</td>"
                t_html += "</tr>"
            t_html += "</table><br>"
            return t_html

        def get_table_html(title_text, config_dict, color_dict, config):
            # We wrap the title in a span with a larger font size (e.g., 12pt)
            t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
            # The table itself remains standard size for precision
            t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
            t_html += "<tr><th width='70' align='left'>Pattern</th><th width='65' align='right'>#Samples</th><th width='45' align='right'>TP</th><th width='40' align='right'>FP</th><th width='40' align='right'>TN</th><th width='40' align='right'>FN</th></tr>"
            
            for c, count in config_dict.items():
                bg = "background-color: #FFFF22;" if c == config else ""
                t_html += f"<tr style='{bg}'>"
                t_html += f"<td>{c[:4]}-{c[4:]}</td>"
                t_html += f"<td align='right'>{count}</td>"
                t_html += f"<td align='right'>{color_dict[c]['TP']}</td>"
                t_html += f"<td align='right'>{color_dict[c]['FP']}</td>"
                t_html += f"<td align='right'>{color_dict[c]['TN']}</td>"
                t_html += f"<td align='right'>{color_dict[c]['FN']}</td>"
                t_html += "</tr>"
            t_html += "</table><br>"
            return t_html

        html.append(get_table_html("Train samples", self.configs, self.color_count, config))
        html.append(get_table_html("Test samples", self.test_configs, self.test_color_count, config))
        
        # --- Set the HTML once ---
        self.info_panel.setHtml("".join(html))       
        
        eq_list, inequalities, poly_class0, poly_class1, inequalities_class0, inequalities_class1, poly_global, all_zeros_winning_class, neuron_eqs, signs, output_contrib_eqs, output_class_eq = \
                            get_eq_list_new(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)
                            # get_eq_list(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)

        self.H_inequalities[config] = inequalities
        self.H_inequalities_class0[config] = inequalities_class0
        self.H_inequalities_class1[config] = inequalities_class1
        
        
        color_index = 0
        color_list = ['blue', 'olive', 'brown', 'orange', 'lightblue', 'pink', 'gray', 'lightseagreen']


        # --- 6. Bottom Panels: HTML Construction ---

        # A. LEFT PANEL: INEQUALITIES
        # We start with the Title
        left_html = "<div style='font-family: monospace; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
        
        # Line 2: The Global Domain Range (separated by <br>)
        left_html += f"&nbsp;&nbsp;{self.x_range[0]:.0{self.decimals}f} <= x{self.subscript_map[1]} <= {self.x_range[1]:.0{self.decimals}f}<br>"
        left_html += f"&nbsp;&nbsp;{self.y_range[0]:.0{self.decimals}f} <= x{self.subscript_map[2]} <= {self.y_range[1]:.0{self.decimals}f}<br>"
        
        # # Line 3+: Each equation on a new line
        for index, eq in enumerate(neuron_eqs):
            bit = config[index]
            # Use orig_sign to ensure the math string shows >= 0 or <= 0 correctly
            eq_str = self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=int(bit))
            left_html += f"&nbsp;&nbsp;N={bit}, HS{index+1}: {eq_str}<br>"

        out_text = f'&nbsp;&nbsp;Y_r >= Y_g, HS9: {self.str_equ(eq, decimals=self.decimals)} {">= 0" if signs[index] > 0 else "<= 0"}'
        
        left_html += out_text
        
        left_html = left_html.replace('<=', '&le;').replace('>=', '&ge;')

        left_html += '</div>'

        # Set the left panel
        self.left_text.setHtml(left_html)

        if all_zeros_winning_class >= 0:
            if all_zeros_winning_class == 0:
                poly_class0 = poly_global
                poly_global = []
                inequalities_class0 = inequalities 
            elif all_zeros_winning_class == 1:
                poly_class1 = poly_global
                poly_global = []
                inequalities_class1 = inequalities
                
        
        # --- 6.1. Middle Panels: HTML Construction ---

        # We start with the Title
        middle_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE ANTECEDENTS</b><br>"
        
        if inequalities_class0 != [] and len(inequalities_class0.array) > 0:
            inequ = np.array(inequalities_class0.array)
            middle_html += '&nbsp;\u2022 <b>Class R (red)</b><br>'

            if self.show_boundaries:
                for index, eq in enumerate(inequ):
                    if not self.eq_in_boundaries(eq):
                        continue
                    middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'

            for index, eq in enumerate(inequ):
                if self.eq_in_boundaries(eq):
                    continue
                middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'
            
            pass
        else:
            middle_html += '&nbsp;\u2022 <b>Class R (red)</b><br>'
            middle_html += '&nbsp;&nbsp;No Apply<br>'


        if inequalities_class1 != [] and len(inequalities_class1.array) > 0:
            inequ = np.array(inequalities_class1.array)
            middle_html += '<br>&nbsp;\u2022 <b>Class G (green)</b><br>'

            if self.show_boundaries:
                for index, eq in enumerate(inequ):
                    if not self.eq_in_boundaries(eq):
                        continue
                    middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'

            for index, eq in enumerate(inequ):
                if self.eq_in_boundaries(eq):
                    continue
                middle_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'
        else:
            middle_html += '<br>&nbsp;\u2022 <b>Class G (green)</b><br>'
            middle_html += '&nbsp;&nbsp;No Apply<br>'


        middle_html = middle_html.replace('<=', '&le;').replace('>=', '&ge;')

        middle_html += '</div>'

        # Set the middle panel
        self.middle_text.setHtml("".join(middle_html))
        
        right_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE CONSEQUENT (Network Output)</b><br>"
           
        for index, eq in enumerate(output_contrib_eqs):
            if self.subscript_map[index+1] == '1':
                y_text = 'Y_r '
            else:
                y_text = 'Y_g'
            right_html += f'&nbsp;&nbsp;{y_text} = {self.str_equ(eq, decimals=self.decimals, normalize=False)}<br>'

        # '''
        # Show Inequalities: global and per class
        # '''
        if len(inequalities.array) > 0:
            inequ = np.array(inequalities.array)
            right_html += '<br><b>ACTIVATION REGION (Classes R&G)</b><br>'
            
            if self.show_boundaries:
                for index, eq in enumerate(inequ):
                    if not self.eq_in_boundaries(eq):
                        continue
                    right_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'


            for index, eq in enumerate(inequ):
                if self.eq_in_boundaries(eq):
                    continue
                right_html += f'&nbsp;&nbsp;{self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}<br>'


        right_html = right_html.replace('<=', '&le;').replace('>=', '&ge;')
        
        right_html += '</div>'
        # Set the right panel
        self.right_text.setHtml("".join(right_html))
        
        if self.show_graph:
            
            self.clear_ui_elements()

            self.color_index = 0 
            
            '''
            Add polygons, scatter points, lines, and arrows
            '''
            for index, eq in enumerate(eq_list):
                if eq[1] == False:
                    print(f'EQ[1] {index} = FALSE!!!')
                    continue
                
                if eq[2] != -2 and eq[1] is None: 
                    # self.append_bottom_text(f'{eq[0]} = 0x + 0y + b = 0', no_trad=True)
                    pass
                else:
                    if eq[2] == 0:
                        '''
                        Para la ecuación de clase, siempre marcamos la dirección 
                        hacia la clase 0 (roja)
                        '''
                        self.add_equation(eq[0], eq[1], self.color_class, 
                                          linestyle='dashdot', direction=0)
                        self.add_orthogonal_arrow(eq[0], arrow_length=0.3, 
                                                  # direction=eq[4], # eq[4] 
                                                  direction=1, # eq[4] 
                                                  color=self.color_class)
                    elif eq[2] == -2: 
                        pass
                    else:
                        if index < self.model.layers[0].out_features:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='solid', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=0.3, direction=eq[2], 
                                                      color=color_list[color_index])
                        else:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='dashed', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=0.3, direction=eq[2], 
                                                      color=color_list[color_index])
    
    
            if inequalities_class0 != []:
                self.add_polygon('Class 1', poly_class0, fill_color='lightsalmon', alpha=0.15)
    
            if inequalities_class1 != []:
                self.add_polygon('Class 2', poly_class1, fill_color='lightgreen', alpha=0.15)
    
            self.set_scatter_points_light(self.X_train,
                                    self.y_colors_light)

            self.set_scatter_points(self.X_train[self.config_samples[config]],
                                    self.y_colors[self.config_samples[config]])
            
            # self.clear_equations()
            # self.replot()


        
    def export_pdf(self, filename='classification_output/export_classification_clean.pdf'):
        # Widgets to hide while exporting
        widgets_to_hide = [self.canvas, self.btn_forward, self.btn_back]
        # widgets_to_hide = [self.canvas]
    
        # Freeze sizes for hidden widgets
        for w in widgets_to_hide:
            w.setFixedSize(w.size())
            w.hide()
    
        # Configure printer (unchanged)
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filename)
        
        # Page rectangle in device (printer) pixels
        page_rect = printer.pageRect()  # QRect in device pixels
        
        # Compute scale to map widget coordinates to page coordinates (widget -> printer device pixels)
        widget_w = float(self.width())
        widget_h = float(self.height())
        scale = min(page_rect.width() / widget_w, page_rect.height() / widget_h)
        
        # Create painter on printer early to query device pixel ratio
        painter = QtGui.QPainter(printer)
        device_ratio = painter.device().devicePixelRatioF() \
                        if hasattr(painter.device(), "devicePixelRatioF") else 1.0
        
        # device_scale maps widget pixels -> device pixels used by the painter
        device_scale = scale * device_ratio
        
        # Create a high-res pixmap sized in device pixels
        pixmap_size = QtCore.QSize(int(widget_w * device_scale), int(widget_h * device_scale))
        
        pix = QtGui.QPixmap(pixmap_size)
        pix.fill(QtCore.Qt.transparent)
        
        # If using devicePixelRatio, set it so drawPixmap behaves correctly
        if hasattr(pix, "setDevicePixelRatio"):
            pix.setDevicePixelRatio(device_ratio)
        
        # Render the widget into the pixmap (use device_scale but render in widget coordinates)
        p_pix = QtGui.QPainter(pix)
        p_pix.setRenderHints(QtGui.QPainter.Antialiasing | 
                             QtGui.QPainter.TextAntialiasing | 
                             QtGui.QPainter.SmoothPixmapTransform)
        p_pix.scale(scale, scale)
        self.render_widget_recursive(self, p_pix)
        
        self.check_widget.adjustSize()
        self.check_widget.ensurePolished()

        # Start with the margin/spacing of your layout
        current_y = self.check_widget.layout().contentsMargins().top()
        spacing = self.check_widget.layout().spacing()
        
        offset_x = 23 #XXXX6  # Moves the line to the right
        offset_y = -195 # Moves the line up

        for name, line_widget in self.line_dict.items():
            row_widget = line_widget.parentWidget()
            
            actual_x = line_widget.x() + offset_x
            actual_y = current_y + line_widget.y() + offset_y
    
            target_pos = QtCore.QPoint(actual_x, actual_y)
            
            line_widget.render(p_pix, target_pos)
            
            # Increase the tally by the height of the row + the gap between rows
            current_y += row_widget.height() + spacing
            
        
        p_pix.end()
        
        # Draw background (optional)
        painter.save()
        
        painter.restore()
        
        # Draw the pixmap at the printable origin (device coords)
        painter.drawPixmap(page_rect.x(), page_rect.y(), pix)
        
        # End drawing
        painter.end()
        
        # Restore hidden widgets
        for w in widgets_to_hide:
            w.show()


    def render_widget_recursive(self, widget, painter):
        
        if not widget.isHidden():
            pos = widget.mapTo(self, QtCore.QPoint(0, 0))
            widget.render(painter, pos)
    
        for child in widget.findChildren(QtWidgets.QWidget, "", QtCore.Qt.FindDirectChildrenOnly):
            self.render_widget_recursive(child, painter)
            




def generate_xor(input_bits=3):
    input_list, output_categorical_list, output_int_list = [], [], []
    for i in range(2**input_bits):
        bi = f'{i:0{input_bits}b}'
        input_list.append([float(i) for i in bi])
        
        o = int(bi[0])
        for j in bi[1:]:
            o = o ^ int(j)
            
        output_int_list.append(o)
        
        output_categorical_list.append(F.one_hot(torch.tensor(o), 2).float().numpy().tolist())
        
    return torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), torch.tensor(output_categorical_list), np.array(output_int_list)


def generate_sign(n_inputs, size):
    
    size1 = size // 4
    
    vals1 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
                     np.random.uniform(-0.25, 0.25, size=size1)]).T

    vals2 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
                     np.random.uniform(-0.25, 0.25, size=size1)]).T

    vals3 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
                     np.random.uniform(-0.25, 0.25, size=size1)]).T

    vals4 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
                     np.random.uniform(-0.25, 0.25, size=size1)]).T

    vals1 = vals1 + [-0.75, -0.75]
    vals2 = vals2 + [-0.75, 0.75]
    vals3 = vals3 + [0.75, -0.75]
    vals4 = vals4 + [0.75, 0.75]
    
    vals = np.concatenate([vals1, vals2, vals3, vals4])
    
    out = (vals[:,0] * vals[:,1] < 0) * 1
    o_categorical = F.one_hot(torch.tensor(out), 2)
    
    return torch.from_numpy(vals).float(), o_categorical.float(), out       
    
        
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
    
        
    
def main(experiment, hidden_struct=[2,2], epochs=70, 
         # point_size=2, polygon_color='gainsboro', 
         # random_seed = 33, lr=0.001, decimals=2):
         point_size=2, light_point_size=2,
         polygon_color='gainsboro', 
         random_seed=33, lr=0.001, decimals=4,
         show_graph=False,
         avoid_training=True,
         n_configs=10, num_inputs=5,
         fix_first_layer=False,
         show_normalized=None):


    np.seed = random_seed
    np.random.seed(random_seed)
    random.seed = random_seed
    torch.manual_seed(random_seed)
    random_generator = torch.Generator()
    random_generator.manual_seed(random_seed)
    
    ds_name = experiment

    use_saved_model_weights = True

    num_inputs = 2
    num_outputs = 2
        
    if experiment == 'xor':
        (X_train, y_train_categorical, y_train) = generate_xor(num_inputs)
        (X_test, y_test_categorical, y_test) = (X_train, y_train_categorical, y_train)
        pass
    elif experiment == 'sign':    
        (X_train, y_train_categorical, y_train) = generate_sign(num_inputs, 1000)
        (X_test, y_test_categorical, y_test) = generate_sign(num_inputs, 100)
    
    elif experiment == 'damero':
        damero_train_fname = 'damero_train_data.pickle'
        damero_test_fname = 'damero_test_data.pickle'
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

    train_dataset = TensorDataset(X_train, y_train_categorical)
   
    train_loader = DataLoader(train_dataset, batch_size=32)

    weights_file_name = f'{ds_name}_inputs_{num_inputs}_hidden_{hidden_struct}_epochs_{epochs}_seed_{random_seed}_lr_{lr}.pth'

    if avoid_training and use_saved_model_weights and os.path.isfile(weights_file_name):
        model = torch.load(weights_file_name, weights_only=False)
        print(
            f'Using pretrained classification model weights from file {weights_file_name}')
    else:
        print('Training classification model ... ', end='')

        model = FNNModule()

        '''
        Create the network 
        '''
        last_len = 0
        for i, len_layer in enumerate(hidden_struct):
            if i == 0: 
                model.add_layer(nn.Linear(num_inputs, len_layer))
                model.add_layer(nn.ReLU())
            else:
                model.add_layer(nn.Linear(last_len, len_layer))
                model.add_layer(nn.ReLU())
                
            last_len = len_layer 
        
        '''
        Ojo con la retirada del último layer lineal
        '''
        model.add_layer(nn.Linear(last_len, num_outputs))

        if not avoid_training:
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), 
                                          lr=lr)
            train_losses = []
            
                    
            if fix_first_layer:
                
                layer = model.layers[0]

                '''
                Último subido al DT de 2x2
                '''
                '''                
                layer.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
                layer.bias.data = torch.Tensor([-0.33, -0.33])

                for param in model.layers[0].parameters():
                    param.requires_grad = False
                '''

                '''
                Inicialización de primer layer para red 4x5
                Neurona 1: w01= -0.33, w11=1, w21=0; 
                Neurona 2: w02= -0.66, w12=1, w22=0; 
                Neurona 3: w03= -0.33, w13=0, w23=1; 
                Neurona 4: w04= -0.66, w14=0, w24=1
                '''
                
                layer.weight.data = torch.Tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
                layer.bias.data = torch.Tensor([-0.33, -0.66, -0.33, -0.66])

                for param in model.layers[0].parameters():
                    param.requires_grad = False
                
                pass
            
            
            print('YYYY')
            
            max_accuracy = 0 
            
            epoch_accuracy_list = []
            
            for epoch in range(epochs):
                running_loss = 0.0
                # running_accuracy = 0.0
                num_batches = 0
                correct = 0
                total = 0

                model.train()
                optimizer.zero_grad()

                outputs = model(X_train)
                
                targets = torch.tensor(y_train, dtype=torch.long)
                
                loss = criterion(outputs, targets)

                correct += (outputs.argmax(dim=1) == targets).sum().item()
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                epoch_accuracy = 100 * correct / len(targets)
                
                train_losses.append(epoch_loss)

                print(f'{epoch_accuracy:.2f}% ', end='')
                
                if epoch_accuracy > max_accuracy:
                    max_accuracy = epoch_accuracy
                
                epoch_accuracy_list.append(epoch_accuracy)
                
            model.eval()
            
            epoch_accuracy_list_fname = 'epoch_accuracy_list.pickle'
            with open(epoch_accuracy_list_fname, 'wb') as f:
                pickle.dump(epoch_accuracy_list, f)

            
            print(f'\nMAX ACCURACY = {max_accuracy:.2f}%')
            
            print()
            
            for param in model.parameters():
                param.requires_grad_(False)
                

                
            torch.save(model, weights_file_name)
            print('OK')

    
    # Define symbolic variables
    x, y = sp.symbols('x y')
    

    '''
    Realizamos predicciones y evaluamos el resultado de la FNN
    '''
    print('Generating predictions for test data ... ', end='')

    mlp_predictions = model(X_test).detach().numpy()
    y_mlp = np.argmax(mlp_predictions, axis=1)

    print('OK')

    test_accuracy = np.sum(y_mlp == y_test) / len(y_test)

    print(f'Test data accuracy = {test_accuracy:.5f}\n')

    '''
    Realizamos predicciones y evaluamos el resultado de la FNN
    '''
    print('Generating predictions for train data ... ', end='')

    mlp_predictions = model(X_train).detach().numpy()
    y_train_mlp = np.argmax(mlp_predictions, axis=1)

    print('OK')

    train_accuracy = np.sum(y_train_mlp == y_train) / len(y_train)

    print(f'Train data accuracy = {train_accuracy:.5f}\n')


    samples = np.random.choice(range(len(X_test)), 10000, replace=True)

    samples = [i for i in range(len(X_train))]

    t_list = []
    for sample in samples:
        t = time.time()    
        get_face_contrib_accelerated(X_train[sample], model)
        t_list.append(time.time() - t)
               
    print(f'Tiempo por ejecución de FACE: {np.mean(t_list):.06f} +/- {np.std(t_list):.06f} s')                 


    configs, config_samples, color_count = count_configurations(model, X_train, 
                                                                get_samples=True, 
                                                                include_last=False,
                                                                y_train=y_train)

    test_configs, test_config_samples, test_color_count = count_configurations(model, X_test, 
                                                                            get_samples=True, 
                                                                            include_last=False,
                                                                            y_train=y_test)

    if show_normalized is None:
            
        mmax = X_train.max(axis=0).values
        mmin = X_train.min(axis=0).values
        
        percent = 0.30
        
        x1_min = mmin[0] - percent * (mmax[0] - mmin[0])
        x1_max = mmax[0] + percent * (mmax[0] - mmin[0])
    
        x2_min = mmin[1] - percent * (mmax[1] - mmin[1])
        x2_max = mmax[1] + percent * (mmax[1] - mmin[1])
        
        x_range = (x1_min, x1_max)
        y_range = (x2_min, x2_max)
    
    else:
        x_range, y_range = show_normalized
    
    matplotlib.rcParams['figure.dpi'] = 100
    
    
    config = '10110011'
    config_index = list(configs).index(config)
    
    
    print_rules(model, X_train, y_train, x_range, y_range,
                 config_samples, decimals, config)
    
    plotter2 = QtImplicitEquationPlotter2(configs, config_samples, color_count,
                                      X_train, y_train, X_test,
                                      y_train_mlp, 
                                      train_accuracy, test_accuracy,
                                      model, point_size=point_size,
                                      polygon_color=polygon_color,
                                      lr=lr, epochs=epochs,
                                      experiment=experiment,
                                      seed=random_seed,
                                      decimals=decimals,
                                      x_range=x_range, y_range=y_range,
                                      show_boundaries=True,
                                      show_graph=True,
                                      color_class='magenta',
                                      light_point_size=light_point_size,
                                      light_point_alpha=0.4,
                                      show_normalized=show_normalized,
                                      test_configs=test_configs, 
                                      test_color_count=test_color_count,
                                      config_index=config_index)
    
    plt.show()
    
    pass 

    # if len(configs) > 0:
        # root = tk.Tk()
        # root.geometry('1100x600')
        # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
        # app = QtWidgets.QApplication(sys.argv)

        # plotter = QtImplicitEquationPlotter(configs, config_samples, color_count,
        #                                   X_train, y_train, X_test,
        #                                   y_train_mlp, 
        #                                   train_accuracy, test_accuracy,
        #                                   model, point_size=point_size,
        #                                   polygon_color=polygon_color,
        #                                   lr=lr, epochs=epochs,
        #                                   experiment=experiment,
        #                                   seed=random_seed,
        #                                   decimals=decimals,
        #                                   x_range=x_range, y_range=y_range,
        #                                   show_boundaries=True,
        #                                   show_graph=False,
        #                                   color_class='magenta',
        #                                   light_point_size=light_point_size,
        #                                   light_point_alpha=0.4,
        #                                   show_normalized=show_normalized,
        #                                   test_configs=test_configs, 
        #                                   test_color_count=test_color_count,
        #                                   config_index=config_index)
        

        # plotter.show() 

        # plotter.clear_equations()
        # plotter.replot()
        
        # plotter.fig.savefig('classification_output/classifcation_output_canvas.svg', bbox_inches='tight') #, pad_inches=0.1)

        
        # sys.exit(app.exec_())
        
    

if __name__ == "__main__":

    '''
    Paint damero
    '''
    

    '''
    Red 4x4 para Paper
    '''
    main(experiment='damero', hidden_struct=[4,4], epochs=20000, #25000 y 37000
         point_size=5, light_point_size=2,
         show_normalized=([0.0, 1.0], [0.0, 1.0]),
         lr=0.001, random_seed=13112, decimals=4, avoid_training=True)
    
    # main(experiment='damero', hidden_struct=[2,2], epochs=10000, 
    #      point_size=5, lr=0.0005, random_seed=22, decimals=4, avoid_training=False,
    #      fix_first_layer=True)
    
    '''
    # Úlitma llevada al DT
    '''
    # main(experiment='damero', hidden_struct=[2,2], epochs=40000, 
    #      show_normalized=([0.0, 1.0], [0.0, 1.0]),
    #      point_size=5, lr=0.0001, random_seed=112, decimals=4, avoid_training=True,
    #      fix_first_layer=False)
    

    '''
    Red de 4x5 con pesos libres. Acc=0.98
    '''
    # main(experiment='damero', hidden_struct=[4,5], epochs=40000, 
    #      show_normalized=([0.0, 1.0], [0.0, 1.0]),
    #      point_size=5, lr=0.001, random_seed=112, decimals=4, avoid_training=True,
    #      fix_first_layer=False)
    
    '''
    Red de 4x5 con pesos fijados. Acc=0.94
    '''
    # main(experiment='damero', hidden_struct=[4,5], epochs=80000, 
    #      point_size=5, lr=0.0001, random_seed=1121, decimals=4, avoid_training=True,
    #      fix_first_layer=True)
    
    '''
    Red de 4x5 con pesos fijados. Acc=0.96
    '''
    # main(experiment='damero', hidden_struct=[4,5], epochs=140000, 
    #      point_size=5, lr=0.0001, random_seed=1121, decimals=4, avoid_training=False,
    #      fix_first_layer=True)
    
    
    '''
    Red de 4x5 con pesos fijados. Acc=0.98
    '''
    # main(experiment='damero', hidden_struct=[4,5], epochs=180000, 
    #      show_normalized=([0.0, 1.0], [0.0, 1.0]),
    #      point_size=5, lr=0.0001, random_seed=1121, decimals=4, avoid_training=True,
    #      fix_first_layer=True)
    
    
    # '''
    # Red de 4x5 con pesos fijados. Acc=0.98
    # '''
    # main(experiment='damero', hidden_struct=[4,5], epochs=220000, 
    #      point_size=5, lr=0.0001, random_seed=1121, decimals=4, avoid_training=False,
    #      fix_first_layer=True)
    
    # main(experiment='sign', hidden_struct=[2,2], epochs=7000, point_size=5, lr=0.0001, random_seed=77, decimals=4, avoid_training=False)


    # main(experiment='sign', hidden1=2, hidden2=2, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=4)

    # main(experiment='sign', hidden1=2, hidden2=2, epochs=4000, point_size=5, lr=0.0001, random_seed=3333, decimals=4)

    # main(experiment='sign', hidden1=2, hidden2=2, epochs=14000, point_size=5, lr=0.0001, random_seed=77, decimals=4)


    # main(experiment='sign', hidden1=2, hidden2=3, epochs=17000, point_size=5, lr=0.0001, random_seed=33, decimals=4)

    # main(experiment='sign', hidden1=3, hidden2=3, epochs=17000, point_size=5, lr=0.0001, random_seed=33, decimals=4)



    # main(experiment='sign', hidden=4, epochs=12000, point_size=5, lr=0.0001, random_seed=3, decimals=4)

    
    # main(experiment='sign', hidden=3, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=4)

    # main(experiment='sign', hidden=2, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=2)

    # main(experiment='sign', hidden=2, epochs=6000, point_size=5, lr=0.0001, random_seed=443, decimals=2)
    
    # main(experiment='sign', hidden=6, epochs=3000, point_size=5, lr=0.0001, random_seed=11, decimals=4)
    
    # main(experiment='sign', hidden=5, epochs=16000, point_size=5, lr=0.0001, random_seed=3, decimals=4)
    
    # main(experiment='xor', hidden=3, epochs=64000, point_size=10, 
    #      random_seed=3, lr=0.0001)
    
    # main(experiment='xor', hidden=4, epochs=20000, point_size=10, 
    #      random_seed=3, lr=0.0001)

    # main(experiment='sign', hidden=6, epochs=30000, point_size=2, lr=0.0001)