# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 06:44:39 2025

@author: Carles
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter, OrderedDict

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


def run_layers(layer_input, model, return_weighted=True):
    H_list = []
    W_list = []
    I_v_list = []
    pseudo_I_list = []    
    O_list = []
    
    x_sample = np.hstack((1, layer_input))
    
    l = 0
    nlayers = len(model.layers)

    while True:
        if isinstance(model.layers[l], torch.nn.modules.linear.Linear):
            linear_layer = model.layers[l]
            
            activation_layer = None 
            
            if l != nlayers - 1:
                activation_layer = model.layers[l + 1]
                l += 1 
        
            w = linear_layer.weight.detach().numpy()
            bias = linear_layer.bias.detach().numpy()
            
            w_T_ext = get_transposed_ext(w, bias)
            W_list.append(w_T_ext)
            
            if return_weighted:
                O = w_T_ext * x_sample
                O_list.append(O[1:])
            
            pseudo_I, H, I_v = get_I_activation(linear_layer, activation_layer, x_sample, w_T_ext)
            
            H_list.append(H)
            
            I_v_list.append(I_v)
            pseudo_I_list.append(pseudo_I)
            
            x_sample = H
        
        l += 1
   
        if l == nlayers:
            break
        
    return W_list, pseudo_I_list, I_v_list, H_list, O_list
    

'''
Obtemos la matriz ampliada con el bias incorporado
'''
def get_transposed_ext(w, bias=None):
    if not isinstance(bias, np.ndarray):
        bias = np.zeros(w.shape[0])
      
    w_ext = np.hstack((bias.reshape(-1,1), w)).T

    zeros = np.zeros(w.shape[1] + 1)
    zeros[0] = 1 

    zeros = zeros.reshape(-1,1)
    
    return np.hstack((zeros, w_ext)).T
    
    


def get_I_linear(x, w): 
    # print('Aquí')
    Sw = w @ x
    I = np.eye(w.shape[0])
    
    return I, Sw, np.ones(w.shape[0])


# def get_I_relu(x, w, alpha=0.3):
    
#     Sw = w @ x
        
#     I_v = Sw > 0
#     I_v = I_v * 1
    
#     if alpha > 0:
#         mul_neg = Sw <= 0
#         mul_neg = mul_neg * alpha  
#         I_v = I_v + mul_neg 

#     return I_v


def get_I_relu(x, w, alpha=0.3):
    
    Z = w @ x
        
    mul_pos = Z > 0
    mul_pos = mul_pos * 1
    
    I_v = mul_pos
    
    if alpha > 0:
        mul_neg = Z <= 0
        mul_neg = mul_neg * alpha  
        I_v = mul_pos + mul_neg 

    I = np.diag(I_v)
    
    H = Z * I_v
    
    return I, H, I_v
    


def get_I_activation(linear_layer, activation_layer, x, w):
    if isinstance(linear_layer, torch.nn.modules.linear.Linear):
        if activation_layer is not None: 
            if isinstance(activation_layer, torch.nn.modules.activation.ReLU):
                return get_I_relu(x, w, alpha=0)
            elif isinstance(activation_layer, torch.nn.modules.activation.LeakyReLU):
                return get_I_relu(x, w, activation_layer.negative_slope)
            else:
                raise ValueError('Error 1')
        else:
            return get_I_linear(x, w)
    else:
        raise ValueError('Error 2')



def get_face_contrib_accelerated(x_sample, model, return_weighted=True, return_lists=True):

    if isinstance(x_sample, (pd.core.series.Series)):
        x_sample = x_sample.to_numpy()
    elif isinstance(x_sample, (torch.Tensor)):
        x_sample = x_sample.numpy()
    
    W_list, I_list, I_v_list, H_list, O_list = run_layers(x_sample, model, 
                                                          return_weighted=return_weighted)

    '''
    Antiguo cálculo de contribuciones 
    '''    
    # contrib_partial = I_v_list[len(I_list)-1][:, None] * W_list[len(I_list)-1] 
    # # if return_weighted:
    # #     contrib_partial[:,1:] = contrib_partial[:,1:] * x_sample
    # contrib_list.insert(0, contrib_partial)
    
    # for I_index in range(len(I_list)-2, -1, -1):
    #     contrib_partial = I_v_list[I_index][:, None] * W_list[I_index]
    #     # if return_weighted:
    #     #     contrib_partial[:,1:] = contrib_partial[:,1:] * x_sample
    #     contrib_list.insert(0, contrib_partial)

    # contrib = contrib_list[-1]
    # for I_index in range(len(I_list)-2, -1, -1):
    #     contrib = contrib @ contrib_list[I_index]
            
    contrib_list = []
    
    contrib_partial = W_list[0]
    contrib_list.append(contrib_partial)
    
    for I_index in range(len(I_list) - 1):
        contrib_partial = (W_list[I_index+1] * I_v_list[I_index][None, :]) @ contrib_partial
        # contrib_partial = W_list[I_index+1] @ I_list[I_index] @ contrib_partial
        contrib_list.append(contrib_partial)
        
    contrib = contrib_partial
    
    if return_weighted:
        contrib[:,1:] = contrib[:,1:] * x_sample
        
    if not return_lists:
        return contrib[1:]
    else:
        return contrib[1:], W_list, I_v_list, contrib_list, H_list, O_list, I_list
    
    
def count_configurations(model, X, y_train, get_samples=True, include_last=False, get_cm=False):
    configs = []
    samples = {}
    color_count = {}
    # cm = {}
    
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


