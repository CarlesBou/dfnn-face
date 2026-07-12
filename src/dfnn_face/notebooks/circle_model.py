# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
import os 
# import matplotlib

# import time
# import numpy as np
# import random
# import cdd
# from collections import Counter, OrderedDict

# from sympy import Symbol
# import sympy as sp


# import cvxpy as cp
from utils.face_torch import FNNModule
# from utils.face_torch import get_face_contrib_accelerated
# from utils.face_torch import count_configurations

# from utils.graph import ImplicitEquationPlotter
# from utils.graph import get_eq_list_new

# from utils.ineq import str_equ_out

# from utils.gface_torch import get_rules

# import pickle


def get_circle_model(num_inputs, num_outputs, hidden_struct, 
                     epochs, random_seed, lr,
                     X_train, y_train, 
                     train_loader, r_circle=0.75):
    
    ds_name = 'circle'
    use_saved_model_weights = True
    avoid_training = False

    weights_file_name = f'models/{ds_name}_inputs_{num_inputs}_hidden_{hidden_struct}_epochs_{epochs}_seed_{random_seed}_lr_{lr}_r_circle_{r_circle}.pth'

    if use_saved_model_weights and os.path.isfile(weights_file_name):
        model = torch.load(weights_file_name, weights_only=False)
        print(
            f'Using pretrained classification model weights from file {weights_file_name}')
        return model
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
            
                    
            # if fix_first_layer:
                
            #     layer = model.layers[0]

            #     '''
            #     Último subido al DT de 2x2
            #     '''
            #     '''                
            #     layer.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
            #     layer.bias.data = torch.Tensor([-0.33, -0.33])

            #     for param in model.layers[0].parameters():
            #         param.requires_grad = False
            #     '''

            #     '''
            #     Inicialización de primer layer para red 4x5
            #     Neurona 1: w01= -0.33, w11=1, w21=0; 
            #     Neurona 2: w02= -0.66, w12=1, w22=0; 
            #     Neurona 3: w03= -0.33, w13=0, w23=1; 
            #     Neurona 4: w04= -0.66, w14=0, w24=1
            #     '''
                
            #     layer.weight.data = torch.Tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
            #     layer.bias.data = torch.Tensor([-0.33, -0.66, -0.33, -0.66])

            #     for param in model.layers[0].parameters():
            #         param.requires_grad = False
                
            #     pass
            
            
            # print('YYYY')
            
            max_accuracy = 0 
            
            epoch_accuracy_list = []
            
            for epoch in range(epochs):
                running_loss = 0.0
                # running_accuracy = 0.0
                # num_batches = 0
                correct = 0
                # total = 0

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
            
            # epoch_accuracy_list_fname = 'epoch_accuracy_list.pickle'
            # with open(epoch_accuracy_list_fname, 'wb') as f:
            #     pickle.dump(epoch_accuracy_list, f)

            
            print(f'\nMAX ACCURACY = {max_accuracy:.2f}%')
            
            print()
            
            for param in model.parameters():
                param.requires_grad_(False)
                

                
            torch.save(model, weights_file_name)
            print('OK')
        
        return model
        


