# -*- coding: utf-8 -*-
"""
Created on 2026

@author: Carles-Bou
"""

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from circle_dataset import get_circle_dataset
from circle_model   import get_circle_model 

from utils.face_torch  import FNNModule
from utils.face_torch  import count_configurations
from utils.set_random  import set_random_seed
from utils.graph       import ImplicitEquationPlotter
from utils.gface_torch import get_rules
from utils.print_texts import print_bold

'''
Set model ininitalization variables
'''
num_inputs = 2
num_outputs = 2

r_circle = 0.50
   
# hidden_struct=[4,4]
# epochs=2000 
# lr=0.001
# random_seed=132312

# Accuracy 0.985
hidden_struct=[4,4]
epochs=7500 
lr=0.001
random_seed=29

# Accuracy 0.987
hidden_struct=[4,4]
epochs=10000 
lr=0.001
random_seed=11

# Accuracy 0.9822
hidden_struct=[4,4]
epochs=10000 
lr=0.001
random_seed=11
r_circle = 0.50


# Test Accuracy 0.9892
hidden_struct=[4,4]
epochs=10000 
lr=0.001
random_seed=291
r_circle = 0.50

# Accuracy 0.9832
# hidden_struct=[5,5]
# epochs=10000 
# lr=0.001
# random_seed=29


'''
Set random seed 
'''
set_random_seed(random_seed)

'''
Load Circle dataset (generate it if it does not exist)
'''
X_train, y_train_categorical, y_train, \
    X_test, y_test_categorical, y_test = get_circle_dataset(num_inputs, random_seed, r_circle=r_circle) 

train_dataset = TensorDataset(X_train, y_train_categorical)
train_loader = DataLoader(train_dataset, batch_size=32)


'''
Load Model (generate it if it does not exist)
'''
model = get_circle_model(num_inputs, num_outputs, hidden_struct, \
                         epochs, random_seed, lr, \
                         X_train, y_train, train_loader, r_circle=r_circle)
    

'''
Get model accuracy for test dataset
'''
print('Generating predictions for test data ... ', end='')

mlp_predictions = model(X_test).detach().numpy()
y_mlp = np.argmax(mlp_predictions, axis=1)

print('OK')

test_accuracy = np.sum(y_mlp == y_test) / len(y_test)

print(f'Test data accuracy = {test_accuracy:.5f}\n')



# '''
# Get model accuracy for train dataset
# '''
# print('Generating predictions for train data ... ', end='')

# mlp_predictions = model(X_train).detach().numpy()
# y_train_mlp = np.argmax(mlp_predictions, axis=1)

# print('OK')

# train_accuracy = np.sum(y_train_mlp == y_train) / len(y_train)

# print(f'Train data accuracy = {train_accuracy:.5f}\n')



'''
Get configurations (activation patterns) for train dataset
'''
# configs, config_samples, color_count = count_configurations(model, X_train, y_train)
configs, config_samples, color_count = count_configurations(model, X_test, y_test)

 

'''
Get rules for specific config (activation pattern)
'''
config = '10110011'
# config = '00100110'
# config = '00110011'
# config = '00110110'
# config = '00110111'
# config = '10100110'
# config = '10110110'
# config = '10110111'

# # config = '1101011110'

config = '11001111'
config = '11110000'
# config = '11011100'
# config = '11110011'
# config = '01111100'

# config = '01110000'

# config = list(configs.keys())[-1]

normalized_range = ([0.0, 1.0], [0.0, 1.0])

# rule_texts = get_rules(model, X_test, y_test, 
#                        normalized_range,
#                        config_samples, config)

# print_bold(rule_texts['rule_antecedents'][0])
# print(rule_texts['rule_antecedents'][1])

# print_bold(rule_texts['rule_consequent'][0])
# print(rule_texts['rule_consequent'][1])

# print_bold(rule_texts['activation_region'][0])
# print(rule_texts['activation_region'][1])



# '''
# Plot config (activation pattern)
# '''

# plotter_options = {
#     'configs': configs,
#     'config_samples': config_samples, 
#     'X': X_test, 
#     'y': y_test, 
#     'model': model,
#     'normalized_range': normalized_range,
#     'point_size': 1.5,
#     'light_point_size': 1.5,
#     'light_point_alpha': 0.5,
#     # 'class_linewidth': 2,
#     # 'neuron_linewidth': 1.6,
#     'class_linewidth': 4,
#     'neuron_linewidth': 3,
#     'hidden_struct': hidden_struct,
#     'arrow_length': 0.5
# }

# plotter = ImplicitEquationPlotter(**plotter_options)

# plotter.show(config)


for config in configs:
    
    print(f'CONFIG = {config}')
    
    rule_texts = get_rules(model, X_test, y_test, 
                           normalized_range,
                           config_samples, config)

    print_bold(rule_texts['rule_antecedents'][0])
    print(rule_texts['rule_antecedents'][1])

    print_bold(rule_texts['rule_consequent'][0])
    print(rule_texts['rule_consequent'][1])

    print_bold(rule_texts['activation_region'][0])
    print(rule_texts['activation_region'][1])
    
    print() 
    
    plotter_options = {
        'configs': configs,
        'config_samples': config_samples, 
        'X': X_test, 
        'y': y_test, 
        'model': model,
        'normalized_range': normalized_range,
        'point_size': 1.5,
        'light_point_size': 1.5,
        'light_point_alpha': 0.5,
        # 'class_linewidth': 2,
        # 'neuron_linewidth': 1.6,
        'class_linewidth': 4,
        'neuron_linewidth': 3,
        'hidden_struct': hidden_struct,
        'arrow_length': 0.4,
        'arrow_head_size': 0.15,
    }
    
    plotter = ImplicitEquationPlotter(**plotter_options)
    
    plotter.show(config)
    