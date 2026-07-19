# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
#from matplotlib import pyplot as plt
import math
#import time
# from keras.datasets import mnist
import numpy as np
import random
import cdd
#from scipy.spatial import ConvexHull
from collections import Counter, OrderedDict
from utils.ineq import get_face_contrib_accelerated

from sympy import Symbol
#from sympy import plot_implicit, plot, N, Float
import sympy as sp

# import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5 import QtPrintSupport,QtSvg
# from PyQt5.QtGui import QPainter
# from PyQt5.QtGui import QPdfWriter, QPageSize
# from PyQt5.QtCore import QSizeF, QRectF, QMarginsF
from PyQt5.QtPrintSupport import QPrinter
# from PyQt5.QtGui import QPainter, QImage, QPixmap
from PyQt5.QtGui import QPainterPath
from PyQt5.QtCore import Qt
# from PyQt5.QtCore import QRect, QSize
# from PyQt5.QtGui import QPainter 
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QTextDocument

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

import sys


# from tensorflow.keras.utils import to_categorical


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
    
    Sw = w @ x
        
    mul_pos = Sw > 0
    mul_pos = mul_pos * 1
    
    I_v = mul_pos
    
    if alpha > 0:
        mul_neg = Sw <= 0
        mul_neg = mul_neg * alpha  
        I_v = mul_pos + mul_neg 

    I = np.diag(I_v)
    
    H = Sw * I_v
    
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
        
  
def run_layers_old(layer_input, model):
    H_list = []
    W_list = []
    I_v_list = []
    pseudo_I_list = []    

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
            
            pseudo_I, H, I_v = get_I_activation(linear_layer, activation_layer, x_sample, w_T_ext)
            
            H_list.append(H)
            
            I_v_list.append(I_v)
            pseudo_I_list.append(pseudo_I)
            
            x_sample = H
        
        l += 1
   
        if l == nlayers:
            break
        
    return W_list, pseudo_I_list, I_v_list, H_list

    
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
            
            if not return_weighted:
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
    



def get_face_contrib_accelerated_old_funciona_con_1_layer(x_sample, model, return_weighted=True, return_lists=False):
    
    contrib_list = []
    
    if isinstance(x_sample, (pd.core.series.Series)):
        x_sample = x_sample.to_numpy()
    elif isinstance(x_sample, (torch.Tensor)):
        x_sample = x_sample.numpy()
    
    W_list, I_list, I_v_list, H_list, O_list = run_layers(x_sample, model, 
                                                          return_weighted=return_weighted)
    
    for I_index in range(len(I_list)):
        if I_index == 0:
            # contrib = I_list[I_index] @ W_list[I_index]
            contrib = I_v_list[I_index][:, None] * W_list[I_index]
            contrib_list.append(contrib[1:])
        else:
            # contrib = I_list[I_index] @ W_list[I_index] @ contrib
            contrib = (I_v_list[I_index][:, None] * W_list[I_index]) @ contrib
            contrib_list.append(contrib[1:])

    if return_weighted:
        contrib[:,1:] = contrib[:,1:] * x_sample
        
    if not return_lists:
        return contrib[1:]
    else:
        return contrib[1:], W_list, I_v_list, contrib_list, H_list, O_list
    



def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets.argmax(dim=1)).sum().item()
    return correct / total

    


def test_valid_config(W_list, config, boundaries):

    w0 = W_list[0][1:].copy()
    
    sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
    sign_vec[sign_vec == 0] = -1
    
    matrix = w0 * sign_vec
    
    borders = boundaries
    
    new_matrix = np.concatenate((matrix, borders))

    H_bounded = cdd.matrix_from_array(new_matrix, rep_type=cdd.RepType.INEQUALITY)
    H_poly_bounded = cdd.polyhedron_from_matrix(H_bounded)
    generator_bounded = cdd.copy_generators(H_poly_bounded)
    
    if len(generator_bounded.array) > 0:
        return True
    
    return False
    

    
def new_count_configurations(model, X, get_samples=False, include_last=True, 
                             order=False, boundaries=None):
    
    
    boundaries = np.array([
                        [-boundaries[0],  1.],
                        [boundaries[1], -1.]])
    
    contrib, W_list, I_vecs, contrib_list, H_list, O_list =  get_face_contrib_accelerated(
                                                        X[0], 
                                                        model, 
                                                        return_weighted=False, 
                                                        return_lists=True) 

    configs = {}
    
    '''
    Compute the total number of neurons in the network
    '''
    neurons = 0 
    for l in model.layers[:-1]:
        if not isinstance(l, torch.nn.modules.activation.ReLU):
             neurons += l.out_features
    
    for c in range(2 ** neurons):
        config = f'{c:#0{neurons+2}b}'[2:]
        
        if test_valid_config(W_list, config, boundaries):
            configs[config] = 0 
            
    
    return configs




def count_configurations(model, X, get_samples=False, include_last=True, order=False):
    configs = []
    samples = {}
    # num_outputs = model.layers[-1].out_features
    num_outputs = 1
    
    for sample, x in enumerate(X):
        contrib, W_list, I_vecs, _, _, _, _ = get_face_contrib_accelerated(x, model, 
                                                                           return_lists=True)
            
        config = ''
        for vec in I_vecs:
            for v in vec[1:]:
                config = config + str(int(v))

        if not include_last:
            config = config[:-num_outputs]
            
        if configs.count(config) == 0:
            samples[config] = [sample]
        else:
            samples[config].append(sample)
                      
        configs.append(config)

    '''
    En samples devolvemos un diccionario con la configuración y una tupla con:
        último ejemplo de la configuración (índice en X)
        cuenta total de ejemplos de la configuración
    '''
    
    counter_configs = Counter(configs)
         
    
    if not get_samples:
        return counter_configs
    else:
        return counter_configs, samples


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



def get_eq_list_regression(config, config_samples, X_train, model, boundaries, decimals=2, bounded=True):
    
    sample = config_samples[config][0]
    
    contrib, W_list, I_vecs, contrib_list, H_list, O_list, I_list =  get_face_contrib_accelerated(
                                                        X_train[sample], 
                                                        model, 
                                                        return_weighted=False, 
                                                        return_lists=True) 

    # w0 = W_list[0][1:].copy()
    
    w0 = contrib_list[0][1:]
    
    for i in range(len(contrib_list) - 2):
        w0 = np.concatenate((w0, contrib_list[i + 1][1:]))
        
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
    
    inequality = np.sort(np.array(generator_bounded.array)[:,1])

    eq_list = []
    
    x = Symbol('x')
    y = Symbol('y')
    
    
    # w1 =  contrib_list[1] @ I_list[0] @ contrib_list[0]
    
    # w0 = np.concatenate((w0[:4], w1))
    
    for i, v in enumerate(w0):
        s = sp.Eq(x + 222, 0.)
        s = s.subs(x, v[1] * x)
        s = s.subs(222, v[0])
        
        # expr = s.lhs
        # coeff_x = abs(expr.coeff(x))
        # new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
        
        # eq_list.append((f'Eq{i+1}', new_equation, sign_vec[i][0], contrib))
        eq_list.append((f'Eq{i+1}', s, sign_vec[i][0], contrib))
        

    return eq_list, \
            w0, contrib, inequality
            

        


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
        # self.setStyleSheet("background-color: lightblue;")
        
    # def sizeHint(self):
    #     print('sizeHint called!!')
    #     # This tells the layout: "I need at least this much space"
    #     return QtCore.QSize(100, 20)

    def paintEvent(self, event):
        # print('Paint event seen')
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QColor(self.background_color))
        # print(f'Rect={self.rect}')
        
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Ensure we use a proper QColor
        pen = QtGui.QPen(QtGui.QColor(self.color))
        pen.setWidth(self.linewidth)
        
        # Explicitly handle Matplotlib's 'dash' string
        if self.linestyle in ['dashed', '--', 'dash']:
            pen.setStyle(QtCore.Qt.DashLine)
        elif self.linestyle in ['dotted', ':', 'dot']:
            pen.setStyle(QtCore.Qt.DotLine)
        else:
            pen.setStyle(QtCore.Qt.SolidLine)
            
        painter.setPen(pen)
        # Draw the line in the vertical center of the widget
        painter.drawLine(5, 10, 45, 10)
        
        
    def paint_to(self, painter, offset):
        print(f'Paint_to called')
        
        painter.save()
    
        # Guardar la transformación actual
        transform = painter.transform()
    
        # Resetear transformaciones (coordenadas reales)
        painter.resetTransform()
    
        # Calcular coordenadas absolutas reales
        x1 = offset.x() + 5
        y  = offset.y() + self.height() // 2
        x2 = offset.x() + self.width() - 5
    
        # Configurar el pen
        pen = QtGui.QPen(QtGui.QColor(self.color))
        pen.setWidth(self.linewidth)
    
        if self.linestyle in ['dashed', '--', 'dash']:
            pen.setStyle(QtCore.Qt.DashLine)
        elif self.linestyle in ['dotted', ':', 'dot']:
            pen.setStyle(QtCore.Qt.DotLine)
        else:
            pen.setStyle(QtCore.Qt.SolidLine)
    
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
    
        # Dibujar la línea en coordenadas absolutas reales
        painter.drawLine(x1, y, x2, y)
    
        # Restaurar la transformación original
        painter.setTransform(transform)
    
        painter.restore()
        
        print(f'Painting event in LineStyle')
    


        
        

        
class DashLineWidget(QtWidgets.QWidget):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = QtGui.QColor(color)
        self.setFixedHeight(12)  # o lo que uses

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(self.color, 2, QtCore.Qt.DashLine)
        painter.setPen(pen)
        y = self.height() // 2
        painter.drawLine(0, y, self.width(), y)
        
class QtImplicitEquationRegressionPlotter(QtWidgets.QMainWindow):
    # def __init__(self, configs, config_samples, 
    #              X_train, y_train, X_test, 
    #              mlp_train, model,
    #              train_mse=0, test_mse=0,
    #              point_size=3, title='Config sin definir',
    #              polygon_color='gainsboro',
    #              experiment='', epochs=5000, lr=0.0001, seed=33,
    #              decimals=2, show_boundaries=False,
    #              x_range=(-2.5,5), noise=None,
    #              train_rmse_dict={},
    #              test_rmse_dict={},
    #              test_configs=None):
    def __init__(self, configs, config_samples, X_train, y_train, X_test, 
             mlp_train, model, train_mse=0, test_mse=0, point_size=3, 
             title='Config sin definir', polygon_color='gainsboro',
             experiment='', epochs=5000, lr=0.0001, seed=33,
             decimals=2, show_boundaries=False, x_range=(-2.5, 5), 
             noise=None, train_rmse_dict={}, test_rmse_dict={}, 
             test_configs=None):
        
        super().__init__()
        
        self.finish = False
        self.index = 0
        self.plotter = None
    
        self.configs = configs
        self.test_configs = test_configs
        self.config_samples = config_samples
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_predictions = mlp_train
        
        self.config_colors = {}
        self.check_dict = {}
        self.line_dict = {}
        
        self.config_struct = self.get_config_structure(model)
        
        # Set colors: green = correctly predicted, red = otherwise
        # self.y_colors = np.where(y_train == mlp_train, 'green', 'red')

        # Set colors: green = class 0, else red
        # self.y_colors = np.where(y_train, 'green', 'red')

        self.train_mse = train_mse
        self.test_mse = test_mse
        self.model = model
        
        # self.root = root
        # self.root.title('Interactive Activation Pattern plot')
        
        self.title = title
        self.point_size = point_size
        self.lr = lr
        self.epochs = epochs
        self.experiment = experiment
        self.seed = seed
        self.decimals = decimals
        self.show_boundaries = show_boundaries
        self.x_range = x_range
        
        if noise is not None:
            self.use_noise = True
            self.noise_mean = noise[0]
            self.noise_sigma = noise[1]
        else:
            self.use_noise = False
            
        self.color_index = 0
        self.color_list = ['blue', 'olive', 'brown', 'magenta', 'lightcoral', 
                      'pink', 'orange', 'salmon', 'lightblue', 'cyan']
        
        # self.boundaries = np.array([
        #                     [3.,  1.,  0.],
        #                     [3., -1.,  0.],
        #                     [3.,  0.,  1.],
        #                     [3.,  0, -1.]])

        # boundaries = [-2.5, 5]
        
        self.boundaries = np.array([
                            [-self.x_range[0],  1.],
                            [self.x_range[1], -1.]])

        # Define symbolic variables
        self.x, self.y = sp.symbols('x y')
        
        # Initialize empty equations dictionary and scatter points
        self.pan_start = [None, None]
        self.var_dict = {}
        self.row_widgets = {}
        self.check_rows_dict = {}
        self.equations = {}
        self.lines = {}
        self.vlines = OrderedDict()
        self.arrows_config = {}
        self.contour_sets = {}
        
        self.scatter_points = None
        # self.contour_sets = {}
        
        self.solve_order = None
        
        # self.arrows_config = {}
        # self.poly = {}
        

        
        # Store button callback
        self.button_callback = None
    
        self.polygon_color = polygon_color
        
        self.train_rmse_dict = train_rmse_dict
        self.test_rmse_dict = test_rmse_dict

        # self.subscript_map = '₀₁₂₃₄₅₆₇₈₉'
        self.subscript_map = '0123456789'
        
        self.get_vlines()
        
        # self.reorder_configs()
        
        # self.set_colors()
        
        
        # Create all GUI elements
        self.setWindowTitle('Interactive Activation Pattern plot')
        self.setMinimumSize(1000, 760)

        self.create_gui()
        
        # # Set window close protocol
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.replot(ini=True)
        
    def on_resize(self, event):
        # event is a ResizeEvent instance
        # event.width and event.height are the new pixel size of the canvas
        # print(f"Canvas resized to {event.width} × {event.height} pixels")
        
        # Common actions:
        # 1. Re-apply tight_layout (very useful to prevent label cutoff)
        event.canvas.figure.tight_layout()
        
        # 2. Redraw (non-blocking version is usually enough)
        event.canvas.draw_idle()          # preferred in interactive apps
        # or event.canvas.draw()          # blocking, use only if needed
        
        # 3. Custom logic (e.g. adjust font sizes, reposition annotations, etc.)
        # event.canvas.figure.axes[0].set_title(f"Size: {event.width}x{event.height}")
        
    def create_gui(self):
        # --- Main Window Setup ---
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # This horizontal container holds the three main pillars
        self.top_container = QtWidgets.QHBoxLayout()
        
        # --- COLUMN 1: Matplotlib Figure (Left) ---
        self.fig = Figure(figsize=(4.5, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.top_container.addWidget(self.canvas, stretch=2)
        
        # Connect Matplotlib Events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        
        # self.canvas.mpl_connect('resize_event', self.on_resize)
        
        # --- COLUMN 2: Middle Box (Equations & Navigation) ---
        self.middle_column_layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Samples Checkbox (Anchored to top)
        self.scatter_checkbox = QtWidgets.QCheckBox("Samples")
        self.scatter_checkbox.setStyleSheet("font-family: monospace; font-size: 12px;")
                                    
        self.scatter_checkbox.setChecked(True)
        self.scatter_checkbox.stateChanged.connect(self.update_plot)
        self.middle_column_layout.addWidget(self.scatter_checkbox)
        
        # 2. Scrollable Area for Dynamic Checkboxes (The "Lines Box")
        self.check_scroll = QtWidgets.QScrollArea()
        self.check_widget = QtWidgets.QWidget()
        self.check_layout = QtWidgets.QVBoxLayout(self.check_widget)
        self.check_layout.setAlignment(QtCore.Qt.AlignTop) 
        self.check_scroll.setWidget(self.check_widget)
        self.check_scroll.setWidgetResizable(True)
        # self.check_scroll
        self.middle_column_layout.addWidget(self.check_scroll)
        
        # self.check_scroll.viewport().setStyleSheet("background-color: transparent;")
        self.check_scroll.setStyleSheet("QScrollArea { border: 1px solid #666666; }")
        
        self.check_widget.setStyleSheet("background-color: #f1f1f1")
        
        # 3. Navigation Buttons (Immediately below the box)
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("◀")
        self.btn_forward = QtWidgets.QPushButton("▶")
        
        btn_style = "background-color: #4499FF; color: white; font-weight: bold; font-size: 14pt; padding: 2px;"
        self.btn_back.setStyleSheet(btn_style)
        self.btn_forward.setStyleSheet(btn_style)
        
        # self.btn_layout.addWidget(self.btn_back)
        # self.btn_layout.addWidget(self.btn_forward)
        # self.middle_column_layout.addLayout(self.btn_layout)
        
        # 1. Create a container widget for the buttons
        nav_container = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 3, 0, 3) # Tight vertical spacing
        nav_layout.setSpacing(10)                # Gap between the two buttons

        # 2. Set the fixed size (adjust 70x32 to your preference)
        self.btn_back.setFixedSize(45, 25)
        self.btn_forward.setFixedSize(45, 25)

        # 3. Build the layout: Stretch - Button - Button - Stretch
        # nav_layout.addStretch(1)  # Pushes from the left
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_forward)
        # nav_layout.addStretch(1)  # Pushes from the right

        # 4. Add the container to your main sidebar layout
        self.middle_column_layout.addWidget(nav_container)
        
        # 4. The Stretch (Pushes the whole group above it to the top)
        self.middle_column_layout.addStretch(1)
        
        # Add the completed middle pillar to the top container
        self.top_container.addLayout(self.middle_column_layout)
        
        
        # --- COLUMN 3: Info Panel (Right) ---
        self.info_panel = QtWidgets.QTextEdit()
        self.info_panel.setReadOnly(True)
        # Narrowed from 300 to 250 to remove excess right space
        self.info_panel.setFixedWidth(275)
        self.top_container.addWidget(self.info_panel)
        
        # Add the top container (Plot + Middle + Right) to main layout
        self.main_layout.addLayout(self.top_container, stretch=3)
        
        # --- 4. Bottom Text Panels (Inequalities & Rules) ---
        self.bottom_layout = QtWidgets.QHBoxLayout()
        self.left_text = QtWidgets.QTextEdit()
        self.right_text = QtWidgets.QTextEdit()
        

        
        self.left_text.setReadOnly(True)
        self.right_text.setReadOnly(True)
        
        # Sync scrollbars
        self.left_text.verticalScrollBar().valueChanged.connect(
            self.right_text.verticalScrollBar().setValue
        )
        self.right_text.verticalScrollBar().valueChanged.connect(
            self.left_text.verticalScrollBar().setValue
        )
        
        self.left_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.right_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.bottom_layout.addWidget(self.left_text)
        self.bottom_layout.addWidget(self.right_text)
        self.main_layout.addLayout(self.bottom_layout, stretch=1)

        # --- Final Signal Connections ---
        self.btn_forward.clicked.connect(lambda: self.handle_action("forward"))
        self.btn_back.clicked.connect(lambda: self.handle_action("backward"))
    

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



        
    def render_text_as_path(self, painter, widget):
        """Render the text of simple widgets (QLabel, QPushButton, etc.) as vector paths."""
        text = getattr(widget, "text", lambda: "")()
        if not text:
            return False  # widget has no text
    
        font = widget.font()
        metrics = widget.fontMetrics()
    
        # Compute baseline
        x = 0
        y = metrics.ascent()
    
        path = QPainterPath()
        path.addText(x, y, font, text)
    
        painter.save()
        painter.translate(widget.rect().topLeft())
        painter.setPen(widget.palette().windowText().color())
        painter.drawPath(path)
        painter.restore()
    
        return True



    def build_eq_block_document(self, eq_items):
        CHECKED_BOX = "☑"
        UNCHECKED_BOX = "☐"
        
        html_rows = []
    
        for item in eq_items:
            label = item["label"]
            checked = item["checked"]
            color = item["color"]
            dash = item["dash"]
            width = 40
    
            box = CHECKED_BOX if checked else UNCHECKED_BOX
    
            line_html = f"""
            <div style="
                width:{width}px;
                border-bottom: {item['linewidth']}px {dash} {color};
                margin-top:4px;
                margin-bottom:4px;
            "></div>
            """
    
            row = f"""
            <tr>
                <td style="padding-right:6pt; text-align:center; width:14pt;">
                    {box}
                </td>
                <td style="padding-right:6pt; width:30pt;">
                    {label}
                </td>
                <td>
                    {line_html}
                </td>
            </tr>
            """
            html_rows.append(row)
    
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                table {{
                    border-collapse: collapse;
                    font-family: DejaVu Sans, Arial, sans-serif;
                    font-size: 9pt;
                }}
                td {{
                    vertical-align: middle;
                }}
            </style>
        </head>
        <body>
            <table>
                {''.join(html_rows)}
            </table>
        </body>
        </html>
        """
    
        doc = QTextDocument()
        doc.setHtml(html)
        return doc
    
    
    
    def render_widget_recursive(self, widget, painter):
    
        # print(f'{widget.__class__}, hidden={widget.isHidden()}')
        
        # if isinstance(widget, QtWidgets.QScrollArea):
        #     print(f'-->Visto ScrollArea')
        #     return
        
        # if isinstance(widget, LineStyleWidget):
        #     print(f'Visto 111 - LineStyleWidget entre los widgets!!')
        #     pos = widget.mapTo(self, QtCore.QPoint(0, 0))
        #     # pos = widget.mapTo(self, QtCore.QPoint(100, 100 + int(np.random.rand() * 100)))
        #     widget.render(painter, pos)
        #     return            
        
        count = 0
        if not widget.isHidden():
    
            count += 1 
                  
            # 1. Pintar fondo del widget si lo tiene
            # if widget.autoFillBackground():
            #     pal = widget.palette()
            #     bg = pal.color(widget.backgroundRole())
            #     painter.save()
            #     painter.setBrush(bg)
            #     painter.setPen(QtCore.Qt.NoPen)
            #     rect = QtCore.QRect(widget.mapTo(self, QtCore.QPoint(0, 0)), widget.size())
            #     painter.drawRect(rect)
            #     painter.restore()
    
            # 2. Pintar el widget (incluye paintEvent)
            pos = widget.mapTo(self, QtCore.QPoint(0, 0))
            
            # if isinstance(widget, QtWidgets.QScrollArea):
            #     widget.render(painter, pos)
            #     for child in widget.findChildren(QtWidgets.QWidget, "", options=QtCore.Qt.FindChildrenRecursively):
            #                                      # FindDirectChildrenOnly):
            #         # if isinstance(widget, LineStyleWidget):
            #         #     print(f'Recursión 22')
                        
            #         if isinstance(widget, LineStyleWidget):
            #             print(f'Visto 222 - LineStyleWidget entre los widgets!!')
            #             pos = widget.mapTo(self, QtCore.QPoint(0, 0))
            #             widget.render(painter, pos)
                        
            #         # self.render_widget_recursive(child, painter)
            #     # if isinstance(widget, LineStyleWidget):
            #     # print(f'Visto LineStyleWidget entre los widgets (paint_to)!!')
            #     # pos = widget.mapTo(self, QtCore.QPoint(0, 0))
            #     # widget.paint_to(painter, pos)
            # else:
                # widget.render(painter, pos)

            widget.render(painter, pos)
    
        # 3. Recursión
        for child in widget.findChildren(QtWidgets.QWidget, "", QtCore.Qt.FindDirectChildrenOnly):
        # for child in widget.findChildren(QtWidgets.QWidget, "", options=QtCore.Qt.FindChildrenRecursively):
                                         # FindDirectChildrenOnly):
            # print(f'Recursión')
            self.render_widget_recursive(child, painter)
            





    def render_to_painter(self, painter, offset):
        painter.save()
        painter.translate(offset)
        self.paintEvent(QtGui.QPaintEvent(self.rect()))
        painter.restore()
        
    # Run this to restore layout and show hidden widgets
    def restore_ui(self):
        # Undo any fixed-size locks you may have set
        for w in self.findChildren(QtWidgets.QWidget):
            try:
                w.setMinimumSize(0, 0)
                w.setMaximumSize(16777215, 16777215)
            except Exception:
                pass
    
        # Show any widgets you hid
        for name in ("canvas", "btn_forward", "btn_back"):
            w = getattr(self, name, None)
            if w is not None:
                w.show()
    
        # Let the layout recompute and the window return to normal
        QtWidgets.QApplication.processEvents()
        self.adjustSize()
        self.repaint()


                        
    def export_pdf(self):
        # Widgets to hide while exporting
        widgets_to_hide = [self.canvas, self.btn_forward, self.btn_back]
        # widgets_to_hide = [self.canvas]
    
        # Freeze sizes for hidden widgets
        # self.check_widget.setFixedSize(self.check_widget.size())
        for w in widgets_to_hide:
            w.setFixedSize(w.size())
            w.hide()
    
        # Configure printer (unchanged)
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName('regression_output/export_clean.pdf')
        
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
        # scale by device_scale/device_ratio to map widget coords -> pixmap logical coords
        # (if device_ratio != 1, pix has devicePixelRatio set, so logical scale should be scale)
        p_pix.scale(scale, scale)
        self.render_widget_recursive(self, p_pix)
        
        # self.check_widget.render()
        # # self.render_widget_recursive(self.check_layout, p_pix)
        # for k in self.check_rows_layout
                
        #     for i in range(self.check_rows_layout[k].count()):
        #         item = self.check_rows_layout[k].itemAt(i)
        #         widget = item.widget()
        #         if widget is not None:
        #             # Perform your logic here
        #             print(f"Found widget: {widget.__class__}")
        #             self.render_widget_recursive(widget, p_pix)

        self.check_widget.adjustSize()
        self.check_widget.ensurePolished()
        # self.check_layout.adjustSize()
        
        # self.check_widget.render(p_pix,  QtCore.QPoint(), 
        #                                  QtGui.QRegion(), 
        #                                  QtWidgets.QWidget.DrawChildren | QtWidgets.QWidget.DrawWindowBackground)

        # current_y = 0 
        
        # for name, line_widget in self.line_dict.items():
        #     row_parent = line_widget.parentWidget()
        #     actual_x = line_widget.x()
        #     actual_y = row_parent.y() + line_widget.y()
    
        #     target_pos = QtCore.QPoint(actual_x, actual_y)
    
        #     # pos = line_widget.mapTo(self.check_rows_dict[name], QtCore.QPoint(0, 0))
            # print(f'target_pos={target_pos}, name={name}')
            # line_widget.render(p_pix, target_pos)
            
        # Start with the margin/spacing of your layout
        current_y = self.check_widget.layout().contentsMargins().top()
        spacing = self.check_widget.layout().spacing()
        
        offset_x = 60  # Moves the line to the right
        offset_y = -195 # Moves the line up

        for name, line_widget in self.line_dict.items():
            row_widget = line_widget.parentWidget()
            
            actual_x = line_widget.x() + offset_x
            actual_y = current_y + line_widget.y() + offset_y
    
            # Calculate pos based on the running tally
            # We use line_widget.x() to keep it horizontally aligned
            target_pos = QtCore.QPoint(actual_x, actual_y)
            
            # print(f'target_pos={target_pos}, name={name}')
            
            line_widget.render(p_pix, target_pos)
            
            # Increase the tally by the height of the row + the gap between rows
            current_y += row_widget.height() + spacing
            
        
        # self.check_widget.self.c
        
        p_pix.end()
        
        # Draw background (optional)
        painter.save()
        
        # painter.setBrush(QtGui.QColor("#f1f1f1"))
        # painter.setPen(QtCore.Qt.NoPen)
        # painter.drawRect(page_rect)
        
        painter.restore()
        
        # Draw the pixmap at the printable origin (device coords)
        painter.drawPixmap(page_rect.x(), page_rect.y(), pix)
        
        # End drawing
        painter.end()
        
        # Restore hidden widgets
        for w in widgets_to_hide:
            # w.setFixedSize(w.sizeHint())
            w.show()


        
        
    def get_vlines(self, reorder=False):
        
        index_color = 0
        solve_list = []
        vlines = []
        
        items = list(self.configs.items())
        
        '''
        Generar lista de contribs para cada config
        '''
        
        eq_list, neuron_eqs, output_contribs, inequality = \
            get_eq_list_regression(items[0][0], self.config_samples, self.X_train, 
                                   self.model, self.boundaries, self.decimals)

        new_vlines = OrderedDict()
        
        ineq_list = []

        for eq in eq_list:
            x = sp.solve(eq[1])[0]
            contrib = output_contribs[0]
            y = contrib[1] * x + contrib[0]
            
            new_vlines[eq[0]] = {
                'x_coords': [x],
                'y_coords': [y],
                'color': self.color_list[index_color],
                'linestyle': 'dashed' if int(eq[0][-1]) > 4 else 'solid' # OJO CON ESTE 3
            }
            
            index_color += 1
            

            
            for config in self.configs:
                eq_list, neuron_eqs, output_contribs, inequality = \
                    get_eq_list_regression(config, self.config_samples, self.X_train, 
                                           self.model, self.boundaries, self.decimals)
                    
                contrib = output_contribs[0]
                
                y1 = contrib[1] * inequality[0] + contrib[0]
                y2 = contrib[1] * inequality[1] + contrib[0]
                
                if np.isclose(inequality[0], np.float64(x)):
                    new_vlines[eq[0]]['y_coords'] = [y1]
                    new_vlines[eq[0]]['config'] = config
                    
                    # print(f'VISTO X=inequality = {x}')
            
            # y1 = contrib[1] * inequality[0] + contrib[0]
            # y2 = contrib[1] * inequality[1] + contrib[0]
            
            # self.vlines[config] = { 
            #     'x_coords': [inequality[1], inequality[0]], 
            #     'y_coords': [y2, y1],
            #     'color': self.color_list[index_color],
            #     'linestyle': 'solid'
            # }
            
            # index_color += 1

        index_color = 0
        
        for config in self.configs:
            eq_list, neuron_eqs, output_contribs, inequality = \
                get_eq_list_regression(config, self.config_samples, self.X_train, 
                                       self.model, self.boundaries, self.decimals)
                
            contrib = output_contribs[0]
            
            y1 = contrib[1] * inequality[0] + contrib[0]
            y2 = contrib[1] * inequality[1] + contrib[0]
            
            
            pass

            self.lines[config] = { 
                'x_coords': [inequality[1], inequality[0]], 
                'y_coords': [y2, y1],
                # 'color': self.color_list[index_color],
                'color': 'gray',
                'linestyle': 'solid'
            }
            
            index_color += 1
    
        new_vlines['Eq5']['x_coords'] = [self.lines['00010001']['x_coords'][0]]
        new_vlines['Eq6']['x_coords'] = [self.lines['10010011']['x_coords'][0]]
        new_vlines['Eq7']['x_coords'] = [self.lines['01000011']['x_coords'][0]]
        new_vlines['Eq7']['color'] = 'orange'
        self.lines['01100010']['color'] = 'green'
        self.lines['01000011']['color'] = 'brown'
        self.lines['01000001']['color'] = 'orange'
        self.lines['00010011']['color'] = 'salmon'
        self.lines['00000001']['color'] = 'olive'
        self.lines['00010001']['color'] = 'magenta'
        self.lines['01100011']['color'] = 'pink'
        self.lines['10010011']['color'] = 'blue'
        self.lines['10010111']['color'] = 'lightcoral'
        
        
        # self.color_list = ['blue'X, 'olive'X, 'brown'X, 'magenta'X, 'lightcoral'x, 
        #               'pink'x, 'orange'x, 'salmon'x, 'lightblue', 'cyan']
        
        
        # for i in range(8):
        #     eq = f'Eq{i+1}'
        #     new_vlines[eq]['x_coords'] = [ineq_list[i]]
            
        if reorder:
            for eq in eq_list:
                solve_list.append(sp.solve(eq[1]))
        
            self.solve_order = np.argsort(np.array(solve_list).flatten())
    
            items = list(new_vlines.items())
            
            for i in self.solve_order: 
                self.vlines[items[i][0]] = items[i][1]
        else:
            self.vlines = new_vlines
            
            
        # color_list = []
        
        
        # for i in self.solve_order:
        #     color_list.append(items[i][1]['color'])
            
        # for i, c in enumerate(color_list):
        #     items[i][1]['color'] = c
        
        pass
        
    
            
        
    def get_vlines_old(self, reorder=False):
        
        index_color = 0
        solve_list = []
        vlines = []
        
        items = list(self.configs.items())
        
        '''
        Generar lista de contribs para cada config
        '''
        
        eq_list, neuron_eqs, output_contribs, inequality = \
            get_eq_list_regression(items[0][0], self.config_samples, self.X_train, 
                                   self.model, self.boundaries, self.decimals)

        new_vlines = OrderedDict()
        
        for eq in eq_list:
            x = sp.solve(eq[1])[0]
            contrib = output_contribs[0]
            y = contrib[1] * x + contrib[0]
            
            new_vlines[eq[0]] = {
                'x_coords': [x],
                'y_coords': [y],
                'color': self.color_list[index_color],
                'linestyle': 'dashed'
            }
            
            index_color += 1
            
            
            for config in self.configs:
                eq_list, neuron_eqs, output_contribs, inequality = \
                    get_eq_list_regression(config, self.config_samples, self.X_train, 
                                           self.model, self.boundaries, self.decimals)
                    
                contrib = output_contribs[0]
                
                y1 = contrib[1] * inequality[0] + contrib[0]
                y2 = contrib[1] * inequality[1] + contrib[0]
                
                if np.isclose(inequality[0], np.float64(x)):
                    new_vlines[eq[0]]['y_coords'] = [y1]
                    new_vlines[eq[0]]['config'] = config
                    
                    # print(f'VISTO X=inequality = {x}')
            
            # y1 = contrib[1] * inequality[0] + contrib[0]
            # y2 = contrib[1] * inequality[1] + contrib[0]
            
            # self.vlines[config] = { 
            #     'x_coords': [inequality[1], inequality[0]], 
            #     'y_coords': [y2, y1],
            #     'color': self.color_list[index_color],
            #     'linestyle': 'solid'
            # }
            
            # index_color += 1

        index_color = 0
        for config in self.configs:
            eq_list, neuron_eqs, output_contribs, inequality = \
                get_eq_list_regression(config, self.config_samples, self.X_train, 
                                       self.model, self.boundaries, self.decimals)
                
            contrib = output_contribs[0]
            
            y1 = contrib[1] * inequality[0] + contrib[0]
            y2 = contrib[1] * inequality[1] + contrib[0]
            
            
            pass

            self.lines[config] = { 
                'x_coords': [inequality[1], inequality[0]], 
                'y_coords': [y2, y1],
                # 'color': self.color_list[index_color],
                'color': 'gray',
                'linestyle': 'solid'
            }
            
            index_color += 1
    
        
        if reorder:
            for eq in eq_list:
                solve_list.append(sp.solve(eq[1]))
        
            self.solve_order = np.argsort(np.array(solve_list).flatten())
    
            items = list(new_vlines.items())
            
            for i in self.solve_order: 
                self.vlines[items[i][0]] = items[i][1]
        else:
            self.vlines = new_vlines
            
            
        # color_list = []
        
        
        # for i in self.solve_order:
        #     color_list.append(items[i][1]['color'])
            
        # for i, c in enumerate(color_list):
        #     items[i][1]['color'] = c
        
        pass
    
    
    def set_colors(self):
        def get_color(vlines, config):
            for c in vlines.keys():
                # print(f'comparando {vlines[c]["config"]} con {config}')
                if vlines[c].get("config", False) and vlines[c]["config"] == config:
                    return vlines[c]["color"]
            return False
        
        for config in self.configs:
            c = get_color(self.vlines, config)
            
            if c == False:
                self.lines[config]['color'] = 'green'
            else:
                self.lines[config]['color'] = c
                # print(f'SET color={c} for config={config}')
                
        
    def reorder_configs(self, reorder=False):
        if reorder:
            new_configs = Counter()
            new_config_samples = {}
            
            print(f'Solve_order={self.solve_order}')
            
            for config, count in self.configs.items():
                print(f'Config antes {config}', end='')
                config_new_order = ''
                for i in self.solve_order:
                    config_new_order += config[i]
                print(f' después {config_new_order}')
                new_configs[config_new_order] = count
                
                new_config_samples[config_new_order] = self.config_samples[config]
        
            self.configs = new_configs 
            self.config_samples = new_config_samples
               
        pass
        
    
    # def get_config_structure(self, model):
    #     return str(model.layers[0].out_features) if hasattr(model.layers[0], 'out_features') else 'Unknown'
        
    def get_config_structure(self, model):
        try:
            return str(model.layers[0].out_features)
        except:
            return 'Unknown'
        
    def _on_scrollbar(self, *args):
        """Handle scrollbar movement"""
        self.left_text.yview(*args)
        self.right_text.yview(*args)

    def _left_scroll_set(self, *args):
        """Handle left text scrolling"""
        self.scrollbar.set(*args)
    
    def _right_scroll_set(self, *args):
        """Handle right text scrolling"""
        self.scrollbar.set(*args)
        self.left_text.yview_moveto(args[0])
        
    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Only process if it's a valid selection
        if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
            self.ax.set_xlim(min(x1, x2), max(x1, x2))
            self.ax.set_ylim(min(y1, y2), max(y1, y2))
            self.canvas.draw_idle()
    
    def clear_equations(self):
        """Surgically remove all dynamic checkboxes from the sidebar."""
        self.var_dict.clear()
        self.arrows_config.clear()
        
        # We must use list() to avoid iteration errors while deleting
        if hasattr(self, 'row_widgets'):
            for name in list(self.row_widgets.keys()):
                widget = self.row_widgets.pop(name)
                widget.setParent(None)
                widget.deleteLater()
        
        self.color_index = 0 
        self.update_plot()
        

    def add_equation(self, name, equation_text, color, linestyle='solid', linewidth=3):        

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
        # line_visual = LineStyleWidget(color, linestyle, linewidth)

        row_layout.addWidget(checkbox)
        row_layout.addStretch(1)
        row_layout.addWidget(line_visual)
        
        self.check_layout.addWidget(row_widget)
        
        # line_visual = LineStyleWidget(color, linestyle, linewidth, parent=self.check_widget)
        # line_visual = LineStyleWidget(color, linestyle, linewidth, parent=checkbox)
        
        # self.check_layout.addWidget(row_layout)
        # self.check_layout.addChildLayout(row_layout)
        
        self.check_dict[name] = checkbox
        self.line_dict[name] = line_visual
        self.check_rows_dict[name] = row_widget
        
        
        
    def collect_eq_items(self):
        eq_items = []
    
        for name, checkbox in self.check_dict.items():
            line_widget = self.line_dict[name]
    
            eq_items.append({
                "label": name,
                "checked": checkbox.isChecked(),
                "color": line_widget.color,     # QColor → "#rrggbb"
                "linestyle": line_widget.linestyle,    # e.g. "solid", "dash", "dot"
                "linewidth": line_widget.linewidth,    # numeric
            })

        return eq_items
    
    def css_dash_pattern(self, linestyle):
        if linestyle == "solid":
            return "solid"
        if linestyle == "dash":
            return "4px 4px"
        if linestyle == "dot":
            return "2px 2px"
        if linestyle == "dashdot":
            return "6px 3px 2px 3px"
        return "solid"    
        
    # def add_equation(self, name, equation, color, linestyle):
    #     if not isinstance(equation, sp.Eq):
    #         raise ValueError("Equation must be a sympy.Eq instance")
            
    #     self.equations[name] = (equation, color, linestyle)

    #     # Container for the row
    #     row_widget = QtWidgets.QWidget()
    #     row_layout = QtWidgets.QHBoxLayout(row_widget)
    #     row_layout.setContentsMargins(0, 2, 0, 2)
    
    #     checkbox = QtWidgets.QCheckBox(f"{name}  ")
    #     checkbox.setChecked(True)
    #     checkbox.stateChanged.connect(self.update_plot)
    #     self.var_dict[name] = checkbox
    
    #     line_label = QtWidgets.QLabel()
    #     line_label.setFixedSize(25, 10)
    #     style = f"border-bottom: 3px dashed {color};" if linestyle == 'dash' else f"background-color: {color};"
    #     line_label.setStyleSheet(style)
    
    #     row_layout.addWidget(checkbox)
    #     row_layout.addWidget(line_label)
    #     row_layout.addStretch()
        
    #     self.check_layout.addWidget(row_widget)
        
    #     if not hasattr(self, 'row_widgets'): self.row_widgets = {}
    #     self.row_widgets[name] = row_widget
        
    # def clear_equations(self):
    #         """Clear all dynamic UI rows (equations/lines) from the sidebar layout."""
    #         # 1. Clear data references
    #         self.var_dict.clear()
    #         self.arrows_config.clear()
            
    #         # 2. Surgically remove the container widgets from the layout
    #         # We use list() to avoid "dictionary changed size during iteration" errors
    #         if hasattr(self, 'row_widgets'):
    #             for name in list(self.row_widgets.keys()):
    #                 widget = self.row_widgets.pop(name)
                    
    #                 # This effectively removes the widget from the sidebar UI
    #                 widget.setParent(None)
                    
    #                 # This schedules the C++ object for safe deletion from memory
    #                 widget.deleteLater()
            
    #         # 3. Reset the color counter so the next pattern starts fresh
    #         self.color_index = 0 
            
    #         # 4. Refresh the plot
    #         self.update_plot()        
        
    # def clear_equations(self):
    #     """Clear all equations and reset the plot"""
    #     # self.equations.clear()
    #     self.var_dict.clear()
    #     # self.lines.clear()
    #     # self.vlines.clear()
    #     self.arrows_config.clear()
    #     # self.contour_sets.clear()
    #     # self.poly.clear()
    
    #     current_callback = self.button_callback
        
    #     # Clear checkbuttons except scatter
    #     for widget in self.check_frame.winfo_children():
    #         if widget not in [self.scatter_checkbutton, self.buttons_frame]:
    #             widget.destroy()
        
    #     if current_callback:
    #         self.set_button_callback(current_callback)
            
    #     # initial_width = self.root.winfo_width()
    #     # initial_height = self.root.winfo_height()
    #     # self.root.minsize(initial_width, initial_height)
        
    #     self.color_index = 0 
        
    #     self.update_plot()
        

    
    # def update_bottom_text(self, text):
    #     """Update the bottom text display with new content."""
    #     self.bottom_text.delete('1.0', tk.END)
    #     self.bottom_text.insert('1.0', text)
    
    def str_equ(self, eq, decimals=2, return_full=False, orig_sign=-2):
        
        eq = eq.copy()
        
        # eq[0] = -eq[0]
        
        eq[abs(eq) < 1e-10] = 0
        
        sign = 1
        
        # if eq[1] != 0:
        #     # if eq[1] < 0:
        #     #     sign = -1
        #     # print(f'Eq={eq} {eq / eq[1]}')
        #     # eq = eq / eq[1]
        # elif eq[2] != 0:
        #     if eq[2] < 0:
        #         sign = -1
            # eq = eq / eq[2]

        
        # b = -eq[0]
        # A = eq[1:]
        
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
                    ret += f' + {abs(val):.{decimals}f}x '
                    # ret += f' + {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'x '
                    # ret += f'x{self.subscript_map[i+1]} '
            elif val == -1:
                if not first:
                    ret += f'- {abs(val):.{decimals}f}x '
                    # ret += f'- {abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
                else:
                    ret += f'-x '
                    # ret += f'-x{self.subscript_map[i+1]} '
            else:
                if val < 0:
                    if first:
                        ret += '-'
                    else:
                        ret += '-'
                else:
                    if not first:
                        ret += '+'
                    else:
                        ret += '&nbsp;'
                    
                ret += f'{abs(val):.{decimals}f}x&nbsp;'
                # ret += f'{abs(val):.{decimals}f}x{self.subscript_map[i+1]} '
            first = False
        if first:
            if eq[0] < 0:
                ret += f'{eq[0]:.{decimals}f}'
            else:
                ret += f'{abs(eq[0]):.{decimals}f}'
        else:
            if eq[0] < 0:
                if not np.isclose(eq[0], 0.0):
                    ret += f'&nbsp;-&nbsp;{abs(eq[0]):.{decimals}f}'
            else:
                if not np.isclose(eq[0], 0.0):
                    ret += f'+&nbsp;{abs(eq[0]):.{decimals}f}'
                
        if return_full:
            r = sign * orig_sign
            # if r 
            ret += f' {">= 0" if r > 0 else "<= 0"}'
            
        return ret
    
    def append_bottom_text(self, text, no_trad=False, bold=False, pos='left'):
        # 1. Pick the correct panel
        target = self.right_text if pos == 'right' else self.left_text
        
        # 2. Clean the string and handle newlines
        formatted_text = text.replace('\n', '<br>')
        
        # 3. Apply bolding if requested
        if bold:
            formatted_text = f"<b>{formatted_text}</b>"
            
        # 4. Insert as a block-level element to prevent "smashing" lines together
        # We use line-height: 1.2 to keep it tight like your original image
        style = "font-family: Arial; font-size: 10pt; line-height: 1.2;"
        target.insertHtml(f"<div style='{style}'>{formatted_text}</div>")
        
        
    
    
    def append_bottom_text2(self, text, no_trad=False, bold=False, pos=None):
        """Append text to the bottom text content using PyQt5 Rich Text."""
        
        # 1. Determine the target text zone (left or right)
        if pos is None:
            # Get actual line count from the Qt document
            left_blocks = self.left_text.document().blockCount()
            
            # Calculate visible lines based on actual font metrics
            font_metrics = self.left_text.fontMetrics()
            line_height = font_metrics.lineSpacing()
            visible_lines = self.left_text.height() // line_height
            
            if left_blocks <= visible_lines:
                text_zone = self.left_text
            else:
                text_zone = self.right_text
        else:
            text_zone = self.right_text if pos == 'right' else self.left_text
    
        # 2. String Translation Logic (Reproducing your original logic)
        letters = 'xyz'
        if no_trad:
            new_text = text
        else:
            # This handles your underscore/subscript conversion logic
            processed_chars = []
            skip_next = False
            for i, t in enumerate(text):
                if skip_next:
                    skip_next = False
                    continue
                if t == '*':
                    continue
                if t == '_':
                    if i + 1 < len(text):
                        pos1 = '0123456789'.find(text[i+1])
                        if pos1 >= 0:
                            processed_chars.append(letters[pos1])
                            skip_next = True
                else:
                    idx = letters.find(t)
                    if idx >= 0:
                        processed_chars.append(f" {letters[idx]}")
                    else:
                        processed_chars.append(t)
            new_text = "".join(processed_chars)
    
        # 3. Apply Formatting using HTML
        # We don't need tag_configure; we just use inline CSS
        font_style = "font-family: Arial; font-size: 11pt;"
        if bold:
            html_output = f"<div style='{font_style} font-weight: bold;'>{new_text}</div>"
        else:
            html_output = f"<div style='{font_style}'>{new_text}</div>"
    
        # 4. Insert into the widget
        # append() handles adding a new line automatically
        text_zone.append(html_output)
        
        
    # def append_bottom_text(self, text, no_trad=False, bold=False, pos=None):
    #     """Append text to the bottom text content."""
        
    #     if pos is None:
    #         """Add text to columns with automatic overflow handling"""
    #         # Get the number of lines in each text widget
    #         left_lines = int(self.left_text.index('end-1c').split('.')[0])
    #         right_lines = int(self.right_text.index('end-1c').split('.')[0])
            
    #         # Calculate visible lines (approximate based on height)
    #         visible_lines = self.left_text.winfo_height() // 19  # Approximate line height
            
    #         # print(f'Visible_lines={visible_lines}, left={left_lines}, right={right_lines}')
    #         text_zone = None
    #         # if left_lines < visible_lines or left_lines <= right_lines:
    #         if left_lines <= visible_lines:
    #             # If left column has space or has fewer lines than right, add to left
    #             text_zone = self.left_text
    #             # self.left_text.insert(tk.END, text + "\n")
    #         else:
    #             # Otherwise add to right column
    #             text_zone = self.right_text
    #             # self.right_text.insert(tk.END, text + "\n")
    #     else:
    #         if pos == 'right':
    #             text_zone = self.right_text
    #         else:
    #             text_zone = self.left_text

        
    #     letters = 'xyz'
    #     # subscript_map = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
        
    #     # subscript_map = '₀₁₂₃₄₅₆₇₈₉'

    #     new_text = ''
        
    #     self.left_text.config(state="normal")
    #     self.right_text.config(state="normal")
        
    #     if no_trad:
    #         new_text = text
    #         if bold:
    #             text_zone.tag_configure('bold', font=('Arial', 11, 'bold'))
    #             text_zone.insert(tk.END, new_text + '\n', 'bold')
    #             # self.bottom_text.tag_configure('bold', font=('Arial', 11, 'bold'))
    #             # self.bottom_text.insert(tk.END, new_text + '\n', 'bold')
    #         else:
    #             text_zone.tag_configure('normalX', font=('Arial', 11))
    #             text_zone.insert(tk.END, new_text + '\n', 'normalX')
    #             # self.bottom_text.tag_configure('normal', font=('Arial', 11))
    #             # self.bottom_text.insert(tk.END, new_text + '\n', 'normal')
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
    #                     # new_text += f' x{subscript_map[pos+1]}'
    #                 else:
    #                     new_text += t
            
    #         text_zone.tag_configure('normalX', font=('Arial', 11))
    #         text_zone.insert(tk.END, new_text + '\n', 'normalX')
    #         # self.bottom_text.tag_configure('normal', font=('Arial', 11))
    #         # self.bottom_text.insert(tk.END, new_text + '\n', 'normal')
        
    #     self.left_text.config(state="disabled")
    #     self.right_text.config(state="disabled")

        
        
    
    # def update_text(self, text):
    #     """Update the text display with new content.
        
    #     Args:
    #         text (str): Text to display
    #     """
    #     self.text_widget.delete('1.0', tk.END)
    #     self.text_widget.insert('1.0', text)
    
    # def append_text_old(self, text):
    #     """Append text to the existing content.
        
    #     Args:
    #         text (str): Text to append
    #     """
    #     # self.text_widget.insert(tk.END, text + '\n')
    #     self.text_widget.config(state="normal")
    #     self.text_widget.insert(tk.END, text + '\n')
    #     self.text_widget.config(state="disabled")        
        
    def append_text(self, text, bold=False, center=False):
        """Encapsulates alignment within the HTML string for perfect consistency."""
        # 1. Clean the text and convert newlines to HTML breaks
        # We strip to avoid double-spacing since <div> creates its own block
        html_content = text.strip().replace('\n', '<br>')
        
        # 2. Determine styles
        align = "center" if center else "left"
        weight = "bold" if bold else "normal"
        
        # 3. Wrap in a styled div
        # This forces the renderer to align this specific block independently
        styled_html = f"<div style='text-align: {align}; font-weight: {weight};'>{html_content}</div>"
        
        self.info_panel.append(styled_html)
        
    # def append_text(self, text, bold=False):
    #     """Uses HTML tags for formatting, much faster than Tkinter tags."""
    #     if bold:
    #         self.info_panel.append(f"<b>{text}</b>")
    #     else:
    #         self.info_panel.append(text)
            
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
    #         # self.text_widget.insert(tk.END, new_text + '\n', 'bold')

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
                

    #         # self.text_widget.config(state="normal")
    #         # self.text_widget.insert(tk.END, text + '\n')
    #         self.text_widget.config(state="disabled")
    #     else:
    #         self.text_widget.config(state="normal")
    #         self.text_widget.insert(tk.END, text + '\n')
    #         self.text_widget.config(state="disabled")    
            
    def set_scatter_points(self, x_points, y_points, colors):
        """Add scatter points to the plot and enable the UI control."""
        # The data processing logic remains identical
        points = np.zeros((len(x_points), 2))
        points[:,0] = x_points[:,0]
        points[:,1] = y_points[:,0]
        
        self.scatter_points = np.array(points)
        self.scatter_colors = colors
        
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
        
    # def set_scatter_points(self, x_points, y_points, colors):
    #     """Add scatter points to the plot.
        
    #     Args:
    #         points: numpy array or list of shape (n, 2) containing x, y coordinates
    #     """
    #     points = np.zeros((len(x_points), 2))
    #     points[:,0] = x_points[:,0]
    #     points[:,1] = y_points[:,0]
        
    #     self.scatter_points = np.array(points)
        
    #     self.scatter_colors = colors
    #     self.scatter_checkbutton.configure(state='disabled')
    #     # self.scatter_checkbutton.configure(state='normal')
    #     self.update_plot()
        
    # def clear_scatter_points(self):
    #     """Remove scatter points from the plot."""
    #     self.scatter_points = None
    #     self.scatter_colors = None
    #     self.scatter_var.set(False)
    #     self.scatter_checkbutton.configure(state='disabled')
    #     self.update_plot()
        

        
    # def add_equation(self, name, equation, color, linestyle):
    #     # print(f'Adding equation {name}, equation {equation}')
    #     if not isinstance(equation, sp.Eq):
    #         raise ValueError("Equation must be a sympy.Eq instance")
            
    #     # Add equation to dictionary
    #     self.equations[name] = (equation, color, linestyle)

    #     # 1. Create a container widget for this specific row
    #     row_widget = QtWidgets.QWidget()
    #     row_layout = QtWidgets.QHBoxLayout(row_widget)
    #     row_layout.setContentsMargins(0, 2, 0, 2) # Tighten the spacing
    
    #     # 2. Setup the checkbox
    #     checkbox = QtWidgets.QCheckBox(f"{name}  ")
    #     checkbox.setChecked(True)
    #     checkbox.stateChanged.connect(self.update_plot)
    #     self.var_dict[name] = checkbox
    
    #     # 3. Setup the color line (using the QLabel/CSS method)
    #     line_label = QtWidgets.QLabel()
    #     line_label.setFixedSize(25, 10)
    #     style = f"border-bottom: 3px dashed {color};" if linestyle == 'dash' else f"background-color: {color};"
    #     line_label.setStyleSheet(style)
    
    #     # 4. Add to the row and then to the main sidebar
    #     row_layout.addWidget(checkbox)
    #     row_layout.addWidget(line_label)
    #     row_layout.addStretch()
        
    #     self.check_layout.addWidget(row_widget)
        
    #     # Store the row_widget so we can kill it later
    #     if not hasattr(self, 'row_widgets'): self.row_widgets = {}
    #     self.row_widgets[name] = row_widget
        
        # # Create new checkbutton variable
        # var = tk.BooleanVar(value=True)
        # self.var_dict[name] = var
        
        # # Create new checkbutton
        # ttk.Checkbutton(
        #     self.check_frame,
        #     text=name + '  ',
        #     variable=var,
        #     command=self.update_plot
        # ).grid(row=len(self.var_dict), column=0, sticky=tk.W)
        
        # line_canvas = tk.Canvas(self.check_frame, width=25, height=10, 
        #                highlightthickness=0, )
        #                # bg=self.check_frame.cget('background'))
        # line_canvas.grid(row=len(self.var_dict), column=1, padx=(5, 0), sticky=tk.W)

        # if linestyle == 'dash':
        #     # line_canvas.create_line(0, 5, 25, 5, fill=color, width=3, dash=(4, 2, 4, 2))
        #     line_canvas.create_line(0, 5, 25, 5, fill=color, width=3, 
        #                             dash=(100, 1))
        #                             # dash=(3, 3, 3, 4))
        # else:
        #     line_canvas.create_line(0, 5, 25, 5, fill=color, width=3)
        
        # # self.update_plot()
    
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
            
    
    # def remove_equation(self, name):
    #     """Remove an equation from the plotter.
        
    #     Args:
    #         name (str): Name of the equation to remove
    #     """
    #     if name in self.equations:
    #         del self.equations[name]
    #         del self.var_dict[name]
            
    #         # Rebuild checkbuttons
    #         for widget in self.check_frame.winfo_children():
    #             widget.destroy()
            
    #         # Recreate scatter checkbutton
    #         self.scatter_checkbutton = ttk.Checkbutton(
    #             self.check_frame,
    #             text="Scatter Points",
    #             variable=self.scatter_var,
    #             command=self.update_plot,
    #             state='normal' if self.scatter_points is not None else 'disabled'
    #         )
    #         self.scatter_checkbutton.grid(row=0, column=0, sticky=tk.W)
            
    #         # Recreate equation checkbuttons
    #         for i, (eq_name, var) in enumerate(self.var_dict.items()):
    #             ttk.Checkbutton(
    #                 self.check_frame,
    #                 text=eq_name,
    #                 variable=var,
    #                 command=self.update_plot
    #             ).grid(row=i+1, column=0, sticky=tk.W)
            
    #         self.update_plot()
    
    
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

    def add_line(self, config, x_coords, y_coords, color, linestyle):
        # Store the color data as before
        self.config_colors[config] = color
        
        # Create the checkbox for this activation pattern
        checkbox = QtWidgets.QCheckBox(f"Pattern {config}")
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.update_plot)
        
        # Store the checkbox in var_dict so update_plot can check its status
        self.var_dict[config] = checkbox
        
        # Create a container widget for the sidebar row
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 2, 0, 2)
        
        # Add a small color indicator next to the checkbox
        color_indicator = QtWidgets.QLabel()
        color_indicator.setFixedSize(20, 10)
        color_indicator.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
        
        row_layout.addWidget(checkbox)
        row_layout.addWidget(color_indicator)
        row_layout.addStretch()
        
        # Add to the sidebar and keep a reference for removal
        self.check_layout.addWidget(row_widget)
        if not hasattr(self, 'row_widgets'): self.row_widgets = {}
        self.row_widgets[config] = row_widget
        
        self.update_plot()
        
    # def add_line(self, config, x_coords, y_coords, color, linestyle):
    #     # print(f'Activating var_dict for config {config}')
        
    #     self.config_colors[config] = color
        
    #     var = tk.BooleanVar(value=True)
    #     self.var_dict[config] = var
               
    #     self.update_plot()
        
        
    def remove_line(self, config):
        """Surgically remove a specific line and its UI elements."""
        # Check both dictionaries as per your original logic
        if config in self.equations or config in self.var_dict:
            # Clean up data references
            if config in self.equations: del self.equations[config]
            if config in self.var_dict: del self.var_dict[config]
            
            # Remove only the specific UI row widget
            if hasattr(self, 'row_widgets') and config in self.row_widgets:
                widget_to_remove = self.row_widgets.pop(config)
                
                # This detaches the widget from the UI and deletes it from memory
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
                
            # Redraw the plot without the removed data
            self.update_plot()
            
        
    # def remove_line(self, config):
    #     """Remove an equation from the plotter.
        
    #     Args:
    #         name (str): Name of the equation to remove
    #     """
    #     if config in self.equations:
    #         del self.equations[config]
    #         del self.var_dict[config]
            
    #         # Rebuild checkbuttons
    #         for widget in self.check_frame.winfo_children():
    #             widget.destroy()
            
    #         # Recreate scatter checkbutton
    #         self.scatter_checkbutton = ttk.Checkbutton(
    #             self.check_frame,
    #             text="Scatter Points",
    #             variable=self.scatter_var,
    #             command=self.update_plot,
    #             state='normal' if self.scatter_points is not None else 'disabled'
    #         )
    #         self.scatter_checkbutton.grid(row=0, column=0, sticky=tk.W)
            
    #         # Recreate equation checkbuttons
    #         for i, (config, var) in enumerate(self.var_dict.items()):
    #             ttk.Checkbutton(
    #                 self.check_frame,
    #                 text=config,
    #                 variable=var,
    #                 command=self.update_plot
    #             ).grid(row=i+1, column=0, sticky=tk.W)
            
    #         self.update_plot()
            
            

    def order_vertices(self, vertices):
        # Calculate centroid
        center = np.mean(vertices, axis=0)
        
        # Calculate angles of points with respect to center
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        
        # Sort vertices based on these angles
        # ordered = [x for _, x in sorted(zip(angles, vertices))]
        
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
            
            # print(f'\n+++\nVertices={vertices}\n---\n')
            # Calculate centroid
            centroid = np.mean(vertices, axis=0)
            
            # Expand vertices outward from centroid
            scale_factor = 3  # Adjust this value to control expansion
            expanded_vertices = centroid + scale_factor * (vertices - centroid)
            
            # print(expanded_vertices)
            # print()
            '''
            Fill polygon with transparency
            '''
            longest_distance = self.longest(vertices)
            area = self.area(vertices)
            
            # print(f'Longest={longest_distance}')
            # print(f'Area={area}')
            # if longest_distance != 0:
            #     print(f'Area/longest={area/longest_distance}')
                
            adjust = (self.x_range[1] - self.x_range[0]) / 6

            if area / (longest_distance * adjust) > 0.15:
            # if area / longest_distance > 0.15:
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
        
        # self.get_vlines()
        
        # 1. State & Setup
        config = list(self.configs.keys())[self.index]
        self.ax.clear()
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(-0.1, 1.15)
        
        # Grid/Axis styling
        self.ax.axhline(y=0, color='black', linestyle='-.', linewidth=0.3)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        
        # 2. Background "Ghost" Function (Fixed Slanted Lines)
        for pattern_id, s in self.lines.items():
            if pattern_id != config:
                # We pass the lists directly: plot([x1, x2], [y1, y2])
                self.ax.plot(s['x_coords'], s['y_coords'], 
                             color='black', linestyle='solid', linewidth=2, alpha=0.8, zorder=1)

        # 3. Active Region Shading
        if config in self.lines:
            s_active = self.lines[config]
            self.ax.axvspan(s_active['x_coords'][0], s_active['x_coords'][1], 
                            color='gray', alpha=0.1, zorder=0)

        # 4. Training Samples
        if self.scatter_checkbox.isChecked():
            # Convert to numpy and flatten to ensure they are simple arrays
            x_data = self.X_train.detach().cpu().numpy().flatten()
            y_data = self.y_train.detach().cpu().numpy().flatten()
            
            # XXXX 0.2
            self.ax.scatter(x_data, y_data, c='green', s=1.5, zorder=2, alpha=0.2)
            
            active_idx = self.config_samples[config]
            self.ax.scatter(x_data[active_idx], y_data[active_idx], 
                            c='green', s=1.5, zorder=3, alpha=0.9)

        # 5. Boundaries (Vertical Lines) and Arrows
        reg_x = self.lines[config]['x_coords']
        reg_mid = (reg_x[0] + reg_x[1]) / 2

        bx_list = []
        for i, (key, vline_data) in enumerate(self.vlines.items()):
            bx = float(vline_data['x_coords'][0])
            bx_list.append(bx)
            
        eq_order = np.argsort(bx_list)
        ranks = np.argsort(eq_order)
        
        # print(f'EQ_Order = {eq_order}')
            
        for i, (key, vline_data) in enumerate(self.vlines.items()):
            bx = float(vline_data['x_coords'][0])
            # print(f'BX={bx}')
            color = vline_data['color']
            # Construct the name to match exactly what you sent to add_equation
            eq_name = f"Eq{i+1}" 
            
            if eq_name in self.check_dict and self.check_dict[eq_name].isChecked():
                # A. Draw the dashed vertical line (Always visible)
                # self.ax.axvline(x=bx, color=color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
                self.ax.axvline(x=bx, color=color, linestyle=vline_data['linestyle'], linewidth=1.5, alpha=0.7, zorder=4)
                
                # B. DRAW ARROW
                
                # Temporary: Remove the 'if' check to see if arrows appear at all
                # if eq_name in self.var_dict and self.var_dict[eq_name].isChecked():
                
                direction = 1 if reg_mid > bx else -1
                dx = 0.3 * direction
                # y_pos = -0.03 if i % 2 == 0 else -0.07
                
                y_pos = -0.03 if ranks[i] % 2 == 0 else -0.07
                
                self.ax.arrow(bx, y_pos, dx, 0, 
                             head_width=0.02, head_length=0.1,
                             fc=color, ec=color, 
                             length_includes_head=True, width=0.01, zorder=10)

        # 6. The Thick Active Segment
        if config in self.lines:
            s_active = self.lines[config]
            self.ax.plot(s_active['x_coords'], s_active['y_coords'],
                         color=s_active['color'], linestyle='solid', linewidth=3, zorder=6)

        # 7. Finalize
        self.ax.set_title(self.title, pad=10, fontsize=10)
        # self.title = f'TRAIN SAMPLES: Activation Pattern {config[:4]}-{config[4:]}'

        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.10)

        self.canvas.draw()
        
        
        
    def update_plot44(self):

        config = list(self.configs.keys())[self.index]

        self.ax.clear()
        self.contour_sets.clear()
        
        # Set plot ranges
        x_range = self.x_range
        y_range = (-0.1, 1.15)
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        
        self.ax.axhline(y=0, color='black', linestyle='-.', linewidth=0.3)
        self.ax.tick_params(axis='both', which='major', labelsize=9) 
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # 1. Background Logic: All Activation Patterns (Gray segments)
        # Only plot if the specific checkbox for that pattern is active
        for c in self.configs:
            try:
                # Check if it exists in the dict AND is not a 'deleted' C++ object
                if c in self.var_dict and self.var_dict[c].isChecked():
                    s = self.lines[c]
                    self.ax.plot([s['x_coords'][0], s['x_coords'][1]], 
                                 [s['y_coords'][0], s['y_coords'][1]],
                                 color='gray', linestyle='solid', linewidth=2, alpha=0.6)
            except RuntimeError:
                # If the checkbox was deleted by Qt, we skip it and remove it from the dict
                if c in self.var_dict:
                    del self.var_dict[c]
                continue
        
        # for c in self.configs:
        #     if c in self.var_dict and self.var_dict[c].isChecked():
        #         s = self.lines[c]
        #         self.ax.plot([s['x_coords'][0], s['x_coords'][1]], 
        #                      [s['y_coords'][0], s['y_coords'][1]],
        #                      color='gray', linestyle='solid', linewidth=2, alpha=0.6)
        
        # 1. Background Plotting: Draw all known segments in Gray
        # We iterate over self.lines directly to bypass the checkbox requirement
        for pattern_id, line_data in self.lines.items():
            # Skip the active pattern here; we draw it thick and green in step 7
            if pattern_id != config:
                self.ax.plot([line_data['x_coords'][0], line_data['x_coords'][1]], 
                             [line_data['y_coords'][0], line_data['y_coords'][1]],
                             color='gray', linestyle='solid', linewidth=2, alpha=0.4)
        
        # 2. Points Logic: Reactive to the "Samples" toggle
        if self.scatter_checkbox.isChecked():
            # Keep data as numpy for faster plotting
            x = self.X_train.numpy().flatten()
            y = self.y_train.numpy().flatten()
            
            # Plot all background samples (low alpha)
            self.ax.scatter(x, y, c='green', s=1.5, zorder=1, alpha=0.2)
            
            # Highlight current region's samples (high alpha)
            active_indices = self.config_samples[config]
            self.ax.scatter(x[active_indices], y[active_indices], 
                            c='green', s=1.5, zorder=1, alpha=0.9)
            
            # active_indices = self.config_samples[config]
            # self.ax.scatter(x[active_indices], y[active_indices], 
                            # c='green', s=1.5, zorder=1, alpha=0.9)
        
        # 3. Vertical Boundary Lines
        for eq, vlines in self.vlines.items():
            self.ax.plot([vlines['x_coords'][0], vlines['x_coords'][0]],
                         y_range,
                         color=vlines['color'], linestyle=vlines['linestyle'], 
                         linewidth=1.75)

        # 4. Arrows for Neuron Inequalities
        # # Re-fetch config to ensure it's synced with self.index
        # config = list(self.configs.keys())[self.index]
        # for eq_name, eq_data in self.equations.items():
        #     if eq_name in self.var_dict and self.var_dict[eq_name].isChecked():
        #         eq_obj = eq_data[0]
        #         eq_color = eq_data[1]
        #         x_val = float(sp.solve(eq_obj)[0])
                
        #         # Arrow logic (N=1 points left, N=0 points right usually)
        #         if self.lines[config]['x_coords'][0] <= x_val and self.lines[config]['x_coords'][1] < x_val:
        #             self.ax.arrow(x_val, -0.03, -0.25, 0, head_width=0.02, head_length= head_length := 0.1,
        #                             fc=eq_color, ec=eq_color, length_includes_head=True, width=0.01, zorder=1000)
        #         else:
        #             self.ax.arrow(x_val, -0.07, 0.25, 0, head_width=0.02, head_length=head_length,
        #                             fc=eq_color, ec=eq_color, length_includes_head=True, width=0.01, zorder=1000)
             
        # Define constants once to avoid syntax errors and redundancy
        h_len = 0.1 
        config = list(self.configs.keys())[self.index]
        
        # Calculate midpoint to determine arrow direction robustly
        region_x = self.lines[config]['x_coords']
        region_mid = (region_x[0] + region_x[1]) / 2

        for eq_name, eq_data in self.equations.items():
            if eq_name in self.var_dict and self.var_dict[eq_name].isChecked():
                eq_obj, eq_color, _ = eq_data
                
                try:
                    x_val = float(sp.solve(eq_obj)[0])
                except (IndexError, TypeError):
                    continue 

                # Determine direction: point toward the region's midpoint
                dx = 0.25 if region_mid > x_val else -0.25
                # Stagger height slightly so arrows don't overlap on the axis
                y_pos = -0.07 if dx > 0 else -0.03

                self.ax.arrow(x_val, y_pos, dx, 0, 
                                head_width=0.02, 
                                head_length=h_len,
                                fc=eq_color, 
                                ec=eq_color, 
                                length_includes_head=True, 
                                width=0.01, 
                                zorder=1000)
                
        # 5. Shading and Active Segment
        if self.lines.get(config, False):
            line_data = self.lines[config]
            # Activation Region Shading
            self.ax.fill([line_data['x_coords'][0], line_data['x_coords'][1], 
                          line_data['x_coords'][1], line_data['x_coords'][0]],
                          [y_range[0], y_range[0], y_range[1], y_range[1]], 
                          color='gray', alpha=0.1)
            # Main regressor segment
            self.ax.plot([line_data['x_coords'][0], line_data['x_coords'][1]], 
                         [line_data['y_coords'][0], line_data['y_coords'][1]],
                         color=line_data['color'], linestyle='solid', linewidth=2.75)

        self.ax.set_title(self.title, pad=10, fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw()
    
    
        
        
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
        """Connect the navigation buttons using PyQt5 signals."""
        self.button_callback = callback
        
        # Disconnect existing connections first to avoid multiple triggers if called again
        try: self.btn_forward.clicked.disconnect()
        except: pass
        try: self.btn_back.clicked.disconnect()
        except: pass
    
        # Connect to the new callback
        self.btn_forward.clicked.connect(lambda: callback("forward"))
        self.btn_back.clicked.connect(lambda: callback("backward"))
        
    # def set_button_callback(self, callback):
    #     """Set the callback function for both buttons"""
    #     self.button_callback = callback
    #     self.forward_button.configure(command=lambda: callback("forward"))
    #     self.backward_button.configure(command=lambda: callback("backward"))

    # def closeEvent(self, event):
    #     """Handle the window close event."""
    #     if hasattr(self, 'button_callback') and self.button_callback:
    #         self.button_callback("end")
        
        # Accept the event to allow the window to close
        # event.accept()

    # def on_closing(self):
    #     """Handle window closing using the same callback"""
    #     if self.button_callback:
    #         self.button_callback("end")
            
    #     self.root.destroy()
        
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

    # Function to handle mouse wheel for zooming
    # def on_scroll(self, event):
    #     if event.inaxes != self.ax:
    #         return
            
    #     # Only process if Ctrl key is pressed
    #     if event.key == 'control':
    #         # Get the current x and y limits
    #         x_min, x_max = self.ax.get_xlim()
    #         y_min, y_max = self.ax.get_ylim()
    #         x_range = x_max - x_min
    #         y_range = y_max - y_min
            
    #         # Zoom factor
    #         scale_factor = 1.2 if event.button == 'up' else 1/1.2
            
    #         # New ranges
    #         new_x_range = x_range / scale_factor
    #         new_y_range = y_range / scale_factor
            
    #         # Center point (where mouse is)
    #         x_center = event.xdata
    #         y_center = event.ydata
            
    #         # Calculate new limits centered on mouse position
    #         x_min_new = x_center - new_x_range * (x_center - x_min) / x_range
    #         x_max_new = x_center + new_x_range * (x_max - x_center) / x_range
    #         y_min_new = y_center - new_y_range * (y_center - y_min) / y_range
    #         y_max_new = y_center + new_y_range * (y_max - y_center) / y_range
            
    #         # Set new limits
    #         self.ax.set_xlim(x_min_new, x_max_new)
    #         self.ax.set_ylim(y_min_new, y_max_new)
    #         self.canvas.draw_idle()

    # Function to handle mouse button press for panning
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left mouse button
            self.pan_start[0] = event.xdata
            self.pan_start[1] = event.ydata
    
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


    def handle_action(self, action):
        if action == "forward":
            self.index = (self.index + 1) % len(self.configs)
        elif action == "backward":
            self.index = (self.index - 1) % len(self.configs)
        elif action == "end":
            self.finish = True
            self.close()
            return

        # self.clear_ui_elements()
        self.replot()
        
        self.export_pdf() 
        
        self.fig.savefig('regression_output/regression_output_canvas.svg', bbox_inches='tight') #, pad_inches=0.1)

        
        # self.export_widget_to_pdf(self.check_scroll, 'regression_output/eqs.pdf')
        # self.export_widget_to_pdf(self.scatter_checkbox, 'regression_output/samples.pdf')
        # self.export_widget_to_pdf(self.right_text, 'regression_output/right_text.pdf')
        # self.export_widget_to_pdf(self.left_text, 'regression_output/left_text.pdf')
        # self.export_widget_to_pdf(self.info_panel, 'regression_output/info_panel_text.pdf')
        # self.save_window_as_svg(filename='regression_output/global.svg')
        
        # # Auto-save SVG
        # import os
        # if not os.path.exists('regression_output'): os.makedirs('regression_output')
        # self.fig.savefig('regression_output/regression_output.svg', bbox_inches='tight', pad_inches=0.1)
        
    def closeEvent(self, event):
        """Standard Qt way to handle window closing."""
        self.handle_action("end")
        event.accept()

    def get_config_structure(self, model):
        try:
            return str(model.layers[0].out_features)
        except:
            return 'Unknown'
        
    # def handle_action(self, action):
    #     print(f'HANDLING ACTION: {action}')
    #     if action == "forward":
    #         self.index = (self.index + 1) % len(self.configs)
    #         self.clear_ui_elements() # Refactored clear_equations
    #         self.replot()
            
    #         # Saving SVG still works exactly the same
    #         import os
    #         if not os.path.exists('regression_output'):
    #             os.makedirs('regression_output')
    #         self.fig.savefig('regression_output/regression_output.svg', bbox_inches='tight', pad_inches=0.1)
    
    #     elif action == "backward":
    #         self.index = (self.index - 1) % len(self.configs)
    #         self.clear_ui_elements()
    #         self.replot()
            
    #     elif action == "end":
    #         self.finish = True
            # self.close() # This triggers closeEvent automatically
            
    # def handle_action(self, action):
    #     print(f'HANDLING ACTION')
    #     if action == "forward":
    #         self.index += 1
    #         if self.index == len(self.configs):
    #             self.index = 0
            
    #         # XX Retirar este comentario
    #         self.clear_equations()
    #         self.replot()
            
    #         if True:
    #             f_name = f'regression_output/regression_output.svg'
    #             self.fig.savefig(f_name, bbox_inches='tight', pad_inches=0.1)

            
    #     elif action == "backward":
    #         self.index -= 1
    #         if self.index < 0:
    #             self.index = len(self.configs) - 1
    
    #         # XX Retirar este comentario
    #         self.clear_equations()
    #         self.replot()
            
    #     elif action == "end":
    #         self.finish = True
    

    def eq_in_boundaries(self, eq):
        for boundary in self.boundaries:
            # print(f'Comparing boundary {boundary} con {eq}')
            # print(f'Result = {np.allclose(eq, boundary)}')
            if np.allclose(eq, boundary):
                return True
        return False
    
    
    def replot(self, ini=False):
        # 1. Prepare data and clear buffers
        config = list(dict(self.configs).keys())[self.index]
        
        self.right_text.clear()
        self.left_text.clear()
        self.info_panel.clear() # This was self.text_widget in Tk
        
        self.title = f'TRAIN SAMPLES: Activation Pattern {config[:4]}-{config[4:]}'
        
        # 2. Main Info Panel: Experiment Details
        # Header Info
        # self.append_text('<b>TOY EXPERIMENT ON 1D REGRESSION DATASET</b>')
        # self.append_text(f'# Neurons = {self.config_struct}')
        # self.append_text(f'# Activation Patterns = {len(self.configs)}')
        # self.append_text(f'# Samples = {len(self.X_train)} (train) {len(self.X_test)} (test)')
        # self.append_text(f'RMSE (train) = {np.sqrt(self.train_mse):.05f}')
        # self.append_text(f'RMSE (test)  = {np.sqrt(self.test_mse):.05f}')
        # self.append_text(f'Seed = {self.seed}<br>')
        
        # --- Build one single HTML string for the sidebar ---
        html = []
        
        # 1. Centered Header
        # html.append("<div style='font-family: \"Cascadia Mono\"; text-align: center; font-size: 11pt; margin-bottom: 10px;'>")
        html.append("<div style='font-family: monospace; text-align: center; font-size: 11pt; margin-bottom: 10px;'>")
        html.append("<b>TOY EXPERIMENT ON 1D<br>REGRESSION DATASET</b>")
        html.append("</div>")
        
        # 2. Left-aligned Statistics
        html.append("<div style='text-align: left; font-family: monospace; font-size: 9pt;'>")
        # html.append("<div style='text-align: left; font-family: \"Cascadia Mono\"; font-size: 9pt;'>")
        # html.append("<div style='text-align: left; font-family: sans-serif; font-size: 10pt;'>")
        html.append(f"# Neurons = {self.config_struct}<br>")
        html.append(f"# Activation Patterns = {len(self.configs)}<br>")
        html.append(f"# Samples = {len(self.X_train)} (train) {len(self.X_test)} (test)<br>")
        html.append(f"# Epochs = {self.epochs}<br>")
        html.append(f"Learning Rate = {self.lr}<br>")
        html.append(f"RMSE (train)&nbsp;= {np.sqrt(self.train_mse):.05f}<br>")
        html.append(f"RMSE (test)&nbsp;&nbsp;= {np.sqrt(self.test_mse):.05f}<br>")
        html.append(f"Seed = {self.seed}<br>")
        html.append("</div>")

        # 3. Helper for perfectly aligned tables
        def get_table_html(title_text, config_dict, rmse_dict, active_c):
            # We wrap the title in a span with a larger font size (e.g., 12pt)
            t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
            # The table itself remains standard size for precision
            t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
            # t_html += "<table width='100%' style='border-collapse: collapse; font-family: \"Cascadia Mono\"; font-size: 9pt;'>"
            t_html += "<tr><th width='90' align='left'>Pattern</th><th width='70' align='right'>#Samples</th><th width='80' align='right'>RMSE</th></tr>"
            
            for c, count in config_dict.items():
                # bg = "background-color: #FFFF22; font-weight: bold;" if c == active_c else ""
                bg = "background-color: #FFFF22;" if c == active_c else ""
                t_html += f"<tr style='{bg}'>"
                t_html += f"<td>{c[:4]}-{c[4:]}</td>"
                t_html += f"<td align='right'>{count}</td>"
                t_html += f"<td align='right'>{rmse_dict[c]:.05f}</td>"
                t_html += "</tr>"
            t_html += "</table><br>"
            return t_html

        html.append(get_table_html("Train samples", self.configs, self.train_rmse_dict, config))
        html.append(get_table_html("Test samples", self.test_configs, self.test_rmse_dict, config))

        # --- Set the HTML once ---
        self.info_panel.setHtml("".join(html))
        
           
        # 5. Math Retrieval
        eq_list, neuron_eqs, output_contribs, inequality = \
            get_eq_list_regression(config, self.config_samples, self.X_train, 
                                   self.model, self.boundaries, self.decimals)
            
        contrib = output_contribs[0]
    
        # --- 6. Bottom Panels: HTML Construction ---

        # A. LEFT PANEL: INEQUALITIES
        # We start with the Title
        # left_html = "<div style='font-family: \"Cascadia Mono\"; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
        left_html = "<div style='font-family: monospace; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
        
        # Line 2: The Global Domain Range (separated by <br>)
        left_html += f"&nbsp;&nbsp;{self.x_range[0]:.0{self.decimals}f} <= x <= {self.x_range[1]:.0{self.decimals}f}<br>"
        
        # Line 3+: Each equation on a new line
        for index, eq in enumerate(neuron_eqs):
            bit = config[index]
            # Use orig_sign to ensure the math string shows >= 0 or <= 0 correctly
            eq_str = self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=int(bit))
            left_html += f"&nbsp;&nbsp;N={bit}, HS{index+1}: {eq_str}<br>"
        
        left_html = left_html.replace('<=', '&le;').replace('>=', '&ge;')

        left_html += '</div>'

        # Set the left panel
        self.left_text.setHtml(left_html)

        # B. RIGHT PANEL: ACTIVATION REGION
        # Title
        # right_html = "<div style='font-family: \"Cascadia Mono\"; font-size: 10pt;'><b>RULE ANTECEDENT (Activation Region)</b><br>"
        right_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE ANTECEDENT (Activation Region)</b><br>"
        
        # # Bullet 1: Input Range (using &bull; for manual bullet)
        # right_html += "<b>&bull; Input Range</b><br>"
        # Using &nbsp; for indentation instead of a list style
        right_html += f"&nbsp;&nbsp;{inequality[0]:.0{self.decimals}f} <= x <= {inequality[1]:.0{self.decimals}f}<br><br>"
        
        # Bullet 2: Network Output
        right_html += "<b>RULE CONSEQUENT (Network Output)</b><br>"        

        # right_html += "<b>&bull; Network Output</b><br>"
        consequent_str = self.str_equ(contrib, decimals=self.decimals)
        right_html += f"&nbsp;&nbsp;Output Y = {consequent_str}"

        right_html = right_html.replace('<=', '&le;').replace('>=', '&ge;')
        
        right_html += '</div>'
        # Set the right panel
        self.right_text.setHtml(right_html)
        
            
        # 7. Sidebar and Scatter Update
        
        self.clear_ui_elements()
        
        self.color_index = 0 
        
        for index, eq in enumerate(eq_list):
            if eq[2] != -2 and eq[1] is not None:
                self.add_equation(eq[0], eq[1], self.color_list[self.color_index], 
                                  linestyle='solid' if index < 4 else 'dash')
                self.color_index = (self.color_index + 1) % len(self.color_list)
    
        # y1 = contrib[1] * inequality[0] + contrib[0]
        # y2 = contrib[1] * inequality[1] + contrib[0]
        
        # # self.add_line(config, [inequality[0], inequality[1]], [y1, y2], 
        # #               self.color_list[self.color_index], linestyle='solid')
    
        active_indices = self.config_samples[config]
        self.set_scatter_points(self.X_train[active_indices].numpy(),
                                self.y_predictions[active_indices], 
                                np.repeat(self.color_list[self.color_index], len(active_indices)))
    

def generate_hello(n_inputs, size, range_x=-2.5, range_y=5):
    
    def f(x): 
        return 0.01*((x+2)*(x-1)*(x-3)*(x-4)+50)
    
    x = np.linspace(range_x, range_y, size)
    y = f(x)
    
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    
    return torch.from_numpy(x).type(torch.float), torch.from_numpy(y).type(torch.float)
    

def generate_new_hello(n_inputs, size, range_x=-2.5, range_y=5, 
                       seed=1, use_noise=False, rng=None, noise=(0.0, 0.05)):
    
    def f(x, rng): 
        val = 0.01*((x+2)*(x-1)*(x-3)*(x-4)+50)
        if use_noise:
            val += rng.normal(noise[0], noise[1], size=x.shape)
        return val
    
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    
    x = np.linspace(range_x, range_y, size)
    y = f(x, rng)
    
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    
    return torch.from_numpy(x).type(torch.float), torch.from_numpy(y).type(torch.float), rng
    
      
    
def main(experiment, hidden, epochs, 
         point_size=2, polygon_color='gainsboro', 
         random_seed = 33, lr=0.001, decimals=2, noise=None):
    
    print(f'Launching regression experiment {experiment} with hidden = {hidden} and epochs = {epochs}')
    
    np.seed = random_seed
    np.random.seed(random_seed)
    random.seed = random_seed
    torch.manual_seed(random_seed)

    ds_name = experiment
    model_version = 0
    use_saved_model_weights = True
    # epochs = 30000

    num_inputs = 1
    num_outputs = 1
    
    # sigma = 0.05
    
    if experiment == 'hello_world':    
        X_train, y_train = generate_hello(num_inputs, 1000)
        X_test, y_test = generate_hello(num_inputs, 100)
    elif experiment == 'new_hello_world':
        ds_name = 'Regression Hello World'
        
        X_train, y_train, rng = generate_new_hello(num_inputs, 1500, 
                                              seed=random_seed, 
                                              use_noise=True,
                                              noise=noise)

        n_train = len(X_train)
        n_test = int((n_train)*((1/(2/3))-1))

        X_test, y_test, _ = generate_new_hello(num_inputs, n_test, 
                                            rng=rng,
                                            use_noise=True,
                                            noise=noise)


    train_dataset = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32)

    weights_file_name = f'{ds_name}_inputs_{num_inputs}_hidden_{hidden}_epochs_{epochs}_seed_{random_seed}_lr_{lr}_noise_mean_{noise[0]}_noice_sigma_{noise[1]}.pth'

    if use_saved_model_weights and os.path.isfile(weights_file_name):
        model = torch.load(weights_file_name, weights_only=False)
        print(
            f'Using pretrained classification model weights from file {weights_file_name}')
    else:
        print('Training classification model ... ', end='')

        model = FNNModule()
        
        model.add_layer(nn.Linear(num_inputs, hidden[0]))
        
        n_inp = hidden[0]
        
        for i, h in enumerate(hidden[1:]):
            model.add_layer(nn.ReLU())
            model.add_layer(nn.Linear(n_inp, h))
            n_inp = h 
                        
        model.add_layer(nn.ReLU())
        model.add_layer(nn.Linear(n_inp, num_outputs))


        # model = FNNModule()
        # model.add_layer(nn.Linear(num_inputs, hidden))
        # model.add_layer(nn.ReLU())
        # model.add_layer(nn.Linear(hidden, num_outputs))

        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_accuracy = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                running_loss += loss.item()
                # running_mse += compute_accuracy(outputs, targets)
                num_batches += 1
                
                if epoch % 1000 == 0: print(f'Epoch {epoch} ... runnnig loss = {running_loss:0.10f}')
            
            # Calculate epoch metrics
            epoch_loss = running_loss / num_batches
            epoch_accuracy = running_accuracy / num_batches
            
        model.eval()
        
        for param in model.parameters():
            param.requires_grad_(False)

        torch.save(model, weights_file_name)
        print('OK')

        
    
    # Define symbolic variables
    x, y = sp.symbols('x y')
    

    '''
    Realizamos predicciones y evaluamos el resultado de la FNN
    '''
    print('Generating predictions for train data ... ', end='')

    train_mlp_predictions = model(X_train).detach().numpy()
    # y_mlp = np.argmax(mlp_predictions, axis=1)
    y_train_mlp = train_mlp_predictions 
    
    print('OK')

    # accuracy = np.sum(y_mlp == y_test.numpy()) / len(y_test)
    # accuracy = np.sum(y_mlp == y_test) / len(y_test)
    
    train_mse = np.mean((train_mlp_predictions - y_train.numpy()) ** 2)
    
    print(f'Train data MSE = {train_mse:.5f}, RMSE = {np.sqrt(train_mse):.5f}\n')

    print('Generating predictions for test data ... ', end='')

    test_mlp_predictions = model(X_test).detach().numpy()
    # y_mlp = np.argmax(mlp_predictions, axis=1)
    y_test_mlp = test_mlp_predictions 
    
    print('OK')

    # accuracy = np.sum(y_mlp == y_test.numpy()) / len(y_test)
    # accuracy = np.sum(y_mlp == y_test) / len(y_test)
    
    test_mse = np.mean((test_mlp_predictions - y_test.numpy()) ** 2)
    
    print(f'Test data MSE = {test_mse:.5f}, RMSE = {np.sqrt(test_mse):.5f}\n')
    
    pass

    # mlp_train_predictions = model(X_train).detach().numpy()
    # y_train_mlp = np.argmax(mlp_train_predictions, axis=1)
    
    # samples = np.random.choice(range(len(X_test)), 10000, replace=True)

    # samples = [i for i in range(len(X_train))]

    # t_list = []
    # for sample in samples:
    #     t = time.time()    
    #     get_face_contrib_accelerated(X_train[sample], model)
    #     t_list.append(time.time() - t)
               
    # print(f'Tiempo por ejecución de FACE: {np.mean(t_list):.06f} +/- {np.std(t_list):.06f} s')                 

    # configs_new = new_count_configurations(model, X_train, get_samples=True, 
    #                                                include_last=False, order=True,
    #                                                boundaries=(-2.5, 5.0))
    


    configs, config_samples = count_configurations(model, X_train, get_samples=True, 
                                                   include_last=False, order=True)
    
    train_rmse_dict = {}
    for c in configs:    
        samples = config_samples[c]
        train_rmse = np.sqrt(np.mean((train_mlp_predictions[samples] - y_train[samples].numpy()) **2))
        train_rmse_dict[c] = train_rmse


    test_configs, test_config_samples = count_configurations(model, X_test, get_samples=True, 
                                                             include_last=False, order=True)
    
    test_rmse_dict = {}
    for c in test_configs:    
        samples = test_config_samples[c]
        test_rmse = np.sqrt(np.mean((test_mlp_predictions[samples] - y_test[samples].numpy()) **2))
        test_rmse_dict[c] = test_rmse

    
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 100
    
    # fig = Figure(figsize=(5.5, 5.5), dpi=100)
    # ax = fig.add_subplot(111)

    # canvas = FigureCanvas(fig)
    
    # x_pos = np.linspace(-2.5, 5, 1000)
    
    # x = X_train.numpy().reshape(1,-1)[0]
    # y = y_train.numpy().reshape(1,-1)[0]
    
    # colors = ['red', 'blue', 'green', 'magenta', 'yellow', 'orange', 'violet', 'salmon', 'blue', 'olive']
        
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, y, '--', linewidth=2)
    

    
    # for (x1, x2), color in zip(x_subspaces, colors):
    #     # Find indices corresponding to the subspace boundaries
    #     # print(f'x_subspace=[{x1}, {x2}]')
    #     # idx_start = np.searchsorted(x, x1)
    #     # idx_end = np.searchsorted(x, x2)
        
    #     # Extract the relevant portion of x and y arrays
    #     # x_sub = x[idx_start:idx_end+1]
        
    #     x_sub = x[x1:x2+1]
    #     # y_sub = y[x1:x2+1]
    #     y_sub = y_train_mlp[x1:x2+1].reshape(1,-1)[0]

        
    #     # print(f'x_sub={x_sub}, y_sub={y_sub}')
    #     # Fill between 0 and the curve
    #     plt.fill_between(x_sub, 0, y_sub, color=color, alpha=0.5)
    
    # plt.show()

    # # print('\nHOLA!')
    
    
    if len(configs) > 0:
        # root = tk.Tk()
        
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

        app = QtWidgets.QApplication(sys.argv)
        
        # root.geometry('1100x600')

        plotter = QtImplicitEquationRegressionPlotter(configs, config_samples, 
                                          X_train, y_train, X_test, 
                                          y_train_mlp, 
                                          model, 
                                          train_mse=train_mse, test_mse=test_mse,
                                          point_size=point_size,
                                          polygon_color=polygon_color,
                                          lr=lr, epochs=epochs,
                                          experiment=experiment,
                                          seed=random_seed,
                                          decimals=decimals, 
                                          x_range=(-2.5, 5.0),
                                          noise=noise,
                                          train_rmse_dict=train_rmse_dict,
                                          test_rmse_dict=test_rmse_dict,
                                          test_configs=test_configs)
        
        plotter.show()

        sys.exit(app.exec_())

        # while not plotter.finish:
        #     root.mainloop()


if __name__ == "__main__":

    # main(experiment='new_hello_world', hidden=8, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=11, decimals=4, noise=(0.0, 0.01))

    # main(experiment='new_hello_world', hidden=[8], epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=11, decimals=4, noise=(0.0, 0.01))

    main(experiment='new_hello_world', hidden=[4,4], epochs=12000, point_size=5, 
            lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.01))



    # main(experiment='hello_world', hidden=10, epochs=8000, point_size=5, 
    #         lr=0.0001, random_seed=1, decimals=4)
    
    # main(experiment='hello_world', hidden=5, epochs=8000, point_size=5, 
    #         lr=0.0001, random_seed=1, decimals=4)
    
    # main(experiment='new_hello_world', hidden=5, epochs=80000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4)

    # main(experiment='new_hello_world', hidden=5, epochs=8000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4)

    # main(experiment='new_hello_world', hidden=8, epochs=45000, point_size=5, 
    #         lr=0.0001, random_seed=1, decimals=4, noise=(0.0, 0.05))

    # main(experiment='new_hello_world', hidden=9, epochs=25000, point_size=5, 
    #         lr=0.001, random_seed=11, decimals=4, noise=(0.0, 0.05))

    # main(experiment='new_hello_world', hidden=8, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.05))

    # main(experiment='new_hello_world', hidden=8, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.001))

    # main(experiment='new_hello_world', hidden=8, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.001))

    # main(experiment='new_hello_world', hidden=8, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=11, decimals=4, noise=(0.0, 0.01))
    
    # main(experiment='new_hello_world', hidden=7, epochs=12000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.025))

    # main(experiment='new_hello_world', hidden=6, epochs=25000, point_size=5, 
    #         lr=0.001, random_seed=1, decimals=4, noise=(0.0, 0.05))

    # # import sys 
    # # print(sys.path())
    # # main(experiment='sign', hidden=8, epochs=2000, point_size=2)
    # # main(experiment='sign', hidden=16, epochs=30000, point_size=2)
    # # main(experiment='xor', hidden=4, epochs=4000, point_size=10)
    # # main(experiment='xor', hidden=3, epochs=14000, point_size=10, 
    # #      random_seed=33, polygon_color='gainsboro', lr=0.0001)
    
    # '''
    # Ok
    # '''
    # '''
    # Para ver zona pequeña
    # '''
    # # main(experiment='sign', hidden=4, epochs=6000, 
    # #      point_size=5, polygon_color='gainsboro', lr=0.0001, random_seed=33)

    # main(experiment='sign', hidden=4, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=4)
    
    # # main(experiment='sign', hidden=3, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=4)

    # # main(experiment='sign', hidden=2, epochs=6000, point_size=5, lr=0.0001, random_seed=1, decimals=2)

    # # main(experiment='sign', hidden=2, epochs=6000, point_size=5, lr=0.0001, random_seed=443, decimals=2)
    
    # # main(experiment='sign', hidden=6, epochs=3000, point_size=5, lr=0.0001, random_seed=11, decimals=4)
    
    # # main(experiment='sign', hidden=5, epochs=3000, point_size=5,
    # #      random_seed=3, lr=0.0001)
    
    # # main(experiment='xor', hidden=3, epochs=64000, point_size=10, 
    # #      random_seed=3, lr=0.0001)
    
    # # main(experiment='xor', hidden=4, epochs=20000, point_size=10, 
    # #      random_seed=3, lr=0.0001)

    # # main(experiment='sign', hidden=6, epochs=30000, point_size=2, lr=0.0001)