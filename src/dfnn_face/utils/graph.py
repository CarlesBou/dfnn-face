# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

# import torch
# import torch.nn as nn

import matplotlib

from matplotlib import pyplot as plt
import math

import numpy as np

import cdd
# from collections import Counter, OrderedDict

from sympy import Symbol
import sympy as sp


import cvxpy as cp
from .face_torch import get_face_contrib_accelerated


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
    
    
# subscript_map = '0RG3456789'





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
    
    # if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0 or contrib_y1_y2[0] != 0:
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
        
        # ex_class0 = False
        if len(generator_unbounded_class0.array) > 0:
            
            if not bounded:
                V_poly_unbounded_class0 = cdd.polyhedron_from_matrix(generator_unbounded_class0) 
                H_inequalities_class0 = cdd.copy_inequalities(V_poly_unbounded_class0)
            else:
                try:
                    V_poly_bounded_class0 = cdd.polyhedron_from_matrix(generator_bounded_class0) 
                    H_inequalities_class0 = cdd.copy_inequalities(V_poly_bounded_class0)
                except:
                    # ex_class0 = True
                    H_inequalities_class0 = []
                    print('EXCEPT 1')

        
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
                    print('EXCEPT 2')
                   
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
        H_bounded = cdd.matrix_from_array(new_matrix, rep_type=cdd.RepType.INEQUALITY)
        # H_bounded = cdd.matrix_from_array(matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_bounded = cdd.polyhedron_from_matrix(H_bounded)
        # H_poly_unbounded = cdd.polyhedron_from_matrix(H_bounded)
        generator_bounded = cdd.copy_generators(H_poly_bounded)
        
        try: 
            V_poly_bounded = cdd.polyhedron_from_matrix(generator_bounded) 
            H_inequalities = cdd.copy_inequalities(V_poly_bounded)
        except:
            H_inequalities = cdd.matrix_from_array([], rep_type=cdd.RepType.INEQUALITY)
            print('EXCEPT 3')


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

        eq_name = 'Eq' + str(len(config) + 1)
        eq_list.append((eq_name, new_equation, 0, generator_bounded, sign2))
        # eq_list.append(('Eq9', new_equation, 0, generator_bounded, sign2))
    else:
        eq_name = 'Eq' + str(len(config) + 1)
        
        sign2 = 1 if contrib_y1_y2[0] >= 0 else -1

        eq_list.append((eq_name, new_equation, 0, generator_bounded, sign2))
        # print(f'CASI 0s!!')
        
    
    return eq_list, \
            H_inequalities, \
            generator_bounded_list_class0, generator_bounded_list_class1, \
            H_inequalities_class0, H_inequalities_class1, \
            generator_bounded.array, \
            all_zeros_winning_class, \
            w0, sign_vec, contrib, ret_y1_y2
    


  
    


class ImplicitEquationPlotter():
    def __init__(self, 
                 config=None, configs=None, config_samples=None,
                 X=None, y=None,
                 model=None, 
                 title='Config sin definir',
                 polygon_color='gainsboro',
                 experiment='', epochs=5000, lr=0.0001, seed=33,
                 decimals=2, show_boundaries=True,
                 show_graph=True, color_class='magenta',
                 point_size=5, light_point_size=2, light_point_alpha=0.4,
                 class_linewidth=2, neuron_linewidth=1.6,
                 normalized_range=None,
                 test_configs=None,
                 test_color_count=None,
                 config_index=-1,
                 hidden_struct=None,
                 arrow_length=0.4, 
                 arrow_head_size=0.15):
        
        self.finish = False
        
        if config_index == -1:
            if config is None:
                self.index = 0
            else:
                self.index = list(configs).index(config)
        else:
            self.index = config_index 
            
        self.plotter = None
    
        self.show_graph = show_graph
        self.color_class = color_class
        
        self.configs = configs
        # self.test_configs = test_configs
        self.config_samples = config_samples
        # self.color_count = color_count
        # self.test_color_count = test_color_count
        self.X_train = X
        self.y_train = y
        # self.X_test = X_test
        
        self.x_range = normalized_range[0]
        self.y_range = normalized_range[1]
    
        self.config_struct = self.get_config_structure(model)

        # Set colors: green = class 0, else red
        self.y_colors = np.where(self.y_train, 'green', 'red')
        self.y_colors_light = np.where(self.y_train, 'green', 'red')
        

        # self.train_accuracy = train_accuracy
        # self.test_accuracy = test_accuracy
        self.model = model
        
        self.title = title
        self.point_size = point_size
        self.light_point_size = light_point_size
        self.light_point_alpha = light_point_alpha
        
        self.class_linewidth = class_linewidth 
        self.neuron_linewidth = neuron_linewidth 
        
        self.arrow_length = arrow_length
        self.arrow_head_size = arrow_head_size
        
        # self.lr = lr
        # self.epochs = epochs
        # self.experiment = experiment
        # self.seed = seed
        self.decimals = decimals
        self.show_boundaries = show_boundaries

        
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
        # self.button_callback = None
    
        self.polygon_color = polygon_color
        
        self.hidden_struct = hidden_struct
        
        

        

    def show(self, config):
        
        matplotlib.rcParams['figure.dpi'] = 100
        
        self.index = list(self.configs).index(config)

        self.create_gui()
        
        # self.subscript_map = '0123456789'
        
        self.replot(ini=True)
        self.update_plot()
        self.replot()
        
        plt.show()
        
        
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
        
        
    
    def create_gui(self):

        self.fig = plt.figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Initialize variable dictionary for checkbuttons
        self.var_dict = {}
                
    
    def clear_ui_elements(self):
        # Clear the dictionary
        self.check_dict = {}
        self.line_dict = {}
        
        
    def clear_equations(self):
        """Surgically remove all dynamic checkboxes from the sidebar."""
        self.equations.clear()
        self.var_dict.clear()
        self.arrows_config.clear()
        self.contour_sets.clear()
        self.poly.clear()
                
        self.color_index = 0 
        self.replot()
        self.update_plot()
        
                  
        
    def set_scatter_points(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points = np.array(list(points))
        self.scatter_colors = colors
        
        self.update_plot()

        
    def set_scatter_points_light(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points_light = np.array(list(points))
        self.scatter_colors_light = colors
        
        self.update_plot()
            
        
        
    def clear_scatter_points(self):
        """Remove scatter points from data and reset the UI checkbox."""
        self.scatter_points = None
        self.scatter_colors = None
        
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
        
        
    
        # # Create new checkbutton variable
        var = BooleanVarReplica(value=True)
        self.var_dict[name] = var
        
        
    def remove_equation(self, name):
        """Remove an equation from the plotter with surgical precision."""
        if name in self.equations:
            # Remove data references
            del self.equations[name]
            del self.var_dict[name]
       
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
                    linewidth = self.neuron_linewidth
                    #1.6  
                    new_color = self.equations[eq_name][1]
                    direction = self.equations[eq_name][3]
                    
                    # # if eq_name.startswith('Eq9'):
                    # if int(eq_name[-1]) == len(self.var_dict.items()):
                    #     linewidth = self.class_linewidth
                    #     new_color = self.color_class
                    #     linestyles = 'dashdot'
                    # elif eq_name[2] in ['1', '2', '3', '4']:
                    #     linestyles = self.equations[eq_name][2]
                    #     linewidth = self.neuron_linewidth - 0.4
                    #     '''
                    #     Nuevo
                    #     '''
                    #     print('ACA1')
                    # else:
                    #     linestyles = self.equations[eq_name][2]
                    #     linewidth = self.neuron_linewidth
                    #     print('ACA2')

                    # if eq_name.startswith('Eq9'):
                    if int(eq_name[-1]) == len(self.var_dict.items()):
                        linewidth = self.class_linewidth
                        new_color = self.color_class
                        linestyles = 'dashdot'
                    elif self.hidden_struct is None:
                        if eq_name[2] in ['1', '2', '3', '4']:
                            linestyles = self.equations[eq_name][2]
                            linewidth = self.neuron_linewidth
                        else:
                            linestyles = self.equations[eq_name][2]
                            linewidth = self.neuron_linewidth
                    else:
                        if int(eq_name[-1]) <= self.hidden_struct[0]:
                            linestyles = self.equations[eq_name][2]
                            linewidth = self.neuron_linewidth
                        else:
                            linestyles = self.equations[eq_name][2]
                            linewidth = self.neuron_linewidth

                        
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
                                        head_width=self.arrow_head_size * adjust,
                                        head_length=self.arrow_head_size * adjust,
                                        fc=self.equations[eq_name][1],
                                        ec=self.equations[eq_name][1],
                                        length_includes_head=True,
                                        width=0.05 * adjust,
                                        zorder=1000)
    
        
                self.plot_polygon()
                
                # config = list(dict(self.configs).keys())[self.index]
        
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
    
    

    def eq_in_boundaries(self, eq):
        for boundary in self.boundaries:
            if np.allclose(eq, boundary):
                return True
        return False
    
    def replot(self, ini=False):        
        config = list(dict(self.configs).keys())[self.index]
      
        self.title = f'Activation Pattern {self.get_config_structure(self.model, config)}'
        
        eq_list, inequalities, poly_class0, poly_class1, inequalities_class0, inequalities_class1, poly_global, all_zeros_winning_class, neuron_eqs, signs, output_contrib_eqs, output_class_eq = \
                            get_eq_list_new(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)
                            # get_eq_list(config, self.config_samples, self.X_train, self.model, self.boundaries, self.decimals, y_train=self.y_train)

        self.H_inequalities[config] = inequalities
        self.H_inequalities_class0[config] = inequalities_class0
        self.H_inequalities_class1[config] = inequalities_class1
                
        color_index = 0
        color_list = ['blue', 'olive', 'brown', 'orange', 'lightblue', 'pink', 'gray', 'lightseagreen']
        color_list = ['blue', 'olive', 'brown', 'cyan', 'orange', 'blue', 'olive', 'brown', 'cyan', 'orange']

        if all_zeros_winning_class >= 0:
            if all_zeros_winning_class == 0:
                poly_class0 = poly_global
                poly_global = []
                inequalities_class0 = inequalities 
            elif all_zeros_winning_class == 1:
                poly_class1 = poly_global
                poly_global = []
                inequalities_class1 = inequalities
                
                
        if self.show_graph:
            
            self.clear_ui_elements()

            self.color_index = 0 
            
            '''
            Add polygons, scatter points, lines, and arrows
            '''
            for index, eq in enumerate(eq_list):
                if eq[1] == False:
                    # print(f'EQ[1] {index} = FALSE!!!')
                    continue
                
                if eq[2] != -2 and eq[1] is None: 
                    # print('AQUÍ 2!!!')
                    pass
                else:
                    if eq[2] == 0:
                        '''
                        Para la ecuación de clase, siempre marcamos la dirección 
                        hacia la clase 0 (roja)
                        '''
                        self.add_equation(eq[0], eq[1], self.color_class, 
                                          linestyle='dashdot', direction=0)
                        self.add_orthogonal_arrow(eq[0], arrow_length=self.arrow_length,
                                                  direction=1, # eq[4] 
                                                  color=self.color_class)
                        # print('AQUÍ!!!')
                    elif eq[2] == -2: 
                        # print(f'AQUÍ 3!!! {eq[0]}')
                        pass
                    else:
                        # print(f'AQUÍ 6!!!! {eq[0]}')
                        if index < self.model.layers[0].out_features:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='solid', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=self.arrow_length, direction=eq[2], 
                                                      color=color_list[color_index])
                        else:
                            self.add_equation(eq[0], eq[1], color_list[color_index], linestyle='dashed', direction=eq[4])
                            color_index = (color_index + 1) % len(color_list)
                            self.add_orthogonal_arrow(eq[0], arrow_length=self.arrow_length, direction=eq[2], 
                                                      color=color_list[color_index])
    
    
            if inequalities_class0 != []:
                self.add_polygon('Class 1', poly_class0, fill_color='lightsalmon', alpha=0.15)
    
            if inequalities_class1 != []:
                self.add_polygon('Class 2', poly_class1, fill_color='lightgreen', alpha=0.15)
    
            self.set_scatter_points_light(self.X_train,
                                    self.y_colors_light)

            self.set_scatter_points(self.X_train[self.config_samples[config]],
                                    self.y_colors[self.config_samples[config]])
            


