# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 09:43:42 2024

@author: Carles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
import os 
import sys
import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib.patches import Circle
import math
import time
# from keras.datasets import mnist
import numpy as np
import random
import cdd
# from scipy.spatial import ConvexHull
from collections import Counter, OrderedDict

from sympy import Symbol
# from sympy import plot_implicit, plot, N, Float
import sympy as sp

import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5 import QtPrintSupport,QtSvg
# from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QColor
# from PyQt5.QtGui import QPdfWriter, QPageSize
# from PyQt5.QtCore import QSizeF, QRectF, QMarginsF
from PyQt5.QtPrintSupport import QPrinter
# from PyQt5.QtGui import QPainter, QImage, QPixmap, QPainterPath
# from PyQt5.QtCore import QRect, QSize
from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPainter, QColor
# from PyQt5.QtGui import QTextDocument

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import cvxpy as cp
# from utils.ineq import get_radius, get_radius_old, analyze_displacement
# from utils.ineq import get_ra_matrix, get_new_ra
from utils.ineq import get_face_contrib_accelerated

import pickle
# from pprint import pprint


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
                # print(f'Keeping equation at index {i}')
                return np.array([])
            else:
                tolerance_radius = 1e-6
                # tolerance_center = 1e-4
                
                radius_change = np.abs(original_radius - new_radius)
                # center_distance = np.sqrt(np.sum((original_center - new_center) ** 2))
                
                print(f'RADIUS CHANGE = {radius_change:.04e}')
                if radius_change < tolerance_radius:
                # if radius_change < tolerance_radius and center_distance < tolerance_center:
                    removal_list.append(i)
                else:
                    # print(f'Different at index {i}, dif={radius_change:0.4e}, center_dis={center_distance:.04e}')
                    # print(f'Keep equation at index {i}')
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
                # print(f'Keeping equation at index {i}')
                return np.array([])
            else:
                tolerance_radius = 1e-10
                tolerance_center = 1e-10
                
                radius_change = np.abs(original_radius - new_radius)
                center_distance = np.sqrt(np.sum((original_center - new_center) ** 2))
                
                
                if radius_change < tolerance_radius and center_distance < tolerance_center:
                    removal_list.append(i)
                else:
                    # print(f'Different at index {i}, dif={radius_change:0.4e}, center_dis={center_distance:.04e}')
                    # print(f'Keep equation at index {i}')
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
                    # break
        
        if not removed_any:
            break
    
    return current_matrix


# import numpy as np
# import cvxpy as cp


# ''' 
# Remove redundant inequations of a system

# IMPORTANT
#  matrix in [b A] form, representing Ax + b >=0 (as used in CDD)
 
#  '''

# def reduce_system(matrix):
#     # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
#     n_constraints, n_dims_plus_one = matrix.shape
#     n_dims = n_dims_plus_one - 1
#     b = matrix[:, 0]  
#     A = matrix[:, 1:]  

#     objective_coeffs = np.ones(n_dims)

#     # Initial LP to find baseline optimal value (optional, for reference)
#     x = cp.Variable(n_dims)
    
#     objective = cp.Maximize(objective_coeffs @ x)
    
#     constraints = [A @ x + b >= 0]
    
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status != "optimal":
#         # print("Initial problem is infeasible or unbounded. Exiting.")
#         return matrix

#     # Check each constraint for redundancy
#     redundant_indices = []
#     for i in range(n_constraints):
#         # Test if constraint i is active by maximizing a_i^T x subject to other constraints
#         x_test = cp.Variable(n_dims)
        
#         mask = np.ones(n_constraints, dtype=bool)
#         mask[i] = False
    
#         A_reduced = A[mask, :]
#         b_reduced = b[mask]
        
#         '''
#         Nuestro sistema está descrito como Ax >= -b (o Ax + b >= 0)
#         Así Max(A_1x) --> Max(-Ax)
#         '''
#         objective_test = cp.Maximize(-A[i, :] @ x_test)

#         constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
        
#         problem_test = cp.Problem(objective_test, constraints_reduced)
        
#         problem_test.solve()

#         if problem_test.status == "optimal":
#             epsilon = 1e-8 # Posible cálculo δ = ε · max(1, ||a_j|| · ||b_j||)
#             threshold = b[i] - epsilon
            
#             if problem_test.value <= threshold:
#                 redundant_indices.append(i)
    
#     # Remove redundant constraints
#     if redundant_indices:
#         keep_indices = [i for i in range(n_constraints) if i not in redundant_indices]
#         return matrix[keep_indices, :]
#     else:
#         return matrix


# def check_redundancy(matrix, objective_coeffs=None):
#     # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
#     n_constraints, n_dims_plus_one = matrix.shape
#     n_dims = n_dims_plus_one - 1
#     b = matrix[:, 0]  # Negate b for Ax <= b to -Ax >= -b
#     A = matrix[:, 1:]  # Negate A for Ax <= b to -Ax >= -b

#     # Set default objective coefficients (sum of all variables) if not provided
#     if objective_coeffs is None:
#         objective_coeffs = np.ones(n_dims)
#     elif len(objective_coeffs) != n_dims:
#         raise ValueError("Objective coefficients must match number of dimensions")

#     # Check if objective matches any constraint (up to scale)
#     for i in range(n_constraints):
#         if np.allclose(A[i, :], objective_coeffs, rtol=1e-5, atol=1e-8):
#             raise ValueError(f"Objective matches constraint {i} coefficients.")

#     # Initial LP to find baseline optimal value (optional, for reference)
#     x = cp.Variable(n_dims)
#     objective = cp.Maximize(objective_coeffs @ x)
#     constraints = [A @ x + b >= 0]  # Constraints as -Ax >= -b
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status != "optimal":
#         print("Initial problem is infeasible or unbounded. Exiting.")
#         return matrix

#     original_value = problem.value
#     # print(f"Original optimal value: {original_value}")

#     # Check each constraint for redundancy
#     redundant_indices = []
#     for i in range(n_constraints):
#         # Test if constraint i is active by maximizing a_i^T x subject to other constraints
#         x_test = cp.Variable(n_dims)
        
#         mask = np.ones(n_constraints, dtype=bool)
#         mask[i] = False
    
#         A_reduced = A[mask, :]
#         b_reduced = b[mask]
        
#         # Maximize a_i^T x (equivalent to minimizing -a_i^T x in -Ax >= -b form)
#         # objective_test = cp.Minimize(A[i, :] @ x_test + b[i])
#         # constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
#         # problem_test = cp.Problem(objective_test, constraints_reduced)
#         # problem_test.solve()

#         objective_test = cp.Maximize(A[i, :] @ x_test - b[i])
#         constraints_reduced = [A_reduced @ x_test - b_reduced <= 0]
#         problem_test = cp.Problem(objective_test, constraints_reduced)
#         problem_test.solve()

#         '''
#         Posible cálculo de la tolerancia, con epsilon = 1e-8
#         ''' 
#         # δ = ε · max(1, ||a_j|| · ||b_j||)
        
#         if problem_test.status == "optimal":
#             tolerance = 1e-8
            
#             if problem_test.value <= +tolerance:
#                 print(f'  Red AA - {problem_test.value} >= {tolerance}')
#                 redundant_indices.append(i)
    
#         #     new_value = problem_test.value
#         #     # print(f"\nTesting constraint {i} (e.g., {matrix[i]}):")
#         #     # print(f'Comparing new_value={new_value} con old_value={b[i]}, diff={abs(new_value - b[i])}')
#         #     # print(f"Max a_{i}^T x: {new_value}, b_{i}: {b[i]}")
#         #     if b[i] < 0:
#         #         if new_value < b[i]:
#         #             redundant_indices.append(i)
#         #             # print(f"Constraint {i} is redundant (max a_{i}^T x < b_{i} + tolerance)")
#         #         else:
#         #             # print(f"Constraint {i} is non-redundant (max a_{i}^T x >= b_{i} + tolerance)")
#         #             print(f'  BB - {new_value} < {b[i]}')
#         #     else:
#         #         if new_value > b[i]:
#         #             redundant_indices.append(i)
#         #             # print(f"Constraint {i} is redundant (max a_{i}^T x < b_{i} + tolerance)")
#         #             print(f'  Red AAA - {new_value} < {b[i]}')
#         #         else:
#         #             # print(f"Constraint {i} is non-redundant (max a_{i}^T x >= b_{i} + tolerance)")
#         #             print(f'  BBB - {new_value} < {b[i]}')
#         # else:
#         #     print(f"\nConstraint {i} (e.g., {matrix[i]}) is essential (infeasible without it)")
#         #     # If reduced problem is infeasible, the constraint is non-redundant
#         #     continue

#     # Remove redundant constraints
#     if redundant_indices:
#         print(f"\nRemoving redundant constraints at indices: {redundant_indices}")
#         keep_indices = [i for i in range(n_constraints) if i not in redundant_indices]
#         return matrix[keep_indices, :]
#     else:
#         print("\nNo redundant constraints found.")
#         return matrix
    
# def check_redundancy33(matrix, objective_coeffs=None):
#     # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
#     n_constraints, n_dims_plus_one = matrix.shape
#     n_dims = n_dims_plus_one - 1
#     b = matrix[:, 0]  # Negate b for Ax <= b to -Ax >= -b
#     A = matrix[:, 1:]  # Negate A for Ax <= b to -Ax >= -b

#     # Set default objective coefficients (sum of all variables) if not provided
#     if objective_coeffs is None:
#         objective_coeffs = np.ones(n_dims)
#     elif len(objective_coeffs) != n_dims:
#         raise ValueError("Objective coefficients must match number of dimensions")

#     # Check if objective matches any constraint (up to scale)
#     for i in range(n_constraints):
#         if np.allclose(A[i, :], objective_coeffs, rtol=1e-5, atol=1e-8):
#             raise ValueError(f"Objective matches constraint {i} coefficients.")

#     # Initial LP to find baseline optimal value (optional, for reference)
#     x = cp.Variable(n_dims)
#     objective = cp.Maximize(objective_coeffs @ x)
#     constraints = [A @ x + b >= 0]  # Constraints as -Ax >= -b
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status != "optimal":
#         print("Initial problem is infeasible or unbounded. Exiting.")
#         return matrix

#     original_value = problem.value
#     print(f"Original optimal value: {original_value}")

#     # Check each constraint for redundancy
#     redundant_indices = []
#     for i in range(n_constraints):
#         # Test if constraint i is active by maximizing a_i^T x subject to other constraints
#         x_test = cp.Variable(n_dims)
#         mask = np.ones(n_constraints, dtype=bool)
#         mask[i] = False
#         A_reduced = A[mask, :]
#         b_reduced = b[mask]
#         # Maximize a_i^T x (equivalent to minimizing -a_i^T x in -Ax >= -b form)
#         objective_test = cp.Maximize(A[i, :] @ x_test)
#         constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
#         problem_test = cp.Problem(objective_test, constraints_reduced)
#         problem_test.solve()

#         if problem_test.status == "optimal":
#             new_value = problem_test.value
#             tolerance = 1e-8
#             print(f"\nTesting constraint {i} (e.g., {matrix[i]}):")
#             print(f'Comparing new_value={new_value} con old_value={b[i]}, diff={abs(new_value - b[i])}')
#             # print(f"Max a_{i}^T x: {new_value}, b_{i}: {b[i]}")
#             if b[i] < 0:
#                 if new_value < b[i]:
#                     redundant_indices.append(i)
#                     # print(f"Constraint {i} is redundant (max a_{i}^T x < b_{i} + tolerance)")
#                     print(f'  Red AA - {new_value} < {b[i]}')
#                 else:
#                     # print(f"Constraint {i} is non-redundant (max a_{i}^T x >= b_{i} + tolerance)")
#                     print(f'  BB - {new_value} < {b[i]}')
#             else:
#                 if new_value > b[i]:
#                     redundant_indices.append(i)
#                     # print(f"Constraint {i} is redundant (max a_{i}^T x < b_{i} + tolerance)")
#                     print(f'  Red AAA - {new_value} < {b[i]}')
#                 else:
#                     # print(f"Constraint {i} is non-redundant (max a_{i}^T x >= b_{i} + tolerance)")
#                     print(f'  BBB - {new_value} < {b[i]}')
#         else:
#             print(f"\nConstraint {i} (e.g., {matrix[i]}) is essential (infeasible without it)")
#             # If reduced problem is infeasible, the constraint is non-redundant
#             continue

#     # Remove redundant constraints
#     if redundant_indices:
#         print(f"\nRemoving redundant constraints at indices: {redundant_indices}")
#         keep_indices = [i for i in range(n_constraints) if i not in redundant_indices]
#         return matrix[keep_indices, :]
#     else:
#         print("\nNo redundant constraints found.")
#         return matrix
    
# def check_redundancy22(matrix, objective_coeffs=None):
#     # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
#     n_constraints, n_dims_plus_one = matrix.shape
#     n_dims = n_dims_plus_one - 1
#     b = matrix[:, 0]  # Negate b for Ax <= b to -Ax >= -b
#     A = matrix[:, 1:]  # Negate A for Ax <= b to -Ax >= -b

#     # Set default objective coefficients (sum of all variables) if not provided
#     if objective_coeffs is None:
#         objective_coeffs = np.ones(n_dims)
#     elif len(objective_coeffs) != n_dims:
#         raise ValueError("Objective coefficients must match number of dimensions")

#     # # Check if objective matches any constraint (up to scale)
#     # for i in range(n_constraints):
#     #     if np.allclose(A[i, :], objective_coeffs, rtol=1e-5, atol=1e-8):
#     #         raise ValueError(f"Objective matches constraint {i} coefficients.")

#     # Initial LP to find baseline optimal value
#     x = cp.Variable(n_dims)
#     objective = cp.Maximize(objective_coeffs @ x)
#     constraints = [A @ x + b >= 0]  # Constraints as -Ax >= -b
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status != "optimal":
#         print("Initial problem is infeasible or unbounded. Exiting.")
#         return matrix

#     original_value = problem.value
#     # original_x = x.value

#     # Track active constraints using indices
#     active_indices = list(range(n_constraints))
 
#     removable_indices = []

#     for i in range(n_constraints):
#         # Test removing constraint i
#         x_test = cp.Variable(n_dims)
#         mask = np.ones(n_constraints, dtype=bool)
#         mask[i] = False
        
#         # active_indices_reduced = [idx for idx, keep in zip(active_indices, mask) if keep]
#         A_reduced = A[mask, :]
#         b_reduced = b[mask]
        
#         constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
        
#         problem_reduced = cp.Problem(cp.Maximize(objective_coeffs @ x_test), 
#                                      constraints_reduced)
        
#         problem_reduced.solve()

#         if problem_reduced.status == "optimal":
#             reduced_value = problem_reduced.value
#             tolerance_value = 1e-6

#             # Feasibility check: Does original_x satisfy the reduced constraints?
#             # feasibility = A_reduced @ original_x + b_reduced
#             # is_feasible = np.all(feasibility <= tolerance_value)

#             print(f"\nTesting removal of constraint {i} (e.g., {matrix[i]}):")
#             print(f"Original value: {original_value}, Reduced value: {reduced_value}")
#             print(f"Value difference: {abs(original_value - reduced_value)}")
#             # print(f"Feasibility of original_x: {is_feasible}")

#             # if abs(original_value - reduced_value) < tolerance_value and is_feasible:
#             if abs(original_value - reduced_value) < tolerance_value:
#                 removable_indices.append(i)
#                 print(f"Constraint {i} marked for removal")
#         else:
#             print(f"\nConstraint {i} (e.g., {matrix[i]}) is essential (infeasible without it)")

#     # Apply removals if any
#     if removable_indices:
#         # active_indices = [idx for idx in active_indices if idx not in removable_indices]
#         keep_indices = [i for i in range(n_constraints) if i not in removable_indices]
#         return matrix[keep_indices, :]
#     else:
#         return matrix




# def check_redundancy2(matrix, objective_coeffs=None):
#     # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
#     n_constraints, n_dims_plus_one = matrix.shape
#     n_dims = n_dims_plus_one - 1
#     b = matrix[:, 0]
#     A = matrix[:, 1:]

#     # Set default objective coefficients (sum of all variables) if not provided
#     if objective_coeffs is None:
#         objective_coeffs = np.ones(n_dims)
#     elif len(objective_coeffs) != n_dims:
#         raise ValueError("Objective coefficients must match number of dimensions")

#     # Check if objective matches any constraint (up to scale and offset)
#     for i in range(n_constraints):
#         constraint_coeffs = A[i, :]
#         if np.allclose(constraint_coeffs, objective_coeffs, rtol=1e-5, atol=1e-8):
#             raise ValueError(f"Objective {objective_coeffs} matches constraint {i} coefficients {constraint_coeffs}. Choose a different objective.")

#     # Initial LP to find baseline optimal value
#     x = cp.Variable(n_dims)
#     objective = cp.Maximize(objective_coeffs @ x)
#     constraints = [A @ x + b >= 0]
#     problem = cp.Problem(objective, constraints)
#     problem.solve()

#     if problem.status != "optimal":
#         print("Initial problem is infeasible or unbounded. Exiting.")
#         return matrix

#     original_value = problem.value
#     original_x = x.value
#     # print(f"Original optimal value (z = {objective_coeffs @ x}): {original_value}")
#     # print(f"Original solution (x): {original_x}")

#     # Test removing each constraint
#     minimal_matrix = matrix.copy()
#     while True:
#         A_current = minimal_matrix[:, 1:]
#         b_current = minimal_matrix[:, 0]
#         x_current = cp.Variable(n_dims)
#         objective_current = cp.Maximize(objective_coeffs @ x_current)
#         constraints_current = [A_current @ x_current + b_current >= 0]
#         problem_current = cp.Problem(objective_current, constraints_current)
#         problem_current.solve()

#         if problem_current.status != "optimal":
#             break

#         original_value_current = problem_current.value
#         original_x_current = x_current.value
#         # print(f"\nCurrent optimal value: {original_value_current}")
#         # print(f"Current solution (x): {original_x_current}")

#         removable_indices = []
#         for i in range(minimal_matrix.shape[0]):
#             x_test = cp.Variable(n_dims)
#             mask = np.ones(minimal_matrix.shape[0], dtype=bool)
#             mask[i] = False
#             A_reduced = minimal_matrix[mask, 1:]
#             b_reduced = minimal_matrix[mask, 0]
#             constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
#             problem_reduced = cp.Problem(cp.Maximize(objective_coeffs @ x_test), constraints_reduced)
#             problem_reduced.solve()

#             if problem_reduced.status == "optimal":
#                 reduced_value = problem_reduced.value
#                 reduced_x = x_test.value
#                 tolerance_value = 1e-6
#                 tolerance_distance = 1e-6

#                 # Feasibility check: Does original_x_current satisfy the reduced constraints?
#                 feasibility = A_reduced @ original_x_current + b_reduced
#                 is_feasible = np.all(feasibility >= -tolerance_value)

#                 # Distance check: How much did the optimal point shift?
#                 distance = np.linalg.norm(original_x_current - reduced_x)

#                 print(f"\nTesting removal of constraint {i} (e.g., {matrix[i]}):")
#                 print(f"Original value: {original_value_current}, Reduced value: {reduced_value}")
#                 print(f"Value difference: {abs(original_value_current - reduced_value)}")
#                 print(f"Feasibility of original_x_current: {is_feasible}")
#                 print(f"Distance between solutions: {distance}")

#                 if (abs(original_value_current - reduced_value) < tolerance_value and 
#                     is_feasible and 
#                     distance < tolerance_distance):
#                     removable_indices.append(i)
#                     print(f"Constraint {i} marked for removal")
#             else:
#                 print(f"\nConstraint {i} (e.g., {matrix[i]}) is essential (infeasible without it)")

#         # Apply removals if any
#         if removable_indices:
#             # Sort indices in descending order to avoid index shift issues
#             removable_indices.sort(reverse=True)
#             for i in removable_indices:
#                 minimal_matrix = np.delete(minimal_matrix, i, axis=0)
#             print(f"Removed constraints at indices: {removable_indices}")
#         else:
#             break

#     return minimal_matrix

# # from tensorflow.keras.utils import to_categorical


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


# '''
# Obtemos la matriz ampliada con el bias incorporado
# '''
# def get_transposed_ext(w, bias=None):
#     if not isinstance(bias, np.ndarray):
#         bias = np.zeros(w.shape[0])
      
#     w_ext = np.hstack((bias.reshape(-1,1), w)).T

#     zeros = np.zeros(w.shape[1] + 1)
#     zeros[0] = 1 

#     zeros = zeros.reshape(-1,1)
    
#     return np.hstack((zeros, w_ext)).T
    
    


# def get_I_linear(x, w): 
#     # print('Aquí')
#     Sw = w @ x
#     I = np.eye(w.shape[0])
    
#     return I, Sw, np.ones(w.shape[0])


# # def get_I_relu(x, w, alpha=0.3):
    
# #     Sw = w @ x
        
# #     I_v = Sw > 0
# #     I_v = I_v * 1
    
# #     if alpha > 0:
# #         mul_neg = Sw <= 0
# #         mul_neg = mul_neg * alpha  
# #         I_v = I_v + mul_neg 

# #     return I_v


# def get_I_relu(x, w, alpha=0.3):
    
#     Sw = w @ x
        
#     mul_pos = Sw > 0
#     mul_pos = mul_pos * 1
    
#     I_v = mul_pos
    
#     if alpha > 0:
#         mul_neg = Sw <= 0
#         mul_neg = mul_neg * alpha  
#         I_v = mul_pos + mul_neg 

#     I = np.diag(I_v)
    
#     H = Sw * I_v
    
#     return I, H, I_v
    


# def get_I_activation(linear_layer, activation_layer, x, w):
#     if isinstance(linear_layer, torch.nn.modules.linear.Linear):
#         if activation_layer is not None: 
#             if isinstance(activation_layer, torch.nn.modules.activation.ReLU):
#                 return get_I_relu(x, w, alpha=0)
#             elif isinstance(activation_layer, torch.nn.modules.activation.LeakyReLU):
#                 return get_I_relu(x, w, activation_layer.negative_slope)
#             else:
#                 raise ValueError('Error 1')
#         else:
#             return get_I_linear(x, w)
#     else:
#         raise ValueError('Error 2')
        
  
# def run_layers_old(layer_input, model):
#     H_list = []
#     W_list = []
#     I_v_list = []
#     pseudo_I_list = []    

#     x_sample = np.hstack((1, layer_input))
    
#     l = 0
#     nlayers = len(model.layers)

#     while True:
#         if isinstance(model.layers[l], torch.nn.modules.linear.Linear):
#             linear_layer = model.layers[l]
            
#             activation_layer = None 
            
#             if l != nlayers - 1:
#                 activation_layer = model.layers[l + 1]
#                 l += 1 
        
#             w = linear_layer.weight.detach().numpy()
#             bias = linear_layer.bias.detach().numpy()
            
#             w_T_ext = get_transposed_ext(w, bias)
            
#             W_list.append(w_T_ext)
            
#             pseudo_I, H, I_v = get_I_activation(linear_layer, activation_layer, x_sample, w_T_ext)
            
#             H_list.append(H)
            
#             I_v_list.append(I_v)
#             pseudo_I_list.append(pseudo_I)
            
#             x_sample = H
        
#         l += 1
   
#         if l == nlayers:
#             break
        
#     return W_list, pseudo_I_list, I_v_list, H_list

    
# def run_layers(layer_input, model, return_weighted=True):
#     H_list = []
#     W_list = []
#     I_v_list = []
#     pseudo_I_list = []    
#     O_list = []
    
#     x_sample = np.hstack((1, layer_input))
    
#     l = 0
#     nlayers = len(model.layers)

#     while True:
#         if isinstance(model.layers[l], torch.nn.modules.linear.Linear):
#             linear_layer = model.layers[l]
            
#             activation_layer = None 
            
#             if l != nlayers - 1:
#                 activation_layer = model.layers[l + 1]
#                 l += 1 
        
#             w = linear_layer.weight.detach().numpy()
#             bias = linear_layer.bias.detach().numpy()
            
#             w_T_ext = get_transposed_ext(w, bias)
#             W_list.append(w_T_ext)
            
#             if return_weighted:
#                 O = w_T_ext * x_sample
#                 O_list.append(O[1:])
            
#             pseudo_I, H, I_v = get_I_activation(linear_layer, activation_layer, x_sample, w_T_ext)
            
#             H_list.append(H)
            
#             I_v_list.append(I_v)
#             pseudo_I_list.append(pseudo_I)
            
#             x_sample = H
        
#         l += 1
   
#         if l == nlayers:
#             break
        
#     return W_list, pseudo_I_list, I_v_list, H_list, O_list
    



# def get_face_contrib_accelerated(x_sample, model, return_weighted=True, return_lists=True):
    

#     if isinstance(x_sample, (pd.core.series.Series)):
#         x_sample = x_sample.to_numpy()
#     elif isinstance(x_sample, (torch.Tensor)):
#         x_sample = x_sample.numpy()
    
#     W_list, I_list, I_v_list, H_list, O_list = run_layers(x_sample, model, 
#                                                           return_weighted=return_weighted)

#     '''
#     Antiguo cálculo de contribuciones 
#     '''    
#     # contrib_partial = I_v_list[len(I_list)-1][:, None] * W_list[len(I_list)-1] 
#     # # if return_weighted:
#     # #     contrib_partial[:,1:] = contrib_partial[:,1:] * x_sample
#     # contrib_list.insert(0, contrib_partial)
    
#     # for I_index in range(len(I_list)-2, -1, -1):
#     #     contrib_partial = I_v_list[I_index][:, None] * W_list[I_index]
#     #     # if return_weighted:
#     #     #     contrib_partial[:,1:] = contrib_partial[:,1:] * x_sample
#     #     contrib_list.insert(0, contrib_partial)

#     # contrib = contrib_list[-1]
#     # for I_index in range(len(I_list)-2, -1, -1):
#     #     contrib = contrib @ contrib_list[I_index]
            
#     contrib_list = []
    
#     contrib_partial = W_list[0]
#     contrib_list.append(contrib_partial)
    
#     for I_index in range(len(I_list) - 1):
#         contrib_partial = (W_list[I_index+1] * I_v_list[I_index][None, :]) @ contrib_partial
#         # contrib_partial = W_list[I_index+1] @ I_list[I_index] @ contrib_partial
#         contrib_list.append(contrib_partial)
        
#     contrib = contrib_partial
    
#     if return_weighted:
#         contrib[:,1:] = contrib[:,1:] * x_sample
        
#     if not return_lists:
#         return contrib[1:]
#     else:
#         return contrib[1:], W_list, I_v_list, contrib_list, H_list, O_list, I_list
    



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
    # num_outputs = model.layers[-2].out_features
    # num_outputs = 2
    
    for sample, x in enumerate(X):
        contrib, W_list, I_vecs, _, H_list, _, _ = get_face_contrib_accelerated(x, model)
            
        config = ''
        for vec in I_vecs:
            for v in vec[1:]:
                config = config + str(int(v))
                
        if not include_last:
            config = config[:-num_outputs]
            
        # if config.startswith('10') or config.startswith('01') or config.startswith('11'):
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

        # print(f'Row = {row}')

        # row[0] = -row[0]
        
        # row[1:] = -row[1:]
        
        # if -row[1] < 0:
        #     sign = '<='
        #     # row[1:] = -row[1:]
        # else:
        #     sign = '>='

        
        # row = row / abs(row[1])
        
        sign = '>='
        
        # print(f'Row antes = {row}')
        if row[1] != 0:
            if row[1] < 0:
                sign = '<='
            row = row / row[1]
        elif row[2] != 0:
            if row[2] < 0:
                sign = '<='
            row = row / row[2]
            
        # print(f'Row después = {row}')

        b = -row[0]
        A = row[1:]

        # b = row[0]
        # A = -row[1:]
        
        # a = -row[1:]
        # b = row[0]
        # a = -row[1:]

        # print(f'[b -A] = {b}, {a}')
        

        
        terms = []
        letters = 'xyz'
        
        for i, A_i in enumerate(A):
            val = eval(f'{A_i:.{decimals}f}')
            # print(f"Val={val}, {A_i:.{2}f}")
            if len(terms) > 0:
                if A_i < 0:
                    # print("11-0")
                    if val == -1:
                        # print("11")
                        terms.append(f" - {letters[i]}")
                    else:
                        terms.append(f" - {abs(A_i):.{decimals}f}{letters[i]}")
                else:
                    # print("11-2")
                    if val == 1:
                        # print("12")
                        terms.append(f" + {letters[i]}")
                    elif val == 0:
                        continue
                    else:
                        terms.append(f" + {abs(A_i):.{decimals}f}{letters[i]}")
            else:
                if i == 0:
                    # print(f"ACÁ ai={ai}, val={val}")
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
                        # print("11-3")
                        if val == -1:
                            # print("13")
                            terms.append(f"-{letters[i]}")
                        else:
                            terms.append(f"{A_i:.{decimals}f}{letters[i]}")
                    else:
                        # print("11-4")
                        if val == 1:
                            # print("14")
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
    
    # print(f'Row antes = {row}')
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
        
    # print(f'Row después = {row}')

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

    # face = FACEExplainerTorch(model)    
    # contrib, W_list, I_vecs, contrib_list, H_list, O_list, I_list = face.explain(
    #                                                                 X_train[sample].numpy()) 

    # print(f'STARTING Getting Eqs from config={config}')
    '''
    Cálculo con las contribuciones
    '''
    
    w0 = contrib_list[0][1:]
    
    for i in range(len(contrib_list) - 2):
        w0 = np.concatenate((w0, contrib_list[i + 1][1:]))
    # w0 = np.concatenate((contrib_list[0][1:], contrib_list[1][1:]))
    

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

    # print(f'1st GENERATOR BOUNDED={len(generator_bounded.array)}')
    
    # from pprint import pprint 
    
    # pprint(generator_bounded.array)
    
    # import sys
    # if len(generator_bounded.array) == 0:
    #     sys.exit(0)
        
    generator_bounded_list_class0, generator_bounded_list_class1 = [], []
    H_inequalities_class0, H_inequalities_class1 = [], []

    
    all_zeros_winning_class = -1
    
    '''
    Añadidmos Y1 >= Y2 ... si existe 
    '''
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    # contrib_y1_y2[0] = -contrib_y1_y2[0]
    
    # sign = np.sum(contrib, axis=1)
    sign = H_list[-1][1:]
    sign = 1 if sign[0] > sign[1] else -1
    # sign = -1 if sign[0] > sign[1] else 1
        
    # print(f'contrib=\n{contrib}, sign={sign}\n')
    
    sign = 1
    
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        # half_matrix_class0 = np.vstack((new_matrix, contrib[0].reshape(1,-1)))
        half_matrix_class0 = np.vstack((new_matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class0 = cdd.matrix_from_array(half_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class0 = cdd.polyhedron_from_matrix(H_bounded_half_class0)
        generator_bounded_class0 = cdd.copy_generators(H_poly_bounded_half_class0)

        # half_matrix_class1 = np.vstack((new_matrix, -contrib[0].reshape(1,-1)))
        half_matrix_class1 = np.vstack((new_matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class1 = cdd.matrix_from_array(half_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class1 = cdd.polyhedron_from_matrix(H_bounded_half_class1)
        generator_bounded_class1 = cdd.copy_generators(H_poly_bounded_half_class1)
        
        # half_unbounded_matrix_class0 = np.vstack((matrix, contrib[0].reshape(1,-1)))
        half_unbounded_matrix_class0 = np.vstack((matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class0 = cdd.matrix_from_array(half_unbounded_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class0 = cdd.polyhedron_from_matrix(H_unbounded_half_class0)
        generator_unbounded_class0 = cdd.copy_generators(H_poly_unbounded_half_class0)

        # half_unbounded_matrix_class1 = np.vstack((matrix, -contrib[0].reshape(1,-1)))
        half_unbounded_matrix_class1 = np.vstack((matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class1 = cdd.matrix_from_array(half_unbounded_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class1 = cdd.polyhedron_from_matrix(H_unbounded_half_class1)
        generator_unbounded_class1 = cdd.copy_generators(H_poly_unbounded_half_class1)
        
        # if len(generator_unbounded_class0.array) > 0:
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
                    # print('EXCEPTION CLASS0')
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
                    # print('EXCEPTION CLASS1')
                    H_inequalities_class1 = []
                #     H_inequalities_class1 = H_inequalities_class0
    
                # if ex_class0:
                #     H_inequalities_class0 = H_inequalities_class1
                    
        generator_bounded_list_class0 = generator_bounded_class0.array
        generator_bounded_list_class1 = generator_bounded_class1.array
    else: 
        all_zeros_winning_class = 0 if contrib_y1_y2[0] >= 0 else 1
        # print("SETTING ALL_ZEROS")

    
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
        # H_bounded = cdd.matrix_from_array(new_matrix, rep_type=cdd.RepType.INEQUALITY)
        
        H_poly_unbounded = cdd.polyhedron_from_matrix(H_bounded)
        generator_bounded = cdd.copy_generators(H_poly_bounded)

        # print(f'2nd GENERATOR BOUNDED={len(generator_bounded.array)}')
        # pprint(generator_bounded.array)
        
        V_poly_bounded = cdd.polyhedron_from_matrix(generator_bounded) 
        H_inequalities = cdd.copy_inequalities(V_poly_bounded)


    # inequalities_global = get_inequalities(H_inequalities, decimals)
    # inequalities_class0 = get_inequalities(H_inequalities_class0, decimals)
    # inequalities_class1 = get_inequalities(H_inequalities_class1, decimals)



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

        
            # s = sp.Eq(x + y + 222, 0)
            # s = s.subs(x, v[1] * x)
            # s = s.subs(y, v[2] * y)
            
            s = s.subs(2222, v[0])
            
        # eq_list.append((f'Eq{i+1}', s, sign_vec[i][0], generator))
        
        # print(f'Adding eq [{s}]')
        expr = s.lhs
        coeff_x = abs(expr.coeff(x_list[0]))
        
        if coeff_x == 0:
            coeff_x = abs(expr.coeff(x_list[1]))
            
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
        
        
        # print(f'Adding new_equation [{new_equation}]\n')
        
        eq_list.append((f'Eq{i+1}', new_equation, sign_vec[i][0], generator_bounded, sign_vec[i][0]))
        


    '''
    Añadimos Y1 e Y2
    '''
    for i, v in enumerate(contrib):
        if np.allclose(v[1:], 0.):
        #     # s = sp.Eq(x - 2, 0)
            # print(f'\nADDING DUMMIE EQUATION 3.1\n')
            continue
        #     # print(f'\nADDING DUMMIE EQUATION 1 v[0]={v[0]}\n')
        #     # eq_list.append((f'Y_{i+1} = {v[0]}', None, -2, generator_bounded))
        #     pass
        else:
            # print(f'\nADDING DUMMIE EQUATION 3.2\n')
            # s = sp.Eq(x + y + 222, 0)
            # s = s.subs(222, v[0])
            # s = s.subs(x, v[1] * x)
            # s = s.subs(y, v[2] * y)
        
            lhs = sum(x_list) + 2222
            
            s = sp.Eq(lhs, 0)
            
            for i in range(len(x_list)):
                s = s.subs(x_list[j], v[j+1] * x_list[j])
 
            # s = sp.Eq(x + y + 222, 0)
            # s = s.subs(x, v[1] * x)
            # s = s.subs(y, v[2] * y)
            
            s = s.subs(2222, v[0])


            # expr = s.lhs
            # coeff_x = abs(expr.coeff(x_list[0]))
            # new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
            # print(f'AAY_{i+1} = {s}')
            # eq_list.append((f'Y_{i+1}', new_equation, 0, generator_bounded, 0))

            

            
    '''
    Añadimos Y1 >= Y2
    '''
    # print(f'Contrib\n{contrib[0] - contrib[1]}\n')
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    ret_y1_y2 = None
    
    if not np.allclose(contrib_y1_y2[1:], 0.):
        # print('ADDING NEW DUMMIE 2')
    # if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        ret_y1_y2 = contrib_y1_y2
        
        lhs = sum(x_list) + 2222
            
        s = sp.Eq(lhs, 0)
            
        for i in range(len(x_list)):
            s = s.subs(x_list[i], contrib_y1_y2[i+1] * x_list[i])
         
        s = s.subs(2222, contrib_y1_y2[0])

        # s = sp.Eq(x + y + 222, 0)
        
        # s = s.subs(222, contrib_y1_y2[0])
        # s = s.subs(x, contrib_y1_y2[1] * x)
        # s = s.subs(y, contrib_y1_y2[2] * y)
     
        # eq_list.append(('Y_1 >= Y_2', s, 0, generator))
        
        expr = s.lhs
        coeff_x = abs(expr.coeff(x_list[0]))
        
        if coeff_x == 0:
            coeff_x = abs(expr.coeff(x_list[1]))
        
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
        sign = np.sum(contrib, axis=1)
        
        sign = 1 if sign[0] > sign[1] else -1
        
        sign2 = 1 if H_list[-1][1] > H_list[-1][2] else -1

        # '''
        # Marcamos el signo de la flecha de clase apuntando a la Clase 0 (red)
        # Para apuntar a la clase contraria (la verde), debemos usar sign2 = -1
        # '''        
        # sign2 = -1
        
        # eq_list.append(('Y_r >= Y_g', new_equation, 0, generator_bounded, sign2))

        eq_list.append(('Eq9', new_equation, 0, generator_bounded, sign2))


        # eq_list.append(('Y_1 >= Y_2', new_equation, 0, generator_bounded, 1 if H_list[-1][1] > H_list[-1][2] else -1))
        
        # print(f'Adding H={H_list[-1]}, contrib_y1_y2={contrib_y1_y2} \n  new_equation [{s} --> {new_equation}]\n')
        
    # minimal_matrix = cvx_get_minimal_matrix_incremental(new_matrix)
    # minimal_matrix = cvx_get_minimal_matrix(new_matrix)
    # minimal_matrix = check_redundancy(new_matrix)
     
    # print(f'check_redundancy={minimal_matrix}')


    # minimal_matrix = reduce_system(new_matrix)
    
    # print(f'reduced_system={minimal_matrix}')
        
    
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
    # w0 = np.concatenate((contrib_list[0][1:], contrib_list[1][1:]))
    

    pass

    '''
    Para el caso de 1 en config, el vector se mantiene [b A] para expresar Ax + b >= 0
    Para el caso de 0 en config, cambiamos a [-b -A] para expresar Ax + b <= 0 --> -Ax - b >=0
    '''
    
    sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
    sign_vec[sign_vec == 0] = -1
    
    # if np.allclose(sign_vec, [[1],[-1],[1],[-1]]):
    #     sign_vec[0] = -1
    #     sign_vec[1] = 1
    #     sign_vec[2] = 1
    #     sign_vec[3] = -1
        
    
    # for i, s in enumerate(sign_pos):
    #     sign_vec[2+s[0]] = s[1]
        
    matrix = w0 * sign_vec
    
    # matrix = matrix[2:]
    
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
    
    # contrib_y1_y2[0] = -contrib_y1_y2[0]
    
    # sign = np.sum(contrib, axis=1)
    sign = H_list[-1][1:]
    sign = 1 if sign[0] > sign[1] else -1
        
    # print(f'contrib=\n{contrib}, sign={sign}\n')
    
    sign = 1
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        # half_matrix_class0 = np.vstack((new_matrix, contrib[0].reshape(1,-1)))
        half_matrix_class0 = np.vstack((new_matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class0 = cdd.matrix_from_array(half_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class0 = cdd.polyhedron_from_matrix(H_bounded_half_class0)
        generator_bounded_class0 = cdd.copy_generators(H_poly_bounded_half_class0)

        # half_matrix_class1 = np.vstack((new_matrix, -contrib[0].reshape(1,-1)))
        half_matrix_class1 = np.vstack((new_matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_bounded_half_class1 = cdd.matrix_from_array(half_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_bounded_half_class1 = cdd.polyhedron_from_matrix(H_bounded_half_class1)
        generator_bounded_class1 = cdd.copy_generators(H_poly_bounded_half_class1)
        
        # half_unbounded_matrix_class0 = np.vstack((matrix, contrib[0].reshape(1,-1)))
        half_unbounded_matrix_class0 = np.vstack((matrix, sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class0 = cdd.matrix_from_array(half_unbounded_matrix_class0, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class0 = cdd.polyhedron_from_matrix(H_unbounded_half_class0)
        generator_unbounded_class0 = cdd.copy_generators(H_poly_unbounded_half_class0)

        # half_unbounded_matrix_class1 = np.vstack((matrix, -contrib[0].reshape(1,-1)))
        half_unbounded_matrix_class1 = np.vstack((matrix, -sign * contrib_y1_y2.reshape(1,-1)))
        H_unbounded_half_class1 = cdd.matrix_from_array(half_unbounded_matrix_class1, rep_type=cdd.RepType.INEQUALITY)
        H_poly_unbounded_half_class1 = cdd.polyhedron_from_matrix(H_unbounded_half_class1)
        generator_unbounded_class1 = cdd.copy_generators(H_poly_unbounded_half_class1)
        
        # if len(generator_unbounded_class0.array) > 0:
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
                    # print('EXCEPTION CLASS0')
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
                    # print('EXCEPTION CLASS1')
                    H_inequalities_class1 = []
                #     H_inequalities_class1 = H_inequalities_class0
    
                # if ex_class0:
                #     H_inequalities_class0 = H_inequalities_class1
                    
        generator_bounded_list_class0 = generator_bounded_class0.array
        generator_bounded_list_class1 = generator_bounded_class1.array
    else: 
        all_zeros_winning_class = 0 if contrib_y1_y2[0] >= 0 else 1
        # print("SETTING ALL_ZEROS")

    
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


    # inequalities_global = get_inequalities(H_inequalities, decimals)
    # inequalities_class0 = get_inequalities(H_inequalities_class0, decimals)
    # inequalities_class1 = get_inequalities(H_inequalities_class1, decimals)



    eq_list = []
    
    x = Symbol('x')
    y = Symbol('y')
    
    for i, v in enumerate(w0):
        if v[1] == 0 and v[2] == 0:
            s = sp.Eq(x - 2, 0)
            # print(f'\nADDING DUMMIE EQUATION 2\n')
            continue
        else:
            s = sp.Eq(x + y + 222, 0)
            s = s.subs(x, v[1] * x)
            s = s.subs(y, v[2] * y)
            
            # # if i == 0 or i == 3:
            # if i == 3:
            #     s = s.subs(222, -v[0])
            # else:
            #     s = s.subs(222, v[0])
            s = s.subs(222, v[0])
        
        # eq_list.append((f'Eq{i+1}', s, sign_vec[i][0], generator))
        
        # print(f'Adding eq [{s}]')
        expr = s.lhs
        coeff_x = abs(expr.coeff(x))
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
        # print(f'Adding new_equation [{new_equation}]\n')
        
        eq_list.append((f'Eq{i+1}', new_equation, sign_vec[i][0], generator_bounded, sign_vec[i][0]))
        


    '''
    Añadimos Y1 e Y2
    '''
    for i, v in enumerate(contrib):
        if v[1] == 0 and v[2] == 0:
            # print(f'\nADDING DUMMIE EQUATION 3.1\n')
            continue

            # print(f'\nADDING DUMMIE EQUATION 3 v[0]={v[0]}\n')
            # # eq_list.append((f'Y_{i+1} = {v[0]}', None, -2, generator_bounded))
            # pass
        else:
            # print(f'\nADDING DUMMIE EQUATION 3.2\n')
            s = sp.Eq(x + y + 222, 0)
            s = s.subs(222, v[0])
            s = s.subs(x, v[1] * x)
            s = s.subs(y, v[2] * y)
        
            # eq_list.append((f'Y_{i+1}', s, -2, generator_bounded, 0))

            expr = s.lhs
            coeff_x = abs(expr.coeff(x))
            new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
            # eq_list.append((f'Y_{i+1}', new_equation, 0, generator_bounded, 0))

            
    '''
    Añadimos Y1 >= Y2
    '''
    # print(f'Contrib\n{contrib[0] - contrib[1]}\n')
    contrib_y1_y2 = contrib[0] - contrib[1]
    
    ret_y1_y2 = None
    
    if contrib_y1_y2[1] != 0 or contrib_y1_y2[2] != 0:
        print('ADDING DUMMIE 2')
        ret_y1_y2 = contrib_y1_y2
        s = sp.Eq(x + y + 222, 0)
        s = s.subs(222, contrib_y1_y2[0])
        s = s.subs(x, contrib_y1_y2[1] * x)
        s = s.subs(y, contrib_y1_y2[2] * y)
     
        # eq_list.append(('Y_1 >= Y_2', s, 0, generator))
        
        expr = s.lhs
        coeff_x = abs(expr.coeff(x))
        new_equation = sp.Eq(s.lhs / coeff_x, s.rhs / coeff_x)
            
        sign = np.sum(contrib, axis=1)
        
        sign = 1 if sign[0] > sign[1] else -1
        
        sign2 = 1 if H_list[-1][1] > H_list[-1][2] else -1
        
        # print(f'sign={sign}, sign2={sign2}')
        
        eq_list.append(('Y_1 >= Y_2', new_equation, 0, generator_bounded, sign2))
        # eq_list.append(('Y_1 >= Y_2', new_equation, 0, generator_bounded, 1 if H_list[-1][1] > H_list[-1][2] else -1))
        
        # print(f'Adding H={H_list[-1]}, contrib_y1_y2={contrib_y1_y2} \n  new_equation [{s} --> {new_equation}]\n')
        
        
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
        elif self.linestyle in ['dashdot']:
            pen.setStyle(QtCore.Qt.DashDotLine)
        else:
            pen.setStyle(QtCore.Qt.SolidLine)
            
        painter.setPen(pen)
        # Draw the line in the vertical center of the widget
        painter.drawLine(0, 10, 50, 10)
        
        
        
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
                 test_color_count=None):
        
        super().__init__()
        
        self.finish = False
        self.index = 0
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

        # Set colors: green = correctly predicted, red = otherwise
        # self.y_colors = np.where(y_train == mlp_train, 'green', 'red')

        # Set colors: green = class 0, else red
        self.y_colors = np.where(y_train, 'green', 'red')
        self.y_colors_light = np.where(y_train, 'green', 'red')
        # self.y_colors_light = np.where(y_train, 'palegreen', 'lightcoral')
        

        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.model = model
        
        # self.root = root
        # self.root.title('Interactive Activation Pattern plot')
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
        
                            # [3.,  1.,  0.],
                            # [3., -1.,  0.],
                            # [3.,  0.,  1.],
                            # [3.,  0, -1.]])

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
        
        # Set window close protocol
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # self.subscript_map = '₀₁₂₃₄₅₆₇₈₉'
        self.subscript_map = '0123456789'
        
        self.replot(ini=True)
        self.update_plot()
        self.replot()
        
        
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
        
        # # # Configure root window to be resizable
        # self.root.rowconfigure(0, weight=1)
        # self.root.columnconfigure(0, weight=1)
        
        # # Create the main container with grid weights
        # self.main_frame = ttk.Frame(self.root, padding="2")
        # self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # # Configure main_frame grid weights
        # self.main_frame.rowconfigure(0, weight=0)  # Keep top row fixed
        # self.main_frame.rowconfigure(1, weight=1)  # Allow bottom row to expand
        # self.main_frame.columnconfigure(0, weight=0)  # Plot column - fixed
        # self.main_frame.columnconfigure(1, weight=0)  # Checkbuttons column - fixed
        # self.main_frame.columnconfigure(2, weight=1)  # Text widget column - expandable
        
        # # Create matplotlib figure (fixed size)
        # self.fig = Figure(figsize=(5.5, 5.5), dpi=100)
        # self.ax = self.fig.add_subplot(111)
        
        # --- COLUMN 1: Matplotlib Figure (Left) ---
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.top_container.addWidget(self.canvas, stretch=2)
        
        # Connect Matplotlib Events
        # self.canvas.mpl_connect('scroll_event', self.on_scroll)
        # self.canvas.mpl_connect('button_press_event', self.on_press)
        # self.canvas.mpl_connect('button_release_event', self.on_release)
        # self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        # self.canvas.mpl_connect('key_press_event', self.on_key)
        # self.canvas.mpl_connect('resize_event', self.on_resize)
        
        # # Create canvas (fixed size)
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        # self.canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky='nw')
        
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
        # self.check_scroll
        self.middle_column_layout.addWidget(self.check_scroll, stretch=0)
        
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
        
        self.btn_back.setFixedSize(45, 25)
        self.btn_forward.setFixedSize(45, 25)
        
        # 3. Build the layout: Stretch - Button - Button - Stretch
        # nav_layout.addStretch(1)  # Pushes from the left
        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_forward)
        # nav_layout.addStretch(1)  # Pushes from the right
        
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
        
        # # Create checkbutton frame (fixed size)
        # self.check_frame = ttk.Frame(self.main_frame, padding="5")
        # self.check_frame.grid(row=0, column=1, sticky='nw', padx=2)
        
        # # Create text frame (expandable)
        # self.text_frame = ttk.Frame(self.main_frame, padding="2")
        # self.text_frame.grid(row=0, column=2, sticky=(tk.N, tk.E, tk.W))
        # self.text_frame.rowconfigure(0, weight=1)
        # self.text_frame.columnconfigure(0, weight=1)
        
        # # Add text widget (expandable)
        # self.text_widget = tk.Text(self.text_frame, width=45, height=33.6, padx=5, pady=5, state="disabled")
        # self.text_widget.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S))
        
        # # Add scrollbar for text
        # self.scrollbar = ttk.Scrollbar(self.text_frame, orient='vertical', 
        #                               command=self.text_widget.yview)
        # self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        # self.text_widget['yscrollcommand'] = self.scrollbar.set
        
        # '''
        # Bottom frame with two panes
        # '''        
        # # Create bottom frame (expandable)
        # self.bottom_frame = ttk.Frame(self.main_frame, padding="2")
        
        # self.bottom_frame.grid(row=1, column=0, columnspan=3, 
        #               sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 2))  # Added N to stick to top
        # self.bottom_frame.columnconfigure(0, weight=1)
        # self.bottom_frame.rowconfigure(0, weight=1)  # Allow vertical expansion
        

        # text_area_height = 12 
        
        # # Create a paned window to contain the two text areas
        # self.paned_window = tk.PanedWindow(self.bottom_frame, orient=tk.HORIZONTAL)
        # self.paned_window.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # # Left text area
        # self.left_text = tk.Text(self.paned_window, height=text_area_height, padx=5, pady=5,
        #                          wrap=tk.WORD, state="disabled")
        # self.paned_window.add(self.left_text, stretch="always", width=180)
        
        # # Middle text area
        # self.middle_text = tk.Text(self.paned_window, height=text_area_height, padx=5, pady=5, 
        #                           wrap=tk.WORD, state="disabled")
        # self.paned_window.add(self.middle_text, stretch="always", width=200)


        # # Right text area
        # self.right_text = tk.Text(self.paned_window, height=text_area_height, padx=5, pady=5, 
        #                           wrap=tk.WORD, state="disabled")
        # self.paned_window.add(self.right_text, stretch="always", width=180)
        
        # # Create a single scrollbar that will control both text widgets
        # self.scrollbar = ttk.Scrollbar(self.bottom_frame, orient='vertical')
        # self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))        
        
        
        
        # self.scrollbar['command'] = self._on_scrollbar
        # self.left_text['yscrollcommand'] = self._left_scroll_set
        # self.right_text['yscrollcommand'] = self._right_scroll_set

        # self.buttons_frame = ttk.Frame(self.main_frame, padding="2")
        # # self.buttons_frame = ttk.Frame(self.check_frame, padding="2")
        # self.buttons_frame.grid(row=0, column=1, sticky='w', pady=(10, 0))  
        
        # # Style for blue buttons
        # button_style = {
        #                     'bg': '#4499FF',  # Light blue color
        #                     'fg': 'white', 
        #                     'width': 7,
        #                     'height': 0,
        #                     'font': ('Arial', 12, 'bold'),
        #                     'pady': 0,
        #                     'padx': 0
        #                 }
        
        # # Forward button (pointing right)
        # self.forward_button = tk.Button(self.buttons_frame, text="\u25B6", **button_style)
        # self.forward_button.grid(row=0, column=0, pady=(0, 2))
        
        # # Backward button (pointing left)
        # self.backward_button = tk.Button(self.buttons_frame, text="\u25C0", **button_style)
        # self.backward_button.grid(row=1, column=0)
        
        # self.forward_button.configure(command=lambda: self.on_button_click("forward"))
        # self.backward_button.configure(command=lambda: self.on_button_click("backward"))
                
        # Initialize variable dictionary for checkbuttons
        self.var_dict = {}
        
        # Add scatter points control
        # self.scatter_var = tk.BooleanVar(value=True)
        # self.scatter_checkbutton = ttk.Checkbutton(
        #     self.check_frame,
        #     text="Samples",
        #     variable=self.scatter_var,
        #     command=self.update_plot,
        #     state='disabled'  # Initially disabled until points are added
        # )
        # self.scatter_checkbutton.grid(row=0, column=0, sticky=tk.W)
        
        # self.button_callback = None
        # self.forward_button.configure(command=lambda: self.button_callback and self.button_callback("forward"))
        # self.backward_button.configure(command=lambda: self.button_callback and self.button_callback("backward"))

        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # self.set_button_callback(self.handle_action)  
        
        # self.root.update()

        # # Get the initial window size and set it as minimum
        # initial_width = self.root.winfo_width()
        # initial_height = self.root.winfo_height()
        # self.root.minsize(initial_width, initial_height)
        
    
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
        
        
    # def clear_equations(self):
    #     """Clear all equations and reset the plot"""
    #     self.equations.clear()
    #     self.var_dict.clear()
    #     self.arrows_config.clear()
    #     self.contour_sets.clear()
    #     self.poly.clear()
        

        
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
        
    #     self.update_plot()
        

    
    # def update_bottom_text(self, text):
    #     """Update the bottom text display with new content."""
    #     self.bottom_text.delete('1.0', tk.END)
    #     self.bottom_text.insert('1.0', text)
    
    def str_equ(self, eq, decimals=2, return_full=False, orig_sign=-2, normalize=True):
        
        eq = eq.copy()
        
        eq[abs(eq) < 1e-10] = 0
        
        sign = 1
        
        # print(f'Row antes = {row}')
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
            
        # print(f'Row después = {row}')

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
                    ret += f'&nbsp;- {abs(eq[0]):.{decimals}f}'
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
        # subscript_map = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
        
        # subscript_map = '₀₁₂₃₄₅₆₇₈₉'

        if pos is None:
            # Get the number of lines in each text widget
            left_lines = int(self.left_text.index('end-1c').split('.')[0])
            # right_lines = int(self.right_text.index('end-1c').split('.')[0])
            
            # Calculate visible lines (approximate based on height)
            visible_lines = self.left_text.winfo_height() // 19  # Approximate line height
            
            # print(f'Visible_lines={visible_lines}, left={left_lines}, right={right_lines}')
            text_zone = None
            # if left_lines < visible_lines or left_lines <= right_lines:
            if left_lines <= visible_lines:
                # If left column has space or has fewer lines than right, add to left
                text_zone = self.left_text
                # self.left_text.insert(tk.END, text + "\n")
            else:
                # Otherwise add to right column
                text_zone = self.right_text
                # self.right_text.insert(tk.END, text + "\n")
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

                # self.bottom_text.tag_configure('bold', font=('Arial', 11, 'bold'))
                # self.bottom_text.insert(tk.END, new_text + '\n', 'bold')
            else:
                text_zone.tag_configure('normalX', font=('Arial', 11))
                text_zone.insert(tk.END, new_text + '\n', 'normalX')

                # self.bottom_text.tag_configure('normal', font=('Arial', 11))
                # self.bottom_text.insert(tk.END, new_text + '\n', 'normal')
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
                        # new_text += f' x{subscript_map[pos+1]}'
                    else:
                        new_text += t

            text_zone.tag_configure('normalX', font=('Arial', 11))
            text_zone.insert(tk.END, new_text + '\n', 'normalX')
        
            # self.bottom_text.tag_configure('normal', font=('Arial', 11))
            # self.bottom_text.insert(tk.END, new_text + '\n', 'normal')
        
        self.left_text.config(state="disabled")
        self.middle_text.config(state="disabled")
        self.right_text.config(state="disabled")
        
    
    # def update_text(self, text):
    #     """Update the text display with new content.
        
    #     Args:
    #         text (str): Text to display
    #     """
    #     self.text_widget.delete('1.0', tk.END)
    #     self.text_widget.insert('1.0', text)
    
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
            # self.text_widget.insert(tk.END, new_text + '\n', 'bold')

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
                

            # self.text_widget.config(state="normal")
            # self.text_widget.insert(tk.END, text + '\n')
            self.text_widget.config(state="disabled")
        else:
            self.text_widget.config(state="normal")
            self.text_widget.insert(tk.END, text + '\n')
            self.text_widget.config(state="disabled")    
        
        
    # def set_scatter_points(self, points, colors):
    #     """Add scatter points to the plot.
        
    #     Args:
    #         points: numpy array or list of shape (n, 2) containing x, y coordinates
    #     """
    #     self.scatter_points = np.array(points)
    #     self.scatter_colors = colors
    #     # self.scatter_checkbutton.configure(state='normal')
    #     self.update_plot()
        
        
    def set_scatter_points(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points = np.array(list(points))
        self.scatter_colors = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        self.scatter_checkbox.setEnabled(True)
        self.update_plot()

        
    def set_scatter_points_light(self, points, colors):
        """Add scatter points to the plot and enable the UI control."""
        
        self.scatter_points_light = np.array(list(points))
        self.scatter_colors_light = colors
        
        # In PyQt, we use setEnabled(True/False) instead of state='normal'/'disabled'
        # I am enabling it here so the user can toggle the points once they exist
        self.scatter_checkbox.setEnabled(True)
        self.update_plot()
            
        
    # def set_scatter_points_light(self, points, colors):
    #     """Add scatter points to the plot.
        
    #     Args:
    #         points: numpy array or list of shape (n, 2) containing x, y coordinates
    #     """
    #     self.scatter_points_light = np.array(points)
    #     self.scatter_colors_light = colors
    #     # self.scatter_checkbutton.configure(state='normal')
    #     self.update_plot()
        

    # def clear_scatter_points(self):
    #     """Remove scatter points from the plot."""
    #     self.scatter_points = None
    #     self.scatter_colors = None
    #     self.scatter_var.set(False)
    #     # self.scatter_checkbutton.configure(state='disabled')
    #     self.update_plot()
        
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
        
        
        # # Create new checkbutton variable
        # var = tk.BooleanVar(value=True)
        var = BooleanVarReplica(value=True)
        self.var_dict[name] = var
        
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

        # # # Draw the colored line
        # # if name.star('Y1'):
        # #     linestyle = '-.'
        # if linestyle == 'dash' or linestyle == 'dashdot':
        #     line_canvas.create_line(0, 5, 25, 5, fill=color, width=3, dash=(4, 2, 4, 2))
        # elif linestyle == 'dashed':
        #     line_canvas.create_line(0, 5, 25, 5, fill=color, width=3, dash=(5, 1))
        # else:
        #     line_canvas.create_line(0, 5, 25, 5, fill=color, width=3)
        # # Update the plot
        
        # self.update_plot()
    
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

        # if vertices.array[-1] and vertices.array[-1][0] == 1:        
        #     # Add first point at the end to close the polygon
        #     x_coords.append(x_coords[0])
        #     y_coords.append(y_coords[0])

        # print(f'Adding coords \nx_coords=[{x_coords}] \ny_coords=[{y_coords}]')
 
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
            # centroid = np.mean(vertices, axis=0)
            
            # Expand vertices outward from centroid
            # scale_factor = 3  # Adjust this value to control expansion
            # expanded_vertices = centroid + scale_factor * (vertices - centroid)
            
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
        
        # Set plot ranges
        # x_range = (-3, 3)
        # y_range = (-3, 3)
        
        if self.show_graph:
            x_range = self.x_range
            y_range = self.y_range
            
            # x_range = (-1, 1)
            # y_range = (-1, 1)
    
            # Ensure equal scaling
            self.ax.set_aspect('equal')
            self.ax.set_xlim(x_range)
            self.ax.set_ylim(y_range)
            
            # self.ax.axhline(y=0, color='black', linestyle='-.', linewidth=0.3)
            # self.ax.axvline(x=0, color='black', linestyle='-.', linewidth=0.3)
            
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
                    
                    # if eq_name.startswith('Y_1 >'):
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
                    
                    # # direction = 1 if reg_mid > bx else -1
                    # direction = 1

                    # dx = 0.3 * direction
                    # # y_pos = -0.03 if i % 2 == 0 else -0.07
                    
                    # # y_pos = -0.03 if ranks[i] % 2 == 0 else -0.07
                    # y_pos = -0.03
                    
                    # self.ax.arrow(bx, y_pos, dx, 0, 
                    #              head_width=0.02, head_length=0.1,
                    #              fc=color, ec=color, 
                    #              length_includes_head=True, width=0.01, zorder=10)

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
            # if self.scatter_points_light is not None and self.scatter_var.get():
                self.ax.scatter(self.scatter_points_light[:, 0], self.scatter_points_light[:, 1], 
                                c=self.scatter_colors_light, s=self.light_point_size, 
                                zorder=0, alpha=self.light_point_alpha)

            if self.scatter_points is not None:
            # if self.scatter_points is not None and self.scatter_var.get():
                self.ax.scatter(self.scatter_points[:, 0], self.scatter_points[:, 1], 
                                c=self.scatter_colors, s=self.point_size, zorder=5)
        
            self.ax.set_title(self.title, pad=7, fontsize=10)
            
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
            # print(f'Comparing boundary {boundary} con {eq}')
            # print(f'Result = {np.allclose(eq, boundary)}')
            if np.allclose(eq, boundary):
                return True
        return False
    
    def replot(self, ini=False):        
        config = list(dict(self.configs).keys())[self.index]
      
        self.right_text.clear()
        self.middle_text.clear()
        self.left_text.clear()
        self.info_panel.clear()

        # self.right_text.config(state="normal")
        # self.right_text.delete('1.0', tk.END)
        # self.right_text.config(state="disabled")

        # self.middle_text.config(state="normal")
        # self.middle_text.delete('1.0', tk.END)
        # self.middle_text.config(state="disabled")
        
        # self.left_text.config(state="normal")
        # self.left_text.delete('1.0', tk.END)
        # self.left_text.config(state="disabled")
        
        # self.text_widget.config(state="normal")
        # self.text_widget.delete('1.0', tk.END)
        # self.text_widget.config(state="disabled")
        
        # self.bottom_text.delete('1.0', tk.END)
        # self.text_widget.delete('1.0', tk.END)
        
        # self.title = f'Activation Pattern {self.get_config_structure(self.model, config)} - Samples = {self.configs[config]}/{len(self.X_train)}'
        self.title = f'TRAIN SAMPLES: Activation Pattern {self.get_config_structure(self.model, config)}'
        
        # --- Build one single HTML string for the sidebar ---
        html = []
        
        # 1. Centered Header
        html.append("<div style='font-family: monospace; text-align: center; font-size: 10pt; margin-bottom: 10px;'>")
        html.append("<b>TOY EXPERIMENT ON 3×3 CHECKERBOARD<br>CLASSIFICATION DATASET</b>")
        html.append("</div>")
        
        # 2. Left-aligned Statistics
        html.append("<div style='text-align: left; font-family: monospace; font-size: 9pt;'>")
        # html.append("<div style='text-align: left; font-family: \"Cascadia Mono\"; font-size: 9pt;'>")
        # html.append("<div style='text-align: left; font-family: sans-serif; font-size: 10pt;'>")
        html.append(f"# Neurons = {self.config_struct}<br>")
        html.append(f"# Activation Patterns = {len(self.configs)}<br>")
        html.append(f"# Samples = {len(self.X_train)} (train)&nbsp;&nbsp;{len(self.X_test)} (test)<br>")
        html.append(f"# Epochs = {self.epochs}<br>")
        html.append(f"Learning Rate = {self.lr}<br>")
        html.append(f"Train Accuracy&nbsp;=&nbsp;{self.train_accuracy:.04f}<br>")
        html.append(f"Test Accuracy&nbsp;&nbsp;=&nbsp;{self.test_accuracy:.04f}<br>")
        html.append(f"Seed = {self.seed}<br>")
        html.append("</div>")

        
        # self.append_text('TOY EXPERIMENT ON 3×3 CHECKERBOARD', bold=True, center=True)
        # self.append_text('CLASSIFICATION DATASET\n', bold=True, center=True)
        # self.append_text(f'# Neurons = {self.config_struct}')
        # self.append_text(f'# Activation Patterns = {len(self.configs)}')
        # self.append_text(f'# Samples = {len(self.X_train)} (train) {len(self.X_test)} (test)')
        # self.append_text(f'# Epochs = {self.epochs}')
        # self.append_text(f'Learning Rate = {self.lr}')
        # self.append_text(f'Train Accuracy = {self.train_accuracy:.04f}')
        # self.append_text(f'Test Accuracy  = {self.test_accuracy:.04f}')
        # self.append_text(f'Seed = {self.seed}')
    

        # 3. Helper for perfectly aligned tables
        def get_table_html_old(title_text, config_dict, rmse_dict, active_c):
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
                # t_html += f"<td align='right'>{rmse_dict[c]:.05f}</td>"
                t_html += "</tr>"
            t_html += "</table><br>"
            return t_html

        def get_table_html(title_text, config_dict, color_dict, config):
            # We wrap the title in a span with a larger font size (e.g., 12pt)
            t_html = f"<br><span style='font-size: 10pt;font-family: monospace; '><b>{title_text}</b></span>"
            
            # The table itself remains standard size for precision
            t_html += "<table width='100%' style='border-collapse: collapse; font-family: monospace; font-size: 10pt;'>"
            # t_html += "<table width='100%' style='border-collapse: collapse; font-family: \"Cascadia Mono\"; font-size: 9pt;'>"
            t_html += "<tr><th width='70' align='left'>Pattern</th><th width='65' align='right'>#Samples</th><th width='45' align='right'>TP</th><th width='40' align='right'>FP</th><th width='40' align='right'>TN</th><th width='40' align='right'>FN</th></tr>"
            
            for c, count in config_dict.items():
                # bg = "background-color: #FFFF22; font-weight: bold;" if c == active_c else ""
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


        # html.append(get_table_html("Train samples", self.configs, self.train_rmse_dict, config))
        # html.append(get_table_html("Test samples", self.test_configs, self.test_rmse_dict, config))

        html.append(get_table_html("Train samples", self.configs, self.color_count, config))
        html.append(get_table_html("Test samples", self.test_configs, self.test_color_count, config))
        
        # --- Set the HTML once ---
        self.info_panel.setHtml("".join(html))


        # self.append_text('\nTrain samples', bold=True)
        
        # len_first = max(len('Pattern   '), len(config) + 2)
        # # self.append_text(f'{"Pattern   ":<{len_first}} #Samples  {"#TP":>4}  {"#FP":>4}  {"#TN":>4}  {"#FN":>4}')
        # self.append_text(f'{"Pattern   ":<{len_first}} #Samples  {"TP":>4}  {"FP":>4}  {"TN":>4}  {"FN":>4}')

                    
        # for c in self.configs:
        #     conv_c = self.get_config_structure(self.model, c)
            
        #     # conv_c += ','
            
        #     len_first = max(len('Pattern   '), len(conv_c) + 3)
            
        #     if c == config:
        #         self.append_text(f'{conv_c:<{len_first}} {self.configs[c]:>6d}  {self.color_count[c]["TP"]:>4d}  {self.color_count[c]["FP"]:>4d}  {self.color_count[c]["TN"]:>4d}  {self.color_count[c]["FN"]:>4d}', bold=True, reverse=True)
        #     else:
        #         self.append_text(f'{conv_c:<{len_first}} {self.configs[c]:>6d}  {self.color_count[c]["TP"]:>4d}  {self.color_count[c]["FP"]:>4d}  {self.color_count[c]["TN"]:>4d}  {self.color_count[c]["FN"]:>4d}')


        # self.append_text('\nTest samples', bold=True)
        
        # len_first = max(len('Pattern   '), len(config) + 2  )
        # # self.append_text('\nActivation Pattern, [#total, #red, #green]')
        # # self.append_text(f'{"Pattern   ":<{len_first}} #Samples  {"#TP":>4}  {"#FP":>4}  {"#TN":>4}  {"#FN":>4}')
        # self.append_text(f'{"Pattern   ":<{len_first}} #Samples  {"TP":>4}  {"FP":>4}  {"TN":>4}  {"FN":>4}')

                    
        # for c in self.test_configs:
        #     conv_c = self.get_config_structure(self.model, c)
            
        #     # conv_c += ','
            
        #     len_first = max(len('Pattern   '), len(conv_c) + 3)
            
        #     if c == config:
        #         # self.append_text(f'{conv_c}, {self.configs[c]:>{k_configs}d}, {self.color_count[c]["count_r"]:>{k_reds}d}r, {self.color_count[c]["count_g"]:>{k_greens}d}g')
        #         self.append_text(f'{conv_c:<{len_first}} {self.test_configs[c]:>6d}  {self.test_color_count[c]["TP"]:>4d}  {self.test_color_count[c]["FP"]:>4d}  {self.test_color_count[c]["TN"]:>4d}  {self.test_color_count[c]["FN"]:>4d}', bold=True, reverse=True)
        #     else:
        #         self.append_text(f'{conv_c:<{len_first}} {self.test_configs[c]:>6d}  {self.test_color_count[c]["TP"]:>4d}  {self.test_color_count[c]["FP"]:>4d}  {self.test_color_count[c]["TN"]:>4d}  {self.test_color_count[c]["FN"]:>4d}')
        
        
        
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
        # left_html = "<div style='font-family: \"Cascadia Mono\"; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
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

        # out_text = f'&nbsp;&nbsp;Output Y_r >= Y_g: {self.str_equ(eq, decimals=self.decimals)} {">= 0" if signs[index] > 0 else "<= 0"}'
        out_text = f'&nbsp;&nbsp;Y_r >= Y_g, HS9: {self.str_equ(eq, decimals=self.decimals)} {">= 0" if signs[index] > 0 else "<= 0"}'
        
        left_html += out_text
        
        left_html = left_html.replace('<=', '&le;').replace('>=', '&ge;')

        left_html += '</div>'

        # Set the left panel
        self.left_text.setHtml(left_html)
        
        
        # '''
        # Show Output Y1 >= Y2
        # '''
        # if isinstance(output_class_eq, np.ndarray):
        #     eq = output_class_eq
        #     self.append_bottom_text(f'  Output Y_r >= Y_g: {self.str_equ(eq, decimals=self.decimals)} {">= 0" if signs[index] > 0 else "<= 0"}', no_trad=True, bold=False, pos='left')
        
        

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
        # left_html = "<div style='font-family: \"Cascadia Mono\"; font-size: 10pt;'><b>ACTIVATION PATTERN INEQUALITIES</b><br>"
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
        # self.middle_text.setHtml(middle_html)
        self.middle_text.setHtml("".join(middle_html))
        
        right_html = "<div style='font-family: monospace; font-size: 10pt;'><b>RULE CONSEQUENT (Network Output)</b><br>"
        
        # # Bullet 1: Input Range (using &bull; for manual bullet)
        # right_html += f"&nbsp;&nbsp;{inequality[0]:.0{self.decimals}f} <= x <= {inequality[1]:.0{self.decimals}f}<br><br>"
        
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


        # right_html += "<b>&bull; Network Output</b><br>"
        # consequent_str = self.str_equ(contrib, decimals=self.decimals)
        # right_html += f"&nbsp;&nbsp;Output Y = {consequent_str}"
        
        right_html = right_html.replace('<=', '&le;').replace('>=', '&ge;')
        
        right_html += '</div>'
        # Set the right panel
        # self.right_text.setHtml(right_html)
        self.right_text.setHtml("".join(right_html))

        # '''
        # Show Individual Output Equations
        # '''
        # self.append_bottom_text('RULE CONSEQUENT (Network Outputs)', no_trad=True, bold=True, pos='right')

        # # self.append_bottom_text(f'  Network Output in the Activation Region', no_trad=True, bold=True)
        
        # for index, eq in enumerate(output_contrib_eqs):
        #     # self.append_bottom_text(f'    Y{self.subscript_map[index+1]} = {self.str_equ(eq, decimals=self.decimals, normalize=False)}', no_trad=True, bold=False, pos='right')
        #     print(f'SUBUSCRIPT {self.subscript_map[index+1]}')
        #     if self.subscript_map[index+1] == '1':
        #         y_text = 'Y_r '
        #     else:
        #         y_text = 'Y_g'
        #     self.append_bottom_text(f'    {y_text} = {self.str_equ(eq, decimals=self.decimals, normalize=False)}', no_trad=True, bold=False, pos='right')

        # '''
        # Show Inequalities: global and per class
        # '''
        # if len(inequalities.array) > 0:
        #     inequ = np.array(inequalities.array)
        #     self.append_bottom_text('\nACTIVATION REGION (Classes R&G)', no_trad=True, bold=True, pos='right')
        #     # self.append_bottom_text('\nActivation Region', no_trad=True, bold=True)
            
        #     # self.append_bottom_text(' \u2022  Classes R&G', no_trad=True, bold=True, pos='right')
            
        #     if self.show_boundaries:
        #         for index, eq in enumerate(inequ):
        #             if not self.eq_in_boundaries(eq):
        #                 continue
        #             self.append_bottom_text(f'    {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}', no_trad=True, bold=False, pos='right')
        #             # self.append_bottom_text(f'  \u25CB {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}', no_trad=True, bold=False)


        #     for index, eq in enumerate(inequ):
        #         if self.eq_in_boundaries(eq):
        #             continue
        #         self.append_bottom_text(f'    {self.str_equ(eq, decimals=self.decimals, return_full=True, orig_sign=1)}', no_trad=True, bold=False, pos='right')
            

        # self.show_graph = False 
        
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
    # a = 0
    input_list, output_categorical_list, output_int_list = [], [], []
    for i in range(2**input_bits):
        bi = f'{i:0{input_bits}b}'
        input_list.append([float(i) for i in bi])
        
        o = int(bi[0])
        for j in bi[1:]:
            o = o ^ int(j)
            
        output_int_list.append(o)
        
        # output_list.append(F.one_hot(torch.tensor(o), 2))
        output_categorical_list.append(F.one_hot(torch.tensor(o), 2).float().numpy().tolist())
        
    # return torch.tensor(input_list), torch.tensor(output_categorical_list), np.array(output_int_list)
    return torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), torch.tensor(output_categorical_list), np.array(output_int_list)


# def generate_sign(n_inputs, size):
#     vals = np.random.rand(size, 2) - 0.5
    
#     out = (vals[:,0] * vals[:,1] < 0) * 1
#     o_categorical = F.one_hot(torch.tensor(out), 2)
    
#     return torch.from_numpy(vals).float(), o_categorical.float(), out 

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
    # sigma = 0.03                       # desviación típica (tendencia a agruparse en el centro)
    # domain = (0.0, 1.0)                # cuadrado [0,1]^2
    # domain = (0.0, 0.80)                # cuadrado [0,1]^2
    
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
    y = np.concatenate(y_list)  #ALmacena los valores de clase de los datos
    
    # size1 = size // 4
    
    # vals1 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
    #                  np.random.uniform(-0.25, 0.25, size=size1)]).T

    # vals2 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
    #                  np.random.uniform(-0.25, 0.25, size=size1)]).T

    # vals3 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
    #                  np.random.uniform(-0.25, 0.25, size=size1)]).T

    # vals4 = np.array([np.random.uniform(-0.25, 0.25, size=size1), 
    #                  np.random.uniform(-0.25, 0.25, size=size1)]).T

    # vals1 = vals1 + [-0.75, -0.75]
    # vals2 = vals2 + [-0.75, 0.75]
    # vals3 = vals3 + [0.75, -0.75]
    # vals4 = vals4 + [0.75, 0.75]
    
    # vals = np.concatenate([vals1, vals2, vals3, vals4])
    
    # out = (vals[:,0] * vals[:,1] < 0) * 1
    
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
    
    # print(f'Launching experiment {experiment} with 2 hidden layers hidden1 = {hidden1}, hidden2 = {hidden2} and epochs = {epochs}')
    
    # np.seed = random_seed
    # np.random.seed(random_seed)
    # random.seed = random_seed
    # torch.manual_seed(random_seed)

    np.seed = random_seed
    np.random.seed(random_seed)
    random.seed = random_seed
    torch.manual_seed(random_seed)
    random_generator = torch.Generator()
    random_generator.manual_seed(random_seed)
    
    ds_name = experiment
    # model_version = 0
    use_saved_model_weights = True
    # epochs = 30000

    num_inputs = 2
    num_outputs = 2
        
    if experiment == 'xor':
        (X_train, y_train_categorical, y_train) = generate_xor(num_inputs)
        (X_test, y_test_categorical, y_test) = (X_train, y_train_categorical, y_train)
    
        # X_train = torch.tensor(X_train)
        # y_train_categorical = torch.tensor(y_train_categorical)
        # X_test = torch.tensor(X_test)
        # y_test_categorical = torch.tensor(y_test_categorical)
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
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

        # model.add_layer(nn.Linear(last_len, 1))

        # second_layer_weights_before = model.layers[2].weight.data.clone()
        
        # for param in model.layers[0].parameters():
        #     param.requires_grad = False

        # for param in model.layers[4].parameters():
        #     param.requires_grad = False

        if not avoid_training:
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), 
                                          lr=lr)
            train_losses = []
            
            # print("XX Trainable parameters in optimizer:")
            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
            
            # print("\nXX Optimizer parameter groups:")
            # for i, group in enumerate(optimizer.param_groups):
            #     print(f"Group {i}: {len(group['params'])} parameters")
            #     for p in group['params']:
            #         print(f"  Shape: {p.shape}, requires_grad: {p.requires_grad}")
                    
                    
            if fix_first_layer:
                # {w11=1, w12=0, b1= - 0.66} y {w21=0, w22=1, b1= - 0.33}
                
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

                # layer = model.layers[4]
                
                # layer.weight.data = torch.eye(2)
                # # layer.weight.data = torch.Tensor([[0.9999, 0.0001], [0.0001, 0.9999]])
                # layer.bias.data = torch.zeros(2)
                
                # # for param in model.layers[4].parameters():
                # #     param.requires_grad = False
                    
                # nn.init.xavier_uniform_(model.layers[2].weight)
                # nn.init.zeros_(model.layers[2].bias)
                
                # nn.init.xavier_uniform_(model.layers[4].weight)
                # nn.init.zeros_(model.layers[4].bias)

                # model.layers[2].weight.data = torch.randn(2, 2) * 0.5
                # model.layers[2].bias.data = torch.ones(2) * 0.5  # Positive bias
                
                pass
            
            # # for l in [2, 4]:
            # for l in [2]:
            #     nn.init.uniform_(model.layers[l].weight, -1/np.sqrt(2), 1/np.sqrt(2))
            #     # nn.init.uniform_(model.layers[l].weight, -1, 1)
            #     # nn.init.uniform_(model.layers[l].weight)
            #     # nn.init.xavier_uniform_(model.layers[l].weight)
            #     # model.layers[l].weight.data.fill_(1.0)
            #     model.layers[l].bias.data.fill_(0.0)
            #     # model.layers[l].weight.data.fill_(0.0)
                
            #     model.layers[l].weight.data = torch.Tensor([[-1.0, 0.0], [0.0, -1.0]])
            #     model.layers[l].bias.data = torch.Tensor([0.6, 0.4])
                
            #     # model.layers[l].bias.data = torch.Tensor([0.5, 0.5])
                
            # for l in [4]:
            #     model.layers[l].bias.data.fill_(0.0)
            #     model.layers[l].weight.data = torch.Tensor([[0, 0.0], [0.0, 0.0]])
            #     model.layers[l].weight.data.fill_(0.0)    
                
            # initial_w0 = model.layers[0].weight.data.clone()
            # initial_b0 = model.layers[0].bias.data.clone()
            # initial_w2 = model.layers[2].weight.data.clone()
            # initial_b2 = model.layers[2].bias.data.clone()
            # initial_w4 = model.layers[4].weight.data.clone()
            # initial_b4 = model.layers[4].bias.data.clone()
            
            # # print('XXXX')
            # print("Initial layer 0 weights:", initial_w0)
            # print("Initial layer 0 bias:", initial_b0)
            # print("Initial layer 2 weights:", initial_w2)
            # print("Initial layer 2 bias:", initial_b2)
            # print("Initial layer 4 weights:", initial_w4)
            # print("Initial layer 4 bias:", initial_b4)
            
            # Train for just a few epochs
            # for epoch in range(10):
            #     model.train()
            #     optimizer.zero_grad()
            #     outputs = model(X_train)
            #     targets = torch.tensor(y_train, dtype=torch.long)
            #     loss = criterion(outputs, targets)
            #     loss.backward()
                
            #     # Check gradients
            #     if epoch == 0:
            #         print(f"\nEpoch 0 gradients:")
            #         print(f"Layer 0 weight grad: {model.layers[0].weight.grad}")
            #         print(f"Layer 0 bias grad: {model.layers[0].bias.grad}")
            #         print(f"Layer 2 weight grad: {model.layers[2].weight.grad}")
            #         print(f"Layer 2 bias grad: {model.layers[2].bias.grad}")
            #         print(f"Layer 4 weight grad: {model.layers[4].weight.grad}")
            #         print(f"Layer 4 bias grad: {model.layers[4].bias.grad}")
                
            #     optimizer.step()
            
            # print("\nAfter 10 epochs:")
            # print("Layer 0 weight change:", (model.layers[0].weight.data - initial_w0).abs().sum().item())
            # print("Layer 0 bias change:", (model.layers[0].bias.data - initial_b0).abs().sum().item())
            # print("Layer 2 weight change:", (model.layers[2].weight.data - initial_w2).abs().sum().item())
            # print("Layer 2 bias change:", (model.layers[2].bias.data - initial_b2).abs().sum().item())
            # print("Layer 4 weight change:", (model.layers[4].weight.data - initial_w4).abs().sum().item())
            # print("Layer 4 bias change:", (model.layers[4].bias.data - initial_b4).abs().sum().item())
            
            # # print("\nFinal layer 0 weights:", model.layers[0].weight.data)
            # # print("Final layer 0 bias:", model.layers[0].bias.data)
            # # print("\nFinal layer 2 weights:", model.layers[2].weight.data)
            # # print("Final layer 2 bias:", model.layers[2].bias.data)
            
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
                
                # targets = F.one_hot(torch.tensor(y_train), 2).float()
                targets = torch.tensor(y_train, dtype=torch.long)
                # targets = torch.tensor(y_train)
                
                loss = criterion(outputs, targets)

                
                # _, predicted = torch.max(outputs.data, dim=1)
                
                # _, new_target = torch.max(targets.data, dim=1)

                # correct += (predicted == new_target).sum().item()

                correct += (outputs.argmax(dim=1) == targets).sum().item()
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                epoch_accuracy = 100 * correct / len(targets)
                
                train_losses.append(epoch_loss)

                # Check gradients
                # if epoch % 10 == 0:
                #     print(f"\nEpoch {epoch} gradients:")
                #     print(f"Layer 0 weight grad: {model.layers[0].weight.grad}")
                #     print(f"Layer 0 bias grad: {model.layers[0].bias.grad}")
                #     print(f"Layer 2 weight grad: {model.layers[2].weight.grad}")
                #     print(f"Layer 2 bias grad: {model.layers[2].bias.grad}")
                #     print(f"Layer 4 weight grad: {model.layers[4].weight.grad}")
                #     print(f"Layer 4 bias grad: {model.layers[4].bias.grad}")
                #     print()
                
                # print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
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
            # print("Trainable parameters in optimizer:")
            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
                
            
            # print("\nFinal layer 0 weights:", model.layers[0].weight.data)
            # print("Final layer 0 bias:", model.layers[0].bias.data)
            # print("\nFinal layer 2 weights:", model.layers[2].weight.data)
            # print("Final layer 2 bias:", model.layers[2].bias.data)
            # print("\nFinal layer 4 weights:", model.layers[4].weight.data)
            # print("Final layer 4 bias:", model.layers[4].bias.data)

            
            # with torch.no_grad():
            #     # Check first layer output distribution
            #     x = model.layers[0](X_train)
            #     x = model.layers[1](x)  # After ReLU
            #     print("First layer output (after ReLU):")
            #     print(f"  Non-zero samples: {(x.abs() > 0).sum().item()} / {x.numel()}")
            #     print(f"  Mean per neuron: {x.mean(dim=0)}")
            #     print(f"  Max per neuron: {x.max(dim=0).values}")
            #     print(f"  Sample outputs:\n{x[:10]}")
                
            #     # Check final outputs
            #     final = model(X_train)
            #     predictions = final.argmax(dim=1)
            #     print(f"\nPrediction distribution: {predictions.unique(return_counts=True)}")
            #     print(f"True label distribution: {torch.tensor(y_train).unique(return_counts=True)}")
                
                
            # with torch.no_grad():
            #     x = X_train
            #     print("YYY Input shape:", x.shape)
                
            #     x = model.layers[0](x)
            #     print("After layer 0 (linear):", x[:3])
                
            #     x = model.layers[1](x)
            #     print("After layer 1 (ReLU):", x[:3])
                
            #     x = model.layers[2](x)
            #     print("After layer 2 (linear):", x[:3])
                
            #     x = model.layers[3](x)
            #     print("After layer 3 (ReLU):", x[:3])
                
            #     # Check if layer 4 exists and is being used
            #     if len(model.layers) > 4:
            #         x = model.layers[4](x)
            #         print("After layer 4 (linear):", x[:3])
                
            #     print("\nFinal output shape:", x.shape)
            #     print("Predictions:", x.argmax(dim=1)[:20])
            #     print("Targets:", y_train[:20])    
                
            for param in model.parameters():
                param.requires_grad_(False)
                

                
            torch.save(model, weights_file_name)
            print('OK')


        # if not avoid_training:
        #     # criterion = nn.CrossEntropyLoss()
        #     criterion = nn.BCEWithLogitsLoss()
        #     optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
        #     train_losses = []
            
        #     for epoch in range(epochs):
        #         model.train()
        #         running_loss = 0.0
        #         # running_accuracy = 0.0
        #         num_batches = 0
        #         correct = 0
        #         total = 0
                
        #         for batch_idx, (inputs, targets) in enumerate(train_loader):
        #             # Forward pass
        #             # inputs = torch.tensor(np.zeros((32, 2)), dtype=torch.float32)
        #             outputs = model(inputs)
        #             loss = criterion(outputs, targets)
                    
        #             #Zero gradients 
        #             optimizer.zero_grad()
                    
        #             # Backward pass and optimization
        #             loss.backward()
        #             optimizer.step()
                    
        #             # Compute metrics
        #             running_loss += loss.item()
        #             _, predicted = torch.max(outputs.data, dim=1)
                    
        #             _, new_target = torch.max(targets.data, dim=1)
                    
        #             total += targets.size(0)
                    
        #             correct += (predicted == new_target).sum().item()
        #             # correct += (predicted == targets).sum().item()
                    
        #             if batch_idx % 200 == 0:
        #                 print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], '
        #                       f'Loss: {loss.item():.4f}')

        #             # num_batches += 1
                    
        #             # if epoch % 1000 == 0: print(f'Epoch {epoch} ... runnnig loss = {running_loss:0.10f}')
                
        #         # Calculate epoch metrics
        #         epoch_loss = running_loss / len(train_loader)
        #         epoch_accuracy = 100 * correct / total
                
        #         train_losses.append(epoch_loss)
        #         print(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
 
                
        #     model.eval()
            
        #     for param in model.parameters():
        #         param.requires_grad_(False)
    
        #     torch.save(model, weights_file_name)
        #     print('OK')

    
    # Define symbolic variables
    x, y = sp.symbols('x y')
    

    '''
    Realizamos predicciones y evaluamos el resultado de la FNN
    '''
    print('Generating predictions for test data ... ', end='')

    mlp_predictions = model(X_test).detach().numpy()
    y_mlp = np.argmax(mlp_predictions, axis=1)

    print('OK')

    # accuracy = np.sum(y_mlp == y_test.numpy()) / len(y_test)
    test_accuracy = np.sum(y_mlp == y_test) / len(y_test)

    print(f'Test data accuracy = {test_accuracy:.5f}\n')

    '''
    Realizamos predicciones y evaluamos el resultado de la FNN
    '''
    print('Generating predictions for train data ... ', end='')

    mlp_predictions = model(X_train).detach().numpy()
    y_train_mlp = np.argmax(mlp_predictions, axis=1)

    print('OK')

    # accuracy = np.sum(y_mlp == y_test.numpy()) / len(y_test)
    train_accuracy = np.sum(y_train_mlp == y_train) / len(y_train)

    print(f'Train data accuracy = {train_accuracy:.5f}\n')


    # mlp_train_predictions = model(X_train).detach().numpy()
    # y_train_mlp = np.argmax(mlp_train_predictions, axis=1)
    
    samples = np.random.choice(range(len(X_test)), 10000, replace=True)

    samples = [i for i in range(len(X_train))]

    t_list = []
    for sample in samples:
        t = time.time()    
        get_face_contrib_accelerated(X_train[sample], model)
        t_list.append(time.time() - t)
               
    print(f'Tiempo por ejecución de FACE: {np.mean(t_list):.06f} +/- {np.std(t_list):.06f} s')                 


    # X_train = torch.cat((X_train, torch.tensor([[-2.75, -2.75]])), dim=0)
    # yy = np.argmax(model(X_train[-1]))
    # y_train = np.concatenate((y_train, [yy]))
    
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
    
    if len(configs) > 0:
        # root = tk.Tk()
        # root.geometry('1100x600')
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
        app = QtWidgets.QApplication(sys.argv)

        plotter = QtImplicitEquationPlotter(configs, config_samples, color_count,
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
                                          test_color_count=test_color_count)

        plotter.show() 
        
        sys.exit(app.exec_())
        
        # while not plotter.finish:
        #     root.mainloop()



# def get_ra_matrix(x, model, full_ra=True) -> dict:
    
#     contrib, _, I_vecs, contrib_list, H_list, _, _ =  \
#         get_face_contrib_accelerated(x, 
#                                      model, 
#                                      return_weighted=False)
        
#     num_outputs = model.layers[-1].out_features
    
#     config = ''
#     for vec in I_vecs:
#         for v in vec[1:]:
#             config = config + str(int(v))

#     config = config[:-num_outputs]
        
#     w0 = contrib_list[0][1:]

#     for i in range(len(contrib_list) - 2):
#         w0 = np.concatenate((w0, contrib_list[i + 1][1:]))

#     sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
#     sign_vec[sign_vec == 0] = -1
    
#     matrix = w0 * sign_vec

#     winner_class = np.argmax(H_list[-1][1:])
    
#     if full_ra:
#         '''
#         Añadidmos Y_winner - Y_looser >= 0  
#         '''
#         for i in range(num_outputs):
#             if i == winner_class:
#                 continue
            
#             # contrib_y1_y2 = contrib[0] - contrib[1]
    
#             contrib_y_winner_y_other = contrib[winner_class] - contrib[i]
            
#             matrix = np.vstack((matrix, contrib_y_winner_y_other.reshape(1,-1)))

#         # contrib, _, I_vecs, contrib_list, H_list, _, _ =  \
#         #     get_face_contrib_accelerated(x, 
#         #                                  model, 
#         #                                  return_weighted=False)
            
#         # num_outputs = model.layers[-1].out_features
        
#         # config = ''
#         # for vec in I_vecs:
#         #     for v in vec[1:]:
#         #         config = config + str(int(v))

#         # config = config[:-num_outputs]      

#     ret = {
#             'matrix': matrix, 
#             'config': config, 
#             'winner_class': winner_class
#           }
    
#     return ret
#     # return matrix, config, winner_class


# def get_new_ra(x, model, walking_tolerance=1e-6):
    
#     orig_ra = get_ra_matrix(x, model)
    
#     original_matrix = orig_ra['matrix']
#     original_config = orig_ra['config']
#     original_class = orig_ra['winner_class']
    
#     # original_matrix, original_config, original_class = get_ra_matrix(x, model)

#     # original_class = np.argmax(model(torch.Tensor(x)))
    
#     disp = analyze_displacement(original_matrix, x)
    
#     new_x = x.copy()
    
#     d = disp['dimension']
#     new_x[d] = x[d] + disp['direction'] * (disp['min_displacement'] + walking_tolerance)

#     new_ra = get_ra_matrix(new_x, model)
    
#     new_matrix = new_ra['matrix']
#     new_config = new_ra['config']
#     new_class = new_ra['winner_class']
    
#     print(f'Original point/class = {x} / {original_class}, config={original_config}, new point/class = {new_x} / {new_class}, config={new_config}')
#     if original_class != new_class:
#         print(f'  CAMBIO DE CLASE {original_class}, config={original_config} --> {new_class}, config={new_config}')
    
    
#     ret = {
#             'new_matrix': new_matrix, 
#             'new_config': new_config, 
#             'new_winner_class': new_class, 
#             'original_matrix': original_matrix, 
#             'original_config': original_config,
#             'original_winner_class': new_class
#           }
    
#     return ret
    
    

if __name__ == "__main__":
    # import sys 
    # print(sys.path())
    # main(experiment='sign', hidden=8, epochs=2000, point_size=2)
    # main(experiment='sign', hidden=16, epochs=30000, point_size=2)
    # main(experiment='xor', hidden=4, epochs=4000, point_size=10)
    # main(experiment='xor', hidden=3, epochs=14000, point_size=10, 
    #      random_seed=33, polygon_color='gainsboro', lr=0.0001)
    
    '''
    Paint damero
    '''
    
    # damero_train_fname = 'damero_train_data.pickle'
    
    # with open(damero_train_fname, 'rb') as f:
    #     (X_train, y_train_categorical, y_train) = pickle.load(f)
        
    # cc = []
    # for f in y_train:
    #     if f == 0:
    #         cc.append('red')
    #     else:
    #         cc.append('green')
    
    # xx = X_train[:, 0]
    # yy = X_train[:, 1]
    
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(xx, yy, s=3, c=cc)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

    # main(experiment='damero', hidden_struct=[6,2], epochs=70000, 
    #      point_size=5, lr=0.001, random_seed=1221, decimals=4, avoid_training=False)

    # main(experiment='damero', hidden_struct=[4,3], epochs=7000, 
    #      point_size=5, lr=0.001, random_seed=123, decimals=4, avoid_training=False)

    # main(experiment='damero', hidden_struct=[5,4], epochs=37000, 
    #      point_size=5, lr=0.0001, random_seed=4112, decimals=4, avoid_training=True)

    # main(experiment='damero', hidden_struct=[5,2], epochs=30000, #25000 y 37000
    #      point_size=5, lr=0.0001, random_seed=3112, decimals=4, avoid_training=False)

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