# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 06:44:39 2025

@author: Carles
"""

import cvxpy as cp
import numpy as np
from typing import Tuple
# from typing import Optional, Union
import torch
from .face_torch import get_face_contrib_accelerated

def reduce_system(matrix):
    # Extract A (coefficients) and b (right-hand side) from matrix [b, a1, a2, ..., an]
    n_constraints, n_dims_plus_one = matrix.shape
    n_dims = n_dims_plus_one - 1
    b = matrix[:, 0]  
    A = matrix[:, 1:]  

    objective_coeffs = np.ones(n_dims)

    # Initial LP to find baseline optimal value (optional, for reference)
    x = cp.Variable(n_dims)
    
    objective = cp.Maximize(objective_coeffs @ x)
    
    constraints = [A @ x + b >= 0]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != "optimal":
        # print("Initial problem is infeasible or unbounded. Exiting.")
        return matrix

    # Check each constraint for redundancy
    redundant_indices = []
    for i in range(n_constraints):
        # Test if constraint i is active by maximizing a_i^T x subject to other constraints
        x_test = cp.Variable(n_dims)
        
        mask = np.ones(n_constraints, dtype=bool)
        mask[i] = False
    
        A_reduced = A[mask, :]
        b_reduced = b[mask]
        
        '''
        Nuestro sistema está descrito como Ax >= -b (o Ax + b >= 0)
        Así Max(A_1x) --> Max(-Ax)
        '''
        objective_test = cp.Maximize(-A[i, :] @ x_test)

        constraints_reduced = [A_reduced @ x_test + b_reduced >= 0]
        
        problem_test = cp.Problem(objective_test, constraints_reduced)
        
        problem_test.solve()

        if problem_test.status == "optimal":
            epsilon = 1e-8 # Posible cálculo δ = ε · max(1, ||a_j|| · ||b_j||)
            threshold = b[i] - epsilon
            
            if problem_test.value <= threshold:
                redundant_indices.append(i)
    
    # Remove redundant constraints
    if redundant_indices:
        keep_indices = [i for i in range(n_constraints) if i not in redundant_indices]
        return matrix[keep_indices, :]
    else:
        return matrix


def get_center(mat_or_A, b=None):
    if b is None:
        '''
        Pasamos la matriz completa, así que la descomponemos
        '''
        A = mat_or_A[:, 1:]
        b = mat_or_A[:, 0]
    else:
        A = mat_or_A
        
    x = cp.Variable(A.shape[1])
    r = cp.Variable()
    
    norms = np.linalg.norm(A, axis=1)
    
    constraints = [A @ x + b >= r * norms]
    
    problem = cp.Problem(cp.Maximize(r), constraints)
    problem.solve()

    if problem.status in ["optimal"]:
        if r.value > 0:
            return x.value, r.value

    return x, -np.inf   


def get_radius_old(point, matrix, b=None, remove_borders=False):
    
    if remove_borders:
        new_matrix = []
        for x in matrix:
            if np.allclose(x[1:], [ 1.,  0.]) or \
               np.allclose(x[1:], [-1.,  0.]) or \
               np.allclose(x[1:], [ 0.,  1.]) or \
               np.allclose(x[1:], [ 0., -1.]):
               continue
            else:
                new_matrix.append(x)
        matrix = np.array(new_matrix)
        
    A = matrix[:, 1:]
    b = matrix[:, 0]
    
    '''
    If we use equations in form Ax <= b --> change to v = b - A @ point
    We are using Ax + b >= 0
    '''
    v = b + A @ point
    l = np.linalg.norm(A, axis=1)
    d = v / l
    
    radius = d.min()
    
    return radius


def get_radius(point, matrix, b=None):
    
    if isinstance(point, torch.Tensor):
        point = point.numpy()
        
    if b is None:
        A = matrix[:, 1:]
        b = matrix[:, 0]
    else:            
        A = matrix
        b = b
        
    # For CDD format: b + A @ point >= 0
    # Distance to constraint i is: (b[i] + A[i] @ point) / ||A[i]||
    v = b + A @ point
    l = np.linalg.norm(A, axis=1)
    d = v / l
    
    radius = d.min()
    
    return radius


def get_radius2(point, matrix, b=None):
    
    if isinstance(point, torch.Tensor):
        point = point.numpy()
        
    if b is None:
        A = matrix[:, 1:]
        b = matrix[:, 0]
    else:            
        A = matrix
        b = b
    
    # Check if point is in the feasible region
    # constraint_violations = -A @ point - b
    # print(f"Constraint violations: max={constraint_violations.max():.6f}, min={constraint_violations.min():.6f}")
    # print(f"Number of violated constraints: {(constraint_violations > 1e-6).sum()}")
    
    # if constraint_violations.max() > 1e-6:
    #     print("WARNING: The point is outside the feasible region!")
    #     return -np.inf
        
    n = 8
    x = cp.Variable(n)
    r = cp.Variable()
    
    constraints = [
                    b + A @ x >= 0,
                    cp.norm(x - point, 2) <= r,
                  ]    
    
    objective = cp.Maximize(r)
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    
    print(f"Problem status: {problem.status}")
    
    radius = -np.inf
    
    if problem.status == "optimal":
        r_optimal = r.value
        x_optimal = x.value
        radius = r_optimal
        print(f"Maximum radius: {r_optimal}")
        print(f"Point achieving this radius: {x_optimal}")
    
    return radius


def get_radius3(point, matrix, b=None):
    
    if isinstance(point, torch.Tensor):
        point = point.numpy()
        
    if b is None:
        A = matrix[:, 1:]
        b = matrix[:, 0]
    else:            
        A = matrix
        b = b
        
    n = 8  # dimension of your problem
    x = cp.Variable(n)
    r = cp.Variable()

    constraints = [
                    A @ x + b >= 0,  # Region π_0 constraints (Z_{π_0}(x) ≤ 0)
                    cp.norm(x - point, 2) <= r,  # Ball constraint
                  ]    
    
    objective = cp.Maximize(r)
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    
    radius = -np.inf
    
    if problem.status == "optimal":
        r_optimal = r.value
        x_optimal = x.value
        
        radius = r_optimal
        
        print(f"Maximum radius: {r_optimal}")
        print(f"Point achieving this radius: {x_optimal}")
    
    return radius
    

def sample_hypersphere(center, radius, n_samples=10):
    """
    Generate a small set of random points uniformly distributed inside a hypersphere.
    
    Parameters:
    -----------
    center : array-like, shape (n_dims,)
        Center point of the hypersphere
    radius : float
        Radius of the hypersphere
    n_samples : int, default=10
        Number of random points to generate (typically small)
        
    Returns:
    --------
    points : ndarray, shape (n_samples, n_dims)
        Random points inside the hypersphere
    """
    center = np.array(center)
    n_dims = len(center)
    
    # Generate random directions and handle zero norms properly
    points = []
    remaining_samples = n_samples
    
    while remaining_samples > 0:
        # Generate enough samples to virtually guarantee we get what we need
        # Even in the astronomically unlikely case of rejections
        batch_size = max(remaining_samples + 10, remaining_samples * 2)
        directions = np.random.standard_normal((batch_size, n_dims))
        
        # Calculate norms and keep only non-zero ones
        norms = np.linalg.norm(directions, axis=1)
        valid_mask = norms > 1e-10
        
        valid_directions = directions[valid_mask]
        valid_norms = norms[valid_mask]
        
        if len(valid_directions) > 0:
            # Take only what we need
            take = min(len(valid_directions), remaining_samples)
            selected_directions = valid_directions[:take]
            selected_norms = valid_norms[:take]
            
            # Normalize to unit directions
            unit_directions = selected_directions / selected_norms.reshape(-1, 1)
            
            # Generate radii for uniform volume distribution
            u = np.random.uniform(0, 1, take)
            
            # Ensure we're strictly inside by using open interval (0,1)
            u = np.where(u == 1.0, 1.0 - 1e-15, u)  # Handle the extremely rare u=1.0 case
 
            radii = radius * np.power(u, 1.0 / n_dims)
            
            # Create points: center + radius * direction
            batch_points = center + radii.reshape(-1, 1) * unit_directions
            points.append(batch_points)
            
            remaining_samples -= take
    
    return np.vstack(points)


def sample_from_sphere(center_point, radius, 
                       num_samples=1, 
                       method='gaussian', 
                       clip_bounds=(0.0, 1.0), 
                       random_seed=None):
    """
    Generate synthetic images by sampling random points from inside a sphere.
    
    Parameters:
    -----------
    center_point : array-like, shape (n_dims,)
        Center of the sphere (your original MNIST image)
    radius : float
        Radius of the sphere (distance to closest hyperplane)
    num_samples : int
        Number of synthetic images to generate
    method : str, one of ['uniform', 'surface', 'gaussian']
        - 'uniform': Uniform distribution inside sphere
        - 'surface': Points on the sphere surface
        - 'gaussian': Gaussian distribution centered at sphere center
    clip_bounds : tuple (min, max) or None
        If provided, clip generated values to stay within bounds
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict containing:
        - samples: array of generated images, shape (num_samples, n_dims)
        - distances: distances from center for each sample
        - valid_samples: boolean mask for samples within clip_bounds
        - clipped_samples: samples after clipping (if clip_bounds provided)
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    center_point = np.array(center_point)
    # n_dims = len(center_point)
    
    if method == 'uniform':
        # Uniform sampling inside n-dimensional sphere
        samples = sample_uniform_sphere(center_point, radius, num_samples)
        
    elif method == 'surface':
        # Sample on sphere surface
        samples = sample_sphere_surface(center_point, radius, num_samples)
        
    elif method == 'gaussian':
        # Gaussian sampling (may go outside sphere)
        samples = sample_gaussian_sphere(center_point, radius, num_samples)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute distances from center
    distances = np.linalg.norm(samples - center_point, axis=1, ord=np.inf)
    
    # Check which samples are within bounds
    valid_samples = np.ones(num_samples, dtype=bool)
    clipped_samples = samples.copy()
    
    if clip_bounds is not None:
        min_bound, max_bound = clip_bounds
        
        # Check validity before clipping
        valid_samples = np.all((samples >= min_bound) & (samples <= max_bound), axis=1)
        
        # Clip samples to bounds
        clipped_samples = np.clip(samples, min_bound, max_bound)
        
        # Compute distances after clipping
        clipped_distances = np.linalg.norm(clipped_samples - center_point, axis=1, ord=np.inf)
        
        # print(f"SAMPLING RESULTS:")
        # print(f"- Samples within bounds: {np.sum(valid_samples)} / {num_samples}")
        # print(f"- Samples outside bounds: {np.sum(~valid_samples)} / {num_samples}")
        # print(f"- Mean distance (original): {np.mean(distances):.6f}")
        # print(f"- Mean distance (clipped): {np.mean(clipped_distances):.6f}")
        # print(f"- Max distance (original): {np.max(distances):.6f}")
        # print(f"- Max distance (clipped): {np.max(clipped_distances):.6f}")
        
        # if np.sum(~valid_samples) > 0:
        #     print(f"⚠️  {np.sum(~valid_samples)} samples were clipped to bounds")
    
    results = {
        'samples': samples,
        'distances': distances,
        'valid_samples': valid_samples,
        'clipped_samples': clipped_samples,
        'method': method,
        'radius': radius,
        'center': center_point
    }
    
    return results

def sample_uniform_sphere_old(center, radius, num_samples):
    """Sample uniformly from inside n-dimensional sphere"""
    n_dims = len(center)
    
    # Generate random direction (unit vector)
    directions = np.random.randn(num_samples, n_dims)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Generate random radii with correct distribution for uniform sampling
    # For L∞ ball (cube), sample each dimension independently
    perturbations = np.random.uniform(-radius, radius, size=(num_samples, n_dims))
    
    return center + perturbations

def sample_uniform_sphere(center, radius, num_samples):
    n_dims = len(center)
    directions = np.random.randn(num_samples, n_dims)
    norms = np.sqrt(np.sum(directions ** 2, axis=1))
    radii = np.random.rand(num_samples) ** (1/n_dims) * radius  # Uniform volume
    perturbations = directions * (radii[:, np.newaxis] / norms[:, np.newaxis])
    return center + perturbations

def sample_uniform_sphere_inf(center, radius, num_samples):
    """Sample uniformly from inside n-dimensional L∞ ball"""
    n_dims = len(center)
    # Sample each dimension independently within [-radius, radius]
    perturbations = np.random.uniform(-radius, radius, size=(num_samples, n_dims))
    return center + perturbations

def sample_sphere_surface_inf(center, radius, num_samples):
    """Sample from L∞ ball surface (faces of hypercube)"""
    n_dims = len(center)
    samples = []
    for _ in range(num_samples):
        # Choose random dimension to be at boundary
        boundary_dim = np.random.randint(0, n_dims)
        boundary_side = np.random.choice([-1, 1])
        
        # Create perturbation with one dimension at ±radius, others within [-radius, radius]
        perturbation = np.random.uniform(-radius, radius, size=n_dims)
        perturbation[boundary_dim] = boundary_side * radius
        samples.append(center + perturbation)
    return np.array(samples)

def sample_gaussian_sphere_inf(center, radius, num_samples):
    """Sample from Gaussian distribution constrained to L∞ ball"""
    n_dims = len(center)
    # Gaussian with std = radius/3, then clip to L∞ ball
    std = radius / 4.0
    perturbations = np.random.normal(0, std, size=(num_samples, n_dims))
    # Ensure all dimensions are within [-radius, radius]
    perturbations = np.clip(perturbations, -radius, radius)
    return center + perturbations

def sample_sphere_surface(center, radius, num_samples):
    """Sample from sphere surface"""
    n_dims = len(center)
    
    # For L∞ ball surface, sample from faces of hypercube
    samples = []
    for _ in range(num_samples):
        # Choose random dimension to be at boundary
        boundary_dim = np.random.randint(0, n_dims)
        boundary_side = np.random.choice([-1, 1])
        
        # Create perturbation
        perturbation = np.random.uniform(-radius, radius, size=n_dims)
        perturbation[boundary_dim] = boundary_side * radius
        
        samples.append(center + perturbation)
    
    return np.array(samples)

def sample_gaussian_sphere(center, radius, num_samples):
    """Sample from Gaussian distribution (may exceed sphere)"""
    n_dims = len(center)
    
    # Gaussian with std = radius/3 (99.7% within radius)
    std = radius / 6.0
    perturbations = np.random.normal(0, std, size=(num_samples, n_dims))
    
    return center + perturbations


# def min_displacement_to_exit(matrix: np.ndarray, x0: np.ndarray, 
#                              tol: float = 1e-10) -> Tuple[float, int, int]:
#     """
#     Calcula el mínimo desplazamiento en una sola dimensión para salir de la región factible.
    
#     La región factible está definida por Ax + b >= 0, donde:
#     - matrix tiene la forma [b A], con b en la primera columna
#     - Un punto es factible si Ax + b >= 0
    
#     Parámetros:
#     -----------
#     matrix : np.ndarray
#         Matriz aumentada [b A] de tamaño (m x n+1)
#     x0 : np.ndarray
#         Punto interior (n,) donde Ax0 + b > 0
#     tol : float
#         Tolerancia para considerar valores como cero
    
#     Retorna:
#     --------
#     min_t : float
#         Mínimo desplazamiento necesario para salir de la región
#     best_j : int
#         Índice de la dimensión donde ocurre el mínimo desplazamiento
#     direction : int
#         Dirección del desplazamiento (+1 o -1)
#     """
#     m, n_plus_1 = matrix.shape
#     n = n_plus_1 - 1
    
#     # Separar b y A de la matriz aumentada
#     b = matrix[:, 0]
#     A = matrix[:, 1:]
    
#     # Verificar que x0 es punto interior: Ax0 + b > 0
#     slack = A @ x0 + b
#     if np.any(slack <= tol):
#         raise ValueError("x0 no es un punto interior")
    
#     min_t = np.inf
#     best_j = -1
#     direction = 0
    
#     # Para cada dimensión j
#     for j in range(n):
#         # Dirección positiva (+e_j)
#         # Al movernos x0 + t*e_j, la restricción i se vuelve: A_i(x0 + t*e_j) + b_i >= 0
#         # Es decir: A_i*x0 + t*A_ij + b_i >= 0
#         # Como slack_i = A_i*x0 + b_i > 0, necesitamos: slack_i + t*A_ij >= 0
#         # Si A_ij < 0, entonces t <= slack_i / |A_ij|
        
#         t_pos = np.inf
#         for i in range(m):
#             if A[i, j] < -tol:  # A_ij < 0 limita movimiento positivo
#                 t_candidate = slack[i] / abs(A[i, j])
#                 t_pos = min(t_pos, t_candidate)
        
#         # Dirección negativa (-e_j)
#         # Al movernos x0 - t*e_j, la restricción i se vuelve: A_i(x0 - t*e_j) + b_i >= 0
#         # Es decir: A_i*x0 - t*A_ij + b_i >= 0
#         # Como slack_i = A_i*x0 + b_i > 0, necesitamos: slack_i - t*A_ij >= 0
#         # Si A_ij > 0, entonces t <= slack_i / A_ij
        
#         t_neg = np.inf
#         for i in range(m):
#             if A[i, j] > tol:  # A_ij > 0 limita movimiento negativo
#                 t_candidate = slack[i] / A[i, j]
#                 t_neg = min(t_neg, t_candidate)
        
#         # Actualizar mínimo global
#         if t_pos < min_t:
#             min_t = t_pos
#             best_j = j
#             direction = 1
        
#         if t_neg < min_t:
#             min_t = t_neg
#             best_j = j
#             direction = -1
    
#     if min_t == np.inf:
#         print("Advertencia: La región no está acotada en ninguna dirección coordenada")
    
#     return min_t, best_j, direction


def min_displacement_to_exit_debug(matrix: np.ndarray, x0: np.ndarray, 
                                   tol: float = 1e-10) -> Tuple[float, int, int, list]:
    """
    Versión con debug que también retorna las restricciones que determinan el mínimo.
    Para la formulación Ax + b >= 0 con matriz [b A].
    """
    m, n_plus_1 = matrix.shape
    n = n_plus_1 - 1
    
    b = matrix[:, 0]
    A = matrix[:, 1:]
    slack = A @ x0 + b
    
    if np.any(slack <= tol):
        raise ValueError("x0 no es un punto interior")
    
    min_t = np.inf
    best_j = -1
    direction = 0
    determining_constraints = []
    
    for j in range(n):
        # Dirección positiva (A_ij < 0 limita)
        candidates_pos = []
        for i in range(m):
            if A[i, j] < -tol:
                t_candidate = slack[i] / abs(A[i, j])
                candidates_pos.append((t_candidate, i))
        
        if candidates_pos:
            t_pos, constraint_pos = min(candidates_pos)
            if t_pos < min_t:
                min_t = t_pos
                best_j = j
                direction = 1
                determining_constraints = [c for t, c in candidates_pos 
                                          if abs(t - t_pos) < max(tol, tol * t_pos)]
        
        # Dirección negativa (A_ij > 0 limita)
        candidates_neg = []
        for i in range(m):
            if A[i, j] > tol:
                t_candidate = slack[i] / A[i, j]
                candidates_neg.append((t_candidate, i))
        
        if candidates_neg:
            t_neg, constraint_neg = min(candidates_neg)
            if t_neg < min_t:
                min_t = t_neg
                best_j = j
                direction = -1
                determining_constraints = [c for t, c in candidates_neg 
                                          if abs(t - t_neg) < max(tol, tol * t_neg)]
    
    return min_t, best_j, direction, determining_constraints


def analyze_displacement(matrix: np.ndarray, x0: np.ndarray, tol: float = 1e-10, 
                         remove_borders=False) -> dict:
    """
    Análisis completo del desplazamiento mínimo para la formulación Ax + b >= 0.
    
    Parámetros:
    -----------
    matrix : np.ndarray
        Matriz aumentada [b A] de tamaño (m x n+1)
    x0 : np.ndarray
        Punto interior donde Ax0 + b > 0
    
    Retorna:
    --------
    Diccionario con análisis completo del desplazamiento
    """           
    if remove_borders:
        new_matrix = []
        for x in matrix:
            if np.allclose(x[1:], [ 1.,  0.]) or \
               np.allclose(x[1:], [-1.,  0.]) or \
               np.allclose(x[1:], [ 0.,  1.]) or \
               np.allclose(x[1:], [ 0., -1.]):
               continue
            else:
                new_matrix.append(x)
        matrix = np.array(new_matrix)
        # print(f'Matrix={matrix}')
               
    m, n_plus_1 = matrix.shape
    n = n_plus_1 - 1
    
    b = matrix[:, 0]
    A = matrix[:, 1:]
    slack = A @ x0 + b
    
    min_t, best_j, direction, determining_constraints = \
        min_displacement_to_exit_debug(matrix, x0, tol)
    
    
    # Encontrar TODAS las restricciones que se activan
    active_constraints = []
    tol_relative = max(tol, tol * min_t) if min_t < np.inf else tol
    
    if min_t < np.inf:
        for i in range(m):
            if direction == 1 and A[i, best_j] < -tol:  # Movimiento positivo
                t_this = slack[i] / abs(A[i, best_j])
                if abs(t_this - min_t) < tol_relative:
                    active_constraints.append(i)
            elif direction == -1 and A[i, best_j] > tol:  # Movimiento negativo
                t_this = slack[i] / A[i, best_j]
                if abs(t_this - min_t) < tol_relative:
                    active_constraints.append(i)
    
    # Calcular punto de salida
    exit_point = x0.copy()
    if min_t < np.inf:
        exit_point[best_j] += direction * min_t
    
    # Calcular todos los desplazamientos para análisis completo
    all_displacements = {}
    for j in range(n):
        t_pos = np.inf
        t_neg = np.inf
        
        # Positivo (limitado por A_ij < 0)
        neg_indices = np.where(A[:, j] < -tol)[0]
        if len(neg_indices) > 0:
            t_pos = np.min(slack[neg_indices] / np.abs(A[neg_indices, j]))
        
        # Negativo (limitado por A_ij > 0)
        pos_indices = np.where(A[:, j] > tol)[0]
        if len(pos_indices) > 0:
            t_neg = np.min(slack[pos_indices] / A[pos_indices, j])
        
        all_displacements[f'x{j}'] = {'positive': t_pos, 'negative': t_neg}
    
    return {
        'min_displacement': min_t,
        'dimension': best_j,
        'direction': direction,
        'exit_point': exit_point,
        'active_constraints': active_constraints,
        'determining_constraints': determining_constraints,
        'all_displacements': all_displacements,
        'slack': slack
    }


# def create_matrix_from_inequalities(A: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """
#     Convierte de la forma estándar Ax <= b a la forma [b' A'] donde A'x + b' >= 0.
    
#     Ax <= b  es equivalente a  -Ax - b >= 0
#     """
#     return np.column_stack([-b, -A])


def create_matrix_from_box(lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    """
    Crea la matriz aumentada [b A] para una caja n-dimensional.
    
    lower_bounds <= x <= upper_bounds
    
    Se convierte a:
    x - lower_bounds >= 0  →  Ix - lower_bounds >= 0
    upper_bounds - x >= 0  →  -Ix + upper_bounds >= 0
    """
    n = len(lower_bounds)
    # Primera mitad: x >= lower_bounds → x - lower_bounds >= 0
    upper_part = np.column_stack([lower_bounds, np.eye(n)])
    # Segunda mitad: x <= upper_bounds → -x + upper_bounds >= 0
    lower_part = np.column_stack([upper_bounds, -np.eye(n)])

    return np.vstack([upper_part, lower_part])


def add_bounding_box(matrix, lb, ub):
    """
    Add bounding box constraints to a CDD matrix.
    
    Parameters:
    -----------
    matrix : array-like, shape (n_constraints, n_dims + 1)
        Original CDD matrix in format [b, a1, a2, ...] representing b + a1*x1 + a2*x2 + ... >= 0
    lb : array-like, shape (n_dims,)
        Lower bounds for each variable: x_i >= lb[i]
    ub : array-like, shape (n_dims,)
        Upper bounds for each variable: x_i <= ub[i]
        
    Returns:
    --------
    extended_matrix : numpy.ndarray
        Extended matrix with original constraints plus bounding box constraints
    """
    
    matrix = np.array(matrix)
    lb = np.array(lb)
    ub = np.array(ub)
    
    n_dims = len(lb)
    
    # Validate inputs
    if len(ub) != n_dims:
        raise ValueError("Lower bounds and upper bounds must have the same length")
    
    if matrix.shape[1] != n_dims + 1:
        raise ValueError(f"Matrix should have {n_dims + 1} columns for {n_dims} dimensions")
    
    # Create bounding box constraints
    bbox_constraints = []
    
    # Lower bound constraints: x_i >= lb[i] --> -lb[i] + x_i >= 0
    for i in range(n_dims):
        if not np.isinf(lb[i]):  # Only add if bound is not infinite
            constraint = np.zeros(n_dims + 1)
            constraint[0] = -lb[i]      # constant term
            constraint[i + 1] = 1.0     # coefficient for x_i
            bbox_constraints.append(constraint)
    
    # Upper bound constraints: x_i <= ub[i] --> ub[i] - x_i >= 0
    for i in range(n_dims):
        if not np.isinf(ub[i]):  # Only add if bound is not infinite
            constraint = np.zeros(n_dims + 1)
            constraint[0] = ub[i]       # constant term
            constraint[i + 1] = -1.0    # coefficient for x_i
            bbox_constraints.append(constraint)
    
    # Combine original matrix with bounding box constraints
    if bbox_constraints:
        bbox_matrix = np.array(bbox_constraints)
        extended_matrix = np.vstack([matrix, bbox_matrix])
    else:
        extended_matrix = matrix.copy()
    
    return extended_matrix


def get_ra_matrix(x, model, full_ra='complete') -> dict:
    
    contrib, _, I_vecs, contrib_list, H_list, _, _ =  \
        get_face_contrib_accelerated(x, 
                                     model, 
                                     return_weighted=False)
        
    num_outputs = model.layers[-1].out_features
    
    winner_class = np.argmax(H_list[-1][1:])
        
    config = ''
    for vec in I_vecs:
        for v in vec[1:]:
            config = config + str(int(v))

    config = config[:-num_outputs]
        
    w0 = contrib_list[0][1:]

    for i in range(len(contrib_list) - 2):
        w0 = np.concatenate((w0, contrib_list[i + 1][1:]))

    sign_vec = np.array(list(config), dtype=float).reshape(-1,1)
    sign_vec[sign_vec == 0] = -1
    
    neuron_matrix = w0 * sign_vec

    output_matrix = []
    
    '''
    Añadidmos Y_winner - Y_looser >= 0  
    '''
    for i in range(num_outputs):
        if i == winner_class:
            continue

        contrib_y_winner_y_other = contrib[winner_class] - contrib[i]
        
        output_matrix.append(contrib_y_winner_y_other)
        

    output_matrix = np.array(output_matrix)
        
    if full_ra == 'complete':
        matrix = np.vstack((neuron_matrix, output_matrix))
    
    elif full_ra == 'neuron_eqs':
        matrix = neuron_matrix
    
    elif full_ra == 'output_eqs':
        matrix = np.array(output_matrix)
        
    else:
        raise Exception(f'get_ra_matrix: error in output description (full_ra={full_ra})')

    
    
    ret = {
            'matrix': matrix, 
            'config': config, 
            'winner_class': winner_class
          }
    
    return ret
    # return matrix, config, winner_class


def get_new_ra(x, model, full_ra='complete', walking_distance=1e-6):
    
    orig_ra = get_ra_matrix(x, model, full_ra=full_ra)
    
    original_matrix = orig_ra['matrix']
    original_config = orig_ra['config']
    original_class = orig_ra['winner_class']
    
    # original_matrix, original_config, original_class = get_ra_matrix(x, model)

    # original_class = np.argmax(model(torch.Tensor(x)))
    
    disp = analyze_displacement(original_matrix, x)
    
    new_x = x.copy()
    
    d = disp['dimension']
    new_x[d] = x[d] + disp['direction'] * (disp['min_displacement'] + walking_distance)

    # new_new_x = np.asarray(new_x)
    
    # new_new_x[:] = new_x
    
    new_ra = get_ra_matrix(new_x, model, full_ra=full_ra)
    
    new_matrix = new_ra['matrix']
    new_config = new_ra['config']
    new_class = new_ra['winner_class']
    
    # print(f'Original point/class = {x} / {original_class}, config={original_config}, new point/class = {new_x} / {new_class}, config={new_config}')
    # if original_class != new_class:
    #     print(f'  CAMBIO DE CLASE {original_class}, config={original_config} --> {new_class}, config={new_config}')
    
    
    ret = {
            'new_matrix': new_matrix, 
            'new_config': new_config, 
            'new_winner_class': new_class, 
            'new_x': new_x,
            'original_matrix': original_matrix, 
            'original_config': original_config,
            'original_winner_class': original_class,
            'displacement': disp
          }
    
    return ret
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd

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
    
#     Z = w @ x
        
#     mul_pos = Z > 0
#     mul_pos = mul_pos * 1
    
#     I_v = mul_pos
    
#     if alpha > 0:
#         mul_neg = Z <= 0
#         mul_neg = mul_neg * alpha  
#         I_v = mul_pos + mul_neg 

#     I = np.diag(I_v)
    
#     H = Z * I_v
    
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
    

    