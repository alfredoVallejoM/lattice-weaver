"""
JIT Compiler - Compilación Just-In-Time con Numba

Compila funciones críticas con Numba para obtener velocidad cercana a C.

Funciones optimizadas:
- Propagación de restricciones
- Cálculo de heurísticas
- Operaciones sobre dominios
- Cálculo de energía

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import List, Set, Dict, Any, Callable
import numpy as np
from numba import jit, njit, prange
import functools


# ============================================================================
# Decoradores JIT
# ============================================================================

def jit_optimize(func: Callable) -> Callable:
    """
    Decorator para optimizar función con Numba JIT.
    
    Usa @jit con opciones óptimas.
    
    Args:
        func: Función a optimizar
    
    Returns:
        Función compilada
    """
    return jit(func, nopython=False, cache=True, fastmath=True)


def njit_optimize(func: Callable) -> Callable:
    """
    Decorator para optimizar función con Numba njit (nopython mode).
    
    Más rápido pero más restrictivo.
    
    Args:
        func: Función a optimizar
    
    Returns:
        Función compilada
    """
    return njit(func, cache=True, fastmath=True, parallel=False)


def njit_parallel(func: Callable) -> Callable:
    """
    Decorator para optimizar función con paralelización.
    
    Args:
        func: Función a optimizar
    
    Returns:
        Función compilada con paralelización
    """
    return njit(func, cache=True, fastmath=True, parallel=True)


# ============================================================================
# Operaciones sobre Dominios (JIT-compiled)
# ============================================================================

@njit_optimize
def domain_intersection_jit(domain1: np.ndarray, domain2: np.ndarray) -> np.ndarray:
    """
    Intersección de dos dominios (JIT-compiled).
    
    Args:
        domain1: Primer dominio (array NumPy)
        domain2: Segundo dominio (array NumPy)
    
    Returns:
        Intersección (array NumPy)
    """
    result = []
    for val in domain1:
        if val in domain2:
            result.append(val)
    return np.array(result)


@njit_optimize
def domain_difference_jit(domain1: np.ndarray, domain2: np.ndarray) -> np.ndarray:
    """
    Diferencia de dos dominios (JIT-compiled).
    
    Args:
        domain1: Primer dominio
        domain2: Segundo dominio
    
    Returns:
        Diferencia
    """
    result = []
    for val in domain1:
        if val not in domain2:
            result.append(val)
    return np.array(result)


@njit_optimize
def domain_size_jit(domain: np.ndarray) -> int:
    """
    Tamaño de dominio (JIT-compiled).
    
    Args:
        domain: Dominio
    
    Returns:
        Tamaño
    """
    return len(domain)


@njit_optimize
def domain_is_empty_jit(domain: np.ndarray) -> bool:
    """
    Verifica si dominio está vacío (JIT-compiled).
    
    Args:
        domain: Dominio
    
    Returns:
        True si vacío
    """
    return len(domain) == 0


# ============================================================================
# Heurísticas (JIT-compiled)
# ============================================================================

@njit_optimize
def mrv_score_jit(domain_sizes: np.ndarray) -> int:
    """
    Calcula score MRV (Minimum Remaining Values).
    
    Args:
        domain_sizes: Array de tamaños de dominios
    
    Returns:
        Índice de variable con MRV
    """
    return np.argmin(domain_sizes)


@njit_optimize
def degree_score_jit(constraint_counts: np.ndarray) -> int:
    """
    Calcula score Degree (número de restricciones).
    
    Args:
        constraint_counts: Array de conteos de restricciones
    
    Returns:
        Índice de variable con mayor degree
    """
    return np.argmax(constraint_counts)


@njit_optimize
def weighted_degree_score_jit(
    domain_sizes: np.ndarray,
    weighted_degrees: np.ndarray
) -> int:
    """
    Calcula score Weighted Degree.
    
    Score = domain_size / weighted_degree (menor es mejor)
    
    Args:
        domain_sizes: Tamaños de dominios
        weighted_degrees: Weighted degrees
    
    Returns:
        Índice de variable con mejor score
    """
    # Evitar división por cero
    weighted_degrees = np.where(weighted_degrees == 0, 0.1, weighted_degrees)
    
    scores = domain_sizes / weighted_degrees
    return np.argmin(scores)


@njit_parallel
def impact_scores_jit(
    impacts: np.ndarray,
    domain_sizes: np.ndarray
) -> np.ndarray:
    """
    Calcula scores de impacto para todas las variables.
    
    Args:
        impacts: Matriz de impactos (variables x valores)
        domain_sizes: Tamaños de dominios
    
    Returns:
        Array de scores
    """
    n_vars = len(domain_sizes)
    scores = np.zeros(n_vars)
    
    for i in prange(n_vars):
        if domain_sizes[i] > 0:
            scores[i] = np.mean(impacts[i, :domain_sizes[i]])
    
    return scores


# ============================================================================
# Propagación (JIT-compiled)
# ============================================================================

@njit_optimize
def ac3_revise_jit(
    domain1: np.ndarray,
    domain2: np.ndarray,
    constraint_matrix: np.ndarray
) -> tuple:
    """
    Revise de AC-3 (JIT-compiled).
    
    Args:
        domain1: Dominio de variable 1
        domain2: Dominio de variable 2
        constraint_matrix: Matriz de restricción (valores compatibles)
    
    Returns:
        (revised, new_domain1)
    """
    revised = False
    new_domain = []
    
    for val1_idx, val1 in enumerate(domain1):
        # Verificar si existe al menos un valor compatible en domain2
        has_support = False
        for val2_idx, val2 in enumerate(domain2):
            if constraint_matrix[val1_idx, val2_idx]:
                has_support = True
                break
        
        if has_support:
            new_domain.append(val1)
        else:
            revised = True
    
    return revised, np.array(new_domain)


@njit_parallel
def forward_checking_jit(
    domains: np.ndarray,
    assignment_var: int,
    assignment_val: int,
    constraint_matrices: np.ndarray
) -> np.ndarray:
    """
    Forward Checking paralelo (JIT-compiled).
    
    Args:
        domains: Array de dominios
        assignment_var: Variable asignada
        assignment_val: Valor asignado
        constraint_matrices: Matrices de restricciones
    
    Returns:
        Nuevos dominios
    """
    n_vars = len(domains)
    new_domains = domains.copy()
    
    for var in prange(n_vars):
        if var != assignment_var:
            # Filtrar dominio según restricción
            constraint_matrix = constraint_matrices[assignment_var, var]
            new_domain = []
            
            for val in domains[var]:
                if constraint_matrix[assignment_val, val]:
                    new_domain.append(val)
            
            new_domains[var] = np.array(new_domain)
    
    return new_domains


# ============================================================================
# Cálculo de Energía (JIT-compiled)
# ============================================================================

@njit_optimize
def energy_unary_jit(
    assignment: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Calcula energía de restricciones unarias (JIT-compiled).
    
    Args:
        assignment: Array de asignación
        weights: Pesos de restricciones
        targets: Valores objetivo
    
    Returns:
        Energía total
    """
    energy = 0.0
    for i in range(len(assignment)):
        if assignment[i] != targets[i]:
            energy += weights[i]
    return energy


@njit_optimize
def energy_binary_jit(
    assignment: np.ndarray,
    constraint_pairs: np.ndarray,
    weights: np.ndarray,
    violation_matrices: np.ndarray
) -> float:
    """
    Calcula energía de restricciones binarias (JIT-compiled).
    
    Args:
        assignment: Array de asignación
        constraint_pairs: Pares de variables en restricciones
        weights: Pesos de restricciones
        violation_matrices: Matrices de violación
    
    Returns:
        Energía total
    """
    energy = 0.0
    for i in range(len(constraint_pairs)):
        var1, var2 = constraint_pairs[i]
        val1 = assignment[var1]
        val2 = assignment[var2]
        
        if violation_matrices[i, val1, val2]:
            energy += weights[i]
    
    return energy


@njit_parallel
def energy_total_parallel_jit(
    assignment: np.ndarray,
    unary_weights: np.ndarray,
    unary_targets: np.ndarray,
    binary_pairs: np.ndarray,
    binary_weights: np.ndarray,
    binary_violations: np.ndarray
) -> float:
    """
    Calcula energía total con paralelización (JIT-compiled).
    
    Args:
        assignment: Asignación
        unary_weights: Pesos unarios
        unary_targets: Targets unarios
        binary_pairs: Pares binarios
        binary_weights: Pesos binarios
        binary_violations: Violaciones binarias
    
    Returns:
        Energía total
    """
    energy_unary = energy_unary_jit(assignment, unary_weights, unary_targets)
    energy_binary = energy_binary_jit(assignment, binary_pairs, binary_weights, binary_violations)
    
    return energy_unary + energy_binary


# ============================================================================
# Utilidades
# ============================================================================

class JITCompiler:
    """
    Gestor de compilación JIT.
    
    Mantiene funciones compiladas y estadísticas.
    """
    
    def __init__(self):
        """Inicializa JIT Compiler."""
        self.compiled_functions: Dict[str, Callable] = {}
        self.stats = {
            'functions_compiled': 0,
            'compilation_time': 0.0,
            'speedup_measured': {}
        }
    
    def compile_function(self, func: Callable, mode: str = 'jit') -> Callable:
        """
        Compila función con Numba.
        
        Args:
            func: Función a compilar
            mode: Modo de compilación ('jit', 'njit', 'parallel')
        
        Returns:
            Función compilada
        """
        func_name = func.__name__
        
        if func_name in self.compiled_functions:
            return self.compiled_functions[func_name]
        
        if mode == 'jit':
            compiled = jit_optimize(func)
        elif mode == 'njit':
            compiled = njit_optimize(func)
        elif mode == 'parallel':
            compiled = njit_parallel(func)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.compiled_functions[func_name] = compiled
        self.stats['functions_compiled'] += 1
        
        return compiled
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        return self.stats.copy()


# Instancia global
_jit_compiler = JITCompiler()


def get_jit_compiler() -> JITCompiler:
    """Retorna instancia global de JIT Compiler."""
    return _jit_compiler

