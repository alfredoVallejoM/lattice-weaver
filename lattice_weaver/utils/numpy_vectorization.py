"""
NumPy Vectorization - Operaciones Vectorizadas para Dominios Grandes

Reemplaza loops Python con operaciones vectorizadas de NumPy para dominios grandes.

Optimizaciones:
- Broadcasting para operaciones elemento-wise
- Operaciones matriciales para restricciones
- Indexing avanzado para filtrado
- Reducción vectorizada

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VectorizedDomains:
    """
    Representación vectorizada de dominios.
    
    Usa arrays NumPy para operaciones eficientes.
    """
    
    # Array de dominios: shape (n_vars, max_domain_size)
    domains: np.ndarray
    
    # Máscaras de validez: shape (n_vars, max_domain_size)
    masks: np.ndarray
    
    # Tamaños de dominios: shape (n_vars,)
    sizes: np.ndarray
    
    # Mapeo: variable name -> índice
    var_to_idx: Dict[str, int]
    
    # Mapeo inverso: índice -> variable name
    idx_to_var: Dict[int, str]


class NumpyVectorizer:
    """
    Vectorizador de operaciones con NumPy.
    
    Convierte operaciones sobre dominios a operaciones vectorizadas.
    """
    
    def __init__(self, max_domain_size: int = 1000):
        """
        Inicializa vectorizador.
        
        Args:
            max_domain_size: Tamaño máximo de dominio
        """
        self.max_domain_size = max_domain_size
        
        # Estadísticas
        self.stats = {
            'vectorized_operations': 0,
            'elements_processed': 0,
            'speedup_measured': 0.0
        }
    
    def vectorize_domains(
        self,
        domains: Dict[str, List[Any]]
    ) -> VectorizedDomains:
        """
        Convierte dominios a representación vectorizada.
        
        Args:
            domains: Dominios como diccionario
        
        Returns:
            Dominios vectorizados
        """
        n_vars = len(domains)
        variables = sorted(domains.keys())
        
        # Crear mapeos
        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        idx_to_var = {idx: var for idx, var in enumerate(variables)}
        
        # Inicializar arrays
        domain_array = np.zeros((n_vars, self.max_domain_size), dtype=np.int32)
        mask_array = np.zeros((n_vars, self.max_domain_size), dtype=bool)
        size_array = np.zeros(n_vars, dtype=np.int32)
        
        # Llenar arrays
        for var, domain in domains.items():
            idx = var_to_idx[var]
            size = min(len(domain), self.max_domain_size)
            
            domain_array[idx, :size] = domain[:size]
            mask_array[idx, :size] = True
            size_array[idx] = size
        
        return VectorizedDomains(
            domains=domain_array,
            masks=mask_array,
            sizes=size_array,
            var_to_idx=var_to_idx,
            idx_to_var=idx_to_var
        )
    
    def devectorize_domains(
        self,
        vectorized: VectorizedDomains
    ) -> Dict[str, List[Any]]:
        """
        Convierte dominios vectorizados a diccionario.
        
        Args:
            vectorized: Dominios vectorizados
        
        Returns:
            Dominios como diccionario
        """
        domains = {}
        
        for idx, var in vectorized.idx_to_var.items():
            size = vectorized.sizes[idx]
            domain = vectorized.domains[idx, :size].tolist()
            domains[var] = domain
        
        return domains
    
    def intersection_vectorized(
        self,
        domains1: VectorizedDomains,
        domains2: VectorizedDomains
    ) -> VectorizedDomains:
        """
        Intersección vectorizada de dominios.
        
        Args:
            domains1: Primeros dominios
            domains2: Segundos dominios
        
        Returns:
            Intersección
        """
        # Usar broadcasting para encontrar intersección
        n_vars = domains1.domains.shape[0]
        
        result_domains = np.zeros_like(domains1.domains)
        result_masks = np.zeros_like(domains1.masks)
        result_sizes = np.zeros_like(domains1.sizes)
        
        for i in range(n_vars):
            # Obtener dominios válidos
            domain1 = domains1.domains[i, domains1.masks[i]]
            domain2 = domains2.domains[i, domains2.masks[i]]
            
            # Intersección usando np.intersect1d
            intersection = np.intersect1d(domain1, domain2)
            
            # Guardar resultado
            size = len(intersection)
            result_domains[i, :size] = intersection
            result_masks[i, :size] = True
            result_sizes[i] = size
        
        self.stats['vectorized_operations'] += 1
        self.stats['elements_processed'] += n_vars
        
        return VectorizedDomains(
            domains=result_domains,
            masks=result_masks,
            sizes=result_sizes,
            var_to_idx=domains1.var_to_idx,
            idx_to_var=domains1.idx_to_var
        )
    
    def filter_by_predicate_vectorized(
        self,
        domains: VectorizedDomains,
        predicate_matrix: np.ndarray
    ) -> VectorizedDomains:
        """
        Filtra dominios usando matriz de predicados.
        
        Args:
            domains: Dominios
            predicate_matrix: Matriz booleana (n_vars, max_domain_size)
        
        Returns:
            Dominios filtrados
        """
        # Aplicar máscara de predicado
        new_masks = domains.masks & predicate_matrix
        
        # Recomputar tamaños
        new_sizes = np.sum(new_masks, axis=1)
        
        # Compactar dominios (eliminar huecos)
        n_vars = domains.domains.shape[0]
        new_domains = np.zeros_like(domains.domains)
        
        for i in range(n_vars):
            valid_values = domains.domains[i, new_masks[i]]
            size = len(valid_values)
            new_domains[i, :size] = valid_values
        
        self.stats['vectorized_operations'] += 1
        self.stats['elements_processed'] += n_vars
        
        return VectorizedDomains(
            domains=new_domains,
            masks=new_masks,
            sizes=new_sizes,
            var_to_idx=domains.var_to_idx,
            idx_to_var=domains.idx_to_var
        )
    
    def compute_mrv_vectorized(
        self,
        domains: VectorizedDomains,
        unassigned_mask: np.ndarray
    ) -> int:
        """
        Calcula MRV vectorizado.
        
        Args:
            domains: Dominios
            unassigned_mask: Máscara de variables no asignadas
        
        Returns:
            Índice de variable con MRV
        """
        # Aplicar máscara de no asignadas
        sizes = np.where(unassigned_mask, domains.sizes, np.inf)
        
        # Encontrar mínimo
        mrv_idx = np.argmin(sizes)
        
        self.stats['vectorized_operations'] += 1
        
        return mrv_idx
    
    def compute_degree_vectorized(
        self,
        constraint_matrix: np.ndarray,
        unassigned_mask: np.ndarray
    ) -> int:
        """
        Calcula Degree vectorizado.
        
        Args:
            constraint_matrix: Matriz de restricciones (n_vars, n_vars)
            unassigned_mask: Máscara de variables no asignadas
        
        Returns:
            Índice de variable con mayor degree
        """
        # Contar restricciones por variable
        degrees = np.sum(constraint_matrix, axis=1)
        
        # Aplicar máscara
        degrees = np.where(unassigned_mask, degrees, -np.inf)
        
        # Encontrar máximo
        degree_idx = np.argmax(degrees)
        
        self.stats['vectorized_operations'] += 1
        
        return degree_idx
    
    def propagate_ac3_vectorized(
        self,
        domains: VectorizedDomains,
        constraint_matrix: np.ndarray
    ) -> Tuple[VectorizedDomains, bool]:
        """
        Propagación AC-3 vectorizada.
        
        Args:
            domains: Dominios
            constraint_matrix: Matriz de compatibilidad (n_vars, n_vars, max_size, max_size)
        
        Returns:
            (nuevos_dominios, changed)
        """
        n_vars = domains.domains.shape[0]
        changed = False
        
        new_domains = domains.domains.copy()
        new_masks = domains.masks.copy()
        new_sizes = domains.sizes.copy()
        
        # Iterar sobre arcos
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                
                # Obtener dominios
                domain_i = domains.domains[i, domains.masks[i]]
                domain_j = domains.domains[j, domains.masks[j]]
                
                if len(domain_i) == 0 or len(domain_j) == 0:
                    continue
                
                # Verificar soporte usando broadcasting
                # support[k] = True si domain_i[k] tiene soporte en domain_j
                support = np.any(
                    constraint_matrix[i, j, :len(domain_i), :len(domain_j)],
                    axis=1
                )
                
                # Filtrar valores sin soporte
                if not np.all(support):
                    new_domain_i = domain_i[support]
                    size = len(new_domain_i)
                    
                    new_domains[i, :] = 0
                    new_domains[i, :size] = new_domain_i
                    new_masks[i, :] = False
                    new_masks[i, :size] = True
                    new_sizes[i] = size
                    
                    changed = True
        
        self.stats['vectorized_operations'] += 1
        self.stats['elements_processed'] += n_vars * n_vars
        
        result = VectorizedDomains(
            domains=new_domains,
            masks=new_masks,
            sizes=new_sizes,
            var_to_idx=domains.var_to_idx,
            idx_to_var=domains.idx_to_var
        )
        
        return result, changed
    
    def compute_energy_vectorized(
        self,
        assignment: np.ndarray,
        constraint_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Calcula energía vectorizada.
        
        Args:
            assignment: Array de asignación (n_vars,)
            constraint_matrix: Matriz de violaciones (n_constraints, n_vars, n_vars)
            weights: Pesos de restricciones (n_constraints,)
        
        Returns:
            Energía total
        """
        n_constraints = constraint_matrix.shape[0]
        
        # Evaluar todas las restricciones en paralelo
        violations = np.zeros(n_constraints, dtype=bool)
        
        for c in range(n_constraints):
            # Obtener pares de variables en restricción c
            # (Simplificado: asumimos restricciones binarias)
            for i in range(len(assignment)):
                for j in range(i + 1, len(assignment)):
                    if constraint_matrix[c, i, j]:
                        val_i = assignment[i]
                        val_j = assignment[j]
                        
                        # Verificar violación
                        # (Simplificado: matriz de compatibilidad)
                        if not constraint_matrix[c, val_i, val_j]:
                            violations[c] = True
                            break
        
        # Sumar pesos de restricciones violadas
        energy = np.sum(weights[violations])
        
        self.stats['vectorized_operations'] += 1
        
        return energy
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats = {
            'vectorized_operations': 0,
            'elements_processed': 0,
            'speedup_measured': 0.0
        }


# Instancia global
_numpy_vectorizer = NumpyVectorizer()


def get_numpy_vectorizer() -> NumpyVectorizer:
    """Retorna instancia global de NumPy Vectorizer."""
    return _numpy_vectorizer

