"""
Constraint Hierarchy Module

Este módulo implementa una jerarquía de restricciones en tres niveles:
- LOCAL: Restricciones binarias/unarias entre variables
- PATTERN: Restricciones sobre grupos de variables
- GLOBAL: Restricciones sobre la solución completa

Parte de la implementación del Flujo de Fibración (Propuesta 2).
"""

from typing import List, Dict, Callable, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class ConstraintLevel(Enum):
    """Niveles de la jerarquía de restricciones."""
    LOCAL = 1      # Restricciones binarias/unarias
    PATTERN = 2    # Restricciones sobre grupos
    GLOBAL = 3     # Restricciones sobre solución completa


class Hardness(Enum):
    """Dureza de una restricción."""
    HARD = "hard"   # Debe satisfacerse siempre
    SOFT = "soft"   # Preferible pero no obligatorio


@dataclass
class Constraint:
    """
    Representación unificada de una restricción.
    
    Attributes:
        level: Nivel en la jerarquía (LOCAL, PATTERN, GLOBAL)
        variables: Lista de IDs de variables involucradas
        predicate: Función que evalúa la restricción
        weight: Peso de la restricción en el funcional de energía
        hardness: Dureza (HARD o SOFT)
        metadata: Información adicional sobre la restricción
    """
    level: ConstraintLevel
    variables: List[str]
    predicate: Callable
    weight: float = 1.0
    hardness: Hardness = Hardness.HARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, assignment: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evalúa la restricción sobre una asignación parcial.
        
        Args:
            assignment: Diccionario {variable: valor}
            
        Returns:
            Tupla (satisfied, violation_degree):
                - satisfied: True si la restricción se satisface
                - violation_degree: Grado de violación (0.0 = satisfecha, 1.0 = máxima violación)
        """
        # Verificar si todas las variables necesarias están asignadas
        assigned_vars = {var: assignment[var] for var in self.variables if var in assignment}
        
        if len(assigned_vars) < len(self.variables):
            # Restricción no evaluable aún (variables no asignadas)
            return (True, 0.0)
        
        try:
            result = self.predicate(assigned_vars)
            
            if isinstance(result, bool):
                # Predicado devuelve bool: True = satisfecha, False = violada
                return (result, 0.0 if result else 1.0)
            elif isinstance(result, (int, float)):
                # Predicado devuelve grado de violación directamente
                violation = float(result)
                return (violation == 0.0, violation)
            else:
                raise ValueError(f"Predicado debe devolver bool o float, no {type(result)}")
                
        except Exception as e:
            # Error en evaluación -> consideramos violada
            print(f"Warning: Error evaluating constraint: {e}")
            return (False, 1.0)
    
    def __repr__(self):
        return (f"Constraint(level={self.level.name}, "
                f"vars={len(self.variables)}, "
                f"weight={self.weight}, "
                f"hardness={self.hardness.value})")


class ConstraintHierarchy:
    """
    Organiza las restricciones del problema en una jerarquía de tres niveles.
    
    Esta jerarquía permite implementar el proceso de hacificación (sheafification)
    donde cada nivel actúa como un filtro de coherencia.
    """
    
    def __init__(self):
        """Inicializa una jerarquía vacía."""
        self.constraints: Dict[ConstraintLevel, List[Constraint]] = {
            ConstraintLevel.LOCAL: [],
            ConstraintLevel.PATTERN: [],
            ConstraintLevel.GLOBAL: []
        }
        
    def add_constraint(self, constraint: Constraint):
        """
        Añade una restricción al nivel apropiado.
        
        Args:
            constraint: Restricción a añadir
        """
        self.constraints[constraint.level].append(constraint)
        
    def add_local_constraint(self, 
                            var1: str, 
                            var2: str, 
                            predicate: Callable[[Dict[str, Any]], bool], 
                            hardness: Hardness = Hardness.HARD,
                            weight: float = 1.0, 
                            metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción local (binaria).
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            predicate: Función que toma un dict {var1: val1, var2: val2} y devuelve bool
            weight: Peso de la restricción
            hardness: Dureza (HARD o SOFT)
            metadata: Información adicional
        """
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=[var1, var2],
            predicate=predicate,
            weight=weight,
            hardness=hardness,
            metadata=metadata or {}
        )
        self.add_constraint(constraint)
        
    def add_unary_constraint(self,
                            variable: str,
                            predicate: Callable[[Dict[str, Any]], bool],
                            weight: float = 1.0,
                            hardness: Hardness = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción unaria (sobre una sola variable).
        
        Args:
            variable: Variable involucrada
            predicate: Función que toma un dict {variable: value} y devuelve bool
            weight: Peso de la restricción
            hardness: Dureza (HARD o SOFT)
            metadata: Información adicional
        """
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=[variable],
            predicate=predicate,
            weight=weight,
            hardness=hardness,
            metadata=metadata or {}
        )
        self.add_constraint(constraint)
        
    def add_pattern_constraint(self, 
                              variables: List[str], 
                              predicate: Callable[[Dict[str, Any]], bool],
                              pattern_type: str = "custom", 
                              weight: float = 2.0,
                              hardness: Hardness = Hardness.HARD,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción de patrón (sobre un grupo de variables).
        
        Args:
            variables: Lista de variables involucradas
            predicate: Función que toma un dict con las variables y devuelve bool o float
            pattern_type: Tipo de patrón (ej. 'all_different', 'cycle', 'clique')
            weight: Peso de la restricción
            hardness: Dureza (HARD o SOFT)
            metadata: Información adicional
        """
        meta = metadata or {}
        meta['pattern_type'] = pattern_type
        
        constraint = Constraint(
            level=ConstraintLevel.PATTERN,
            variables=variables,
            predicate=predicate,
            weight=weight,
            hardness=hardness,
            metadata=meta
        )
        self.add_constraint(constraint)
        
    def add_global_constraint(self, 
                             variables: List[str], 
                             predicate: Callable[[Dict[str, Any]], float],
                             objective: str = "satisfy", 
                             weight: float = 3.0,
                             hardness: Hardness = Hardness.SOFT,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción global (sobre la solución completa).
        
        Args:
            variables: Lista de variables involucradas (puede ser todas)
            predicate: Función que toma un dict con las variables y devuelve float (coste/violación)
            objective: Objetivo ('satisfy', 'minimize', 'maximize')
            weight: Peso de la restricción
            hardness: Dureza (típicamente SOFT para restricciones globales)
            metadata: Información adicional
        """
        meta = metadata or {}
        meta['objective'] = objective
        
        constraint = Constraint(
            level=ConstraintLevel.GLOBAL,
            variables=variables,
            predicate=predicate,
            weight=weight,
            hardness=hardness,
            metadata=meta
        )
        self.add_constraint(constraint)
        
    def get_constraints_at_level(self, level: ConstraintLevel) -> List[Constraint]:
        """
        Obtiene todas las restricciones de un nivel.
        
        Args:
            level: Nivel de la jerarquía
            
        Returns:
            Lista de restricciones en ese nivel
        """
        return self.constraints[level]
    
    def get_constraints_involving(self, variable: str) -> List[Constraint]:
        """
        Obtiene todas las restricciones que involucran una variable.
        
        Args:
            variable: ID de la variable
            
        Returns:
            Lista de restricciones que involucran la variable
        """
        result = []
        for level_constraints in self.constraints.values():
            for constraint in level_constraints:
                if variable in constraint.variables:
                    result.append(constraint)
        return result
    
    def classify_by_hardness(self) -> Dict[Hardness, List[Constraint]]:
        """
        Clasifica restricciones por dureza.
        
        Returns:
            Diccionario {Hardness: [restricciones]}
        """
        result = {Hardness.HARD: [], Hardness.SOFT: []}
        for level_constraints in self.constraints.values():
            for constraint in level_constraints:
                result[constraint.hardness].append(constraint)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la jerarquía.
        
        Returns:
            Diccionario con estadísticas
        """
        total = sum(len(constraints) for constraints in self.constraints.values())
        by_hardness = self.classify_by_hardness()
        
        return {
            'total_constraints': total,
            'by_level': {
                level.name: len(constraints) 
                for level, constraints in self.constraints.items()
            },
            'by_hardness': {
                hardness.value: len(constraints)
                for hardness, constraints in by_hardness.items()
            }
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"ConstraintHierarchy(total={stats['total_constraints']}, "
                f"local={stats['by_level']['LOCAL']}, "
                f"pattern={stats['by_level']['PATTERN']}, "
                f"global={stats['by_level']['GLOBAL']})")

