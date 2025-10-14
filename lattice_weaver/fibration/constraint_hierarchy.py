from typing import List, Dict, Callable, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from lattice_weaver.fibration.constraint_hierarchy_api import ConstraintHierarchyAPI


class ConstraintLevel(Enum):
    """Niveles de la jerarquía de restricciones."""
    LOCAL = "LOCAL"      # Restricciones binarias/unarias
    PATTERN = "PATTERN"    # Restricciones sobre grupos
    GLOBAL = "GLOBAL"     # Restricciones sobre solución completa


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
        expression: La expresión original de la restricción (para serialización/deserialización)
    """
    level: ConstraintLevel
    variables: List[str]
    predicate: Callable
    weight: float = 1.0
    hardness: Hardness = Hardness.HARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    expression: Any = None # Añadido para almacenar la expresión original
    
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
            # Si el predicado espera un diccionario de asignaciones, pasarlo directamente.
            # Si espera argumentos individuales, esto podría necesitar un ajuste más sofisticado
            # o una convención clara para la definición de predicados.
            # Pasar el diccionario de asignaciones directamente al predicado.
             # Los predicados deben estar diseñados para aceptar un único argumento de tipo diccionario.
            # Se pasa el diccionario `assigned_vars` directamente.
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


class ConstraintHierarchy(ConstraintHierarchyAPI):
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

    def add_hard_constraint(self, constraint_expression: Any, level: str = "GLOBAL") -> None:
        # Aquí se asume que constraint_expression es una tupla (variables, predicate, metadata)
        # o un objeto que puede ser convertido a Constraint
        if isinstance(constraint_expression, Constraint):
            constraint = constraint_expression
        else:
            # Asumimos que constraint_expression es una función o una tupla (variables, predicate)
            variables = constraint_expression[0] if isinstance(constraint_expression, tuple) else []
            predicate = constraint_expression[1] if isinstance(constraint_expression, tuple) else constraint_expression
            metadata = constraint_expression[2] if isinstance(constraint_expression, tuple) and len(constraint_expression) > 2 else {}
            constraint = Constraint(
                level=ConstraintLevel[level],
                variables=variables,
                predicate=predicate,
                hardness=Hardness.HARD,
                metadata=metadata,
                expression=constraint_expression
            )
        self.add_constraint(constraint)

    def add_soft_constraint(self, constraint_expression: Any, weight: float, level: str = "GLOBAL") -> None:
        # Similar a add_hard_constraint, pero con dureza SOFT y peso
        if isinstance(constraint_expression, Constraint):
            constraint = constraint_expression
            constraint.hardness = Hardness.SOFT
            constraint.weight = weight
        else:
            variables = constraint_expression[0] if isinstance(constraint_expression, tuple) else []
            predicate = constraint_expression[1] if isinstance(constraint_expression, tuple) else constraint_expression
            metadata = constraint_expression[2] if isinstance(constraint_expression, tuple) and len(constraint_expression) > 2 else {}
            constraint = Constraint(
                level=ConstraintLevel[level],
                variables=variables,
                predicate=predicate,
                weight=weight,
                hardness=Hardness.SOFT,
                metadata=metadata,
                expression=constraint_expression
            )
        self.add_constraint(constraint)

    def evaluate_solution(self, solution: Dict[str, Any]) -> Tuple[bool, float]:
        all_hard_satisfied = True
        total_energy = 0.0

        for level_constraints in self.constraints.values():
            for constraint in level_constraints:
                satisfied, violation_degree = constraint.evaluate(solution)
                if constraint.hardness == Hardness.HARD:
                    if not satisfied:
                        all_hard_satisfied = False
                else:  # Soft constraint
                    total_energy += violation_degree * constraint.weight
        return all_hard_satisfied, total_energy

    def get_constraints_by_level(self, level: str) -> List[Any]:
        return [c.expression for c in self.constraints[ConstraintLevel[level]]]

    def get_all_constraints(self) -> Dict[str, List[Any]]:
        all_constraints = {}
        for level_enum in ConstraintLevel:
            all_constraints[level_enum.value] = self.get_constraints_by_level(level_enum.value)
        return all_constraints

    def to_json(self) -> Dict[str, Any]:
        # Implementación placeholder para serialización
        # Necesitaría un mecanismo para serializar Callables (predicates)
        serialized_constraints = {
            level.value: [
                {
                    "variables": c.variables,
                    "expression": str(c.expression), # Convertir callable a string, idealmente serializar de otra forma
                    "weight": c.weight,
                    "hardness": c.hardness.value,
                    "metadata": c.metadata,
                    "level": c.level.value
                } for c in self.constraints[level]
            ] for level in ConstraintLevel
        }
        return serialized_constraints

    def from_json(self, json_data: Dict[str, Any]) -> None:
        # Implementación placeholder para deserialización
        # Reconstruir Callables a partir de strings es complejo y potencialmente inseguro (eval)
        # En una implementación real, se usaría un registro de predicados o un DSL.
        print("Deserialización placeholder: Los predicados no se reconstruyen completamente desde strings.")
        self.constraints = {
            ConstraintLevel[level_str]: [
                Constraint(
                    level=ConstraintLevel[c_data["level"]],
                    variables=c_data["variables"],
                    predicate=lambda x: True, # Placeholder, no se puede reconstruir de forma segura
                    weight=c_data["weight"],
                    hardness=Hardness[c_data["hardness"].upper()],
                    metadata=c_data["metadata"],
                    expression=c_data["expression"]
                ) for c_data in constraints_list
            ] for level_str, constraints_list in json_data.items()
        }

    # Métodos de conveniencia existentes, adaptados para usar la nueva API internamente
    def add_local_constraint(self, 
                            var1: str, 
                            var2: str, 
                            predicate: Callable[[Dict[str, Any]], bool], 
                            weight: float = 1.0, 
                            hardness: Hardness = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None):
        constraint_expression = ([var1, var2], predicate, metadata)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.LOCAL.value)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.LOCAL.value)
        
    def add_unary_constraint(self,
                            variable: str,
                            predicate: Callable[[Dict[str, Any]], bool],
                            weight: float = 1.0,
                            hardness: Hardness = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None):
        constraint_expression = ([variable], predicate, metadata)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.LOCAL.value)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.LOCAL.value)
        
    def add_pattern_constraint(self, 
                              variables: List[str], 
                              predicate: Callable[[Dict[str, Any]], bool],
                              pattern_type: str = "custom", 
                              weight: float = 2.0,
                              hardness: Hardness = Hardness.HARD,
                              metadata: Optional[Dict[str, Any]] = None):
        meta = metadata or {}
        meta["pattern_type"] = pattern_type
        constraint_expression = (variables, predicate, meta)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.PATTERN.value)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.PATTERN.value)
        
    def add_global_constraint(self, 
                             variables: List[str], 
                             predicate: Callable[[Dict[str, Any]], float],
                             objective: str = "satisfy", 
                             weight: float = 3.0,
                             hardness: Hardness = Hardness.SOFT,
                             metadata: Optional[Dict[str, Any]] = None):
        meta = metadata or {}
        meta["objective"] = objective
        constraint_expression = (variables, predicate, meta)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.GLOBAL.value)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.GLOBAL.value)
        
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
            "total_constraints": total,
            "by_level": {
                level.name: len(constraints) 
                for level, constraints in self.constraints.items()
            },
            "by_hardness": {
                hardness.value: len(constraints)
                for hardness, constraints in by_hardness.items()
            }
        }
        
    def __repr__(self):
        stats = self.get_statistics()
        return (f'ConstraintHierarchy(total={stats["total_constraints"]}, '
                f'local={stats["by_level"]["LOCAL"]}, '
                f'pattern={stats["by_level"]["PATTERN"]}, '
                f'global={stats["by_level"]["GLOBAL"]})')

