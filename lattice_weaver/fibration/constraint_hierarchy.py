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
    level: ConstraintLevel | str
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
            # Estandarización: Los predicados deben aceptar un único argumento de tipo diccionario.
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
        level_name = self.level.name if isinstance(self.level, ConstraintLevel) else self.level
        return (f"Constraint(level={level_name}, "
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
        self.constraints: Dict[ConstraintLevel | str, List[Constraint]] = {}
        for level in ConstraintLevel:
            self.constraints[level] = []

    def add_level(self, level_name: str):
        """
        Añade un nuevo nivel de restricción dinámicamente.
        Args:
            level_name: Nombre del nuevo nivel.
        """
        if level_name not in self.constraints:
            self.constraints[level_name] = []
        else:
            print(f"Warning: El nivel '{level_name}' ya existe.")
        
    def add_constraint(self, constraint: Constraint):
        """
        Añade una restricción al nivel apropiado.
        
        Args:
            constraint: Restricción a añadir
        """
        if constraint.level not in self.constraints:
            self.add_level(constraint.level)
        self.constraints[constraint.level].append(constraint)

    def add_hard_constraint(self, constraint_expression: Any, level: ConstraintLevel | str = ConstraintLevel.GLOBAL) -> None:
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
                level=level,
                variables=variables,
                predicate=predicate,
                hardness=Hardness.HARD,
                metadata=metadata,
                expression=constraint_expression
            )
        self.add_constraint(constraint)

    def add_soft_constraint(self, constraint_expression: Any, weight: float, level: ConstraintLevel | str = ConstraintLevel.GLOBAL) -> None:
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
                level=level,
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

    def get_constraints_by_level(self, level: ConstraintLevel | str) -> List[Any]:
        return [c.expression for c in self.constraints.get(level, [])]

    def get_all_constraints(self) -> Dict[str, List[Any]]:
        all_constraints = {}
        for level_enum in self.constraints.keys():
            level_name = level_enum.value if isinstance(level_enum, ConstraintLevel) else level_enum
            all_constraints[level_name] = self.get_constraints_by_level(level_enum)
        return all_constraints

    def to_json(self) -> Dict[str, Any]:
        # Implementación placeholder para serialización
        # Necesitaría un mecanismo para serializar Callables (predicates)
        serialized_constraints = {
            (level.value if isinstance(level, ConstraintLevel) else level): [
                {
                    "variables": c.variables,
                    "expression": str(c.expression), # Convertir callable a string, idealmente serializar de otra forma
                    "weight": c.weight,
                    "hardness": c.hardness.value,
                    "metadata": c.metadata,
                    "level": c.level.value if isinstance(c.level, ConstraintLevel) else c.level
                } for c in self.constraints[level]
            ] for level in self.constraints.keys()
        }
        return serialized_constraints

    def from_json(self, json_data: Dict[str, Any]) -> None:
        # Implementación placeholder para deserialización
        # Reconstruir Callables a partir de strings es complejo y potencialmente inseguro (eval)
        # En una implementación real, se usaría un registro de predicados o un DSL.
        print("Deserialización placeholder: Los predicados no se reconstruyen completamente desde strings.")
        self.constraints = {}
        for level_str, constraints_list in json_data.items():
            level_key = ConstraintLevel[level_str] if level_str in ConstraintLevel.__members__ else level_str
            self.constraints[level_key] = [
                Constraint(
                    level=level_key,
                    variables=c_data["variables"],
                    predicate=lambda x: True, # Placeholder, no se puede reconstruir de forma segura
                    weight=c_data["weight"],
                    hardness=Hardness[c_data["hardness"].upper()],
                    metadata=c_data["metadata"],
                    expression=c_data["expression"]
                ) for c_data in constraints_list
            ]

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
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.LOCAL)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.LOCAL)
        
    def add_unary_constraint(self,
                            variable: str,
                            predicate: Callable[[Dict[str, Any]], bool],
                            weight: float = 1.0,
                            hardness: Hardness = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None):
        constraint_expression = ([variable], predicate, metadata)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.LOCAL)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.LOCAL)
        
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
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.PATTERN)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.PATTERN)
        
    def add_global_constraint(self, 
                             variables: List[str], 
                             predicate: Callable[[Dict[str, Any]], bool],
                             weight: float = 5.0,
                             hardness: Hardness = Hardness.HARD,
                             metadata: Optional[Dict[str, Any]] = None):
        constraint_expression = (variables, predicate, metadata)
        if hardness == Hardness.HARD:
            self.add_hard_constraint(constraint_expression, level=ConstraintLevel.GLOBAL)
        else:
            self.add_soft_constraint(constraint_expression, weight, level=ConstraintLevel.GLOBAL)

    def get_constraints_by_level_name(self, level_name: str) -> List[Constraint]:
        level_key = ConstraintLevel[level_name] if level_name in ConstraintLevel.__members__ else level_name
        return self.constraints.get(level_key, [])

    def set_constraints_for_level(self, level_name: str, constraints: List[Constraint]):
        level_key = ConstraintLevel[level_name] if level_name in ConstraintLevel.__members__ else level_name
        self.constraints[level_key] = constraints

    def get_global_constraints(self) -> List[Constraint]:
        return self.constraints.get(ConstraintLevel.GLOBAL, [])

    def get_local_constraints(self) -> List[Constraint]:
        return self.constraints.get(ConstraintLevel.LOCAL, [])

    def get_pattern_constraints(self) -> List[Constraint]:
        return self.constraints.get(ConstraintLevel.PATTERN, [])

