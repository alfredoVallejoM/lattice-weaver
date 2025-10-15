from typing import List, Dict, Callable, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from lattice_weaver.fibration.constraint_hierarchy_api import ConstraintHierarchyAPI


class ConstraintLevel(Enum):
    """
    Define los niveles de la jerarquía de restricciones en LatticeWeaver.
    Estos niveles permiten una evaluación estructurada de la coherencia y la energía.

    - LOCAL: Restricciones que involucran un número pequeño de variables (e.g., binarias, unarias).
    - PATTERN: Restricciones que aplican a patrones o grupos de variables (e.g., all-different, sum-constraints).
    - GLOBAL: Restricciones que afectan a la solución completa o a un gran subconjunto de variables.
    """
    LOCAL = "LOCAL"
    PATTERN = "PATTERN"
    GLOBAL = "GLOBAL"


class Hardness(Enum):
    """
    Define la dureza de una restricción, indicando si es obligatoria o preferencial.

    - HARD: La restricción debe satisfacerse siempre. Su violación implica una solución no válida.
    - SOFT: La restricción es preferencial. Su violación contribuye a la energía total de la solución
            pero no la invalida, permitiendo la búsqueda de soluciones óptimas en lugar de solo factibles.
    """
    HARD = "hard"
    SOFT = "soft"


@dataclass
class Constraint:
    """
    Representación unificada de una restricción en el sistema LatticeWeaver.
    Cada restricción se define por su nivel, las variables que involucra, un predicado
    para su evaluación, un peso (para restricciones SOFT), su dureza y metadatos adicionales.

    Attributes:
        level (ConstraintLevel | str): El nivel de la restricción en la jerarquía.
                                       Puede ser un miembro de ConstraintLevel o un string para niveles personalizados.
        predicate (Callable[[Dict[str, Any]], Any]): Función que evalúa la restricción.
                                                    Debe aceptar un diccionario de asignaciones parciales
                                                    y devolver un booleano (satisfecha/violada) o un float
                                                    (grado de violación).
        variables (Tuple[str, ...]): Una tupla de IDs de variables que participan en la restricción.
                                     Se usa una tupla para asegurar inmutabilidad y hashability.
        weight (float): Peso de la restricción. Solo relevante para restricciones SOFT.
                        Un peso mayor indica una mayor penalización por violación.
        hardness (Hardness): La dureza de la restricción (HARD o SOFT).
        metadata (Dict[str, Any]): Diccionario para almacenar información adicional sobre la restricción,
                                   como su nombre, descripción o tipo específico.
        expression (Any): La expresión original de la restricción, útil para serialización
                          o depuración. Puede ser el predicado mismo o una representación simbólica.
    """
    level: ConstraintLevel | str
    predicate: Callable[[Dict[str, Any]], Any]
    variables: Tuple[str, ...] = field(default_factory=tuple)
    weight: float = 1.0
    hardness: Hardness = Hardness.HARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    expression: Any = None

    def evaluate(self, assignment: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evalúa la restricción sobre una asignación parcial de variables.
        Si no todas las variables de la restricción están presentes en la asignación,
        se considera que la restricción no es evaluable y se devuelve (True, 0.0).

        Args:
            assignment (Dict[str, Any]): Un diccionario que mapea IDs de variables a sus valores asignados.

        Returns:
            Tuple[bool, float]: Una tupla que contiene:
                                - `satisfied` (bool): True si la restricción se satisface (o no es evaluable).
                                                      False si la restricción es violada.
                                - `violation_degree` (float): El grado de violación de la restricción.
                                                              0.0 si está satisfecha o no es evaluable.
                                                              Un valor positivo si está violada.

        Raises:
            ValueError: Si el predicado devuelve un tipo de dato no soportado (ni bool ni float/int).
        """
        # Filtrar las variables de la asignación que son relevantes para esta restricción.
        assigned_vars_for_constraint = {var: assignment[var] for var in self.variables if var in assignment}

        # Si no todas las variables de la restricción están asignadas, no se puede evaluar completamente.
        # Se asume que no hay violación por el momento (comportamiento para asignaciones parciales).
        if len(assigned_vars_for_constraint) < len(self.variables):
            return True, 0.0

        try:
            # El predicado debe aceptar un diccionario de asignaciones y devolver un booleano o un float.
            result = self.predicate(assigned_vars_for_constraint)

            if isinstance(result, bool):
                # Si el predicado devuelve un booleano: True significa satisfecha, False significa violada.
                return result, 0.0 if result else 1.0
            elif isinstance(result, (int, float)):
                # Si el predicado devuelve un número: 0.0 significa satisfecha, cualquier otro valor es el grado de violación.
                violation = float(result)
                return violation == 0.0, violation
            else:
                raise ValueError(f"El predicado debe devolver un booleano o un número (int/float), pero se obtuvo {type(result)}.")

        except Exception as e:
            # En caso de error durante la evaluación del predicado, se considera la restricción como violada.
            print(f"Advertencia: Error al evaluar la restricción {self.metadata.get('name', 'sin nombre')}: {e}")
            return False, 1.0

    def __repr__(self) -> str:
        """
        Representación en cadena de la restricción para depuración y logging.
        """
        level_name = self.level.name if isinstance(self.level, ConstraintLevel) else self.level
        return (f"Constraint(level={level_name}, "
                f"vars={', '.join(self.variables)}, "
                f"weight={self.weight}, "
                f"hardness={self.hardness.value})")


class ConstraintHierarchy(ConstraintHierarchyAPI):
    """
    Gestiona una colección de restricciones organizadas en una jerarquía de niveles.
    Esta estructura es fundamental para el proceso de hacificación y la evaluación
    multinivel de la coherencia en LatticeWeaver.

    La jerarquía permite aplicar diferentes estrategias de búsqueda y optimización
    basadas en la importancia o el alcance de las restricciones.
    """

    def __init__(self):
        """
        Inicializa una nueva instancia de ConstraintHierarchy con niveles de restricción predefinidos.
        """
        self.constraints: Dict[ConstraintLevel | str, List[Constraint]] = {}
        for level in ConstraintLevel:
            self.constraints[level] = []

    def add_level(self, level_name: str):
        """
        Añade un nuevo nivel de restricción a la jerarquía dinámicamente.
        Si el nivel ya existe, se emite una advertencia.

        Args:
            level_name (str): El nombre del nuevo nivel a añadir.
        """
        if level_name not in self.constraints:
            self.constraints[level_name] = []
        else:
            print(f"Advertencia: El nivel \'{level_name}\' ya existe en la jerarquía.")

    def add_constraint(self, constraint: Constraint):
        """
        Añade una restricción a su nivel correspondiente dentro de la jerarquía.
        Si el nivel de la restricción no existe, se crea automáticamente.

        Args:
            constraint (Constraint): La instancia de Constraint a añadir.
        """
        if constraint.level not in self.constraints:
            self.add_level(constraint.level)
        self.constraints[constraint.level].append(constraint)

    def evaluate_solution(self, solution: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evalúa una solución completa (o parcial) contra todas las restricciones en la jerarquía.
        Calcula si todas las restricciones HARD están satisfechas y la energía total de las SOFT.

        Args:
            solution (Dict[str, Any]): Un diccionario que representa la asignación de variables.

        Returns:
            Tuple[bool, float]: Una tupla que contiene:
                                - `all_hard_satisfied` (bool): True si todas las restricciones HARD están satisfechas.
                                                               False si al menos una HARD está violada.
                                - `total_energy` (float): La suma ponderada de las violaciones de las restricciones SOFT.
                                                          Si `all_hard_satisfied` es False, este valor suele ser 0.0
                                                          o ignorado, dependiendo de la interpretación del solver.
        """
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

    def get_constraints_by_level(self, level: ConstraintLevel | str) -> List[Constraint]:
        """
        Recupera todas las restricciones asociadas a un nivel específico.

        Args:
            level (ConstraintLevel | str): El nivel de las restricciones a recuperar.

        Returns:
            List[Constraint]: Una lista de objetos Constraint para el nivel especificado.
                              Devuelve una lista vacía si el nivel no existe.
        """
        return self.constraints.get(level, [])

    def get_all_constraints(self) -> Dict[str, List[Constraint]]:
        """
        Recupera todas las restricciones organizadas por el nombre de su nivel.

        Returns:
            Dict[str, List[Constraint]]: Un diccionario donde las claves son los nombres de los niveles
                                        (como strings) y los valores son listas de Constraint.
        """
        all_constraints = {}
        for level_enum in self.constraints.keys():
            level_name = level_enum.value if isinstance(level_enum, ConstraintLevel) else level_enum
            all_constraints[level_name] = self.get_constraints_by_level(level_enum)
        return all_constraints

    def to_json(self) -> Dict[str, Any]:
        """
        Serializa la jerarquía de restricciones a un formato JSON.
        Nota: Los predicados (Callables) no se pueden serializar directamente a JSON.
        Aquí se guarda una representación de su expresión, que debería ser reconstruible
        por un mecanismo externo (e.g., un registro de funciones o un DSL).

        Returns:
            Dict[str, Any]: Un diccionario que representa la jerarquía de restricciones serializada.
        """
        serialized_constraints = {
            (level.value if isinstance(level, ConstraintLevel) else level): [
                {
                    "variables": c.variables,
                    "expression": str(c.expression),  # Placeholder: idealmente, se serializaría de forma más robusta
                    "weight": c.weight,
                    "hardness": c.hardness.value,
                    "metadata": c.metadata,
                    "level": c.level.value if isinstance(c.level, ConstraintLevel) else c.level
                } for c in self.constraints[level]
            ] for level in self.constraints.keys()
        }
        return serialized_constraints

    def from_json(self, json_data: Dict[str, Any]) -> None:
        """
        Deserializa una jerarquía de restricciones desde un formato JSON.
        Nota: Los predicados (Callables) no se pueden reconstruir de forma segura desde strings.
        Esta implementación usa un placeholder (lambda x: True) para los predicados.
        En una aplicación real, se necesitaría un mecanismo para mapear las expresiones
        serializadas a funciones ejecutables de forma segura.

        Args:
            json_data (Dict[str, Any]): El diccionario JSON que contiene la jerarquía serializada.
        """
        print("Advertencia: Deserialización placeholder. Los predicados no se reconstruyen completamente desde strings y se usan placeholders.")
        self.constraints = {}
        for level_str, constraints_list in json_data.items():
            level_key = ConstraintLevel[level_str] if level_str in ConstraintLevel.__members__ else level_str
            self.constraints[level_key] = [
                Constraint(
                    level=level_key,
                    variables=tuple(c_data["variables"]),
                    predicate=lambda x: True,  # Placeholder para el predicado
                    weight=c_data["weight"],
                    hardness=Hardness[c_data["hardness"].upper()],
                    metadata=c_data["metadata"],
                    expression=c_data["expression"]
                ) for c_data in constraints_list
            ]

    def add_local_constraint(self,
                             var1: str,
                             var2: str,
                             predicate: Callable[[Dict[str, Any]], bool],
                             weight: float = 1.0,
                             hardness: str = Hardness.HARD,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción local (binaria) a la jerarquía.

        Args:
            var1 (str): La primera variable involucrada.
            var2 (str): La segunda variable involucrada.
            predicate (Callable[[Dict[str, Any]], bool]): La función de evaluación de la restricción.
            weight (float): Peso de la restricción (para SOFT).
            hardness (str): Dureza de la restricción ('HARD' o 'SOFT').
            metadata (Optional[Dict[str, Any]]): Metadatos adicionales.
        """
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=(var1, var2),
            predicate=predicate,
            weight=weight,
            hardness=Hardness[hardness.upper()] if isinstance(hardness, str) else hardness,
            metadata=metadata or {},
            expression=predicate
        )
        self.add_constraint(constraint)

    def add_unary_constraint(self,
                             variable: str,
                             predicate: Callable[[Dict[str, Any]], bool],
                             weight: float = 1.0,
                             hardness: str = Hardness.HARD,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción unaria a la jerarquía.

        Args:
            variable (str): La variable involucrada.
            predicate (Callable[[Dict[str, Any]], bool]): La función de evaluación de la restricción.
            weight (float): Peso de la restricción (para SOFT).
            hardness (str): Dureza de la restricción ('HARD' o 'SOFT').
            metadata (Optional[Dict[str, Any]]): Metadatos adicionales.
        """
        constraint = Constraint(
            level=ConstraintLevel.LOCAL,
            variables=(variable,),
            predicate=predicate,
            weight=weight,
            hardness=Hardness[hardness.upper()] if isinstance(hardness, str) else hardness,
            metadata=metadata or {},
            expression=predicate
        )
        self.add_constraint(constraint)

    def add_pattern_constraint(self,
                               variables: List[str],
                               predicate: Callable[[Dict[str, Any]], bool],
                               pattern_type: str = "custom",
                               weight: float = 2.0,
                               hardness: str = Hardness.HARD,
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción de patrón a la jerarquía.

        Args:
            variables (List[str]): Lista de variables que forman el patrón.
            predicate (Callable[[Dict[str, Any]], bool]): La función de evaluación de la restricción.
            pattern_type (str): Tipo de patrón (e.g., 'all_different', 'sum', 'custom').
            weight (float): Peso de la restricción (para SOFT).
            hardness (str): Dureza de la restricción ('HARD' o 'SOFT').
            metadata (Optional[Dict[str, Any]]): Metadatos adicionales.
        """
        meta = metadata or {}
        meta["pattern_type"] = pattern_type
        constraint = Constraint(
            level=ConstraintLevel.PATTERN,
            variables=tuple(variables),
            predicate=predicate,
            weight=weight,
            hardness=Hardness[hardness.upper()] if isinstance(hardness, str) else hardness,
            metadata=meta,
            expression=predicate
        )
        self.add_constraint(constraint)

    def add_global_constraint(self,
                              variables: List[str],
                              predicate: Callable[[Dict[str, Any]], bool],
                              weight: float = 5.0,
                              hardness: str = Hardness.HARD,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Añade una restricción global a la jerarquía.

        Args:
            variables (List[str]): Lista de variables que forman el alcance global.
            predicate (Callable[[Dict[str, Any]], bool]): La función de evaluación de la restricción.
            weight (float): Peso de la restricción (para SOFT).
            hardness (str): Dureza de la restricción ('HARD' o 'SOFT').
            metadata (Optional[Dict[str, Any]]): Metadatos adicionales.
        """
        constraint = Constraint(
            level=ConstraintLevel.GLOBAL,
            variables=tuple(variables),
            predicate=predicate,
            weight=weight,
            hardness=Hardness[hardness.upper()] if isinstance(hardness, str) else hardness,
            metadata=metadata or {},
            expression=predicate
        )
        self.add_constraint(constraint)

    def get_constraints_by_level_name(self, level_name: str) -> List[Constraint]:
        """
        Recupera restricciones por el nombre de su nivel (string).

        Args:
            level_name (str): El nombre del nivel (e.g., 'LOCAL', 'PATTERN', 'GLOBAL' o personalizado).

        Returns:
            List[Constraint]: Lista de restricciones para el nivel especificado.
        """
        level_key = ConstraintLevel[level_name] if level_name in ConstraintLevel.__members__ else level_name
        return self.constraints.get(level_key, [])

    def set_constraints_for_level(self, level_name: str, constraints: List[Constraint]):
        """
        Establece la lista completa de restricciones para un nivel dado.

        Args:
            level_name (str): El nombre del nivel.
            constraints (List[Constraint]): La nueva lista de restricciones para ese nivel.
        """
        level_key = ConstraintLevel[level_name] if level_name in ConstraintLevel.__members__ else level_name
        self.constraints[level_key] = constraints

    def get_global_constraints(self) -> List[Constraint]:
        """
        Recupera todas las restricciones de nivel GLOBAL.

        Returns:
            List[Constraint]: Lista de restricciones globales.
        """
        return self.constraints.get(ConstraintLevel.GLOBAL, [])

    def get_local_constraints(self) -> List[Constraint]:
        """
        Recupera todas las restricciones de nivel LOCAL.

        Returns:
            List[Constraint]: Lista de restricciones locales.
        """
        return self.constraints.get(ConstraintLevel.LOCAL, [])

    def get_pattern_constraints(self) -> List[Constraint]:
        """
        Recupera todas las restricciones de nivel PATTERN.

        Returns:
            List[Constraint]: Lista de restricciones de patrón.
        """
        return self.constraints.get(ConstraintLevel.PATTERN, [])

