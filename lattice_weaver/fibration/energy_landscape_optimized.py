import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import OrderedDict
from .constraint_hierarchy import ConstraintHierarchy, Constraint, ConstraintLevel, Hardness
from .energy_landscape_api import EnergyLandscapeAPI

class EnergyLandscapeOptimized(EnergyLandscapeAPI):
    """
    Implementación optimizada del paisaje de energía para el espacio de búsqueda de LatticeWeaver.
    Calcula la energía total de una asignación de variables, considerando restricciones duras (HARD)
    y blandas (SOFT) a través de una jerarquía de restricciones. Utiliza un sistema de caché
    para optimizar los cálculos repetitivos y permite el cálculo incremental de la energía.

    Attributes:
        hierarchy (ConstraintHierarchy): La jerarquía de restricciones que define el problema.
        level_weights (Dict[ConstraintLevel, float]): Pesos aplicados a cada nivel de restricción
                                                     para modular su impacto en la energía total.
        _energy_cache (OrderedDict): Caché para almacenar resultados de cálculos de energía.
        cache_max_size (int): Tamaño máximo del caché.
        _var_to_constraints (Dict[str, List[Constraint]]): Índice que mapea variables a las restricciones
                                                           en las que participan, para cálculos incrementales eficientes.
        cache_hits (int): Contador de aciertos en el caché.
        cache_misses (int): Contador de fallos en el caché.
        incremental_calculations (int): Contador de cálculos de energía incrementales.
        full_calculations (int): Contador de cálculos de energía completos.
    """
    
    def __init__(self, hierarchy: ConstraintHierarchy):
        """
        Inicializa el paisaje de energía optimizado.

        Args:
            hierarchy (ConstraintHierarchy): La jerarquía de restricciones a utilizar.
        """
        self.hierarchy = hierarchy
        self.level_weights = {
            ConstraintLevel.LOCAL: 1.0,
            ConstraintLevel.PATTERN: 1.0,
            ConstraintLevel.GLOBAL: 1.0
        }
        self._energy_cache: OrderedDict[str, Tuple[bool, float, float, float, float]] = OrderedDict()
        self.cache_max_size = 100000
        self._var_to_constraints: Dict[str, List[Constraint]] = {}
        self._build_constraint_index()
        self.cache_hits = 0
        self.cache_misses = 0
        self.incremental_calculations = 0
        self.full_calculations = 0

    def _build_constraint_index(self):
        """
        Construye un índice que mapea cada variable a la lista de restricciones en las que participa.
        Esto es crucial para la eficiencia de los cálculos de energía incrementales, ya que permite
        identificar rápidamente las restricciones afectadas por el cambio de una variable.
        """
        self._var_to_constraints.clear()
        for level_name, constraints_list in self.hierarchy.get_all_constraints().items():
            for constraint in constraints_list:
                for var in constraint.variables:
                    if var not in self._var_to_constraints:
                        self._var_to_constraints[var] = []
                    self._var_to_constraints[var].append(constraint)

    def compute_energy(self, assignment: Dict[str, Any], use_cache: bool = True) -> Tuple[bool, float, float, float, float]:
        """
        Calcula la energía total de una asignación de variables, así como el desglose por niveles.
        Utiliza un caché para evitar recálculos si la asignación ya ha sido evaluada.

        Args:
            assignment (Dict[str, Any]): La asignación de variables a evaluar.
            use_cache (bool): Si es True, intenta usar el caché y almacena el resultado.

        Returns:
            Tuple[bool, float, float, float, float]: Una tupla que contiene:
                - `all_hard_satisfied` (bool): True si todas las restricciones HARD están satisfechas.
                - `total_energy` (float): La suma ponderada de las violaciones de las restricciones SOFT.
                - `local_energy` (float): La energía total de las restricciones LOCAL SOFT.
                - `pattern_energy` (float): La energía total de las restricciones PATTERN SOFT.
                - `global_energy` (float): La energía total de las restricciones GLOBAL SOFT.
        """
        cache_key = self._assignment_to_key(assignment)
        if use_cache and cache_key in self._energy_cache:
            self.cache_hits += 1
            return self._energy_cache[cache_key] # Cache now stores (all_hard_satisfied, total_energy, local_energy, pattern_energy, global_energy)

        self.cache_misses += 1
        self.full_calculations += 1

        local_energy = 0.0
        pattern_energy = 0.0
        global_energy = 0.0
        all_hard_satisfied = True

        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_by_level(level):
                satisfied, violation = constraint.evaluate(assignment)
                if constraint.hardness == Hardness.HARD:
                    if not satisfied:
                        all_hard_satisfied = False
                else: # Soft constraint
                    if not satisfied:
                        weighted_violation = violation * constraint.weight * self.level_weights[level]
                        if level == ConstraintLevel.LOCAL:
                            local_energy += weighted_violation
                        elif level == ConstraintLevel.PATTERN:
                            pattern_energy += weighted_violation
                        elif level == ConstraintLevel.GLOBAL:
                            global_energy += weighted_violation

        total_energy = local_energy + pattern_energy + global_energy

        if use_cache:
            self._energy_cache[cache_key] = (all_hard_satisfied, total_energy, local_energy, pattern_energy, global_energy)

        return all_hard_satisfied, total_energy, local_energy, pattern_energy, global_energy

    def compute_energy_incremental(self, base_assignment: Dict[str, Any], base_energy: Tuple[bool, float, float, float, float], new_var: str, new_value: Any) -> Tuple[bool, float, float, float, float]:
        """
        Calcula la energía de una nueva asignación de forma incremental, basándose en una asignación
        previa y el cambio de una única variable. Esto es más eficiente que un recálculo completo
        cuando solo una variable ha cambiado.

        Args:
            base_assignment (Dict[str, Any]): La asignación de variables antes del cambio.
            base_energy (Tuple[bool, float, float, float, float]): La energía calculada para `base_assignment`.
            new_var (str): La variable que ha cambiado.
            new_value (Any): El nuevo valor asignado a `new_var`.

        Returns:
            Tuple[bool, float, float, float, float]: La nueva energía y el estado de satisfacción de las
                                                     restricciones HARD para la `new_assignment`.
        """
        self.incremental_calculations += 1
        new_assignment = base_assignment.copy()
        new_assignment[new_var] = new_value

        if new_var not in self._var_to_constraints:
            # Si la variable no afecta a ninguna restricción, la energía no cambia.
            return base_energy

        base_all_hard_satisfied, base_total_energy, base_local_energy, base_pattern_energy, base_global_energy = base_energy

        delta_local = 0.0
        delta_pattern = 0.0
        delta_global = 0.0
        new_all_hard_satisfied = base_all_hard_satisfied

        affected_constraints = self._var_to_constraints[new_var]

        for constraint in affected_constraints:
            old_satisfied, old_violation = constraint.evaluate(base_assignment)
            new_satisfied, new_violation = constraint.evaluate(new_assignment)

            if constraint.hardness == Hardness.HARD:
                # Si una hard constraint pasa de satisfecha a violada, la nueva asignación no satisface todas las hard constraints.
                if old_satisfied and not new_satisfied:
                    new_all_hard_satisfied = False
                # Si una hard constraint violada se satisface, necesitamos re-evaluar todas las hard constraints
                # para determinar si *todas* están ahora satisfechas. Esto es para ser robustos.
                elif not old_satisfied and new_satisfied and not self._check_other_hard_violations(new_assignment, constraint):
                    # Si esta era la única hard constraint violada y ahora está satisfecha
                    # (requiere una re-evaluación más profunda o un seguimiento de todas las hard violations)
                    # Por simplicidad, si una hard constraint se satisface, no asumimos que todas lo están.
                    # Solo podemos decir que si una hard constraint se viola, entonces new_all_hard_satisfied es False.
                    pass # No podemos asumir True sin re-evaluar todas las hard constraints
            else: # Soft constraint
                weighted_old_violation = old_violation * constraint.weight * self.level_weights[constraint.level]
                weighted_new_violation = new_violation * constraint.weight * self.level_weights[constraint.level]
                delta_weighted_violation = weighted_new_violation - weighted_old_violation

                if constraint.level == ConstraintLevel.LOCAL:
                    delta_local += delta_weighted_violation
                elif constraint.level == ConstraintLevel.PATTERN:
                    delta_pattern += delta_weighted_violation
                elif constraint.level == ConstraintLevel.GLOBAL:
                    delta_global += delta_weighted_violation

        new_local_energy = base_local_energy + delta_local
        new_pattern_energy = base_pattern_energy + delta_pattern
        new_global_energy = base_global_energy + delta_global
        new_total_energy = new_local_energy + new_pattern_energy + new_global_energy

        # Si new_all_hard_satisfied se volvió False, se mantiene False.
        # Si era True y ninguna hard constraint se violó en este paso, se mantiene True.
        # Si una hard constraint se satisfizo, necesitamos re-evaluar todas las hard constraints para saber si todas están satisfechas.
        # Para ser robustos, re-evaluamos todas las hard constraints si new_all_hard_satisfied podría cambiar de False a True.
        if not base_all_hard_satisfied and new_all_hard_satisfied: # Si antes no estaban todas satisfechas y ahora podrían estarlo
            new_all_hard_satisfied, _ = self.hierarchy.evaluate_solution(new_assignment)

        cache_key = self._assignment_to_key(new_assignment)
        self._energy_cache[cache_key] = (new_all_hard_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy)

        return new_all_hard_satisfied, new_total_energy, new_local_energy, new_pattern_energy, new_global_energy

    def _check_other_hard_violations(self, assignment: Dict[str, Any], current_constraint: Constraint) -> bool:
        """
        Verifica si hay otras hard constraints violadas en la asignación, excluyendo la actual.
        Esto es útil para determinar si una hard constraint que acaba de ser satisfecha
        era la única violada, lo que permitiría que `all_hard_satisfied` volviera a ser True.

        Args:
            assignment (Dict[str, Any]): La asignación de variables actual.
            current_constraint (Constraint): La restricción que se acaba de evaluar y se quiere excluir.

        Returns:
            bool: True si hay otras hard constraints violadas, False en caso contrario.
        """
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_by_level(level):
                if constraint.hardness == Hardness.HARD and constraint != current_constraint:
                    satisfied, _ = constraint.evaluate(assignment)
                    if not satisfied:
                        return True
        return False

    def compute_energy_gradient_optimized(self, assignment: Dict[str, Any], base_energy: Tuple[bool, float, float, float, float], variable: str, domain: List[Any]) -> Dict[Any, float]:
        """
        Calcula el gradiente de energía para una variable dada, evaluando la energía
        resultante de asignar cada valor posible del dominio a esa variable.

        Args:
            assignment (Dict[str, Any]): La asignación base de variables.
            base_energy (Tuple[bool, float, float, float, float]): La energía de la asignación base.
            variable (str): La variable para la cual se calculará el gradiente.
            domain (List[Any]): El dominio de valores posibles para la variable.

        Returns:
            Dict[Any, float]: Un diccionario donde las claves son los valores del dominio
                              y los valores son las energías resultantes de asignar cada valor.
        """
        gradient = {}
        for value in domain:
            _, energy, _, _, _ = self.compute_energy_incremental(assignment, base_energy, variable, value)
            gradient[value] = energy
        return gradient

    def _assignment_to_key(self, assignment: Dict[str, Any]) -> str:
        """
        Convierte una asignación de variables en una clave de cadena para el caché.
        Asegura un orden consistente de las variables para que asignaciones idénticas
        produzcan la misma clave.

        Args:
            assignment (Dict[str, Any]): La asignación de variables.

        Returns:
            str: Una representación en cadena de la asignación.
        """
        return "|".join(f"{k}:{assignment[k]}" for k in sorted(assignment.keys()))

    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Devuelve estadísticas sobre el uso del caché.

        Returns:
            Dict[str, int]: Un diccionario con el número de aciertos, fallos, cálculos incrementales,
                            cálculos completos y el tamaño actual del caché.
        """
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "incremental_calculations": self.incremental_calculations,
            "full_calculations": self.full_calculations,
            "cache_size": len(self._energy_cache)
        }

    def clear_cache(self):
        """
        Limpia el caché de energía y reinicia los contadores de estadísticas.
        """
        self._energy_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.incremental_calculations = 0
        self.full_calculations = 0

    def to_json(self) -> Dict[str, Any]:
        """
        Serializa el paisaje de energía a un formato JSON.
        Serializa la jerarquía de restricciones y los pesos de nivel.

        Returns:
            Dict[str, Any]: Un diccionario que representa el paisaje de energía serializado.
        """
        return {
            "hierarchy": self.hierarchy.to_json(),
            "level_weights": {level.value if isinstance(level, ConstraintLevel) else level: weight for level, weight in self.level_weights.items()}
        }

    def from_json(self, json_data: Dict[str, Any]) -> None:
        """
        Deserializa el paisaje de energía desde un formato JSON.
        Reconstruye la jerarquía de restricciones y los pesos de nivel.

        Args:
            json_data (Dict[str, Any]): El diccionario JSON que contiene el paisaje de energía serializado.
        """
        self.hierarchy.from_json(json_data["hierarchy"])
        self.level_weights = {
            (ConstraintLevel[level_str] if level_str in ConstraintLevel.__members__ else level_str):
            weight for level_str, weight in json_data["level_weights"].items()
        }
        self._build_constraint_index()

