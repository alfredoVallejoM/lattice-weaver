from typing import Dict, Any, List, Tuple, Callable
import random

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized
from .landscape_modulator import LandscapeModulator

class AutoperturbationSystem:
    """
    Sistema de autoperturbación para Fibration Flow.
    Permite aplicar perturbaciones controladas al EnergyLandscape para extraer información
    relevante y modelar de forma más activa el paisaje de energía.
    """

    def __init__(self, 
                 hierarchy: ConstraintHierarchy, 
                 landscape: EnergyLandscapeOptimized, 
                 modulator: LandscapeModulator,
                 variables_domains: Dict[str, List[Any]]):
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.modulator = modulator
        self.variables_domains = variables_domains

    def apply_perturbation(self, 
                           current_assignment: Dict[str, Any],
                           level: ConstraintLevel = ConstraintLevel.LOCAL,
                           perturbation_type: str = "random_variable_change",
                           magnitude: float = 0.1) -> Dict[str, Any]:
        """
        Aplica una perturbación al estado actual de la asignación.
        La granularidad de la perturbación puede variar.
        """
        perturbed_assignment = current_assignment.copy()

        if perturbation_type == "random_variable_change":
            # Perturbación a nivel de variable individual
            if not self.variables_domains: # No hay variables para perturbar
                return perturbed_assignment
            
            var_to_perturb = random.choice(list(self.variables_domains.keys()))
            if self.variables_domains[var_to_perturb]:
                new_value = random.choice(self.variables_domains[var_to_perturb])
                perturbed_assignment[var_to_perturb] = new_value

        elif perturbation_type == "soft_constraint_weight_change":
            # Perturbación a nivel de restricciones SOFT (cambio de peso)
            soft_constraints = self.hierarchy.get_constraints_at_level(level, hardness=Hardness.SOFT)
            if soft_constraints:
                constraint_to_perturb = random.choice(soft_constraints)
                original_weight = constraint_to_perturb.weight
                new_weight = max(0.0, original_weight + (random.uniform(-1, 1) * magnitude))
                constraint_to_perturb.weight = new_weight # Modifica el peso directamente en la jerarquía
                print(f"Perturbación: Peso de restricción SOFT {constraint_to_perturb.metadata.get('name', 'unnamed')} cambiado de {original_weight:.2f} a {new_weight:.2f}")

        elif perturbation_type == "remove_hard_constraint":
            # Perturbación a nivel de restricciones HARD (eliminación temporal)
            # Esto es más drástico y debe usarse con cautela.
            hard_constraints = self.hierarchy.get_constraints_at_level(level, hardness=Hardness.HARD)
            if hard_constraints:
                constraint_to_remove = random.choice(hard_constraints)
                # Para una eliminación temporal, necesitaríamos un mecanismo para reintroducirla.
                # Por ahora, solo se registra la acción.
                print(f"Perturbación: Restricción HARD {constraint_to_remove.metadata.get('name', 'unnamed')} temporalmente ignorada.")
                # En una implementación real, se necesitaría una forma de deshabilitar/habilitar la restricción
                # o una copia de la jerarquía para la perturbación.

        # Otros tipos de perturbación podrían incluir:
        # - "swap_variable_values": Intercambiar valores entre dos variables.
        # - "subproblem_reinitialization": Reinicializar un subconjunto de variables.
        # - "modulator_parameter_change": Ajustar parámetros del LandscapeModulator.

        return perturbed_assignment

    def observe_and_learn(self, 
                          original_assignment: Dict[str, Any],
                          perturbed_assignment: Dict[str, Any],
                          original_energy: float,
                          perturbed_energy: float) -> Dict[str, Any]:
        """
        Observa el efecto de la perturbación y extrae información.
        Esta información puede ser usada para ajustar el LandscapeModulator o para futuras IAs.
        """
        feedback = {}
        energy_change = perturbed_energy - original_energy
        feedback["energy_change"] = energy_change
        feedback["perturbation_successful"] = energy_change < 0 # Si la energía disminuyó

        # Aquí se podría implementar lógica para:
        # - Registrar qué tipo de perturbación fue efectiva en qué contextos.
        # - Ajustar dinámicamente los parámetros del LandscapeModulator.
        # - Generar datos de entrenamiento para un modelo de ML que prediga perturbaciones óptimas.

        if feedback["perturbation_successful"]:
            self.modulator.adapt_to_success(original_assignment, perturbed_assignment, energy_change)
        else:
            self.modulator.adapt_to_failure(original_assignment, perturbed_assignment, energy_change)

        return feedback

    def get_potential_actions(self) -> List[str]:
        """
        Devuelve una lista de las acciones de perturbación que el sistema puede llevar a cabo.
        """
        return [
            "random_variable_change",
            "soft_constraint_weight_change",
            "remove_hard_constraint",
            # Añadir aquí otros tipos de perturbación a medida que se implementen
        ]

    def get_perturbation_levels(self) -> List[ConstraintLevel]:
        """
        Devuelve los niveles de granularidad a los que se pueden aplicar perturbaciones.
        """
        return list(ConstraintLevel)

