from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents

@dataclass
class HacificationResult:
    is_coherent: bool
    level_results: Dict[ConstraintLevel, bool]
    energy: EnergyComponents
    violated_constraints: List[str]
    has_hard_violation: bool

class HacificationEngine:
    def __init__(self, hierarchy: ConstraintHierarchy, landscape: EnergyLandscapeOptimized):
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.energy_thresholds = {
            ConstraintLevel.LOCAL: 0.0,
            ConstraintLevel.PATTERN: 0.0,
            ConstraintLevel.GLOBAL: 0.1
        }

    def hacify(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
        energy_components = self.landscape.compute_energy(assignment)
        level_results = {}
        all_violated_constraints = []
        has_hard_violation = False

        for level in ConstraintLevel:
            level_energy = getattr(energy_components, f"{level.name.lower()}_energy")
            threshold = self.energy_thresholds.get(level, 0.0)
            level_has_hard_violation = False

            for constraint in self.hierarchy.get_constraints_at_level(level):
                satisfied, violation = constraint.evaluate(assignment)
                if not satisfied or violation > 0:
                    constraint_name = constraint.metadata.get("name", "unnamed")
                    all_violated_constraints.append(f"{level.name}:{constraint_name}")
                    if constraint.hardness == Hardness.HARD:
                        level_has_hard_violation = True
                        has_hard_violation = True
            
            level_results[level] = not level_has_hard_violation and level_energy <= threshold

        if strict:
            is_coherent = not has_hard_violation
        else:
            is_coherent = not has_hard_violation and all(level_results.values())

        return HacificationResult(
            is_coherent=is_coherent,
            level_results=level_results,
            energy=energy_components,
            violated_constraints=list(set(all_violated_constraints)),
            has_hard_violation=has_hard_violation
        )

    def filter_coherent_extensions(self, base_assignment: Dict[str, Any], variable: str, domain: List[Any], strict: bool = True) -> List[Any]:
        """
        Filtra valores del dominio que son coherentes con la asignación base.
        
        Args:
            base_assignment: Asignación parcial de variables
            variable: Variable cuyo dominio se está filtrando
            domain: Dominio de valores posibles para la variable
            strict: Si True, solo considera violaciones HARD; si False, considera también SOFT
            
        Returns:
            Lista de valores del dominio que producen asignaciones coherentes
        """
        coherent_values = []
        for value in domain:
            temp_assignment = base_assignment.copy()
            temp_assignment[variable] = value
            h_result = self.hacify(temp_assignment, strict=strict)
            if h_result.is_coherent:
                coherent_values.append(value)
        return coherent_values



    def get_statistics(self) -> Dict:
        return {
            "energy_thresholds": {level.name: threshold for level, threshold in self.energy_thresholds.items()}
        }

