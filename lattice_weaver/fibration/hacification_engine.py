from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents

@dataclass
class HacificationResult:
    is_coherent: bool
    level_results: Dict[ConstraintLevel, bool]
    energy: EnergyComponents
    violated_constraints: List[str]

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
        overall_is_coherent = True

        for level in ConstraintLevel:
            level_energy = getattr(energy_components, f"{level.name.lower()}_energy")
            threshold = self.energy_thresholds.get(level, 0.0)
            level_is_coherent = True

            hard_violation_in_level = False
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.HARD:
                    satisfied, violation = constraint.evaluate(assignment)
                    if not satisfied or violation > 0:
                        hard_violation_in_level = True
                        all_violated_constraints.append(f"{level.name}:{constraint.metadata.get('name', 'unnamed')}")
                        break

            if hard_violation_in_level:
                level_is_coherent = False
            elif level_energy > threshold:
                level_is_coherent = False
                # Solo añadir violaciones SOFT a violated_constraints si no hubo violaciones HARD
                # o si no estamos en modo estricto (donde las SOFT pueden ser la causa de incoherencia)
                if not hard_violation_in_level or not strict:
                    for constraint in self.hierarchy.get_constraints_at_level(level):
                        if constraint.hardness == Hardness.SOFT:
                            satisfied, violation = constraint.evaluate(assignment)
                            if not satisfied or violation > 0:
                                all_violated_constraints.append(f"{level.name}:{constraint.metadata.get('name', 'unnamed')}")

            level_results[level] = level_is_coherent
            if not level_is_coherent:
                overall_is_coherent = False

        return HacificationResult(
            is_coherent=overall_is_coherent,
            level_results=level_results,
            energy=energy_components,
            violated_constraints=list(set(all_violated_constraints))
        )

    def filter_coherent_extensions(self, base_assignment: Dict[str, Any], variable: str, domain: List[Any], strict: bool = True) -> List[Any]:
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

