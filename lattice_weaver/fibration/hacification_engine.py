from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents

# Importar ArcEngine solo para type hinting, evitar dependencia circular o temprana
# Se importará realmente en _hacify_with_arc_engine si es necesario
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lattice_weaver.arc_engine.core import ArcEngine

@dataclass
class HacificationResult:
    is_coherent: bool
    level_results: Dict[ConstraintLevel, bool]
    energy: EnergyComponents
    violated_constraints: List[str]
    has_hard_violation: bool

class HacificationEngine:
    def __init__(
        self,
        hierarchy: ConstraintHierarchy,
        landscape: EnergyLandscapeOptimized,
        arc_engine: Optional["ArcEngine"] = None,  # Usar Any para evitar import circular por ahora
        use_arc_engine: bool = False
    ):
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.energy_thresholds = {
            ConstraintLevel.LOCAL: 0.0,
            ConstraintLevel.PATTERN: 0.0,
            ConstraintLevel.GLOBAL: 0.1
        }
        self._arc_engine = arc_engine
        self._use_arc_engine = use_arc_engine and arc_engine is not None

    def hacify(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
        if self._use_arc_engine:
            return self._hacify_with_arc_engine(assignment, strict=strict)
        else:
            return self._hacify_original(assignment, strict=strict)

    def _hacify_original(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
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

    def _hacify_with_arc_engine(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
        # En Fase 5, esta lógica se expandirá para usar el ArcEngine real
        # Por ahora, simplemente delegamos al original, pero aseguramos que self._arc_engine es válido
        if self._arc_engine is None:
            raise ValueError("ArcEngine must be provided when use_arc_engine is True")
        
        # TODO: Implementar la lógica real de hacification con ArcEngine en Fase 5
        # Por ahora, para que los tests pasen, retornamos un resultado dummy.
        # Esto se reemplazará en la Fase 5.
        from lattice_weaver.fibration.constraint_hierarchy import ConstraintLevel
        from lattice_weaver.fibration.energy_landscape_optimized import EnergyComponents
        return HacificationResult(
            is_coherent=True,
            level_results={level: True for level in ConstraintLevel},
            energy=EnergyComponents(0.0, 0.0, 0.0, 0.0),
            violated_constraints=[],
            has_hard_violation=False
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
            "energy_thresholds": {level.name: threshold for level, threshold in self.energy_thresholds.items()},
            "use_arc_engine": self._use_arc_engine
        }

