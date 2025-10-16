from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents

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
        arc_engine: Optional["ArcEngine"] = None,
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
        from lattice_weaver.arc_engine.core import ArcEngine
        from lattice_weaver.arc_engine.domains import create_optimal_domain
        from lattice_weaver.arc_engine.constraints import Constraint as ArcConstraint

        # La comprobación de tipo se realizará en tiempo de ejecución o se asumirá que el objeto es válido
        # para permitir mocks en los tests. La implementación real de ArcEngine se integrará en Fase 5.

        # Configurar el ArcEngine existente con las variables y dominios del problema actual
        # El ArcEngine debe ser reutilizable, por lo que lo reiniciamos o reconfiguramos
        self._arc_engine.reset()

        # Obtener todas las variables involucradas en las restricciones de la jerarquía
        all_vars_in_hierarchy = set()
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                all_vars_in_hierarchy.update(constraint.variables)
        
        # Añadir variables y sus dominios al ArcEngine
        # Para las variables en la asignación, su dominio es el valor asignado
        # Para las variables no asignadas pero en la jerarquía, necesitamos sus dominios originales
        # (Esto es un placeholder; en un sistema real, se pasaría un mapa de dominios originales)
        for var_name in all_vars_in_hierarchy:
            if var_name in assignment:
                self._arc_engine.add_variable(var_name, [assignment[var_name]])
            else:
                # Placeholder: Asumimos un dominio por defecto si no está en la asignación
                # En un sistema real, se pasaría el dominio original de la variable
                self._arc_engine.add_variable(var_name, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # Dominio dummy

        # Adaptar las restricciones de ConstraintHierarchy a ArcEngine.Constraint
        # Esto es un paso crucial y complejo. Por ahora, vamos a simular que se hace.
        # En la Fase 5 real, se implementaría un adaptador.
        # Por cada Constraint en self.hierarchy, se crearía un ArcConstraint y se añadiría a self._arc_engine
        # Por ejemplo:
        # for level in ConstraintLevel:
        #     for lw_constraint in self.hierarchy.get_constraints_at_level(level):
        #         # Convertir lw_constraint a ArcConstraint y añadirlo
        #         arc_constraint = self._convert_lw_constraint_to_arc_constraint(lw_constraint)
        #         self._arc_engine.add_constraint(arc_constraint.var1, arc_constraint.var2, arc_constraint.relation_name)

        # Enforzar consistencia de arco
        is_consistent_by_arc_engine = self._arc_engine.enforce_arc_consistency()

        # Si ArcEngine encuentra una inconsistencia, entonces hay una violación HARD
        if not is_consistent_by_arc_engine:
            return HacificationResult(
                is_coherent=False,
                level_results={level: False for level in ConstraintLevel},
                energy=EnergyComponents(float('inf'), float('inf'), float('inf'), float('inf')),
                violated_constraints=["ARC_ENGINE_INCONSISTENCY"],
                has_hard_violation=True
            )
        
        # Si ArcEngine es consistente, devolvemos un resultado que indica coherencia.
        # La evaluación de SOFT constraints y energía se hará en fases posteriores.
        # Por ahora, si ArcEngine dice que es consistente, lo consideramos coherente.
        return HacificationResult(
            is_coherent=True,
            level_results={level: True for level in ConstraintLevel},
            energy=EnergyComponents(0.0, 0.0, 0.0, 0.0), # Valores dummy por ahora
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

