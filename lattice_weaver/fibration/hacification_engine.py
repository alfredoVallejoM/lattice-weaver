from typing import Dict, List, Tuple, Optional, Any, Callable
import uuid
from dataclasses import dataclass
from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents
from lattice_weaver.arc_engine.core import ArcEngine # Importar ArcEngine
from lattice_weaver.arc_engine.csp_solver import CSPProblem, CSPSolver # Importar CSPProblem y CSPSolver

@dataclass
class HacificationResult:
    is_coherent: bool
    has_hard_violation: bool
    level_results: Dict[ConstraintLevel, bool]
    energy: EnergyComponents
    violated_constraints: List[str]

class HacificationEngine:
    def __init__(self, hierarchy: ConstraintHierarchy, landscape: EnergyLandscapeOptimized, arc_engine: ArcEngine):
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.arc_engine = arc_engine
        self.energy_thresholds = {
            ConstraintLevel.LOCAL: 0.0,
            ConstraintLevel.PATTERN: 0.0,
            ConstraintLevel.GLOBAL: 0.1
        }

    def hacify(self, assignment: Dict[str, Any], strict: bool = True) -> HacificationResult:
        temp_arc_engine = ArcEngine(use_tms=self.arc_engine.use_tms, parallel=self.arc_engine.parallel, parallel_mode=self.arc_engine.parallel_mode)

        for var_name, value in assignment.items():
            temp_arc_engine.add_variable(var_name, [value])
        
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.HARD:
                    # Solo añadir restricciones binarias al ArcEngine
                    if len(constraint.variables) == 2 and constraint.variables[0] in assignment and constraint.variables[1] in assignment:
                        # Generar un nombre de relación único si no está en los metadatos
                        relation_name = f"fibration_constraint_{id(constraint.predicate)}_{id(self)}_{id(temp_arc_engine)}_{uuid.uuid4()}"
                        
                        # Adaptar el predicado para que acepte val1, val2, metadata
                        def adapted_predicate(val1, val2, meta, constraint=constraint):
                            temp_assignment = {constraint.variables[0]: val1, constraint.variables[1]: val2}
                            return constraint.predicate(temp_assignment)

                        # Registrar la función de predicado adaptada como una relación en el ArcEngine temporal
                        temp_arc_engine.register_relation(relation_name, adapted_predicate)
                        temp_arc_engine.add_constraint(constraint.variables[0], constraint.variables[1], relation_name, metadata=constraint.metadata)
                    # Las restricciones unarias y de mayor aridad no se manejan directamente con ArcEngine.add_constraint
                    # Su coherencia se verificará a través de la evaluación de la energía o el hacify posterior.

        is_consistent = temp_arc_engine.enforce_arc_consistency()

        if not is_consistent:
            # Si ArcEngine detecta una inconsistencia, toda la asignación es incoherente
            # y todos los niveles se consideran no coherentes.
            energy_components_for_inconsistency = self.landscape.compute_energy(assignment)
            level_results_for_inconsistency = {
                level: (getattr(energy_components_for_inconsistency, f"{level.name.lower()}_energy") <= self.energy_thresholds.get(level, 0.0)) 
                for level in ConstraintLevel
            }
            
            # Intentar identificar la restricción HARD específica que causó la inconsistencia
            all_violated_constraints_from_arc_engine = []
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.HARD:
                        # Para restricciones binarias, verificar si la asignación actual las viola
                        if len(constraint.variables) == 2 and constraint.variables[0] in assignment and constraint.variables[1] in assignment:
                            val1 = assignment[constraint.variables[0]]
                            val2 = assignment[constraint.variables[1]]
                            if not constraint.predicate({constraint.variables[0]: val1, constraint.variables[1]: val2}):
                                all_violated_constraints_from_arc_engine.append(f"{level.name}:{constraint.metadata.get('name', 'unnamed')}")
                        # Para restricciones unarias, verificar si la asignación actual las viola
                        elif len(constraint.variables) == 1 and constraint.variables[0] in assignment:
                            val = assignment[constraint.variables[0]]
                            if not constraint.predicate({constraint.variables[0]: val}):
                                all_violated_constraints_from_arc_engine.append(f"{level.name}:{constraint.metadata.get('name', 'unnamed')}")

            return HacificationResult(
                is_coherent=False,
                has_hard_violation=True,
                level_results=level_results_for_inconsistency,
                energy=energy_components_for_inconsistency,
                violated_constraints=["Inconsistencia detectada por ArcEngine"] + all_violated_constraints_from_arc_engine
            )
        
        # Si ArcEngine es consistente, entonces procedemos a evaluar la energía
        # para determinar la coherencia de los niveles y la coherencia general.
        
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
            has_hard_violation=False, # Si ArcEngine fue consistente, no hay violación HARD
            level_results=level_results,
            energy=energy_components,
            violated_constraints=list(set(all_violated_constraints))
        )

    def filter_coherent_extensions(self, base_assignment: Dict[str, Any], variable: str, domain: List[Any], strict: bool = True) -> List[Any]:
        temp_arc_engine = ArcEngine(use_tms=self.arc_engine.use_tms, parallel=self.arc_engine.parallel, parallel_mode=self.arc_engine.parallel_mode)

        for var_name, value in base_assignment.items():
            temp_arc_engine.add_variable(var_name, [value])
        
        temp_arc_engine.add_variable(variable, domain)

        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.HARD:
                    # Solo añadir restricciones binarias al ArcEngine
                    if len(constraint.variables) == 2 and constraint.variables[0] in temp_arc_engine.variables and constraint.variables[1] in temp_arc_engine.variables:
                        # Generar un nombre de relación único si no está en los metadatos
                        relation_name = f"fibration_constraint_{id(constraint.predicate)}_{id(self)}_{id(temp_arc_engine)}_{uuid.uuid4()}"
                        
                        # Adaptar el predicado para que acepte val1, val2, metadata
                        def adapted_predicate(val1, val2, meta, constraint=constraint):
                            temp_assignment = {constraint.variables[0]: val1, constraint.variables[1]: val2}
                            return constraint.predicate(temp_assignment)

                        # Registrar la función de predicado adaptada como una relación en el ArcEngine temporal
                        temp_arc_engine.register_relation(relation_name, adapted_predicate)
                        temp_arc_engine.add_constraint(constraint.variables[0], constraint.variables[1], relation_name, metadata=constraint.metadata)
                    # Las restricciones unarias y de mayor aridad no se manejan directamente con ArcEngine.add_constraint
                    # Su coherencia se verificará a través de la evaluación de la energía o el hacify posterior.

        if not temp_arc_engine.enforce_arc_consistency():
            return []

        coherent_values = list(temp_arc_engine.variables[variable].get_values())

        final_coherent_values = []
        for value in coherent_values:
            temp_assignment = base_assignment.copy()
            temp_assignment[variable] = value
            h_result = self.hacify(temp_assignment, strict=strict)
            if h_result.is_coherent:
                final_coherent_values.append(value)
        
        return final_coherent_values

    def get_statistics(self) -> Dict:
        return {
            "energy_thresholds": {level.name: threshold for level, threshold in self.energy_thresholds.items()}
        }

