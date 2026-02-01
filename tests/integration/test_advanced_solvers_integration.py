import pytest
from typing import Dict, List, Any

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized,
    HacificationEngine,
    HacificationResult
)
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.fibration.solvers.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
from lattice_weaver.fibration.solvers.fibration_search_solver_adaptive import FibrationSearchSolverAdaptive

# --- Fixtures para un problema de integración simple --- #

@pytest.fixture
def integration_hierarchy():
    hierarchy = ConstraintHierarchy()
    # Variables: A, B, C
    # Dominios: {A: [0, 1], B: [0, 1], C: [0, 1]}

    # Restricción HARD local: A != B
    hierarchy.add_local_constraint(
        "A", "B",
        lambda a, var_a="A", var_b="B": a[var_a] != a[var_b],
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "A_ne_B"}
    )
    # Restricción HARD local: B != C
    hierarchy.add_local_constraint(
        "B", "C",
        lambda a, var_b="B", var_c="C": a[var_b] != a[var_c],
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "B_ne_C"}
    )
    # Restricción SOFT global: Preferir A=0
    hierarchy.add_global_constraint(
        ["A"],
        lambda a, var_a="A": 1.0 if a[var_a] != 0 else 0.0,
        objective="minimize",
        weight=1.0,
        hardness=Hardness.SOFT,
        metadata={"name": "A_is_0"}
    )
    return hierarchy

@pytest.fixture
def integration_landscape(integration_hierarchy):
    return EnergyLandscapeOptimized(integration_hierarchy)

@pytest.fixture
def integration_arc_engine():
    return ArcEngine()

@pytest.fixture
def integration_hacification_engine(integration_hierarchy, integration_landscape, integration_arc_engine):
    return HacificationEngine(integration_hierarchy, integration_landscape, integration_arc_engine, use_arc_engine=True)

@pytest.fixture
def integration_variables():
    return ["A", "B", "C"]

@pytest.fixture
def integration_domains():
    return {"A": [0, 1], "B": [0, 1], "C": [0, 1]}

# --- Tests de Integración --- #

class TestAdvancedSolversIntegration:

    def test_enhanced_solver_finds_solution(self,
                                            integration_hierarchy,
                                            integration_landscape,
                                            integration_arc_engine,
                                            integration_hacification_engine,
                                            integration_variables,
                                            integration_domains):
        solver = FibrationSearchSolverEnhanced(
            hierarchy=integration_hierarchy,
            landscape=integration_landscape,
            arc_engine=integration_arc_engine,
            hacification_engine=integration_hacification_engine,
            variables=integration_variables,
            domains=integration_domains,
            time_limit_seconds=5.0 # Asegurar que no se quede colgado
        )
        solution = solver.solve()
        
        assert solution is not None
        assert solution["A"] != solution["B"]
        assert solution["B"] != solution["C"]
        # Verificar que la solución minimiza la restricción SOFT (A=0)
        assert solution["A"] == 0
        
        # Verificar que la solución es consistente
        h_result = integration_hacification_engine.hacify(solution, strict=True)
        assert h_result.is_coherent is True
        assert h_result.has_hard_violation is False
        assert h_result.energy.total_energy == 0.0 # Con A=0, la soft constraint es 0

    def test_adaptive_solver_finds_solution(self,
                                           integration_hierarchy,
                                           integration_landscape,
                                           integration_arc_engine,
                                           integration_hacification_engine,
                                           integration_variables,
                                           integration_domains):
        solver = FibrationSearchSolverAdaptive(
            hierarchy=integration_hierarchy,
            landscape=integration_landscape,
            arc_engine=integration_arc_engine,
            hacification_engine=integration_hacification_engine,
            variables=integration_variables,
            domains=integration_domains,
            time_limit_seconds=5.0 # Asegurar que no se quede colgado
        )
        solution = solver.solve()
        
        assert solution is not None
        assert solution["A"] != solution["B"]
        assert solution["B"] != solution["C"]
        # Verificar que la solución minimiza la restricción SOFT (A=0)
        assert solution["A"] == 0
        
        # Verificar que la solución es consistente
        h_result = integration_hacification_engine.hacify(solution, strict=True)
        assert h_result.is_coherent is True
        assert h_result.has_hard_violation is False
        assert h_result.energy.total_energy == 0.0 # Con A=0, la soft constraint es 0

    # TODO: Añadir tests para hybrid_search una vez que se refactorice para inyección de dependencias

