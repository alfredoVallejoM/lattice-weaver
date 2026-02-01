import pytest
from unittest.mock import MagicMock

from lattice_weaver.fibration.solvers import FibrationSearchSolverEnhanced, FibrationSearchSolverAdaptive, HybridSearch, HybridSearchConfig, SearchStrategy
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine

@pytest.fixture
def mock_hierarchy():
    hierarchy = MagicMock(spec=ConstraintHierarchy)
    hierarchy.get_constraints_at_level.return_value = []
    hierarchy.get_constraints_involving.return_value = []
    return hierarchy

@pytest.fixture
def mock_landscape():
    landscape = MagicMock(spec=EnergyLandscapeOptimized)
    landscape.compute_energy.return_value.total_energy = 0.0
    return landscape

@pytest.fixture
def mock_arc_engine():
    arc_engine = MagicMock(spec=ArcEngine)
    arc_engine.tms = MagicMock()
    arc_engine.tms.record_decision.return_value = None
    arc_engine.tms.backtrack_to_decision.return_value = None
    arc_engine.tms.get_conflict_set.return_value = []
    return arc_engine

@pytest.fixture
def sample_variables():
    return ["x", "y", "z"]

@pytest.fixture
def sample_domains():
    return {"x": [1, 2], "y": [1, 2], "z": [1, 2]}

from lattice_weaver.fibration.hacification_engine import HacificationEngine

class TestAdvancedSolvers:

    @pytest.fixture
    def mock_hacification_engine(mock_arc_engine):
        h_engine = MagicMock(spec=HacificationEngine)
        h_engine.hacify.return_value = MagicMock(is_coherent=True, has_hard_violation=False, energy=MagicMock(total_energy=0.0), violated_constraints=[])
        h_engine.arc_engine = mock_arc_engine # Ensure it has an arc_engine attribute
        return h_engine

    def test_fibration_search_solver_enhanced_instantiation(self, mock_hierarchy, mock_landscape, mock_arc_engine, mock_hacification_engine, sample_variables, sample_domains):
        solver = FibrationSearchSolverEnhanced(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            arc_engine=mock_arc_engine,
            hacification_engine=mock_hacification_engine,
            variables=sample_variables,
            domains=sample_domains
        )
        assert solver is not None
        # Asegurarse de que el hacification_engine interno también reciba un mock de ArcEngine
        # que tenga el atributo 'tms' si es necesario.
        # En este caso, el HacificationEngine ya está siendo instanciado con mock_arc_engine
        # y el test ya no espera que sea un MagicMock, sino una instancia real.
        assert isinstance(solver.hacification_engine, HacificationEngine)

    def test_fibration_search_solver_adaptive_instantiation(self, mock_hierarchy, mock_landscape, mock_arc_engine, mock_hacification_engine, sample_variables, sample_domains):
        solver = FibrationSearchSolverAdaptive(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            arc_engine=mock_arc_engine,
            hacification_engine=mock_hacification_engine,
            variables=sample_variables,
            domains=sample_domains
        )
        assert solver is not None

    def test_hybrid_search_instantiation(self, mock_hierarchy, mock_landscape, sample_variables, sample_domains):
        config = HybridSearchConfig(strategy=SearchStrategy.HILL_CLIMBING)
        solver = HybridSearch(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            variables=sample_variables,
            domains=sample_domains,
            config=config
        )
        assert solver is not None

    def test_fibration_search_solver_enhanced_solve_no_crash(self, mock_hierarchy, mock_landscape, mock_arc_engine, mock_hacification_engine, sample_variables, sample_domains):
        solver = FibrationSearchSolverEnhanced(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            arc_engine=mock_arc_engine,
            hacification_engine=mock_hacification_engine,
            variables=sample_variables,
            domains=sample_domains,
            time_limit_seconds=0.1 # Short time limit to prevent long runs
        )
        solution = solver.solve()
        # We don't expect a solution with mocked components, just no crash
        assert solution is None or isinstance(solution, dict)

    def test_fibration_search_solver_adaptive_solve_no_crash(self, mock_hierarchy, mock_landscape, mock_arc_engine, mock_hacification_engine, sample_variables, sample_domains):
        solver = FibrationSearchSolverAdaptive(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            arc_engine=mock_arc_engine,
            hacification_engine=mock_hacification_engine,
            variables=sample_variables,
            domains=sample_domains,
            time_limit_seconds=0.1 # Short time limit
        )
        solution = solver.solve()
        assert solution is None or isinstance(solution, dict)

    def test_hybrid_search_solve_no_crash(self, mock_hierarchy, mock_landscape, sample_variables, sample_domains):
        config = HybridSearchConfig(strategy=SearchStrategy.HILL_CLIMBING, time_limit_seconds=0.1)
        solver = HybridSearch(
            hierarchy=mock_hierarchy,
            landscape=mock_landscape,
            variables=sample_variables,
            domains=sample_domains,
            config=config
        )
        solution = solver.search()
        assert solution is None or isinstance(solution, dict)

