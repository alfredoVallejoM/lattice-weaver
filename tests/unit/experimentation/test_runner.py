import pytest
import os
import json
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

from lattice_weaver.experimentation.config import ExperimentConfig, SolverConfig
from lattice_weaver.experimentation.runner import BenchmarkRunner
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import ProblemCatalog

# Mock ProblemFamily para pruebas
class MockProblemFamily(ProblemFamily):
    def __init__(self):
        super().__init__(
            name="mock_problem",
            description="A mock problem for testing"
        )

    def generate(self, **params) -> CSP:
        # Simula la generación de un CSP
        variables = {"V1", "V2"}
        domains = {"V1": frozenset({1, 2, 3}), "V2": frozenset({1, 2, 3})}
        constraints = [Constraint(scope=frozenset({"V1", "V2"}), relation=lambda x, y: x != y, name="neq_v1v2")]
        return CSP(variables=variables, domains=domains, constraints=constraints)

    def validate_solution(self, solution: Dict[str, Any], **params) -> bool:
        return True  # Siempre válida para el mock

    def get_metadata(self, **params) -> Dict[str, Any]:
        return {"n_variables": 2, "n_constraints": 1}

@pytest.fixture(scope="module")
def real_catalog():
    catalog = ProblemCatalog()
    # Asegurarse de que el catálogo esté limpio antes de registrar
    catalog.clear()
    catalog.register(MockProblemFamily())
    return catalog


@pytest.fixture(scope="function")
def cleanup_output_dir():
    output_dir = "./experiments_results"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)
    yield
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)

@patch("lattice_weaver.experimentation.runner.CSPSolver")
@patch("lattice_weaver.problems.catalog.get_catalog")
def test_benchmark_runner_run(mock_get_catalog, mock_csp_solver, real_catalog, cleanup_output_dir):
    mock_get_catalog.return_value = real_catalog

    # Configurar el mock del CSPSolver
    mock_solver_instance = MagicMock()
    # El método solve_all de CSPSolver devuelve una lista de soluciones (diccionarios)
    mock_solver_instance.solve_all.return_value = [{
        "V1": 1, "V2": 2
    }]
    # CSPSolutionStats es un dataclass, no un objeto con atributos directos en el mock
    # Necesitamos mockear el retorno de solve_all para que sea una lista de diccionarios
    # y luego el runner extraerá las estadísticas de un objeto CSPSolutionStats si se usa solve
    # Para este test, simplificamos y asumimos que solve_all devuelve directamente las soluciones
    # y las estadísticas se calculan o se mockean por separado si el runner las necesita.
    
    # Mockear el retorno de solve para que devuelva un CSPSolutionStats
    mock_stats = MagicMock(spec=CSPSolutionStats)
    mock_stats.solutions = [{
        "V1": 1, "V2": 2
    }]
    mock_stats.nodes_explored = 10
    mock_stats.backtracks = 5
    mock_stats.constraints_checked = 20
    mock_solver_instance.solve.return_value = mock_stats

    mock_csp_solver.return_value = mock_solver_instance

    config = ExperimentConfig(
        problem_family="mock_problem",
        problem_params={"size": 3},
        solvers=["default", "tms_enabled"],
        repetitions=2,
        output_dir="./experiments_results"
    )

    runner = BenchmarkRunner(config)
    results = runner.run()

    assert len(results) == 4  # 2 solvers * 2 repetitions
    assert os.path.exists("./experiments_results")
    assert len(os.listdir("./experiments_results")) == 1  # Un archivo JSON de resultados

    # Verificar algunas métricas
    for result in results:
        assert "time_taken" in result
        assert result["solutions_found"] == 1
        assert result["nodes_visited"] == 10
        assert result["backtracks"] == 5
        assert result["constraints_checked"] == 20
        assert result["problem_family"] == "mock_problem"
        assert result["solution_valid"] is True

    # Verificar que CSPSolver fue llamado con los parámetros correctos
    assert mock_csp_solver.call_count == 4
    # Primer solver: default
    assert mock_csp_solver.call_args_list[0].kwargs["use_tms"] is False
    assert mock_csp_solver.call_args_list[0].kwargs["parallel"] is False
    # Segundo solver: tms_enabled
    assert mock_csp_solver.call_args_list[2].kwargs["use_tms"] is True

    # Cargar y verificar el archivo JSON
    json_file = os.path.join(config.output_dir, os.listdir(config.output_dir)[0])
    with open(json_file, "r") as f:
        loaded_results = json.load(f)
    assert len(loaded_results) == 4
    assert loaded_results[0]["solver_name"] == "default"
    assert loaded_results[2]["solver_name"] == "tms_enabled"




# Registrar el MockProblemFamily en el catálogo global para que get_family lo encuentre
from lattice_weaver.problems.catalog import register_family
register_family(MockProblemFamily())

