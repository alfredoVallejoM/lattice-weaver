import time
import json
import os
import logging
from typing import List, Dict, Any, Optional

from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolution
from lattice_weaver.core.csp_problem import CSP as CSPProblem
from lattice_weaver.problems.catalog import get_catalog, get_family
from .config import ExperimentConfig, SolverConfig, ProblemInstanceConfig

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Clase para ejecutar experimentos de benchmarking en problemas CSP.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.catalog = get_catalog()
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _run_single_experiment(self, problem_instance: CSPProblem, solver_config: SolverConfig) -> Dict[str, Any]:
        """
        Ejecuta un único experimento con un problema y un solver.
        """
        solver = CSPSolver(
            use_tms=solver_config.params.get("use_tms", False),
            parallel=solver_config.params.get("parallel", False),
            parallel_mode=solver_config.params.get("parallel_mode", "thread")
        )

        start_time = time.perf_counter()
        solutions = solver.solve(problem_instance, return_all=True, max_solutions=1)
        end_time = time.perf_counter()

        metrics = {
            "solver_name": solver_config.name,
            "time_taken": end_time - start_time,
            "solutions_found": len(solutions.solutions),
            "nodes_explored": solutions.nodes_explored,
            "backtracks": solutions.backtracks,
            "constraints_checked": solutions.constraints_checked,
            "initial_domain_size": sum(len(d) for d in problem_instance.domains.values()),
            "final_domain_size": sum(len(d.get_values()) for d in solver.csp_engine.variables.values()),
            "problem_family": self.config.problem_family,
            "problem_params": self.config.problem_params,
            "solver_params": solver_config.params,
            "solution_valid": False # Se verificará después
        }

        if solutions:
            # Validar la primera solución encontrada
            from ..problems.catalog import get_family
            problem_family_instance = get_family(self.config.problem_family)

            metrics["solution_valid"] = problem_family_instance.validate_solution(
                solutions.solutions[0].assignment, **self.config.problem_params
            )

        return metrics

    def run(self) -> List[Dict[str, Any]]:
        """
        Ejecuta todos los experimentos definidos en la configuración.
        """
        all_results = []
        problem_family_instance = get_family(self.config.problem_family)


        logger.info(f"Generando problema para {self.config.problem_family} con parámetros {self.config.problem_params}")
        # Generar el ArcEngine, que luego se envuelve en CSPProblem
        csp_instance = problem_family_instance.generate(**self.config.problem_params)
        # CSPProblem espera variables como List[str] y dominios como Dict[str, List[Any]]
        # Convertir csp_instance.variables (Dict[str, Domain]) a los formatos esperados
        csp_variables = list(csp_instance.variables.keys())
        csp_domains = {name: list(domain.get_values()) for name, domain in csp_instance.variables.items()}
        
        # CSPProblem espera constraints como List[Tuple[str, str, Any]]
        # Convertir csp_instance.constraints (Dict[str, Constraint]) a los formatos esperados
        # Nota: La relación en CSPProblem es solo un placeholder, el CSPSolver usa la relación registrada
        csp_constraints = [
            (c.var1, c.var2, c.relation_name) for c in csp_instance.constraints.values()
        ]

        problem_instance = CSPProblem(csp_variables, csp_domains, csp_constraints)


        for solver_name in self.config.solvers:
            solver_config = SolverConfig(name=solver_name)
            if solver_name == "default":
                # Usar configuración por defecto del CSPSolver
                pass
            elif solver_name == "tms_enabled":
                solver_config.params["use_tms"] = True
            elif solver_name == "parallel_thread":
                solver_config.params["parallel"] = True
                solver_config.params["parallel_mode"] = "thread"
            elif solver_name == "parallel_topological":
                solver_config.params["parallel"] = True
                solver_config.params["parallel_mode"] = "topological"
            else:
                logger.warning(f"Solver desconocido: {solver_name}. Usando configuración por defecto.")

            logger.info(f"Ejecutando {self.config.repetitions} repeticiones para solver {solver_config.name}")
            for i in range(self.config.repetitions):
                logger.debug(f"Repetición {i+1}/{self.config.repetitions}")
                result = self._run_single_experiment(problem_instance, solver_config)
                all_results.append(result)

        # Guardar resultados en un archivo JSON
        output_file = os.path.join(self.config.output_dir, f"experiment_{self.config.problem_family}_{int(time.time())}.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Resultados guardados en {output_file}")

        return all_results


# Actualizar __init__.py para el módulo experimentation

